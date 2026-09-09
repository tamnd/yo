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
use super::keyspec::{self, Begin, Find, KeySpec};
use super::table::{self, Spec};
use super::{
    DATABASES, Flow, Server, Session, acl, auth, backup, cpu, debug, multi, notify, persist,
};
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
///
/// [`super::backup`] writes it into the `redis-ver` aux field of the base file
/// it produces, so a server told to load one reads the same version out of the
/// file that a client reads off the connection.
pub(super) const REPORTED_VERSION: &str = "8.8.0";

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
    // Where `BACKUP` writes, under `dir`. Fixed here where a real server takes
    // it at startup, because nothing in this build reads it from a file.
    ("backupdirname", backup::DIR_NAME),
    ("databases", "16"),
    ("io-threads", "1"),
    ("proto-max-bulk-len", "536870912"),
    // How much of the command stream a master keeps for a replica that comes
    // back, which is a compiled in size here and happens to be the size Redis
    // ships with. Fixed rather than writable because the backlog is one buffer
    // that is allocated once and resizing it under a replica that is reading
    // out of it is a change of its own. There is a test below that this number
    // and `repl::BACKLOG_BYTES` are the same number.
    ("repl-backlog-size", "1048576"),
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

/// How much the server is allowed to keep on the file before it starts evicting.
///
/// The other half of the eviction inversion `14` section 4.1 describes, and the
/// only setting here that has no counterpart in Redis. `maxmemory` is a limit on
/// memory, and the right answer to a memory limit on a system with a file under
/// it is to move data to the file. Throwing data away is the right answer to a
/// limit on the file, and this is that limit.
///
/// Minus one is no limit and is the default, so a server that never sets this
/// grows until the disk is full and then refuses writes, which is what a
/// database does. Zero is a real setting and it means the file may hold nothing,
/// so migration cannot make room and eviction is all that is left, which is
/// Redis exactly and is the documented setting for a drop in cache.
const MAXSTORE: &str = "maxstore";

/// Where the server writes, which `BACKUP LIST` answers paths under.
///
/// On its own for a fourth reason: it is readable and not writable, and it is
/// not writable in a way of its own. Redis calls it a protected config, which
/// means `CONFIG SET dir` is refused with a sentence about protection rather
/// than about immutability unless the server was started with protected configs
/// enabled. That distinction is copied, because the two messages are what an
/// operator reads when a `CONFIG SET` does not take.
const DIR: &str = "dir";

/// What `SAVE` writes, under [`DIR`].
///
/// Protected in the same way and for a weaker version of the same reason: a
/// server whose file name moved under a running backup script leaves a file
/// nothing goes looking for. Redis protects it too, so `CONFIG SET dbfilename`
/// is refused there as well without protected configs turned on.
const DBFILENAME: &str = "dbfilename";

/// The password every connection is asked for, empty when none is.
///
/// Writable, and the one setting here whose value is a secret. It reads back in
/// the clear, which is what a real server does and is not an oversight of one:
/// an operator who can send `CONFIG GET` on this server can already read
/// everything in it.
const REQUIREPASS: &str = "requirepass";

/// How long a sealed backup is kept before it cleans itself up.
///
/// Seconds, and zero is the default and means it is kept until somebody says
/// `BACKUP CLEANUP`. Writable, since a backup taken by a script that then died
/// is exactly the thing this is for and setting it afterwards has to work.
const SEALED_TTL: &str = "backup-sealed-ttl";

/// The file the users are read from and written back to, empty when there is
/// none.
///
/// Immutable, which is Redis's rule for it and is the right one: an operator who
/// could point a running server at a different ACL file would have a way of
/// changing who may reach it that is invisible to everything watching the file
/// it was started with.
const ACLFILE: &str = "aclfile";

/// How many refusals `ACL LOG` keeps, and nought keeps none.
///
/// Writable, because the reason to change it is that something is happening
/// right now and the log is either too short to see it or long enough to be in
/// the way.
const ACLLOG_MAX_LEN: &str = "acllog-max-len";

/// Whether a new selector starts out allowed every channel.
///
/// Writable, and writing it changes nothing that already exists: it is read at
/// the moment a selector is made and never looked at again. Redis 6 behaved as
/// `allchannels` and Redis 7 changed the default to `resetchannels`, which is
/// what this setting is for, and yo starts where Redis 7 did.
const ACL_PUBSUB_DEFAULT: &str = "acl-pubsub-default";

/// The two words `acl-pubsub-default` is allowed to be, in Redis's order.
const CHANNEL_DEFAULTS: [&str; 2] = ["allchannels", "resetchannels"];

/// Whether a client's write is refused while this server follows a master.
///
/// Writable, and on by default, which is Redis's default and is the only safe
/// one: a write that lands on a replica is a write the master never hears about
/// and that the next full resync throws away. Turning it off is a real thing to
/// do and is what a cache in front of a slow master wants.
///
/// The `slave` spelling is the name this had before Redis renamed it and it
/// still answers to both, so this does too, the same way the size ladder answers
/// to `ziplist`. Two names, one setting.
const REPLICA_READ_ONLY: [&str; 2] = ["replica-read-only", "slave-read-only"];

/// The password and user the link to a master authenticates with.
///
/// Writable and both empty by default, which is a master that asks for nothing.
/// A user without a password is not a thing to send, so an empty `masterauth`
/// means the link sends no `AUTH` at all whatever `masteruser` says. They read
/// back in the clear for the same reason `requirepass` does.
const MASTERAUTH: &str = "masterauth";
/// The user half of [`MASTERAUTH`], for a master with an ACL rather than a
/// password.
const MASTERUSER: &str = "masteruser";

/// The two words a yes or no setting is allowed to be.
const BOOLS: [&str; 2] = ["yes", "no"];

/// The two cluster settings that really move, both of them a yes or a no.
///
/// `cluster-require-full-coverage` decides whether a node with a hole somewhere
/// in the cluster refuses everything or only the keys in the hole, and
/// `cluster-allow-reads-when-down` decides whether a node that has decided the
/// cluster is down still answers reads. Both take effect on the next command,
/// which is what an operator digging a cluster out of a hole wants.
const CLUSTER_COVERAGE: [&str; 2] = [
    "cluster-require-full-coverage",
    "cluster-allow-reads-when-down",
];

/// Whether this server is a cluster node, which is fixed for the life of the
/// process and is Redis's rule.
///
/// A server that could be turned into a cluster node while it was holding keys
/// would be a server whose keys were suddenly in slots it does not own, so
/// `CONFIG SET` refuses it and the only way to set it is at startup.
const CLUSTER_ENABLED: &str = "cluster-enabled";

/// Where a cluster node writes its table, which it was given at startup.
const CLUSTER_CONFIG_FILE: &str = "cluster-config-file";

/// Which classes of keyspace change are published, and on which two channels.
///
/// On its own for a fifth reason: it is the only setting whose value is neither
/// a number nor one of a fixed list of words, but a set of characters that reads
/// back in a different spelling from the one it was written in. `CONFIG SET
/// notify-keyspace-events KEA` reads back as `AKE`. See the `notify` module for
/// what each character means and why the order is what it is.
const NOTIFY: &str = "notify-keyspace-events";

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
///
/// Public because `yodb serve` takes the same limits on the command line that
/// `CONFIG SET` takes at runtime, and a server that accepts `100mb` from one and
/// not the other, or reads it as a different number, is a server that gets
/// misconfigured. One parser, one answer.
#[must_use]
pub fn parse_memory(value: &[u8]) -> Option<u64> {
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
    server: &Server,
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
            // A RESP2 connection in subscribe mode is answered a two element
            // array with `pong` in front, so that everything reaching a
            // subscribed client on RESP2 has the same shape. The one place a
            // command in this file cares what the connection has subscribed to.
            if super::pubsub::ping(session, args, out) {
                return Ok(Flow::Continue);
            }
            if args.len() == 2 {
                out.bulk(args.get(1));
            } else {
                out.simple(b"PONG");
            }
        }
        "echo" => out.bulk(args.get(1)),
        "acl" => acl::execute(server, session, args, out)?,
        "auth" => auth::execute(server, session, args, out)?,
        "debug" => debug::execute(server, session, args, out)?,
        "memory" => super::memory::execute(server, session, args, out)?,
        "replconf" => super::repl::replconf(server, session, args, out)?,
        "psync" | "sync" => super::repl::psync(server, session, args, out)?,
        "replicaof" | "slaveof" => super::follow::replicaof(server, args, out)?,
        "failover" => super::failover::execute(server, args, out)?,
        "cluster" => super::cluster::execute(server, session.db, args, out)?,
        // The three connection commands cluster mode adds. `ASKING` is the one
        // that does anything: it says the next command is allowed into a slot
        // this node is receiving and does not own yet, which is how a client
        // follows an `ASK` it was told.
        //
        // `READONLY` and `READWRITE` say whether this connection will take
        // reads from a replica rather than being redirected to the master, and
        // on a node that is nobody's replica, which is every node here until the
        // bus is in, both of them are an `OK` and nothing else. That is what a
        // real master answers too, so a client library that sends `READONLY` on
        // connect gets the same answer from both.
        "asking" => {
            if !server.cluster_enabled() {
                return Err(super::cluster::disabled());
            }
            session.ask_next();
            out.ok();
        }
        "readonly" | "readwrite" => {
            if !server.cluster_enabled() {
                return Err(super::cluster::disabled());
            }
            out.ok();
        }
        "hello" => hello(server, session, args, out)?,
        "select" => {
            // A cluster has one database and the slots are how it is cut up, so
            // moving to another one would be moving to a database no slot points
            // at. Database nought is still allowed, because a client library
            // that sends SELECT 0 on connect is asking for where it already is.
            let n = args.int(1)?;
            if server.cluster_enabled() && n != 0 {
                return Err(Error::new(
                    Code::Invalid,
                    "SELECT is not allowed in cluster mode",
                ));
            }
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
            //
            // The transaction and the watches go first because letting go of a
            // watch is a change to the server and not to the connection, so
            // clearing the list here without saying so would leave rows on the
            // server that nobody is watching. `RESET` inside `MULTI` answers
            // `+RESET` and leaves no transaction, which is why it is one of the
            // six commands a transaction does not queue.
            // The subscriptions go with them, and for the same reason: a
            // subscription is a row on the server naming this connection, so
            // clearing the connection's list alone would leave the server
            // delivering into a slot that is not listening any more.
            // And the monitor with them, which is the one way out of monitor
            // mode short of closing the socket. `RESET` still answers `+RESET`
            // on a connection that was one, because the reply belongs to the
            // client the connection has just gone back to being.
            multi::release(server, session);
            super::pubsub::release(server, session);
            if session.monitoring() {
                server.watch_no_more(session.row());
            }
            session.reset();
            // Including the password, which is what `RESET` means by putting
            // the connection back the way it was accepted: on a server with a
            // password the client has to send `AUTH` again, and on a server
            // without one it never had to.
            session.admit(!server.guarded());
            out.set_proto(Proto::Resp2);
            out.simple(b"RESET");
        }
        // The reply goes out before the socket closes, which is why this is a
        // flow answer and not something the body does to the connection.
        "quit" => {
            out.ok();
            return Ok(Flow::Close);
        }
        // Every command on the server, from here on, on this connection. The
        // reply is `OK` once and nothing after it, and a connection that sends
        // it twice is answered nothing at all the second time, which is a real
        // server's behaviour and not an oversight of one.
        "monitor" => {
            // A transaction replaying this has been promised a reply for every
            // command it queued, and a connection that has turned into a feed
            // cannot give one. A real server refuses it in the same words.
            if session.running() {
                return Err(Error::new(
                    Code::Invalid,
                    "MONITOR isn't allowed for DENY BLOCKING client",
                ));
            }
            if server.watch_all(session.row()) {
                out.ok();
            }
        }
        "client" => return super::client::execute(server, session, spec, args, out),
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
            for db in &server.dbs {
                db.clear();
            }
            server.search.lock().clear();
            server.cursors.lock().wipe();
            out.ok();
        }
        // The search indexes go too, and they go whichever database this is.
        // An index that only ever followed keys on database zero is dropped by
        // a `FLUSHDB` on database nine, which is measured against a real server
        // rather than reasoned about: the module hangs its callback on the
        // flush event without looking at which database flushed.
        "flushdb" => {
            flush_mode(args)?;
            server.dbs[session.db].clear();
            server.search.lock().clear();
            server.cursors.lock().wipe();
            out.ok();
        }
        // Two databases change places and no key moves. What is in the stripes
        // is exchanged and the databases stay where they are, so this costs two
        // pointer sized writes per stripe whatever is in either of them, which
        // is what makes `SWAPDB` fast and dangerous at the same time.
        //
        // No connection is told. A client on database zero is still on database
        // zero and is now looking at what used to be database one, which is the
        // whole point of the command and is why Redis calls it dangerous. A
        // client parked in `BLPOP` remembers the database index it blocked on
        // and not the database, so it wakes up against the swapped in one, which
        // is Redis's behaviour and falls out of the index being what is stored.
        "swapdb" => {
            if server.cluster_enabled() {
                return Err(Error::new(
                    Code::Invalid,
                    "SWAPDB is not allowed in cluster mode",
                ));
            }
            let first = db_index(args.get(1), "invalid first DB index")?;
            let second = db_index(args.get(2), "invalid second DB index")?;
            server.striped(first).swap_with(server.striped(second));
            out.ok();
        }
        "time" => time(out),
        // The four commands about writing the dataset to a file and the one
        // about who this server is, all in the `persist` module because a client
        // asks them together.
        "save" | "bgsave" | "bgrewriteaof" | "lastsave" | "role" => {
            persist::execute(server, session, spec, args, out)?;
        }
        "backup" => backup::execute(server, args, out)?,
        "shutdown" => return shutdown(server, args),
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

// ---------------------------------------------------------------- SHUTDOWN

/// `SHUTDOWN [NOSAVE | SAVE] [NOW] [FORCE] [ABORT]`.
///
/// On success this writes nothing at all and the connection closes under the
/// client, which is what a server that has stopped looks like from the outside
/// and is what every client library already expects. There is no `OK`, because
/// an `OK` would be a promise made by a process that is about to not exist.
///
/// `SAVE` writes the file [`persist`] writes, and it is the only word here that
/// does anything. `NOSAVE` is the default rather than an instruction, which is
/// the same answer `save` gets from `CONFIG GET`: this server has no save points
/// and never will, because what durability there is belongs to the file
/// underneath and is already on disk by the time a command returns. So there is
/// nothing for `NOSAVE` to skip and the file `SAVE` asks for is an export
/// somebody wants a copy of on the way down. `NOW` and `FORCE` are about not
/// waiting for replicas and about going anyway when a save failed, and neither
/// has anything to wait for or to fail here.
///
/// # Errors
///
/// [`Code::Invalid`] for a word that is not one of the five, for `SAVE` and
/// `NOSAVE` in the same call, and for `ABORT` alongside any other flag, all of
/// which is what 8.10.1 says. `ABORT` on its own gets Redis's message for a
/// cancel with nothing to cancel, and here that is not a state that can be
/// reached rather than one that happens to be empty: a shutdown is decided and
/// done inside one turn of the loop, so there is never a window in which one is
/// in progress and a second client could call it off.
fn shutdown(server: &Server, args: Args<'_>) -> Result<Flow> {
    let (mut save, mut nosave, mut abort, mut other) = (false, false, false, false);
    for at in 1..args.len() {
        let arg = args.get(at);
        match () {
            () if is(arg, b"save") => save = true,
            () if is(arg, b"nosave") => nosave = true,
            () if is(arg, b"abort") => abort = true,
            () if is(arg, b"now") || is(arg, b"force") => other = true,
            () => return Err(args::syntax()),
        }
    }
    // Repeating one is fine and contradicting yourself is not, and `ABORT` says
    // to do nothing so it cannot be combined with a word about how to do it.
    if (save && nosave) || (abort && (save || nosave || other)) {
        return Err(args::syntax());
    }
    if abort {
        return Err(Error::new(Code::Invalid, "No shutdown in progress."));
    }
    if save {
        persist::on_shutdown(server);
    }
    server.stop();
    // Closing is what stops anything the client pipelined behind this from
    // being answered by a server that is on its way out.
    Ok(Flow::Close)
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
///
/// The order of the three things that can go wrong here is the reference's and
/// is worth writing down, because it is not the order they appear in. The
/// protocol version is read and refused first, so `HELLO 9 AUTH default right`
/// on a connection that has not authenticated is a `NOPROTO` and leaves the
/// connection unauthenticated. The `AUTH` option is applied next, so a wrong
/// password is a `WRONGPASS` and the protocol stays where it was. Only then does
/// the connection have to be authenticated at all, which is what makes a bare
/// `HELLO` on a server with a password a `NOAUTH` rather than a greeting.
fn hello(server: &Server, session: &mut Session, args: Args<'_>, out: &mut Out) -> Result<()> {
    // The version this call agreed on, if it named one. Held rather than applied
    // where it is read, because the reply buffer must not change protocol until
    // the password below has been asked for and answered.
    let mut agreed = None;
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
                if !acl::authenticate(server, session, args.get(i + 1), args.get(i + 2), args, out)
                {
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
        agreed = Some(proto);
    }

    if server.guarded() && !session.authenticated() {
        // Its own sentence rather than the one every other command gets, because
        // a client that speaks RESP3 has to send `HELLO` before it can send
        // `AUTH` and would otherwise be told to do the thing it is doing.
        out.error(auth::HELLO_NOAUTH.as_bytes());
        return Ok(());
    }
    // The reply is written in the protocol that was just agreed, not the one the
    // request arrived in.
    if let Some(proto) = agreed {
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
        getkeys(args, out, false)?;
    } else if is(sub, b"GETKEYSANDFLAGS") {
        getkeys(args, out, true)?;
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

/// `COMMAND GETKEYS <full command>` and `COMMAND GETKEYSANDFLAGS <full command>`.
///
/// This is how a cluster aware client routes a command it does not have a rule
/// for, so a wrong answer here is a client that sends a write to the wrong
/// node. The answer comes off the key specs, which is the same place the ACL
/// reads, so the two can never drift apart.
///
/// The three errors are the reference's own and they mean different things. A
/// name nobody registered is one, a command that never takes a key whatever it
/// is sent is another, and a command that does take keys and was handed
/// arguments the specs cannot resolve is the third. Only the last is about what
/// was actually typed.
fn getkeys(args: Args<'_>, out: &mut Out, flags: bool) -> Result<()> {
    let sub = if flags { "getkeysandflags" } else { "getkeys" };
    if args.len() < 3 {
        return Err(args::wrong_arity_sub("command", sub));
    }
    let inner = args.get(2);
    let spec = table::lookup(inner)
        .ok_or_else(|| Error::new(Code::Unsupported, "Invalid command specified"))?;
    if !keyspec::takes_keys(spec, args, 2) {
        return Err(Error::new(
            Code::Invalid,
            "The command has no key arguments",
        ));
    }
    let argc = args.len() - 2;
    if !table::arity_ok(spec, argc) {
        return Err(Error::new(
            Code::Invalid,
            "Invalid number of arguments specified for command",
        ));
    }
    // Three specs at most a command and one run each, so the answer is worked
    // out into a fixed array rather than a list that grows. A run is a first
    // argument and a count, so a hundred keys behind a count is still one of
    // these.
    let mut runs = [None; 4];
    let mut at = 0;
    let whole = keyspec::find(spec, args, 2, &mut |run| {
        if at < runs.len() {
            runs[at] = Some(run);
            at += 1;
        }
    });
    let found: usize = runs.iter().flatten().map(|r| r.count).sum();
    // A command that resolves to nothing is a syntax error, unless it is one of
    // the six that may honestly have no keys, which is the script family: `EVAL
    // body 0` is an ordinary thing to write and answers an empty list.
    if (!whole || found == 0) && !spec.flags.contains(&"no_mandatory_keys") {
        return Err(Error::new(
            Code::Invalid,
            "Invalid arguments specified for command",
        ));
    }
    let found = if whole { found } else { 0 };
    out.array(found);
    if found == 0 {
        return Ok(());
    }
    for run in runs.iter().flatten() {
        for i in 0..run.count {
            let key = args.get(run.first + i * run.step);
            if flags {
                out.array(2);
                out.bulk(key);
                out.set(run.flags.len());
                for f in run.flags {
                    out.simple(f.as_bytes());
                }
            } else {
                out.bulk(key);
            }
        }
    }
    Ok(())
}

/// One command, in the ten field shape `COMMAND INFO` has had since 7.0.
///
/// The tips and the subcommands are still empty, which is what is left of
/// divergence D-13. The key specs are not: they say where the keys are for
/// everything in this table, including the commands the triple above them
/// cannot describe.
///
/// Five of the ten fields are sets rather than arrays, which only shows on
/// RESP3 and shows there on every command. A set is what the reference sends
/// for all five, and it is the honest type for them: nothing in a flag list or
/// an acl category list is ordered or repeated.
fn write_spec(out: &mut Out, spec: &Spec) {
    out.array(10);
    out.bulk(spec.name.as_bytes());
    out.int(i64::from(spec.arity));
    out.set(spec.flags.len());
    for f in spec.flags {
        out.simple(f.as_bytes());
    }
    out.int(i64::from(spec.first_key));
    out.int(i64::from(spec.last_key));
    out.int(i64::from(spec.step));
    out.set(spec.acl.len());
    for a in spec.acl {
        out.simple(a.as_bytes());
    }
    out.set(0);
    out.set(spec.keys.len());
    for key in spec.keys {
        write_key_spec(out, key);
    }
    out.set(0);
}

/// One key spec, as the map `COMMAND INFO` reports it.
///
/// The notes come first and only when there are any, which is why the map is
/// three long or four rather than always four.
fn write_key_spec(out: &mut Out, key: &KeySpec) {
    out.map(if key.notes.is_empty() { 3 } else { 4 });
    if !key.notes.is_empty() {
        out.bulk(b"notes");
        out.bulk(key.notes.as_bytes());
    }
    out.bulk(b"flags");
    out.set(key.flags.len());
    for f in key.flags {
        out.simple(f.as_bytes());
    }
    out.bulk(b"begin_search");
    out.map(2);
    out.bulk(b"type");
    match key.begin {
        Begin::At(index) => {
            out.bulk(b"index");
            out.bulk(b"spec");
            out.map(1);
            out.bulk(b"index");
            out.int(i64::from(index));
        }
        Begin::After(word, from) => {
            out.bulk(b"keyword");
            out.bulk(b"spec");
            out.map(2);
            out.bulk(b"keyword");
            out.bulk(word);
            out.bulk(b"startfrom");
            out.int(i64::from(from));
        }
        Begin::Unknown => {
            out.bulk(b"unknown");
            out.bulk(b"spec");
            out.map(0);
        }
    }
    out.bulk(b"find_keys");
    out.map(2);
    out.bulk(b"type");
    match key.find {
        Find::Range { last, step, limit } => {
            out.bulk(b"range");
            out.bulk(b"spec");
            out.map(3);
            out.bulk(b"lastkey");
            out.int(i64::from(last));
            out.bulk(b"keystep");
            out.int(i64::from(step));
            out.bulk(b"limit");
            out.int(i64::from(limit));
        }
        Find::Counted { count, first, step } => {
            out.bulk(b"keynum");
            out.bulk(b"spec");
            out.map(3);
            out.bulk(b"keynumidx");
            out.int(i64::from(count));
            out.bulk(b"firstkey");
            out.int(i64::from(first));
            out.bulk(b"keystep");
            out.int(i64::from(step));
        }
        Find::Unknown => {
            out.bulk(b"unknown");
            out.bulk(b"spec");
            out.map(0);
        }
    }
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
fn config(server: &Server, args: Args<'_>, out: &mut Out) -> Result<()> {
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
        let store = wanted(MAXSTORE);
        let where_ = wanted(DIR);
        let file = wanted(DBFILENAME);
        let pass = wanted(REQUIREPASS);
        let ttl = wanted(SEALED_TTL);
        let events = wanted(NOTIFY);
        let acls = wanted(ACLFILE);
        let logged = wanted(ACLLOG_MAX_LEN);
        let channels = wanted(ACL_PUBSUB_DEFAULT);
        // Both spellings are two settings by the same rule the ladder follows,
        // so `CONFIG GET *read-only*` sends the replica name and the slave name
        // and the same word under both.
        let readonly = REPLICA_READ_ONLY.map(wanted);
        let mauth = wanted(MASTERAUTH);
        let muser = wanted(MASTERUSER);
        // The four cluster settings, which are there on every server and not
        // only on a node, the same way a real server answers them: a tool asking
        // `CONFIG GET cluster-enabled` wants a no rather than nothing back.
        let coverage = CLUSTER_COVERAGE.map(wanted);
        let clustered = wanted(CLUSTER_ENABLED);
        let nodes_file = wanted(CLUSTER_CONFIG_FILE);
        out.map(
            fixed.clone().count()
                + ladder.clone().count()
                + usize::from(policy)
                + usize::from(limit)
                + usize::from(store)
                + usize::from(where_)
                + usize::from(file)
                + usize::from(pass)
                + usize::from(ttl)
                + usize::from(events)
                + usize::from(acls)
                + usize::from(logged)
                + usize::from(channels)
                + usize::from(readonly[0])
                + usize::from(readonly[1])
                + usize::from(mauth)
                + usize::from(muser)
                + usize::from(coverage[0])
                + usize::from(coverage[1])
                + usize::from(clustered)
                + usize::from(nodes_file),
        );
        for (k, v) in fixed {
            out.bulk(k.as_bytes());
            out.bulk(v.as_bytes());
        }
        for (k, knob) in ladder {
            out.bulk(k.as_bytes());
            out.bulk_int(read_knob(&server.settings(), *knob) as i64);
        }
        if policy {
            out.bulk(MAXMEMORY_POLICY.as_bytes());
            out.bulk(server.settings().policy().name().as_bytes());
        }
        if limit {
            // Back as a plain number of bytes whatever the client typed to set
            // it, which is what a real server does: `CONFIG SET maxmemory 1gb`
            // reads back as 1073741824.
            out.bulk(MAXMEMORY.as_bytes());
            out.bulk_int(server.maxmemory() as i64);
        }
        if store {
            // Minus one for no limit, and a plain number of bytes otherwise.
            // Zero cannot mean no limit here the way it does for `maxmemory`,
            // because zero is the setting that says the file holds nothing.
            out.bulk(MAXSTORE.as_bytes());
            out.bulk_int(server.maxstore().map_or(-1, |n| n as i64));
        }
        if where_ {
            // Absolute, which is what a real server answers too: it resolves the
            // directory at startup and reports the resolved one, so a client can
            // tell where the files are without knowing where the process was
            // launched from.
            out.bulk(DIR.as_bytes());
            yo_alloc::allow(|| out.bulk(server.dir().to_string_lossy().as_bytes()));
        }
        if file {
            // The name on its own and not the path, which is how a real server
            // answers it too: the two settings are joined by whoever reads them.
            out.bulk(DBFILENAME.as_bytes());
            out.bulk(persist::FILE.as_bytes());
        }
        if pass {
            out.bulk(REQUIREPASS.as_bytes());
            server.with_password(|p| out.bulk(p));
        }
        if ttl {
            out.bulk(SEALED_TTL.as_bytes());
            out.bulk_int(server.backup().ttl() as i64);
        }
        if events {
            // The flags and not the string that set them, which is what a real
            // server answers too and is why the parser has a formatter next to
            // it rather than the text being kept.
            out.bulk(NOTIFY.as_bytes());
            let (buf, len) = notify::format(server.notify_flags());
            out.bulk(&buf[..len]);
        }
        if acls {
            // Exactly what the server was started with, which for nearly every
            // server is nothing at all. Not resolved to an absolute path the way
            // `dir` is, because a real server answers what it was given here.
            out.bulk(ACLFILE.as_bytes());
            yo_alloc::allow(|| {
                out.bulk(
                    server
                        .aclfile()
                        .map(|p| p.to_string_lossy())
                        .unwrap_or_default()
                        .as_bytes(),
                );
            });
        }
        if logged {
            out.bulk(ACLLOG_MAX_LEN.as_bytes());
            out.bulk_int(server.acl_log().max_len() as i64);
        }
        if channels {
            out.bulk(ACL_PUBSUB_DEFAULT.as_bytes());
            out.bulk(CHANNEL_DEFAULTS[usize::from(!server.users().open_channels())].as_bytes());
        }
        for (name, asked) in REPLICA_READ_ONLY.iter().zip(readonly) {
            if asked {
                out.bulk(name.as_bytes());
                out.bulk(BOOLS[usize::from(!server.replica_read_only_setting())].as_bytes());
            }
        }
        if coverage[0] {
            out.bulk(CLUSTER_COVERAGE[0].as_bytes());
            out.bulk(BOOLS[usize::from(!server.cluster_full_coverage())].as_bytes());
        }
        if coverage[1] {
            out.bulk(CLUSTER_COVERAGE[1].as_bytes());
            out.bulk(BOOLS[usize::from(!server.cluster_reads_when_down())].as_bytes());
        }
        if clustered {
            out.bulk(CLUSTER_ENABLED.as_bytes());
            out.bulk(BOOLS[usize::from(!server.cluster_enabled())].as_bytes());
        }
        if nodes_file {
            out.bulk(CLUSTER_CONFIG_FILE.as_bytes());
            yo_alloc::allow(|| out.bulk(server.cluster_file().as_bytes()));
        }
        if mauth || muser {
            server.with_master_auth(|user, pass| {
                if mauth {
                    out.bulk(MASTERAUTH.as_bytes());
                    out.bulk(pass);
                }
                if muser {
                    out.bulk(MASTERUSER.as_bytes());
                    out.bulk(user);
                }
            });
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
        let mut store = None;
        let mut ttl = None;
        let mut events = None;
        let mut password = None;
        let mut logged = None;
        let mut channels = None;
        let mut readonly = None;
        let mut coverage: [Option<bool>; 2] = [None, None];
        let mut mauth = None;
        let mut muser = None;
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
            if is(name, MAXSTORE.as_bytes()) {
                // `-1` before the memory parser sees it, because that parser
                // refuses a sign and should keep refusing one: `maxmemory -1`
                // is not a very large number and never was.
                let parsed = if value == b"-1" {
                    Some(None)
                } else {
                    parse_memory(value).map(Some)
                };
                let Some(bytes) = parsed else {
                    return Err(Error::fmt(
                        Code::Invalid,
                        format_args!(
                            "CONFIG SET failed (possibly related to argument '{MAXSTORE}') - argument must be a memory value or -1"
                        ),
                    ));
                };
                store = Some(bytes);
                continue;
            }
            if is(name, MAXMEMORY_POLICY.as_bytes()) {
                // Named twice in one command, the last one wins, which is the
                // same rule the ladder settings follow here and is not what a
                // real server does with a setting repeated in a single `CONFIG
                // SET`. It refuses the command instead, which is D-138.
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
            // Refused whatever the value is, including the one they are already
            // set to, which is the one place a setting here does not take the
            // write that changes nothing. That is the reference's answer: a
            // protected config is refused before anybody looks at what was
            // asked for.
            if let Some(protected) = [DIR, DBFILENAME]
                .into_iter()
                .find(|p| is(name, p.as_bytes()))
            {
                return Err(Error::fmt(
                    Code::Unsupported,
                    format_args!(
                        "CONFIG SET failed (possibly related to argument '{protected}') - can't set protected config"
                    ),
                ));
            }
            // Refused whatever the value is, including the one it is already
            // set to, which is how a real server answers every immutable
            // config: the check is on the name and never reaches the value.
            // The names in `SETTINGS` take the value that changes nothing,
            // which is a difference and is registered as one.
            if is(name, ACLFILE.as_bytes()) {
                return Err(Error::fmt(
                    Code::Unsupported,
                    format_args!(
                        "CONFIG SET failed (possibly related to argument '{ACLFILE}') - can't set immutable config"
                    ),
                ));
            }
            if is(name, ACLLOG_MAX_LEN.as_bytes()) {
                let Some(n) = parse_i64(value).filter(|&n| n >= 0) else {
                    return Err(bad_setting(ACLLOG_MAX_LEN, parse_i64(value).is_some()));
                };
                logged = Some(n as u64);
                continue;
            }
            if is(name, ACL_PUBSUB_DEFAULT.as_bytes()) {
                let Some(at) = CHANNEL_DEFAULTS
                    .iter()
                    .position(|w| is(value, w.as_bytes()))
                else {
                    return Err(Error::fmt(
                        Code::Invalid,
                        format_args!(
                            "CONFIG SET failed (possibly related to argument '{ACL_PUBSUB_DEFAULT}') - argument(s) must be one of the following: {}, {}",
                            CHANNEL_DEFAULTS[0], CHANNEL_DEFAULTS[1]
                        ),
                    ));
                };
                channels = Some(at == 0);
                continue;
            }
            if is(name, REQUIREPASS.as_bytes()) {
                // Anything at all is a password, including an empty one, which
                // is how a password is taken off again. There is nothing to
                // refuse here and a real server refuses nothing either.
                password = Some(value);
                continue;
            }
            if let Some(spelling) = REPLICA_READ_ONLY.iter().find(|k| is(name, k.as_bytes())) {
                let Some(at) = BOOLS.iter().position(|w| is(value, w.as_bytes())) else {
                    return Err(Error::fmt(
                        Code::Invalid,
                        format_args!(
                            "CONFIG SET failed (possibly related to argument '{spelling}') - argument must be 'yes' or 'no'"
                        ),
                    ));
                };
                readonly = Some(at == 0);
                continue;
            }
            if let Some(at) = CLUSTER_COVERAGE.iter().position(|k| is(name, k.as_bytes())) {
                let Some(word) = BOOLS.iter().position(|w| is(value, w.as_bytes())) else {
                    return Err(Error::fmt(
                        Code::Invalid,
                        format_args!(
                            "CONFIG SET failed (possibly related to argument '{}') - argument must be 'yes' or 'no'",
                            CLUSTER_COVERAGE[at]
                        ),
                    ));
                };
                coverage[at] = Some(word == 0);
                continue;
            }
            if is(name, CLUSTER_ENABLED.as_bytes()) || is(name, CLUSTER_CONFIG_FILE.as_bytes()) {
                return Err(yo_alloc::allow(|| {
                    Error::fmt(
                        Code::Unsupported,
                        format_args!(
                            "CONFIG SET failed (possibly related to argument '{}') - can't set immutable config",
                            String::from_utf8_lossy(name).to_lowercase()
                        ),
                    )
                }));
            }
            if is(name, MASTERAUTH.as_bytes()) {
                // Anything at all, including nothing, which is how the password
                // is taken off again. It is read at the next dial rather than
                // now, so setting it on a replica whose link is already up takes
                // effect the next time that link breaks and comes back.
                mauth = Some(value);
                continue;
            }
            if is(name, MASTERUSER.as_bytes()) {
                muser = Some(value);
                continue;
            }
            if is(name, NOTIFY.as_bytes()) {
                // The only setting here whose error names what was wrong with
                // the value rather than what the value should have been, and it
                // quotes the accepted characters in the reference's order.
                let Some(flags) = notify::parse(value) else {
                    return Err(Error::fmt(
                        Code::Invalid,
                        format_args!(
                            "CONFIG SET failed (possibly related to argument '{NOTIFY}') - Invalid event class character. Use '{}'.",
                            notify::ACCEPTED
                        ),
                    ));
                };
                events = Some(flags);
                continue;
            }
            if is(name, SEALED_TTL.as_bytes()) {
                let Some(n) = parse_i64(value).filter(|&n| n >= 0) else {
                    return Err(bad_setting(SEALED_TTL, parse_i64(value).is_some()));
                };
                ttl = Some(n as u64);
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
        // Every stripe of every database, because these are one server wide
        // number in Redis and the fact that a `Keyspace` carries its own copy is
        // ours and not the client's problem. A stripe that missed one would put
        // a key in a different shape from the same key on the stripe next to it,
        // which `OBJECT ENCODING` would then answer differently for depending on
        // where the key happened to land.
        // The whole database is held while its stripes are set rather than one
        // stripe at a time, for the same reason they all get the same number: a
        // client that read `OBJECT ENCODING` in the middle of a half done change
        // would be told two different things about two keys depending on nothing
        // it can see.
        for (knob, n) in writes.iter().flatten() {
            for at in 0..DATABASES {
                let db = server.striped(at);
                let mut held = db.hold_many(0..db.width());
                for i in 0..db.width() {
                    write_knob(held.stripe_mut(i), *knob, *n);
                }
            }
        }
        if let Some(p) = policy {
            for at in 0..DATABASES {
                let db = server.striped(at);
                let mut held = db.hold_many(0..db.width());
                for i in 0..db.width() {
                    held.stripe_mut(i).set_policy(p);
                }
            }
        }
        if let Some(seconds) = ttl {
            server.backup().set_ttl(seconds);
        }
        if let Some(flags) = events {
            server.set_notify_flags(flags);
        }
        if let Some(n) = logged {
            server.acl_log().set_max_len(n);
        }
        if let Some(open) = channels {
            server.users().set_open_channels(open);
        }
        if let Some(yes) = readonly {
            server.set_replica_read_only(yes);
        }
        // One write of the pair whichever of the two was named, for the same
        // reason the master credentials are written together: they are read as a
        // pair and a set of one has to leave the other where it was.
        if coverage[0].is_some() || coverage[1].is_some() {
            server.set_cluster_coverage(
                coverage[0].unwrap_or_else(|| server.cluster_full_coverage()),
                coverage[1].unwrap_or_else(|| server.cluster_reads_when_down()),
            );
        }
        // One write of the pair whichever of the two was named, because they
        // live together and a set of one has to leave the other where it was.
        if mauth.is_some() || muser.is_some() {
            let (user, pass) =
                yo_alloc::allow(|| server.with_master_auth(|u, p| (u.to_vec(), p.to_vec())));
            server.master_auth(muser.unwrap_or(&user), mauth.unwrap_or(&pass));
        }
        if let Some(value) = password {
            // The connections that are already open are left where they are,
            // including the one that sent this. See the `auth` module for why
            // that is the reference's rule and not an accident of it.
            server.set_password(value);
        }
        // Last, so that a `CONFIG SET maxmemory 1mb maxmemory-policy allkeys-lru`
        // has the policy in place before the limit that will act on it. The two
        // in the other order would run the first eviction under whatever the
        // policy used to be, which for a fresh server is `noeviction` and would
        // refuse the next write instead of making room for it.
        if let Some(bytes) = store {
            server.set_maxstore(bytes);
        }
        if let Some(bytes) = limit {
            server.set_maxmemory(bytes);
        }
        out.ok();
    } else if is(sub, b"RESETSTAT") {
        server.reset_stats();
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
    // Redis keeps two lists: the sections a bare `INFO` hands back, and the ones
    // that have to be asked for by name or by `all`. `commandstats` is in the
    // second, along with `latencystats` and `errorstats`, because they grow with
    // the number of distinct commands a server has seen and a monitoring tool
    // polling `INFO` every second does not want them.
    //
    // `unit/info-command` is exactly this distinction written down: it asks for
    // `INFO default` and insists `rejected_calls` is not in the answer, then
    // asks for `INFO all` and insists that it is.
    let named = |section: &str| (1..args.len()).any(|i| is(args.get(i), section.as_bytes()));
    let everything = (1..args.len()).any(|i| {
        let a = args.get(i);
        is(a, b"all") || is(a, b"everything")
    });
    let by_default = args.len() == 1 || (1..args.len()).any(|i| is(args.get(i), b"default"));
    let want = |section: &str| by_default || everything || named(section);
    let extra = |section: &str| everything || named(section);
    // One string, built once and written once. It allocates, which is allowed
    // here and nowhere near the commands that count: `INFO` is a monitoring
    // call and it is not on the path M2 is measured on.
    let text = yo_alloc::allow(|| {
        let mut s = String::with_capacity(1024);
        if want("server") {
            let _ = write!(
                s,
                "# Server\r\nredis_version:{REPORTED_VERSION}\r\nyo_version:{}\r\n\
                 redis_mode:{}\r\narch_bits:{}\r\nprocess_id:0\r\n\
                 run_id:0000000000000000000000000000000000000000\r\ntcp_port:{}\r\n\
                 uptime_in_seconds:{}\r\nio_threads_active:0\r\n\r\n",
                env!("CARGO_PKG_VERSION"),
                if server.cluster_enabled() {
                    "cluster"
                } else {
                    "standalone"
                },
                usize::BITS,
                // The port the socket was actually bound to, which whoever bound
                // it told the server. Nought on an embedded caller that never
                // opened one, which is honest: there is no port.
                server.announced_port(),
                server.uptime_secs(),
            );
        }
        if want("clients") {
            let _ = write!(
                s,
                "# Clients\r\nconnected_clients:{}\r\nblocked_clients:{}\r\n\
                 pubsub_clients:{}\r\ncluster_connections:0\r\n\r\n",
                server
                    .totals()
                    .clients
                    .saturating_sub(server.replica_count()),
                server.parked(),
                server.pubsub_counts().clients,
            );
        }
        if want("memory") {
            // Both the cap and the quarter of it, because the quarter is an
            // empirical number and somebody surprised by it should be able to
            // see what it was a quarter of without reading the source. The
            // reasoning is written out in `cap`.
            let cap = crate::cap::cap();
            let compact = server.compaction();
            // Read out of its stripe before the write, because an argument list
            // keeps every temporary in it alive until the whole call is over
            // and one of the other arguments walks that same stripe.
            let policy = server.settings().policy().name();
            let _ = write!(
                s,
                "# Memory\r\nused_memory:{}\r\nused_memory_dataset:{}\r\n\
                 used_memory_overhead:{}\r\nmem_arena_bytes:{}\r\n\
                 mem_arena_segments:{}\r\nmem_compact_walked:{}\r\n\
                 mem_compact_moved:{}\r\nmem_compact_bytes:{}\r\n\
                 mem_index_bytes:{}\r\n\
                 mem_client_buffers:{}\r\ntotal_system_memory:{}\r\n\
                 mem_cgroup_limit:{}\r\nmem_limit:{}\r\nmem_budget:{}\r\n\
                 maxmemory:{}\r\nmaxmemory_policy:{}\r\n\
                 maxstore:{}\r\nyo_store_bytes:{}\r\nyo_memory_regime:{}\r\n\r\n",
                server.memory_bytes(),
                server.dataset_bytes(),
                server.memory_bytes() - server.dataset_bytes(),
                server.arena_bytes(),
                server.segment_count(),
                compact.walked,
                compact.moved,
                compact.bytes,
                server.index_bytes(),
                server.conn_bytes(),
                cap.host.unwrap_or(0),
                cap.cgroup.unwrap_or(0),
                cap.limit().unwrap_or(0),
                cap.budget(),
                server.maxmemory(),
                policy,
                server.maxstore().map_or(-1, |n| n as i64),
                server.store_bytes(),
                server.regime(),
            );
        }
        if want("persistence") {
            persist::info(server, &mut s);
        }
        if want("stats") {
            // The cold counters live here and not in the memory section,
            // because they are totals since the server started and everything
            // in that section is a level right now. `yo_cold_faults` over the
            // point reads a run issued is the ratio G9 is a gate on, and it
            // cannot be worked out from outside the server.
            let cold = server.cold_stats();
            let totals = server.totals();
            let subs = server.pubsub_counts();
            // The five ACL counters last, which is where a real server puts them
            // too: they are appended after the rest of the section rather than
            // written with it.
            let denied = server.acl_log().counters();
            let _ = write!(
                s,
                "# Stats\r\ntotal_connections_received:{}\r\n\
                 total_commands_processed:{}\r\nexpired_subkeys:{}\r\n\
                 expired_subkeys_active:{}\r\nexpired_keys:{}\r\n\
                 evicted_keys:{}\r\nkeyspace_hits:{}\r\nkeyspace_misses:{}\r\n\
                 yo_cold_demoted:{}\r\nyo_cold_promoted:{}\r\n\
                 yo_cold_faults:{}\r\nyo_cold_served:{}\r\nyo_cold_bytes_out:{}\r\n\
                 yo_cold_bytes_in:{}\r\npubsub_channels:{}\r\n\
                 pubsub_patterns:{}\r\npubsubshard_channels:{}\r\n\
                 acl_access_denied_auth:{}\r\nacl_access_denied_cmd:{}\r\n\
                 acl_access_denied_key:{}\r\nacl_access_denied_channel:{}\r\n\
                 acl_access_denied_tls_cert:{}\r\n\r\n",
                totals.connections,
                totals.commands,
                server.expired_fields(),
                server.expired_fields_active(),
                server.expired_keys(),
                server.evicted_keys(),
                server.keyspace_hits(),
                server.keyspace_misses(),
                cold.demoted,
                cold.promoted,
                cold.faults,
                cold.served,
                cold.bytes_out,
                cold.bytes_in,
                subs.channels,
                subs.patterns,
                subs.shard,
                denied[0],
                denied[1],
                denied[2],
                denied[3],
                denied[4],
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
            super::repl::info(server, &mut s);
        }
        if want("cluster") {
            // One field, which is the one every client library reads on connect
            // to decide whether it needs a slot map at all.
            let _ = write!(
                s,
                "# Cluster\r\ncluster_enabled:{}\r\n\r\n",
                u8::from(server.cluster_enabled()),
            );
        }
        if extra("threads") {
            // An extra rather than a default section for the reason above: it
            // grows with the thread count, and a tool polling `INFO` every
            // second on a thirty two thread server does not want thirty two
            // more lines every time.
            //
            // The three numbers are the three questions worth asking of a
            // server where a connection belongs to the thread that accepted it.
            // `clients` says whether the connections open right now are shared
            // out. `connections` says whether they were shared out as they
            // arrived, which is a different question, because a split that was
            // fair at the start and is unfair now is clients hanging up rather
            // than an accept race. `commands` says whether an even split of
            // connections turned into an even split of work, which it does not
            // when the clients are not all asking for the same thing.
            let per = server.per_thread();
            let _ = write!(s, "# Threads\r\nio_threads:{}\r\n", per.len());
            for (at, thread) in per.iter().enumerate() {
                let _ = write!(
                    s,
                    "thread_{at}:clients={},connections={},commands={}\r\n",
                    thread.clients, thread.connections, thread.commands,
                );
            }
            s.push_str("\r\n");
        }
        if extra("commandstats") {
            s.push_str("# Commandstats\r\n");
            for (name, row) in server.command_stats() {
                let _ = write!(
                    s,
                    "cmdstat_{name}:calls={},rejected_calls={},failed_calls={}\r\n",
                    row.calls, row.rejected, row.failed,
                );
            }
            s.push_str("\r\n");
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

#[cfg(test)]
mod tests {
    use super::{BOOLS, REPLICA_READ_ONLY, SETTINGS};

    /// The backlog setting is a compiled in number written out twice, and the
    /// two have to be the same number: a client that reads `repl-backlog-size`
    /// and then works out how far behind a replica may fall before a full resync
    /// is reading this to answer that question.
    #[test]
    fn the_backlog_setting_is_the_size_the_backlog_actually_is() {
        let said = SETTINGS
            .iter()
            .find(|(k, _)| *k == "repl-backlog-size")
            .expect("the setting is there")
            .1;
        assert_eq!(
            said.parse::<usize>().expect("a number"),
            super::super::repl::BACKLOG_BYTES
        );
    }

    /// The two spellings are one setting and the reference answers to both, so a
    /// script written against either works here.
    #[test]
    fn the_read_only_setting_answers_to_both_of_its_names() {
        assert_eq!(REPLICA_READ_ONLY, ["replica-read-only", "slave-read-only"]);
        assert_eq!(BOOLS, ["yes", "no"]);
    }
}
