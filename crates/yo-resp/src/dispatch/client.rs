//! The `CLIENT` container.
//!
//! Half of it is about the connection sending the command and half of it is
//! about the other ones. The first half reads and writes the session it was
//! handed. The second half, which is `LIST` and `KILL`, cannot: on a server with
//! more than one thread the other connections live inside another thread's
//! front, and this thread has no borrow of that and must never take one. So
//! every connection publishes the part of itself these two report into a row any
//! thread can read, and both work off the table of those rows. See the `clients`
//! module for the row.
//!
//! `PAUSE` and `UNPAUSE` are the third shape. They are not a report and not a
//! close, they are a gate in front of every command on every thread, so all
//! that is here is the parse of the timeout and the mode. What they arm is one
//! word on the server, which the `clients` module keeps beside the rows and the
//! funnel reads once per command.
//!
//! # Why a report nobody on the server reads
//!
//! `CLIENT INFO` is forty fields and not one of them changes what the server
//! does. It is here because it is the first thing anybody runs when a server is
//! behaving oddly, and because a good deal of tooling parses it: `redis-cli
//! --stat`, every dashboard, and the connection pool in more than one client
//! library, which reads `id` back to name itself in a later `CLIENT KILL`.
//!
//! So the fields are the fields Redis 8.10.1 emits, in Redis's order, with the
//! same spelling and the same trailing newline. Where yo has nothing that
//! matches, the field is still there with the honest number in it rather than
//! being left out, because a parser that splits on spaces and expects a name it
//! knows breaks on a missing field and copes with a zero. Divergence D-123 says
//! which those are.
//!
//! # How a connection on another thread is closed
//!
//! It is not closed by the thread that ran the command. A kill sets a bit on the
//! row and counts itself on the server, and the thread that owns the connection
//! sees the count on its next turn, finds its own killed rows and lets go of
//! them there. That costs one relaxed load per turn on a server nobody has ever
//! killed a client on, and it means the close happens where the buffers are.

use super::args::{self, Args, is};
use super::clients::{self, Client};
use super::table::{self, Spec};
use super::{Flow, Reply, Server, Session};
use crate::proto::Proto;
use crate::reply::Out;
use core::fmt::Write;
use std::sync::atomic::Ordering::{Acquire, Relaxed};
use yo_common::{Code, Error, Result};

/// `CLIENT <subcommand> [arg ...]`.
pub(super) fn execute(
    server: &Server,
    session: &mut Session,
    _spec: &Spec,
    args: Args<'_>,
    out: &mut Out,
) -> Result<Flow> {
    let sub = args.get(1);
    if is(sub, b"ID") {
        one(args, "id")?;
        out.int(session.id as i64);
    } else if is(sub, b"GETNAME") {
        one(args, "getname")?;
        if session.name.is_empty() {
            out.nil();
        } else {
            out.bulk(&session.name);
        }
    } else if is(sub, b"SETNAME") {
        two(args, "setname")?;
        let name = args.get(2);
        if !printable(name) {
            return Err(Error::new(
                Code::Invalid,
                "Client names cannot contain spaces, newlines or special characters.",
            ));
        }
        session.set_name(name);
        out.ok();
    } else if is(sub, b"SETINFO") {
        setinfo(session, args)?;
        out.ok();
    } else if is(sub, b"INFO") {
        one(args, "info")?;
        let text = report(server, session, out.proto(), out.len());
        out.verbatim(b"txt", text.as_bytes());
    } else if is(sub, b"LIST") {
        list(server, args, out)?;
    } else if is(sub, b"KILL") {
        if kill(server, session, args, out)? {
            // The reply goes out and then the socket goes away, which is what
            // the engine does with a close: the buffer is written first.
            return Ok(Flow::Close);
        }
    } else if is(sub, b"PAUSE") {
        pause(server, args)?;
        out.ok();
    } else if is(sub, b"UNPAUSE") {
        one(args, "unpause")?;
        server.unpause();
        out.ok();
    } else if is(sub, b"REPLY") {
        reply(session, args)?;
        // Only `ON` says anything, and it says it here. The other two are
        // silent because the engine takes the reply back after the command, so
        // writing the `+OK` and having it thrown away is the same thing as not
        // writing it, and it keeps this arm the same shape as the others.
        out.ok();
    } else if is(sub, b"NO-EVICT") {
        let on = on_off(args, "no-evict")?;
        session.set_no_evict(on);
        out.ok();
    } else if is(sub, b"NO-TOUCH") {
        let on = on_off(args, "no-touch")?;
        session.set_no_touch(on);
        out.ok();
    } else if is(sub, b"HELP") {
        super::server::help(out, CLIENT_HELP);
    } else {
        return Err(args::unknown_subcommand(sub, "CLIENT"));
    }
    Ok(Flow::Continue)
}

/// A subcommand that takes nothing after it.
fn one(args: Args<'_>, sub: &str) -> Result<()> {
    if args.len() == 2 {
        Ok(())
    } else {
        Err(args::wrong_arity_sub("client", sub))
    }
}

/// A subcommand that takes exactly one argument.
fn two(args: Args<'_>, sub: &str) -> Result<()> {
    if args.len() == 3 {
        Ok(())
    } else {
        Err(args::wrong_arity_sub("client", sub))
    }
}

/// Whether every byte of a name is one a client is allowed to send.
///
/// Redis's `validateClientAttr`, which is printable ASCII and nothing else, so
/// no spaces and no newlines. It is not fussiness: these go out again inside
/// `CLIENT LIST`, one client to a line and one field to a space, and a name
/// with either in it would make that report unparseable.
fn printable(value: &[u8]) -> bool {
    value.iter().all(|b| (b'!'..=b'~').contains(b))
}

/// `CLIENT SETINFO <LIB-NAME|LIB-VER> <value>`.
fn setinfo(session: &mut Session, args: Args<'_>) -> Result<()> {
    if args.len() != 4 {
        return Err(args::wrong_arity_sub("client", "setinfo"));
    }
    let what = args.get(2);
    let value = args.get(3);
    let name = if is(what, b"LIB-NAME") {
        "lib-name"
    } else if is(what, b"LIB-VER") {
        "lib-ver"
    } else {
        return Err(Error::fmt(
            Code::Invalid,
            format_args!("Unrecognized option '{}'", String::from_utf8_lossy(what)),
        ));
    };
    if !printable(value) {
        return Err(Error::fmt(
            Code::Invalid,
            format_args!("{name} cannot contain spaces, newlines or special characters."),
        ));
    }
    if name == "lib-name" {
        session.set_lib_name(value);
    } else {
        session.set_lib_ver(value);
    }
    Ok(())
}

/// `CLIENT REPLY <ON|OFF|SKIP>`.
fn reply(session: &mut Session, args: Args<'_>) -> Result<()> {
    if args.len() != 3 {
        return Err(args::wrong_arity_sub("client", "reply"));
    }
    let mode = args.get(2);
    session.reply = if is(mode, b"ON") {
        Reply::On
    } else if is(mode, b"OFF") {
        Reply::Off
    } else if is(mode, b"SKIP") {
        // Not `SkipNow`: what this skips is the command after it, and the
        // engine steps the state on once per command, so this arrives at
        // `SkipNow` exactly when that command runs.
        Reply::SkipNext
    } else {
        return Err(args::syntax());
    };
    Ok(())
}

/// A subcommand whose one argument is `ON` or `OFF`.
fn on_off(args: Args<'_>, sub: &str) -> Result<bool> {
    if args.len() != 3 {
        return Err(args::wrong_arity_sub("client", sub));
    }
    let arg = args.get(2);
    if is(arg, b"ON") {
        Ok(true)
    } else if is(arg, b"OFF") {
        Ok(false)
    } else {
        Err(args::syntax())
    }
}

/// Which connections a `TYPE` word picks out.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// No `TYPE` was given, so all of them.
    Any,
    /// A client that is not in subscribe mode.
    Normal,
    /// A client that is.
    Pubsub,
    /// A replication link in one direction or the other, of which there are
    /// none here yet, so naming one is a filter nothing matches rather than an
    /// error.
    Link,
}

impl Kind {
    /// Redis's `getClientTypeByName`, which knows both spellings of a replica.
    fn parse(word: &[u8]) -> Result<Kind> {
        if is(word, b"normal") {
            Ok(Kind::Normal)
        } else if is(word, b"pubsub") {
            Ok(Kind::Pubsub)
        } else if is(word, b"master") || is(word, b"replica") || is(word, b"slave") {
            Ok(Kind::Link)
        } else {
            Err(Error::fmt(
                Code::Invalid,
                format_args!("Unknown client type '{}'", String::from_utf8_lossy(word)),
            ))
        }
    }

    /// Whether this connection is one of them.
    fn covers(self, row: &Client) -> bool {
        match self {
            Kind::Any => true,
            Kind::Normal => !row.flag(clients::SUBSCRIBED),
            Kind::Pubsub => row.flag(clients::SUBSCRIBED),
            Kind::Link => false,
        }
    }
}

/// `CLIENT LIST [TYPE <type>] [ID <id> ...]`.
///
/// One line per connection, in the order they were opened, and each line is the
/// line `CLIENT INFO` gives for one. The two are one formatter with two callers,
/// which is the only way they stay the same line as the command grows.
fn list(server: &Server, args: Args<'_>, out: &mut Out) -> Result<()> {
    let mut want = Kind::Any;
    let mut ids: Vec<u64> = Vec::new();
    // The three shapes Redis accepts, and nothing in between: the bare command,
    // one `TYPE` and its word, or `ID` and every argument after it.
    if args.len() == 4 && is(args.get(2), b"TYPE") {
        want = Kind::parse(args.get(3))?;
    } else if args.len() > 3 && is(args.get(2), b"ID") {
        for at in 3..args.len() {
            // Any integer, including zero and negative ones, which are ids no
            // connection has and so are a filter that matches nothing rather
            // than an error. That is a real server's parse.
            let id = args
                .int(at)
                .map_err(|_| Error::new(Code::Invalid, "Invalid client ID"))?;
            yo_alloc::allow(|| ids.push(id as u64));
        }
    } else if args.len() != 2 {
        return Err(args::syntax());
    }

    let now = server.now_ms();
    let text = yo_alloc::allow(|| {
        let mut text = String::new();
        for row in server.client_rows() {
            if !want.covers(&row) || (!ids.is_empty() && !ids.contains(&row.id)) {
                continue;
            }
            let obl = row.obl.load(Relaxed);
            let resp = row.resp.load(Relaxed);
            line(&row, now, obl, resp, &mut text);
        }
        text
    });
    out.verbatim(b"txt", text.as_bytes());
    Ok(())
}

/// What a `CLIENT KILL` was asked to match on.
///
/// Every field is a condition and an unset one matches everything, so a kill
/// with no filters at all would take every connection down. Redis allows that
/// and so does this, which is why `SKIPME` defaults to leaving the caller alone.
#[derive(Default)]
struct Filter<'a> {
    id: Option<u64>,
    addr: Option<&'a [u8]>,
    laddr: Option<&'a [u8]>,
    kind: Option<Kind>,
    /// The youngest connection this will take, in seconds.
    maxage: Option<u64>,
    /// Whether the connection asking is spared, which the old form turns off.
    skipme: bool,
}

impl Filter<'_> {
    /// Whether this connection is one the filter names.
    ///
    /// The address comparison is over the string the report prints, because that
    /// is the string the operator read it out of, and a kill has to name a
    /// client the same way the listing did.
    fn covers(&self, row: &Client, now: u64, me: u64) -> bool {
        if self.skipme && row.id == me {
            return false;
        }
        if self.id.is_some_and(|id| id != row.id) {
            return false;
        }
        if self.kind.is_some_and(|kind| !kind.covers(row)) {
            return false;
        }
        if let Some(want) = self.maxage {
            let since = row.since_ms.load(Relaxed);
            let age = if since == 0 {
                0
            } else {
                now.saturating_sub(since) / 1000
            };
            if age < want {
                return false;
            }
        }
        if self.addr.is_none() && self.laddr.is_none() {
            return true;
        }
        let text = row.text.lock();
        self.addr.is_none_or(|want| want == text.peer.as_slice())
            && self.laddr.is_none_or(|want| want == text.local.as_slice())
    }
}

/// `CLIENT KILL <addr>` and `CLIENT KILL <filter> <value> ...`.
///
/// The two forms differ in more than their arguments. The old one names one
/// address, will take the caller's own connection, and answers `OK` or an error
/// saying it found nobody. The new one takes any number of conditions, spares
/// the caller unless told not to, and answers with how many it took. Both are
/// Redis's, kept apart the same way it keeps them apart, by the argument count.
fn kill(server: &Server, session: &mut Session, args: Args<'_>, out: &mut Out) -> Result<bool> {
    let old = args.len() == 3;
    let mut filter = Filter {
        skipme: !old,
        ..Filter::default()
    };
    if old {
        filter.addr = Some(args.get(2));
    } else if args.len() > 3 {
        let mut at = 2;
        while at + 1 < args.len() {
            let word = args.get(at);
            let value = args.get(at + 1);
            if is(word, b"ID") {
                filter.id = Some(client_id(value)?);
            } else if is(word, b"ADDR") {
                filter.addr = Some(value);
            } else if is(word, b"LADDR") {
                filter.laddr = Some(value);
            } else if is(word, b"TYPE") {
                filter.kind = Some(Kind::parse(value)?);
            } else if is(word, b"USER") {
                // There is no ACL yet, so there is one user and it is `default`.
                // Naming it filters nothing out and naming any other is the same
                // error a server with an ACL gives for a name it has never seen.
                if !is(value, b"default") {
                    return Err(Error::fmt(
                        Code::Invalid,
                        format_args!("No such user '{}'", String::from_utf8_lossy(value)),
                    ));
                }
            } else if is(word, b"MAXAGE") {
                filter.maxage = Some(maxage(value)?);
            } else if is(word, b"SKIPME") {
                filter.skipme = if is(value, b"yes") {
                    true
                } else if is(value, b"no") {
                    false
                } else {
                    return Err(args::syntax());
                };
            } else {
                return Err(args::syntax());
            }
            at += 2;
        }
        // A trailing word with no value, which the loop above walked past.
        if at != args.len() {
            return Err(args::syntax());
        }
    } else {
        return Err(args::wrong_arity_sub("client", "kill"));
    }

    let now = server.now_ms();
    let me = session.id;
    let mut killed = 0u64;
    let mut myself = false;
    let mut posted = 0;
    for row in server.client_rows() {
        if !filter.covers(&row, now, me) {
            continue;
        }
        killed += 1;
        if row.id == me {
            // The caller's own connection is closed here rather than by the
            // sweep, because the reply to this command still has to go out and
            // the connection that carries it is this one. Redis calls it
            // CLOSE_AFTER_REPLY and it is the same thing.
            myself = true;
        } else if row.kill() {
            posted += 1;
        }
    }
    server.note_kills(posted);

    if old {
        if killed == 0 {
            return Err(Error::new(Code::Invalid, "No such client"));
        }
        out.ok();
    } else {
        out.uint(killed);
    }
    Ok(myself)
}

/// The `ID` a kill was given, which has to be a client id and not any number.
fn client_id(value: &[u8]) -> Result<u64> {
    let text = core::str::from_utf8(value).ok();
    let id = text.and_then(|t| t.parse::<i64>().ok());
    match id {
        Some(id) if id > 0 => Ok(id as u64),
        _ => Err(Error::new(
            Code::Invalid,
            "client-id should be greater than 0",
        )),
    }
}

/// The `MAXAGE` a kill was given, in seconds.
///
/// The two errors are two checks and not one, which is worth keeping apart
/// because a real server keeps them apart: a word that is not a number is out of
/// range and a number that is not positive is too small. Zero is too small, so
/// there is no way to write a kill that takes every connection by age.
fn maxage(value: &[u8]) -> Result<u64> {
    let text = core::str::from_utf8(value).ok();
    let Some(age) = text.and_then(|t| t.parse::<i64>().ok()) else {
        return Err(Error::new(
            Code::Invalid,
            "maxage is not an integer or out of range",
        ));
    };
    if age <= 0 {
        return Err(Error::new(Code::Invalid, "maxage should be greater than 0"));
    }
    Ok(age as u64)
}

/// `CLIENT PAUSE <timeout> [WRITE|ALL]`.
///
/// The timeout is in milliseconds and it is how long from now rather than a
/// point in time, so a client that means to hold the server for a second asks
/// for a thousand. `ALL` is the default and holds every command; `WRITE` holds
/// the ones that change something or would replicate.
///
/// Nothing here is exempt from the gate this arms, including `CLIENT UNPAUSE`,
/// so a long `ALL` pause cannot be called off before it runs out. That is
/// Redis's behaviour and it is the reason the timeout is a required argument
/// rather than something with a sensible default.
fn pause(server: &Server, args: Args<'_>) -> Result<()> {
    if args.len() < 3 {
        return Err(args::wrong_arity_sub("client", "pause"));
    }
    if args.len() > 4 {
        // Not a wrong arity, which is the one place `CLIENT` says this instead:
        // a real server reads the mode itself and complains about the whole
        // subcommand when there is something after it.
        return Err(args::subcommand_syntax(args.get(1), "CLIENT"));
    }
    let text = core::str::from_utf8(args.get(2)).ok();
    let Some(ms) = text.and_then(|t| t.parse::<i64>().ok()) else {
        return Err(Error::new(
            Code::Invalid,
            "timeout is not an integer or out of range",
        ));
    };
    if ms < 0 {
        return Err(Error::new(Code::Invalid, "timeout is negative"));
    }
    let all = if args.len() == 3 {
        true
    } else {
        let mode = args.get(3);
        if is(mode, b"ALL") {
            true
        } else if is(mode, b"WRITE") {
            false
        } else {
            return Err(Error::new(
                Code::Invalid,
                "CLIENT PAUSE mode must be WRITE or ALL",
            ));
        }
    };
    server.pause(server.now_ms().saturating_add(ms as u64), all);
    Ok(())
}

/// The one line `CLIENT INFO` answers with and `CLIENT LIST` prints one of per
/// connection.
///
/// `obl` and `resp` are passed in rather than read off the row because the two
/// callers know different things. `CLIENT INFO` is describing the connection it
/// is running on, so it knows what the reply buffer held before this command
/// started writing and which protocol the reply it is composing is in. `CLIENT
/// LIST` is describing somebody else, so all it can have is what that connection
/// last published.
fn line(row: &Client, now_ms: u64, obl: u64, resp: u32, into: &mut String) {
    // A connection nobody told when it was opened has no age and no idle time,
    // rather than the whole of the epoch. That is every embedded caller, which
    // has no socket for either number to be about.
    let since = row.since_ms.load(Relaxed);
    let now = if since == 0 { 0 } else { now_ms };
    let qbuf = row.qbuf.load(Relaxed);
    let qbuf_free = row.qbuf_free.load(Relaxed);
    let rbs = row.rbs.load(Relaxed);
    // The room the connection is holding, which is both buffers, and not the
    // bytes in use: they keep their capacity between batches on purpose.
    let tot_mem = qbuf + qbuf_free + rbs;
    let named = row.has_sub.load(Acquire) == 1;
    let text = row.text.lock();
    let _ = writeln!(
        into,
        "id={id} addr={addr} laddr={laddr} fd={fd} name={name} age={age} idle={idle} \
         flags={flags} db={db} sub={sub} psub={psub} ssub={ssub} multi={multi} watch={watch} \
         qbuf={qbuf} qbuf-free={qbuf_free} argv-mem={argv_mem} multi-mem={multi_mem} \
         rbs={rbs} rbp={rbp} obl={obl} oll=0 omem=0 omem-shared=0 omem-unshared=0 \
         tot-mem={tot_mem} events=r cmd={cmd} user={user} redir=-1 resp={resp} \
         lib-name={lib_name} lib-ver={lib_ver} io-thread={io_thread} tot-net-in={net_in} \
         tot-net-out={net_out} tot-cmds={cmds} read-events={reads} \
         avg-pipeline-len-sum={cmds} avg-pipeline-len-cnt={reads}",
        id = row.id,
        addr = String::from_utf8_lossy(&text.peer),
        laddr = String::from_utf8_lossy(&text.local),
        fd = row.fd.load(Relaxed),
        name = String::from_utf8_lossy(&text.name),
        user = String::from_utf8_lossy(if text.user.is_empty() {
            super::acl::DEFAULT
        } else {
            &text.user
        }),
        age = (now.saturating_sub(since)) / 1000,
        idle = (now.saturating_sub(row.last_ms.load(Relaxed))) / 1000,
        flags = Flags(row.flags.load(Relaxed)),
        db = row.db.load(Relaxed),
        sub = row.sub.load(Relaxed),
        psub = row.psub.load(Relaxed),
        ssub = row.ssub.load(Relaxed),
        multi = row.multi.load(Relaxed),
        watch = row.watch.load(Relaxed),
        argv_mem = row.argv_mem.load(Relaxed),
        multi_mem = row.multi_mem.load(Relaxed),
        rbp = row.rbp.load(Relaxed),
        cmd = Named(row.spec.load(Relaxed), named.then_some(&text.sub)),
        lib_name = String::from_utf8_lossy(&text.lib_name),
        lib_ver = String::from_utf8_lossy(&text.lib_ver),
        io_thread = row.thread.load(Relaxed),
        net_in = row.net_in.load(Relaxed),
        net_out = row.net_out.load(Relaxed),
        cmds = row.cmds.load(Relaxed),
        reads = row.reads.load(Relaxed),
    );
}

/// The whole report for one connection, as its own string.
fn report(server: &Server, session: &Session, proto: Proto, mark: usize) -> String {
    yo_alloc::allow(|| {
        let mut s = String::with_capacity(512);
        line(
            session.row(),
            server.now_ms(),
            mark as u64,
            proto.version() as u32,
            &mut s,
        );
        s
    })
}

/// The letters in the `flags` field, in Redis's order.
///
/// Redis has nineteen of them and yo can be in five of the states they name. A
/// connection in none of them reads `N`, which is Redis's spelling for no flags
/// rather than for a flag called none.
struct Flags(u32);

impl core::fmt::Display for Flags {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let letters = [
            // In front of the rest because that is the order a real server
            // writes them in, so a monitor that has also subscribed reads `OP`
            // and never `PO`.
            (clients::MONITOR, 'O'),
            (clients::SUBSCRIBED, 'P'),
            (clients::IN_MULTI, 'x'),
            (clients::UNIX, 'U'),
            (clients::NO_EVICT, 'e'),
            (clients::NO_TOUCH, 'T'),
        ];
        let mut wrote = false;
        for (bit, letter) in letters {
            if self.0 & bit != 0 {
                f.write_char(letter)?;
                wrote = true;
            }
        }
        if wrote { Ok(()) } else { f.write_char('N') }
    }
}

/// The `cmd` field, which is the command name with the subcommand after a bar
/// when there was one.
///
/// A formatting shim rather than a built string, so that the common case of a
/// command with no subcommand writes the name straight into the report. The
/// command is carried as its place in the table, since that is a word another
/// thread can read without taking anything.
struct Named<'a>(u32, Option<&'a Vec<u8>>);

impl core::fmt::Display for Named<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let Ok(at) = usize::try_from(self.0) else {
            return f.write_str("NULL");
        };
        if at >= table::count() {
            // A connection that has not sent a command yet, which is what a real
            // server spells `NULL` rather than leaving empty.
            return f.write_str("NULL");
        }
        f.write_str(table::name_at(at))?;
        if let Some(sub) = self.1.filter(|sub| !sub.is_empty()) {
            f.write_str("|")?;
            // Lowercased, because the report is a name and not an echo: a
            // client that sent `CLIENT Info` is still running `client|info`.
            for b in sub {
                f.write_char(b.to_ascii_lowercase() as char)?;
            }
        }
        Ok(())
    }
}

/// What `CLIENT HELP` says.
///
/// Redis lists every subcommand it has and this lists the ones that are here,
/// which is the one place the two differ on purpose: a client reading this to
/// find out what it can send should not be told about `CLIENT TRACKING` by a
/// server that would answer it with an unknown subcommand.
const CLIENT_HELP: &[&str] = &[
    "CLIENT <subcommand> [<arg> [value] [opt] ...]. Subcommands are:",
    "GETNAME",
    "    Return the name of the current connection.",
    "ID",
    "    Return the ID of the current connection.",
    "INFO",
    "    Return information about the current client connection.",
    "KILL <ip:port>",
    "    Close the connection from the specified address and port.",
    "KILL <option> <value> [<option> <value> [...]]",
    "    Kill connections. Options are:",
    "    * ADDR (<ip:port>|<unixsocket>:0)",
    "      Kill connections made from the specified address",
    "    * LADDR (<ip:port>|<unixsocket>:0)",
    "      Kill connections made to specified local address",
    "    * TYPE (NORMAL|PUBSUB|MASTER|REPLICA)",
    "      Kill connections by type.",
    "    * USER <username>",
    "      Kill connections authenticated by <username>.",
    "    * SKIPME (YES|NO)",
    "      Skip killing current connection (default: yes).",
    "    * ID <client-id>",
    "      Kill connections by client id.",
    "    * MAXAGE <maxage>",
    "      Kill connections older than the specified age.",
    "LIST [options ...]",
    "    Return information about client connections. Options:",
    "    * TYPE (NORMAL|PUBSUB|MASTER|REPLICA)",
    "      Return clients of specified type.",
    "NO-EVICT (ON|OFF)",
    "    Protect current client connection from eviction.",
    "NO-TOUCH (ON|OFF)",
    "    Will not touch LRU/LFU stats when this mode is on.",
    "UNPAUSE",
    "    Stop the current client pause, resuming traffic.",
    "PAUSE <timeout> [WRITE|ALL]",
    "    Suspend all, or just write, clients for <timeout> milliseconds.",
    "REPLY (ON|OFF|SKIP)",
    "    Control the replies sent to the current connection.",
    "SETINFO <option> <value>",
    "    Set client meta attr. Options are:",
    "    * LIB-NAME: the client lib name.",
    "    * LIB-VER: the client lib version.",
    "SETNAME <name>",
    "    Assign the name <name> to the current connection.",
    "HELP",
    "    Print this help.",
];
