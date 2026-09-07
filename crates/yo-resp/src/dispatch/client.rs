//! The `CLIENT` container, for the half of it that is about the connection
//! sending the command.
//!
//! Everything here answers about or changes the one connection it arrived on,
//! which is why none of it needs to reach another thread. `CLIENT LIST`,
//! `CLIENT KILL`, `CLIENT PAUSE` and `CLIENT UNPAUSE` are the other half, they
//! need a table of every live connection on every thread, and they are not here
//! yet.
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
//! knows breaks on a missing field and copes with a zero. Divergence D-122 says
//! which those are.

use super::args::{self, Args, is};
use super::table::Spec;
use super::{Flow, Reply, Server, Session};
use crate::proto::Proto;
use crate::reply::Out;
use core::fmt::Write;
use yo_common::{Code, Error, Result};

/// `CLIENT <subcommand> [arg ...]`.
pub(super) fn execute(
    server: &Server,
    session: &mut Session,
    spec: &Spec,
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
        let text = report(server, session, spec, &args, out.proto(), out.len());
        out.verbatim(b"txt", text.as_bytes());
    } else if is(sub, b"REPLY") {
        reply(session, args)?;
        // Only `ON` says anything, and it says it here. The other two are
        // silent because the engine takes the reply back after the command, so
        // writing the `+OK` and having it thrown away is the same thing as not
        // writing it, and it keeps this arm the same shape as the others.
        out.ok();
    } else if is(sub, b"NO-EVICT") {
        session.no_evict = on_off(args, "no-evict")?;
        out.ok();
    } else if is(sub, b"NO-TOUCH") {
        session.no_touch = on_off(args, "no-touch")?;
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
    let into = if name == "lib-name" {
        &mut session.lib_name
    } else {
        &mut session.lib_ver
    };
    yo_alloc::allow(|| {
        into.clear();
        into.extend_from_slice(value);
    });
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

/// The one line `CLIENT INFO` answers with, which is also the line `CLIENT
/// LIST` will print one of per connection.
///
/// `obl` is the reply buffer's length as of before this command wrote anything,
/// which is what `mark` is: by the time the report is read back the report
/// itself is in the buffer, and reporting the buffer with the report in it
/// would be a number that changes because it was asked for.
fn report(
    server: &Server,
    session: &Session,
    spec: &Spec,
    args: &Args<'_>,
    proto: Proto,
    mark: usize,
) -> String {
    // A session nobody told when it was opened has no age and no idle time,
    // rather than the whole of the epoch. That is every embedded caller, which
    // has no socket for either number to be about.
    let now = if session.sock.since_ms == 0 {
        0
    } else {
        server.now_ms()
    };
    let sock = &session.sock;
    let (sub, psub, ssub) = session.sub_counts();
    let (multi, multi_mem) = session.queued();
    let argv_mem: usize = (0..args.len()).map(|i| args.get(i).len()).sum();
    // The room the connection is holding, which is both buffers, and not the
    // bytes in use: they keep their capacity between batches on purpose.
    let tot_mem = sock.qbuf + sock.qbuf_free + sock.rbs;

    yo_alloc::allow(|| {
        let mut s = String::with_capacity(512);
        let _ = writeln!(
            s,
            "id={id} addr={addr} laddr={laddr} fd={fd} name={name} age={age} idle={idle} \
             flags={flags} db={db} sub={sub} psub={psub} ssub={ssub} multi={multi} watch={watch} \
             qbuf={qbuf} qbuf-free={qbuf_free} argv-mem={argv_mem} multi-mem={multi_mem} \
             rbs={rbs} rbp={rbp} obl={obl} oll=0 omem=0 omem-shared=0 omem-unshared=0 \
             tot-mem={tot_mem} events=r cmd={cmd} user=default redir=-1 resp={resp} \
             lib-name={lib_name} lib-ver={lib_ver} io-thread={io_thread} tot-net-in={net_in} \
             tot-net-out={net_out} tot-cmds={cmds} read-events={reads} \
             avg-pipeline-len-sum={cmds} avg-pipeline-len-cnt={reads}",
            id = session.id,
            addr = String::from_utf8_lossy(&sock.peer),
            laddr = String::from_utf8_lossy(&sock.local),
            fd = sock.fd,
            name = String::from_utf8_lossy(&session.name),
            age = (now.saturating_sub(sock.since_ms)) / 1000,
            idle = (now.saturating_sub(sock.last_ms)) / 1000,
            flags = flags(session),
            db = session.db,
            watch = session.watching.len(),
            qbuf = sock.qbuf,
            qbuf_free = sock.qbuf_free,
            rbs = sock.rbs,
            rbp = sock.rbp,
            obl = mark,
            cmd = Named(spec, &sock.sub),
            resp = proto.version(),
            lib_name = String::from_utf8_lossy(&session.lib_name),
            lib_ver = String::from_utf8_lossy(&session.lib_ver),
            io_thread = server.my_slot(),
            net_in = sock.net_in,
            net_out = sock.net_out,
            cmds = sock.cmds,
            reads = sock.reads,
        );
        s
    })
}

/// The letters in the `flags` field, in Redis's order.
///
/// Redis has nineteen of them and yo can be in four of the states they name. A
/// connection in none of them reads `N`, which is Redis's spelling for no flags
/// rather than for a flag called none.
fn flags(session: &Session) -> String {
    let mut s = String::new();
    if session.subscribed() {
        s.push('P');
    }
    if session.in_multi() {
        s.push('x');
    }
    if session.sock.unix {
        s.push('U');
    }
    if session.no_evict {
        s.push('e');
    }
    if session.no_touch {
        s.push('T');
    }
    if s.is_empty() {
        s.push('N');
    }
    s
}

/// The `cmd` field, which is the command name with the subcommand after a bar
/// when there was one.
///
/// A formatting shim rather than a built string, so that the common case of a
/// command with no subcommand writes the name straight into the report.
struct Named<'a>(&'a Spec, &'a [u8]);

impl core::fmt::Display for Named<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.0.name)?;
        if !self.1.is_empty() {
            f.write_str("|")?;
            // Lowercased, because the report is a name and not an echo: a
            // client that sent `CLIENT Info` is still running `client|info`.
            for b in self.1 {
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
    "NO-EVICT (ON|OFF)",
    "    Protect current client connection from eviction.",
    "NO-TOUCH (ON|OFF)",
    "    Will not touch LRU/LFU stats when this mode is on.",
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
