//! `MONITOR`, which is every command the server runs, echoed to whoever asked.
//!
//! A connection that sends this stops being a client and becomes an audience.
//! It is answered `OK` once and then never again, and from that moment every
//! command any connection runs arrives on it as one line. That is the whole
//! feature, and it is the first thing anybody reaches for when a client library
//! is sending something other than what its documentation says it sends.
//!
//! # It is the pub/sub problem again
//!
//! The command runs on the thread its connection landed on and the monitor is on
//! whichever thread it landed on, so the feed cannot write into the monitor's
//! reply buffer any more than a `PUBLISH` can. It does not try. The line is
//! rendered once, however many monitors there are, and goes into the mailbox of
//! each monitor's thread as an [`Envelope`](super::pubsub::Envelope), which that
//! thread drains at the end of its next batch while it holds its own
//! connections. So one command with three monitors watching renders one line and
//! hands out three refcount bumps.
//!
//! A monitor also counts itself on its thread's mailbox, the way a subscriber
//! does, so a thread with a monitor on it keeps the short poller wait and finds
//! its mail rather than sleeping through it.
//!
//! # What it costs a server nobody is watching
//!
//! One relaxed load of a count that is zero, per command. That is the same shape
//! the pub/sub registry and the watch table use, and for the same reason: this
//! is a debugging tool and nearly every server in the world is running without
//! one attached.
//!
//! # Where the line comes from
//!
//! After the command has run, not before, which is not a detail: `SELECT 3` is
//! reported on database three because by the time the line is rendered the
//! connection is on it. A command that was refused before it ever reached its
//! body is not reported at all, so an unknown command, a wrong arity and a
//! command a paused server is holding produce nothing. A transaction reports
//! `MULTI` when it opens, then every command it queued as `EXEC` replays them,
//! and then `EXEC` itself last, because each of those is its own trip through
//! the funnel and `EXEC` is the outermost.
//!
//! The six script commands are the exception and are reported before they run
//! rather than after, because a script's own calls come through here too and a
//! script whose effects arrived before the `EVAL` that caused them would be
//! unreadable. Redis makes the same exception in the same place and says so in
//! the same words.
//!
//! # Why a monitor may not touch the keyspace
//!
//! Because Redis flags one as a replica, and a replica sending `GET` is refused
//! with `Replica can't interact with the keyspace`. It reads like an accident of
//! the implementation and it is load bearing: a monitor is exempt from `CLIENT
//! PAUSE`, so a client that could pause the server, become a monitor and then
//! keep writing would have a way around the pause that nothing else has.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::{Relaxed, Release};

use super::args::Args;
use super::clients::{self, Client};
use super::pubsub::Envelope;
use super::{Server, Session};
use std::io::Write;
use yo_common::lock::Lock;

/// What `MONITOR` is spelled as in the six commands that report themselves
/// before they run rather than after.
pub(super) const SCRIPTS: [&str; 6] = [
    "eval",
    "eval_ro",
    "evalsha",
    "evalsha_ro",
    "fcall",
    "fcall_ro",
];

/// What a monitor is told when it sends a command that reaches a key.
///
/// Redis's sentence, word for word, and it says replica rather than monitor
/// because the flag it is checking is the replica one. Changing it to something
/// clearer would break every test and every tool that matches on it.
pub(super) fn replica() -> yo_common::Error {
    yo_common::Error::new(
        yo_common::Code::Invalid,
        "Replica can't interact with the keyspace",
    )
}

/// The connections listening to everything, and how many there are.
///
/// The count is beside the list rather than read off it, because the count is
/// what every command loads and the list is what only a feed touches.
#[derive(Default)]
pub(crate) struct Monitors {
    rows: Lock<Vec<Arc<Client>>>,
    live: AtomicUsize,
}

impl Server {
    /// Whether anybody is watching, which is the whole cost of this file on a
    /// server where nobody is.
    #[must_use]
    pub(crate) fn monitored(&self) -> bool {
        self.monitors.live.load(Relaxed) != 0
    }

    /// Take a connection on as a monitor, and say whether that was news.
    ///
    /// `MONITOR` sent by a connection that is already one is not an error and is
    /// not answered at all, so the answer here is what decides whether anything
    /// is written.
    pub(crate) fn watch_all(&self, row: &Arc<Client>) -> bool {
        let mut rows = self.monitors.rows.lock();
        if rows.iter().any(|r| r.id == row.id) {
            return false;
        }
        yo_alloc::allow(|| rows.push(Arc::clone(row)));
        self.monitors.live.store(rows.len(), Release);
        row.set_flag(clients::MONITOR, true);
        self.note_here(row.thread.load(Relaxed), 1);
        true
    }

    /// Let one go, which is `RESET` or the connection closing.
    pub(crate) fn watch_no_more(&self, row: &Arc<Client>) {
        let mut rows = self.monitors.rows.lock();
        let Some(at) = rows.iter().position(|r| r.id == row.id) else {
            return;
        };
        rows.remove(at);
        self.monitors.live.store(rows.len(), Release);
        row.set_flag(clients::MONITOR, false);
        self.note_here(row.thread.load(Relaxed), -1);
    }

    /// A handle to every monitor, copied out so the line can be rendered and
    /// posted with the lock let go of.
    ///
    /// The same trade `CLIENT LIST` makes for the same reason: a connection that
    /// stops watching while a line is being rendered leaves a row that is still
    /// readable, and the line lands in a mailbox that is about to be told the
    /// slot has moved on.
    fn monitor_rows(&self) -> Vec<Arc<Client>> {
        let rows = self.monitors.rows.lock();
        yo_alloc::allow(|| rows.clone())
    }

    /// The time now to the microsecond, which is what a monitor line carries.
    ///
    /// Off the fine clock and not the coarse one, because a line's whole use is
    /// telling somebody when a command arrived and a millisecond of a busy
    /// server is thousands of commands.
    fn now_us(&self) -> u64 {
        self.clock.now_us()
    }
}

/// Report a command to everybody watching.
///
/// The caller has already checked [`Server::monitored`], which is what keeps
/// this whole file off the path of a server with no monitor on it.
pub(super) fn feed(server: &Server, session: &Session, args: Args<'_>) {
    let rows = server.monitor_rows();
    if rows.is_empty() {
        return;
    }
    let line = yo_alloc::allow(|| Arc::new(render(server, session, args)));
    for row in rows {
        server.post(
            row.thread.load(Relaxed),
            Envelope::line(row.conn.load(Relaxed), row.id, Arc::clone(&line)),
        );
    }
}

/// One line, without the `+` and the newline the reply buffer adds.
///
/// The shape is Redis's: the time to the microsecond, then the database and who
/// sent it in brackets, then the command and its arguments quoted one by one.
/// Tooling parses this, `redis-cli monitor` included, so the spacing and the
/// quoting are part of it.
fn render(server: &Server, session: &Session, args: Args<'_>) -> Vec<u8> {
    yo_alloc::allow(|| {
        let mut line = Vec::with_capacity(64 + args.len() * 16);
        let us = server.now_us();
        let _ = write!(
            line,
            "{}.{:06} [{} ",
            us / 1_000_000,
            us % 1_000_000,
            session.db
        );
        who(session, &mut line);
        line.extend_from_slice(b"] ");
        let hidden = redacted(args);
        for i in 0..args.len() {
            if i != 0 {
                line.push(b' ');
            }
            if hidden & (1u32 << i.min(31)) != 0 {
                line.extend_from_slice(b"\"(redacted)\"");
            } else {
                quote(args.get(i), &mut line);
            }
        }
        line
    })
}

/// Who sent it, which is an address for a client and the word `lua` for a
/// command a script is making.
///
/// A Unix connection is named by its socket rather than by an address, because
/// it has not got one. The row carries the path with a `:0` after it, which is
/// how `CLIENT LIST` reports it and is not how this line does, so the port comes
/// off again here.
///
/// A caller with no socket at all reads `?:0`, which is Redis's spelling for a
/// client it made up rather than accepted. Here that is an embedded caller, and
/// a server can have one of those and a monitor watching at the same time.
fn who(session: &Session, line: &mut Vec<u8>) {
    if session.scripted() {
        line.extend_from_slice(b"lua");
        return;
    }
    let row = session.row();
    let text = row.text.lock();
    if text.peer.is_empty() {
        line.extend_from_slice(b"?:0");
    } else if row.flag(clients::UNIX) {
        line.extend_from_slice(b"unix:");
        let path = text.peer.strip_suffix(b":0").unwrap_or(&text.peer);
        line.extend_from_slice(path);
    } else {
        line.extend_from_slice(&text.peer);
    }
}

/// Which arguments must not be echoed, as a bit per argument.
///
/// A password is not a thing to put in a log, and a monitor is a log. Redis
/// replaces the argument with `(redacted)` rather than leaving it out, so the
/// line still has the shape the command had. `HELLO ... AUTH` is the only place
/// yo has to do it: `AUTH` is not a command here yet, and every other command
/// that carries a secret is a `CONFIG` subcommand, which no monitor is shown at
/// all.
fn redacted(args: Args<'_>) -> u32 {
    if !super::args::is(args.name(), b"hello") {
        return 0;
    }
    let mut mask = 0;
    for i in 2..args.len() {
        if super::args::is(args.get(i), b"AUTH") && i + 2 < args.len() {
            mask |= (1 << (i + 1)) | (1 << (i + 2));
        }
    }
    mask
}

/// One argument, quoted the way Redis quotes it.
///
/// Which is `sdscatrepr`: a backslash for a quote and a backslash, the five
/// short escapes, and `\xHH` for everything else outside printable ASCII. A
/// space is not escaped, so an argument with one in it is only told apart from
/// two arguments by the quotes around it, and that is the whole reason there are
/// quotes.
fn quote(arg: &[u8], line: &mut Vec<u8>) {
    line.push(b'"');
    for &b in arg {
        match b {
            b'\\' | b'"' => {
                line.push(b'\\');
                line.push(b);
            }
            b'\n' => line.extend_from_slice(b"\\n"),
            b'\r' => line.extend_from_slice(b"\\r"),
            b'\t' => line.extend_from_slice(b"\\t"),
            0x07 => line.extend_from_slice(b"\\a"),
            0x08 => line.extend_from_slice(b"\\b"),
            0x20..=0x7e => line.push(b),
            other => {
                let _ = write!(line, "\\x{other:02x}");
            }
        }
    }
    line.push(b'"');
}

/// Whether this command is one no monitor is shown.
///
/// Redis hides the administrative commands, on the grounds that they are too
/// dangerous to echo, and it decides that per subcommand rather than per
/// container: `CLIENT ID` is shown and `CLIENT LIST` is not. yo's table has one
/// row per container, so the subcommands are named here instead, and they were
/// read off 8.10.1's own table rather than guessed at.
pub(super) fn hidden(spec: &super::table::Spec, args: Args<'_>) -> bool {
    if spec.flags.contains(&"admin") {
        return true;
    }
    let subs: &[&[u8]] = match spec.name {
        "client" => &[b"kill", b"list", b"no-evict", b"pause", b"unpause"],
        "config" => &[b"get", b"set", b"resetstat", b"rewrite"],
        "backup" => &[b"start", b"seal", b"abort", b"cleanup", b"status", b"list"],
        _ => return false,
    };
    args.len() > 1 && subs.iter().any(|sub| super::args::is(args.get(1), sub))
}
