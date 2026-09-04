//! `BACKUP`, the container Redis 8.10 added for taking a copy of the dataset.
//!
//! A backup on a real server is three files in one directory: a base RDB of the
//! dataset as it was when `BACKUP START` was answered, an incremental append
//! only file holding every write since, and a manifest naming the two. The
//! point of the shape is that neither file is ever rewritten, so a copy tool can
//! be pointed at the directory while the server carries on writing, and `BACKUP
//! SEAL` is what says the set is closed and complete.
//!
//! ```text
//! backupdir/appendonly.aof.1.base.rdb    the dataset at START
//! backupdir/appendonly.aof.1.incr.aof    every write between START and SEAL
//! backupdir/appendonly.aof.manifest      which of the two is which
//! ```
//!
//! # What a backup is here
//!
//! The base file is [`yo_kv::Snapshot`] over all sixteen databases, written
//! inside `BACKUP START` before it answers. The incremental file is empty, and
//! it is empty because there is no append only file underneath this server to
//! carry the writes in between: durability here belongs to the one file the
//! keyspace already sits on. So a sealed backup is the dataset as it was at
//! `START` rather than as it was at `SEAL`, which is D-46 and is the one thing
//! about this that a client could notice.
//!
//! The state machine, the file names, the manifest, the paths `LIST` answers,
//! the words `STATUS` uses and every error sentence are the reference's, read
//! off a running 8.10.1 with a raw socket rather than taken from the
//! documentation. `snapshotting` is the one state that is never seen, because
//! the writing happens before `START` answers rather than in a fork behind it.
//!
//! # Where the files go
//!
//! `<dir>/<backupdirname>/`, which is `dir` from the command line and the
//! [`DIR_NAME`] this server cannot be moved off. `BACKUP LIST` answers absolute
//! paths, which is why the server has to know its directory at all.

use std::fs;
use std::path::PathBuf;

use yo_common::{Code, Error, Result};
use yo_kv::Snapshot;

use super::args::{self, Args};
use super::server::{REPORTED_VERSION, help};
use super::{DATABASES, Server};
use crate::reply::Out;

/// The directory under `dir` that a backup is written into.
///
/// `backupdirname` on a real server, where it is a setting that can be given at
/// startup. It is fixed here, and `CONFIG SET backupdirname backupdir` is
/// accepted as the write that changes nothing that every immutable setting
/// takes.
pub(super) const DIR_NAME: &str = "backupdir";

/// The one file whose name does not carry a sequence number.
const MANIFEST: &str = "appendonly.aof.manifest";

/// A second `START` while one is running.
const IN_PROGRESS: &str = "A backup is already in progress, ABORT it first";
/// A `START` while a sealed one is still on disk.
const SEALED_EXISTS: &str = "A sealed backup exists, CLEANUP it first";
/// A `SEAL` from any state but `incrementing`.
const NOT_READY: &str = "No backup ready to seal (must be in the incrementing state)";
/// An `ABORT` with nothing to abort.
const NO_BACKUP: &str = "No backup in progress";
/// A `CLEANUP` while one is running.
const RUNNING: &str = "Backup is in progress";
/// What `STATUS` reports as the error after an `ABORT`.
const ABORTED: &str = "aborted by user";

/// What state a backup is in, which is the word `BACKUP STATUS` answers.
///
/// Five on a real server and four here. The fifth is `snapshotting`, which is
/// the window between the fork starting and the base file being complete, and
/// there is no window here because the file is written before `START` answers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    /// Nothing has been started, or the last one was cleaned up.
    Idle,
    /// A base file is on disk and the backup is open.
    Incrementing,
    /// The set is closed and complete.
    Sealed,
    /// Somebody aborted it, or it could not be finished.
    Failed,
}

impl Phase {
    /// The word `STATUS` puts in the `state` field.
    const fn name(self) -> &'static str {
        match self {
            Phase::Idle => "idle",
            Phase::Incrementing => "incrementing",
            Phase::Sealed => "sealed",
            Phase::Failed => "failed",
        }
    }
}

/// Everything one server remembers about backups.
///
/// One at a time, which is the reference's rule and not a simplification: the
/// file names carry a sequence number but the manifest does not, so a second
/// backup running into the same directory would write over the first one's
/// manifest and leave a set that names files nobody can tell apart.
pub(super) struct State {
    /// Which of the four states this is in.
    phase: Phase,
    /// The second `START` was answered on, and zero when idle.
    start_time: i64,
    /// The second `SEAL` was answered on, and zero until then.
    end_time: i64,
    /// What went wrong, which `STATUS` reports and nothing branches on.
    error: String,
    /// The number this backup's two files carry in their names.
    seq: u64,
    /// The number the next `START` will take.
    ///
    /// It only ever goes up, so a directory that somebody copied a half
    /// finished backup out of cannot have a later one land on the same names.
    next: u64,
    /// `backup-sealed-ttl`: how many seconds a sealed backup is kept before it
    /// is cleaned up on its own. Zero is the default and means it is kept.
    ttl: u64,
}

impl Default for State {
    fn default() -> State {
        State {
            phase: Phase::Idle,
            start_time: 0,
            end_time: 0,
            error: String::new(),
            seq: 0,
            // One and not zero, so the first backup's files read the way the
            // reference's first backup reads on a fresh server.
            next: 1,
            ttl: 0,
        }
    }
}

impl State {
    /// What `CONFIG GET backup-sealed-ttl` answers.
    pub(super) const fn ttl(&self) -> u64 {
        self.ttl
    }

    /// What `CONFIG SET backup-sealed-ttl` writes.
    ///
    /// Setting it does not reach back over a backup that has already outlived
    /// it: the next maintenance turn is what notices, and it compares against
    /// whatever the setting says then. That is how the reference behaves and it
    /// is also the only reading that makes sense, since the setting is a policy
    /// and not a deadline stamped onto the files.
    pub(super) const fn set_ttl(&mut self, seconds: u64) {
        self.ttl = seconds;
    }
}

/// Run one `BACKUP` subcommand.
///
/// The table's arity is exactly two, so every subcommand here has already had
/// its argument count checked and none of them takes anything else.
pub(super) fn execute(server: &mut Server, args: Args<'_>, out: &mut Out) -> Result<()> {
    let sub = args.get(1);
    if args::is(sub, b"start") {
        start(server)?;
        out.ok();
    } else if args::is(sub, b"seal") {
        seal(server)?;
        out.ok();
    } else if args::is(sub, b"abort") {
        abort(server)?;
        out.ok();
    } else if args::is(sub, b"cleanup") {
        cleanup(server)?;
        out.ok();
    } else if args::is(sub, b"status") {
        status(server, out);
    } else if args::is(sub, b"list") {
        list(server, out);
    } else if args::is(sub, b"help") {
        help(out, BACKUP_HELP);
    } else {
        return Err(args::unknown_subcommand(sub, "BACKUP"));
    }
    Ok(())
}

/// `BACKUP START`, which writes the base file and opens the backup.
///
/// The whole dataset is written before this answers. On a real server the fork
/// does it behind the reply and the state is `snapshotting` while it runs, and
/// here there is no fork and nothing to wait for, so a client that sends
/// `START` and then `STATUS` sees `incrementing` and never the state in
/// between.
///
/// Nothing is left behind by a `START` that fails. The state stays where it
/// was, which for the usual case is idle, and the error goes back to the client
/// rather than into the `error` field: a command that answers `-ERR` has told
/// the caller already.
fn start(server: &mut Server) -> Result<()> {
    match server.backup.phase {
        Phase::Incrementing => return Err(Error::new(Code::Invalid, IN_PROGRESS)),
        Phase::Sealed => return Err(Error::new(Code::Invalid, SEALED_EXISTS)),
        Phase::Idle | Phase::Failed => {}
    }
    let seq = server.backup.next;
    let now = seconds(server);
    yo_alloc::allow(|| {
        let dir = dir(server);
        let (file, skipped) = image(server);
        fs::create_dir_all(&dir)
            .and_then(|()| fs::write(dir.join(base_name(seq)), &file))
            .map_err(failed)?;

        let b = &mut server.backup;
        b.phase = Phase::Incrementing;
        b.seq = seq;
        b.next = seq + 1;
        b.start_time = now;
        b.end_time = 0;
        // A graph or a vector index has no Redis type byte, so it cannot go in
        // an RDB file at all. The backup is still worth having and is still
        // taken, and the count goes in the field the reference uses to say that
        // what is on disk is not everything that was asked for. Refusing the
        // whole backup over one key of a type this format cannot carry would
        // leave a server with a vector set in it unable to back up anything.
        b.error = if skipped == 0 {
            String::new()
        } else {
            format!("{skipped} key(s) have no RDB shape and are not in this backup")
        };
        Ok(())
    })
}

/// `BACKUP SEAL`, which closes the set and writes the manifest.
///
/// The incremental file is created here and is empty. It is in the set because
/// the manifest names it and a loader reads the manifest, so a set without it
/// is a set a real `redis-server` will not start on.
fn seal(server: &mut Server) -> Result<()> {
    if server.backup.phase != Phase::Incrementing {
        return Err(Error::new(Code::Invalid, NOT_READY));
    }
    let seq = server.backup.seq;
    let now = seconds(server);
    yo_alloc::allow(|| {
        let dir = dir(server);
        fs::write(dir.join(incr_name(seq)), [])
            .and_then(|()| fs::write(dir.join(MANIFEST), manifest(seq)))
            .map_err(failed)?;
        server.backup.phase = Phase::Sealed;
        server.backup.end_time = now;
        Ok(())
    })
}

/// `BACKUP ABORT`, which throws away a backup that was never sealed.
///
/// The base file goes immediately, because the point of aborting is to stop
/// paying for it. What is left is the `failed` state and the sentence saying
/// who did it, which is what a client polling `STATUS` from somewhere else
/// needs to see rather than an idle server that looks like nothing happened.
fn abort(server: &mut Server) -> Result<()> {
    if server.backup.phase != Phase::Incrementing {
        return Err(Error::new(Code::Invalid, NO_BACKUP));
    }
    let seq = server.backup.seq;
    yo_alloc::allow(|| {
        let _ = fs::remove_file(dir(server).join(base_name(seq)));
        server.backup.phase = Phase::Failed;
        server.backup.error = ABORTED.to_string();
    });
    Ok(())
}

/// `BACKUP CLEANUP`, which takes the files away and goes back to idle.
///
/// Legal from every state but `incrementing`, and from idle it is a way of
/// saying so rather than an error, which is the reference's answer too.
fn cleanup(server: &mut Server) -> Result<()> {
    if server.backup.phase == Phase::Incrementing {
        return Err(Error::new(Code::Invalid, RUNNING));
    }
    yo_alloc::allow(|| discard(server));
    Ok(())
}

/// Take the current backup's files away and forget it.
///
/// Shared by `CLEANUP` and by the sealed timeout, which are the same act with
/// two different things asking for it. A file that is not there is not a
/// problem: `ABORT` already removed the base one, and a backup that was never
/// sealed never had the other two.
fn discard(server: &mut Server) {
    let seq = server.backup.seq;
    let dir = dir(server);
    for name in [base_name(seq), incr_name(seq), MANIFEST.to_string()] {
        let _ = fs::remove_file(dir.join(name));
    }
    let b = &mut server.backup;
    b.phase = Phase::Idle;
    b.start_time = 0;
    b.end_time = 0;
    b.error = String::new();
}

/// Drop a sealed backup that has outlived `backup-sealed-ttl`.
///
/// Called once per batch of commands rather than on a timer, which means a
/// server nobody is talking to keeps a sealed backup past its time. That is the
/// right trade here: the whole engine is turned by the same loop, and a
/// timeout that fires on an idle server would mean a thread whose only job is
/// to delete three files.
pub(super) fn expire(server: &mut Server) {
    let b = &server.backup;
    if b.phase != Phase::Sealed || b.ttl == 0 {
        return;
    }
    // Saturating, because the setting is a `u64` of seconds and a client is
    // free to send one that would run off the end of the clock. A deadline that
    // cannot be reached is a backup that is kept, which is the safe way round.
    let deadline = b.end_time.saturating_add_unsigned(b.ttl);
    if seconds(server) < deadline {
        return;
    }
    yo_alloc::allow(|| discard(server));
}

/// `BACKUP STATUS`, which is a map on RESP3 and the same pairs flat on RESP2.
///
/// The two times are seconds and not milliseconds, and they are zero rather
/// than absent when there is nothing to report, which is what makes this a map
/// of a fixed four pairs whatever state the server is in.
fn status(server: &Server, out: &mut Out) {
    let b = &server.backup;
    out.map(4);
    out.bulk(b"state");
    out.bulk(b.phase.name().as_bytes());
    out.bulk(b"error");
    out.bulk(b.error.as_bytes());
    out.bulk(b"start_time");
    out.int(b.start_time);
    out.bulk(b"end_time");
    out.int(b.end_time);
}

/// `BACKUP LIST`, the absolute paths of the files that are pinned so far.
///
/// One while the backup is open and three once it is sealed, in the order the
/// manifest names them. Nothing at all from idle or failed, since there is
/// nothing on disk to copy.
fn list(server: &Server, out: &mut Out) {
    let seq = server.backup.seq;
    let count = match server.backup.phase {
        Phase::Incrementing => 1,
        Phase::Sealed => 3,
        Phase::Idle | Phase::Failed => 0,
    };
    out.array(count);
    if count == 0 {
        return;
    }
    yo_alloc::allow(|| {
        let dir = dir(server);
        out.bulk(dir.join(base_name(seq)).to_string_lossy().as_bytes());
        if count == 3 {
            out.bulk(dir.join(incr_name(seq)).to_string_lossy().as_bytes());
            out.bulk(dir.join(MANIFEST).to_string_lossy().as_bytes());
        }
    });
}

/// The whole dataset as one RDB image, and how many keys could not go in it.
///
/// The three aux fields are the ones a real base file carries that mean
/// something to whoever reads it. `aof-base` is the one a loader acts on: it
/// says this file is the base of an append only file rather than a standalone
/// dump, which is what lets the manifest point at it.
fn image(server: &mut Server) -> (Vec<u8>, usize) {
    let bits: &[u8] = if usize::BITS == 64 { b"64" } else { b"32" };
    let mut snap = Snapshot::new();
    snap.aux(b"redis-ver", REPORTED_VERSION.as_bytes());
    snap.aux(b"redis-bits", bits);
    snap.aux(b"aof-base", b"1");
    for i in 0..DATABASES {
        snap.database(i, server.striped(i));
    }
    let skipped = snap.skipped();
    (snap.finish(), skipped)
}

/// The two lines that say which file is which.
///
/// One sequence number for both files, which is what a real server writes when
/// the backup is the first thing its append only file has been asked to do. The
/// offsets are replication offsets and are zero here for the same reason
/// `INFO replication` reports a zero: nothing has been written to a stream that
/// does not exist.
fn manifest(seq: u64) -> String {
    format!(
        "file {} seq {seq} type b\nfile {} seq {seq} type i startoffset 0 endoffset 0\n",
        base_name(seq),
        incr_name(seq),
    )
}

/// Where this server's backups go, as an absolute path.
fn dir(server: &Server) -> PathBuf {
    server.dir().join(DIR_NAME)
}

/// The base file's name for one sequence number.
fn base_name(seq: u64) -> String {
    format!("appendonly.aof.{seq}.base.rdb")
}

/// The incremental file's name for one sequence number.
fn incr_name(seq: u64) -> String {
    format!("appendonly.aof.{seq}.incr.aof")
}

/// The wall clock second, which is the unit both times in `STATUS` are in.
fn seconds(server: &Server) -> i64 {
    (server.clock.now_ms() / 1_000) as i64
}

/// What a client is told when the files could not be written.
///
/// The path is not in it. A client that asked for a backup does not necessarily
/// know where the server keeps them and does not necessarily have any business
/// knowing, and the reason a write failed is the same reason whichever file it
/// was.
fn failed(e: std::io::Error) -> Error {
    Error::fmt(Code::Io, format_args!("Backup failed: {e}"))
}

/// What `BACKUP HELP` says.
///
/// Seventeen lines with `HELP` in it twice, which is an upstream bug and is
/// copied as it is. The sentence a client sees should be the sentence every
/// other server sends, and a tidied up help text here would be the one server
/// in the world whose output does not match the rest.
const BACKUP_HELP: &[&str] = &[
    "BACKUP <subcommand> [<arg> [value] [opt] ...]. Subcommands are:",
    "START",
    "    Start a new backup into the configured 'backupdirname'.",
    "SEAL",
    "    Freeze the current backup (BASE + INCR + manifest).",
    "ABORT",
    "    Cancel a backup that has not been sealed yet.",
    "CLEANUP",
    "    Remove a sealed backup's files and return to idle.",
    "STATUS",
    "    Report the current backup state.",
    "LIST",
    "    List the immutable files pinned so far.",
    "HELP",
    "    Return this help.",
    "HELP",
    "    Print this help.",
];
