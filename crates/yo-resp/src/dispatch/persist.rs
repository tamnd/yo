//! `SAVE`, `BGSAVE`, `BGREWRITEAOF`, `LASTSAVE` and `ROLE`.
//!
//! Four commands about writing the dataset to a file and one about who this
//! server is. They are together because a client asks them together: a backup
//! script says `BGSAVE`, polls `LASTSAVE` until the number moves and then reads
//! the file, and a tool that is about to do any of that asks `ROLE` first so it
//! does not take a backup off a replica by accident.
//!
//! # There is already a file, so why write another one
//!
//! Because the two files are for different readers. The one the keyspace sits on
//! is this server's own, it is written as the commands run and it is what makes
//! a restart cheap. What `SAVE` writes is an RDB, which is the format every
//! other thing in the Redis world can read: `redis-check-rdb`, a real
//! `redis-server` told to start on it, a migration tool, a replica being seeded.
//! So this is an export and not a checkpoint, and the durability of the data was
//! never waiting on it.
//!
//! That is also why nothing here ever runs on its own. Redis writes a snapshot
//! when enough keys have changed because the snapshot is the only copy; here it
//! would be a second copy of something already on disk, so `save` is empty and
//! stays empty, and a file appears when somebody asks for one.
//!
//! # Why the background one is not in the background
//!
//! Redis forks, and the child writes the file out of a copy on write image while
//! the parent carries on. There is no fork here, so `BGSAVE` writes the file
//! before it answers and then says `Background saving started`, which is
//! D-125. The reply is the reply a client is waiting for and the file is on disk
//! by the time it arrives, so a script that polls `LASTSAVE` afterwards sees the
//! number it was waiting for on the first read rather than the third. What a
//! client cannot see is `rdb_bgsave_in_progress` going to one, because there is
//! no window in which it is true.
//!
//! # Why the rewrite writes nothing
//!
//! There is no append only file to rewrite. `BGREWRITEAOF` answers the sentence
//! a real server answers, counts itself and does nothing, which is D-126.
//! `appendonly` reads `no` here and cannot be set to `yes`, so a client that
//! looks before it asks already knows there is nothing to rewrite, and the
//! command is here because tooling calls it blind on the way to something else.

use std::fs;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize};

use yo_common::Result;
use yo_kv::Snapshot;

use super::args::{self, Args, is};
use super::server::REPORTED_VERSION;
use super::table::Spec;
use super::{DATABASES, Server, Session};
use crate::reply::Out;

/// What the file is called, which is also what `CONFIG GET dbfilename` answers.
///
/// Fixed, because `dbfilename` is a protected config on a real server too: it
/// is refused by `CONFIG SET` with a sentence about protection unless the server
/// was started with protected configs turned on, and nothing in this build turns
/// them on.
pub(super) const FILE: &str = "dump.rdb";

/// What a save has done, so the persistence section has something to report.
///
/// Every field is a number a client polls rather than one a command reads, so
/// they are relaxed atomics next to each other rather than one lock around the
/// set of them. Nothing here is read in the same breath as anything else: a
/// client asking `INFO persistence` is asking five separate questions that
/// happen to arrive in one string.
#[derive(Debug)]
pub(crate) struct Persistence {
    /// When the last save that worked finished, in whole seconds, or zero for a
    /// server that has not saved.
    at: AtomicI64,
    /// How many saves have been asked for, whichever spelling asked.
    ///
    /// Attempts and not successes, which is Redis's rule: it bumps the counter
    /// on the way in and reports the failures separately.
    saves: AtomicU64,
    /// Whether the last one worked.
    ok: AtomicBool,
    /// Keys the last file could not carry.
    skipped: AtomicUsize,
    /// How many rewrites have been asked for.
    rewrites: AtomicU64,
}

impl Default for Persistence {
    fn default() -> Persistence {
        Persistence {
            at: AtomicI64::new(0),
            saves: AtomicU64::new(0),
            // A server that has never saved reports the last one as having
            // worked, which is Redis's answer too and is the only one that lets
            // a monitoring rule read the field without a special case for a
            // server that has just started.
            ok: AtomicBool::new(true),
            skipped: AtomicUsize::new(0),
            rewrites: AtomicU64::new(0),
        }
    }
}

impl Server {
    /// When the dataset was last written out, in seconds since the epoch.
    ///
    /// A server that has not saved answers the second it started, because Redis
    /// does: it considers the dataset saved at startup on the grounds that what
    /// is in memory came off a file and has not moved since.
    #[must_use]
    pub(crate) fn last_save(&self) -> i64 {
        match self.persist.at.load(Relaxed) {
            0 => (self.started_ms / 1_000) as i64,
            at => at,
        }
    }
}

/// One of the five, by name.
///
/// # Errors
///
/// [`yo_common::Code::Invalid`] for a word after `BGSAVE` that is not
/// `SCHEDULE`. A save that could not write its file does not come back through
/// here at all: it writes its own error line, for the reason [`save`] gives.
pub(super) fn execute(
    server: &Server,
    session: &Session,
    spec: &Spec,
    args: Args<'_>,
    out: &mut Out,
) -> Result<()> {
    match spec.name {
        "save" => save(server, out),
        "bgsave" => {
            // The only argument, and it changes nothing here. On a real server
            // it says to queue the save rather than fail when another child is
            // already running, and there is no child to be waiting on.
            if args.len() > 1 && !(args.len() == 2 && is(args.get(1), b"schedule")) {
                return Err(args::syntax());
            }
            save_in_background(server, session, out);
        }
        "bgrewriteaof" => rewrite(server, session, out),
        "lastsave" => out.int(server.last_save()),
        "role" => role(out),
        _ => return Err(args::unknown_command(args)),
    }
    Ok(())
}

/// `SAVE`, which writes the file and waits for it.
///
/// The failure reply is a bare `-ERR` with nothing after it. That is not a
/// message this file forgot to write: it is `shared.err`, the reply Redis has
/// for a save that failed, and the reason it says nothing is that whatever went
/// wrong went wrong in the file system and is in the server's log rather than in
/// a sentence a client could act on. It is written here rather than returned as
/// an error because the error writer puts a space after `ERR` and this line has
/// nothing to put after the space.
fn save(server: &Server, out: &mut Out) {
    if write_file(server) {
        out.ok();
    } else {
        out.error(b"ERR");
    }
}

/// `BGSAVE [SCHEDULE]`, which writes the file and says it started.
///
/// Two sentences and not one. Inside a transaction a real server cannot fork,
/// because the fork would land in the middle of a batch that has been promised
/// to run without anything in between, so it queues the save for the next tick
/// and answers `scheduled` instead of `started`. Nothing here forks and so
/// nothing here has that problem, and the two sentences are still told apart,
/// because a client that reads them apart is a client reading them for a reason.
///
/// A failure is not reported. Redis answers `-ERR` when the fork itself fails
/// and `+Background saving started` when the fork succeeds and the child then
/// fails, and the second is the one this is: the work that could fail happens
/// after the point where a real server has already answered. So a failed save
/// shows up where a failed background save shows up on a real server, which is
/// `rdb_last_bgsave_status`.
fn save_in_background(server: &Server, session: &Session, out: &mut Out) {
    let scheduled = session.running();
    write_file(server);
    if scheduled {
        out.simple(b"Background saving scheduled");
    } else {
        out.simple(b"Background saving started");
    }
}

/// `BGREWRITEAOF`, which counts itself and does nothing else.
///
/// The same two sentences for the same reason as [`save_in_background`].
fn rewrite(server: &Server, session: &Session, out: &mut Out) {
    server.persist.rewrites.fetch_add(1, Relaxed);
    if session.running() {
        out.simple(b"Background append only file rewriting scheduled");
    } else {
        out.simple(b"Background append only file rewriting started");
    }
}

/// `ROLE`, which is what a client asks before it trusts anything else it reads.
///
/// Three elements: the word, then a number whose meaning depends on the word,
/// then a list whose shape depends on it too. A master answers its replication
/// offset and the replicas attached to it, and this server is a master with no
/// replicas and nothing written to a stream that does not exist yet, which is
/// the same zero `INFO replication` reports next to it.
fn role(out: &mut Out) {
    out.array(3);
    out.bulk(b"master");
    out.int(0);
    out.array(0);
}

/// Write the whole dataset out, and say whether it worked.
///
/// Into a temporary name first and then renamed over the old file, which is what
/// a real server does and is the only way the file is either the old dataset or
/// the new one and never half of each. A reader that opens `dump.rdb` while this
/// is running gets whichever of the two the rename has got to, and both of them
/// are files that load.
///
/// The image is built in memory before any of it is written. That costs the size
/// of the dataset and it is the same trade [`super::backup`] makes, for the same
/// reason: the writer hands back a buffer rather than taking a sink, and turning
/// it into one is the borrowing walk that is a bigger change than this file
/// should make.
fn write_file(server: &Server) -> bool {
    server.persist.saves.fetch_add(1, Relaxed);
    let done = yo_alloc::allow(|| {
        let (image, skipped) = build(server);
        server.persist.skipped.store(skipped, Relaxed);
        let dir = server.dir().to_path_buf();
        let temp = dir.join(format!("temp-{}.rdb", std::process::id()));
        match spill(&temp, &image) {
            Ok(()) => fs::rename(&temp, dir.join(FILE)).is_ok(),
            Err(()) => {
                // A half written temporary file is not the old dataset and not
                // the new one, and leaving it behind would mean the next save
                // has to think about what it found.
                let _ = fs::remove_file(&temp);
                false
            }
        }
    });
    if done {
        server.persist.at.store(seconds(server), Relaxed);
    }
    server.persist.ok.store(done, Relaxed);
    done
}

/// The bytes on disk, flushed and synced before the rename sees them.
///
/// Synced because the point of the file is that it survives the machine and not
/// only the process, and a rename over an unsynced file is a file that can come
/// back empty after a power cut.
fn spill(temp: &PathBuf, image: &[u8]) -> core::result::Result<(), ()> {
    let mut file = fs::File::create(temp).map_err(|_| ())?;
    file.write_all(image).map_err(|_| ())?;
    file.sync_all().map_err(|_| ())
}

/// The whole dataset as one RDB image, and how many keys could not go in it.
///
/// Every database in turn, and the stripes of each are taken one at a time. So
/// this is not one instant of the whole keyspace: a write to database nine while
/// database two is being walked is in the file and a write to database two after
/// it has been walked is not. A real server forks and gets an instant for free,
/// and buying one here would mean holding every stripe of every database at once
/// while megabytes are written, which would stop the server for as long as the
/// save takes. The same trade [`super::backup`] makes and the same reason.
fn build(server: &Server) -> (Vec<u8>, usize) {
    let bits: &[u8] = if usize::BITS == 64 { b"64" } else { b"32" };
    let mut snap = Snapshot::new();
    snap.aux(b"redis-ver", REPORTED_VERSION.as_bytes());
    snap.aux(b"redis-bits", bits);
    snap.aux(b"ctime", seconds(server).to_string().as_bytes());
    // Zero and not one, which is the difference between this file and the one
    // `BACKUP` writes: that one is the base of an append only file and this one
    // stands on its own.
    snap.aux(b"aof-base", b"0");
    for i in 0..DATABASES {
        snap.database(i, server.striped(i));
    }
    let skipped = snap.skipped();
    (snap.finish(), skipped)
}

/// The wall clock second, which is the unit every time in here is in.
fn seconds(server: &Server) -> i64 {
    (server.clock.now_ms() / 1_000) as i64
}

/// The `Persistence` section of `INFO`.
///
/// Redis reports thirty odd fields here and this reports thirteen. What is
/// missing is missing because it is about a fork, an append only file or a load,
/// and there is no fork, no append only file and no load off one of these files:
/// `rdb_bgsave_in_progress` would be a zero that is never anything else,
/// `rdb_last_cow_size` would be a zero about a copy that never happens, and
/// `rdb_last_load_keys_loaded` would be a zero about a file this server has
/// never started from. A field that is not there is a client falling back, and a
/// field that is there and always zero is a client believing it.
///
/// `rdb_changes_since_last_save` is the one absence that is a gap rather than a
/// decision. It is Redis's dirty counter, which every command adds the number of
/// changes it made to, and yo has no such counter yet. Reporting the number of
/// write commands under that name would be a different number wearing the same
/// label, so it waits for D-113.
pub(super) fn info(server: &Server, s: &mut String) {
    use core::fmt::Write as _;
    let p = &server.persist;
    let status = |ok: bool| if ok { "ok" } else { "err" };
    let _ = write!(
        s,
        "# Persistence\r\nloading:0\r\nasync_loading:0\r\n\
         rdb_bgsave_in_progress:0\r\nrdb_last_save_time:{}\r\n\
         rdb_last_bgsave_status:{}\r\nrdb_saves:{}\r\n\
         yo_rdb_last_save_skipped_keys:{}\r\n\
         aof_enabled:0\r\naof_rewrite_in_progress:0\r\n\
         aof_rewrite_scheduled:0\r\naof_last_bgrewrite_status:ok\r\n\
         aof_rewrites:{}\r\naof_last_write_status:ok\r\n\r\n",
        server.last_save(),
        status(p.ok.load(Relaxed)),
        p.saves.load(Relaxed),
        p.skipped.load(Relaxed),
        p.rewrites.load(Relaxed),
    );
}

/// Write the dataset out because a client asked for it on the way out.
///
/// `SHUTDOWN SAVE` and nothing else. A real server saves on the way down when it
/// has save points configured or when the client said so, and this one has no
/// save points and never will, so the word is the whole of it.
pub(super) fn on_shutdown(server: &Server) {
    write_file(server);
}
