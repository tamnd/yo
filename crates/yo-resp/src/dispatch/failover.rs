//! `FAILOVER`: handing the master's job to one of its replicas, on purpose.
//!
//! Everything else about replication is a machine that keeps running when a
//! server dies. This is the other case, the one where nobody has died and an
//! operator wants the master to be somewhere else: a kernel upgrade on the box
//! the master is on, a move to a bigger machine, a rack going away at a time
//! somebody chose. Doing that by killing the master and letting a sentinel
//! notice costs a window where writes are refused and a window before that where
//! writes are accepted and then lost. Doing it with this command costs neither.
//!
//! # How it is safe
//!
//! Three steps, and the order is the whole trick.
//!
//! First the master stops accepting writes. Not by closing anything, which would
//! turn every client into an error, but with the same pause `CLIENT PAUSE`
//! arms: a write that arrives is held on its connection until the pause lifts,
//! and the client sees a slow command rather than a failure. Reads carry on.
//!
//! Then the master waits for a replica to acknowledge every byte it has ever
//! written. Nothing is being written any more, so this is a wait that finishes,
//! and when it does there are two servers holding exactly the same history to
//! exactly the same offset.
//!
//! Then the master becomes a replica of that one and tells it to become the
//! master, both in the same breath: the `PSYNC` it sends carries a fourth word,
//! `FAILOVER`, and a replica that gets a `PSYNC` like that promotes itself and
//! then answers it. The old master is already caught up, so what it gets back is
//! `+CONTINUE` and no snapshot at all, and the pause lifts. From a client's point
//! of view the whole thing is one slow write.
//!
//! # What can go wrong, and what happens then
//!
//! A replica that never catches up, because it is far behind or because it is
//! gone. `TIMEOUT` puts a limit on the wait, and without `FORCE` running out of
//! time puts everything back the way it was. With `FORCE` it hands over anyway,
//! which loses whatever the replica had not got, and is a thing to ask for
//! deliberately, which is why the reference refuses `FORCE` unless a timeout and
//! a target are both named.
//!
//! A target that refuses the `PSYNC`, because it is a master already or because
//! it is following a different history. The old master takes its job back and
//! the pause lifts.
//!
//! Anything else, including a target that has stopped answering the phone
//! between being picked and being dialled. There is no timeout on that on
//! purpose: the server is now a replica of a server that is not there, which is
//! a state an operator can see in `INFO` and get out of with `FAILOVER ABORT`,
//! and guessing on their behalf would mean two servers deciding to be the master
//! at the same time.
//!
//! `FAILOVER ABORT` is the only way out, which is why `REPLICAOF` is refused
//! while this is going on. Two commands that both decide who the master is would
//! be two commands that can disagree.

use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64};
use std::time::Duration;

use yo_common::lock::Lock;
use yo_common::{Code, Error, Result};

use crate::reply::Out;

use super::Server;
use super::args::{self, Args};

/// How often the wait looks at whether a replica has caught up.
///
/// Redis checks in its cron, which runs at ten milliseconds. This is a thread
/// rather than a cron and could look as often as it likes, but the thing it is
/// waiting for is a round trip to another machine, so anything faster than the
/// network is a spin for nothing.
const WATCH: Duration = Duration::from_millis(10);

/// How long the pause on writes lasts, which is until something lifts it.
///
/// The pause is a deadline rather than a flag, so forever has to be a number.
/// Two hundred and eighty million years is the largest one that fits beside the
/// mode bit, and every way this command can end lifts the pause itself.
const FOREVER: u64 = u64::MAX >> 1;

/// Where a failover has got to, which is what `master_failover_state` reports.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(super) enum Stage {
    /// Nothing is going on, which is every server nearly all of the time.
    None = 0,
    /// Writes are paused and a replica is being waited for.
    WaitForSync = 1,
    /// A replica was picked and this server is now dialling it as its master.
    InProgress = 2,
}

impl Stage {
    fn from(n: u8) -> Stage {
        match n {
            1 => Stage::WaitForSync,
            2 => Stage::InProgress,
            _ => Stage::None,
        }
    }

    /// The word `INFO` prints, which is Redis's spelling of all three.
    fn word(self) -> &'static str {
        match self {
            Stage::None => "no-failover",
            Stage::WaitForSync => "waiting-for-sync",
            Stage::InProgress => "failover-in-progress",
        }
    }
}

/// Everything about a failover, all of it idle on a server that is not having
/// one, which is every server that is not being worked on right now.
#[derive(Default)]
pub(crate) struct Failover {
    /// Which of the three [`Stage`]s this is.
    stage: AtomicU8,
    /// Who is being handed the job.
    ///
    /// Set by the command when a target was named, and filled in by the wait
    /// when it was not and the first replica to catch up gets picked.
    target: Lock<Option<(String, u16)>>,
    /// When to stop waiting, in milliseconds, and zero for never.
    end_ms: AtomicU64,
    /// Whether running out of time means hand over anyway rather than give up.
    force: AtomicBool,
    /// Which attempt this is, so the watching thread of one that was called off
    /// lets itself go. The same arrangement the replica link uses.
    epoch: AtomicU64,
}

impl Server {
    /// Whether a failover is going on, which is what refuses `REPLICAOF` and
    /// what keeps `CLIENT UNPAUSE` from lifting the pause.
    #[must_use]
    pub(crate) fn failing_over(&self) -> bool {
        self.failover.stage.load(Relaxed) != Stage::None as u8
    }

    /// Which stage it is at, for `INFO` and for the two places that branch on
    /// the difference between waiting and having committed.
    #[must_use]
    pub(super) fn failover_stage(&self) -> Stage {
        Stage::from(self.failover.stage.load(Relaxed))
    }

    /// The word `master_failover_state` reports.
    #[must_use]
    pub(super) fn failover_word(&self) -> &'static str {
        self.failover_stage().word()
    }
}

/// Put everything back to no failover and let the writes through.
///
/// Called down every path that ends one, including the ones that ended it well.
/// Bumping the epoch is what tells a watching thread that the wait it is in is
/// no longer the wait anybody wants.
fn clear(server: &Server) {
    server.failover.epoch.fetch_add(1, Relaxed);
    server.failover.end_ms.store(0, Relaxed);
    server.failover.force.store(false, Relaxed);
    {
        let mut target = server.failover.target.lock();
        yo_alloc::allow(|| *target = None);
    }
    server.failover.stage.store(Stage::None as u8, Relaxed);
    server.unpause();
}

/// The failover worked: this server is a replica of the new master now.
///
/// Called by the link when the `PSYNC` it sent with `FAILOVER` on the end was
/// answered. What it was answered with does not matter here. `+CONTINUE` is what
/// should happen, since the whole point of the wait was that the two sides were
/// in step, but a full resync is not wrong, only expensive, and either way the
/// handover has happened and there is nothing left to undo.
pub(super) fn landed(server: &Server) {
    if server.failover_stage() == Stage::InProgress {
        clear(server);
    }
}

/// Give up, and take the job back if it had already been handed over.
///
/// The two callers are `FAILOVER ABORT` and the two ways it can fail on its own,
/// which are running out of time without `FORCE` and a target that would not
/// take the `PSYNC`.
pub(super) fn abort(server: &Arc<Server>) {
    if server.failover_stage() == Stage::InProgress {
        // Already a replica of the target, so undo that. The promotion is the
        // one `REPLICAOF NO ONE` does and is right for the same reason: this
        // server is about to start writing history again and the replicas that
        // were following it through the target have to be able to tell the two
        // apart.
        server.stop_following();
    }
    clear(server);
}

// ------------------------------------------------------------- the command

/// `FAILOVER [TO host port [FORCE]] [ABORT] [TIMEOUT ms]`.
///
/// The parsing is the reference's, quirks and all, because the quirks are
/// reachable and somebody's script is written against them. Each of the three
/// words is taken at most once and a second one is a syntax error rather than an
/// overwrite, `TO` needs both of its words present as words and not just as an
/// argument count, and `ABORT` is only `ABORT` when it is the whole command.
pub(super) fn execute(server: &Server, args: Args<'_>, out: &mut Out) -> Result<()> {
    if args.len() == 2 && args.get(1).eq_ignore_ascii_case(b"abort") {
        // Whether anything is going on is asked first, so that a server with no
        // handle behind it, which can never be failing over, answers the sentence
        // the reference answers rather than one about how it was started.
        if !server.failing_over() {
            return Err(Error::new(Code::Invalid, "No failover in progress."));
        }
        let Some(shared) = server.myself() else {
            return Err(embedded());
        };
        abort(&shared);
        out.ok();
        return Ok(());
    }
    let mut timeout = 0i64;
    let mut force = false;
    let mut target: Option<(&[u8], i64)> = None;
    let mut at = 1;
    while at < args.len() {
        let word = args.get(at);
        if word.eq_ignore_ascii_case(b"timeout") && at + 1 < args.len() && timeout == 0 {
            timeout = args.int(at + 1)?;
            if timeout <= 0 {
                return Err(Error::new(
                    Code::Invalid,
                    "FAILOVER timeout must be greater than 0",
                ));
            }
            at += 2;
        } else if word.eq_ignore_ascii_case(b"to") && at + 2 < args.len() && target.is_none() {
            // The port is read before the host is kept, which is the order the
            // reference reads them in and is why a bad port is answered with the
            // integer sentence rather than with a syntax error.
            let port = args.int(at + 2)?;
            target = Some((args.get(at + 1), port));
            at += 3;
        } else if word.eq_ignore_ascii_case(b"force") && !force {
            force = true;
            at += 1;
        } else {
            return Err(args::syntax());
        }
    }
    if server.failing_over() {
        return Err(Error::new(Code::Invalid, "FAILOVER already in progress."));
    }
    if server.following() {
        return Err(Error::new(
            Code::Invalid,
            "FAILOVER is not valid when server is a replica.",
        ));
    }
    if server.replica_count() == 0 {
        return Err(Error::new(
            Code::Invalid,
            "FAILOVER requires connected replicas.",
        ));
    }
    if force && (timeout == 0 || target.is_none()) {
        return Err(Error::new(
            Code::Invalid,
            "FAILOVER with force option requires both a timeout and target HOST and IP.",
        ));
    }
    // A named target has to be a replica of ours right now, and has to be one
    // that has finished its snapshot, because a replica still loading one has no
    // offset to catch up to and nothing to promote.
    let named = match target {
        Some((host, port)) => {
            let host = String::from_utf8_lossy(host).into_owned();
            let port = u16::try_from(port).unwrap_or(0);
            match server.replica_online_at(&host, port) {
                None => {
                    return Err(Error::new(
                        Code::Invalid,
                        "FAILOVER target HOST and PORT is not a replica.",
                    ));
                }
                Some(false) => {
                    return Err(Error::new(
                        Code::Invalid,
                        "FAILOVER target replica is not online.",
                    ));
                }
                Some(true) => Some((host, port)),
            }
        }
        None => None,
    };
    let Some(shared) = server.myself() else {
        return Err(embedded());
    };
    {
        let mut held = server.failover.target.lock();
        yo_alloc::allow(|| *held = named);
    }
    if timeout > 0 {
        let end = server.clock.now_ms() + timeout as u64;
        server.failover.end_ms.store(end, Relaxed);
    }
    server.failover.force.store(force, Relaxed);
    server
        .failover
        .stage
        .store(Stage::WaitForSync as u8, Relaxed);
    // Writes stop here and not a moment later, because everything after this
    // depends on the offset the replicas are chasing standing still.
    server.pause(FOREVER, false);
    let epoch = server.failover.epoch.load(Relaxed);
    let watching = Arc::clone(&shared);
    yo_alloc::allow(|| {
        let _ = std::thread::Builder::new()
            .name(String::from("yo-failover"))
            .spawn(move || watch(&watching, epoch));
    });
    out.ok();
    Ok(())
}

/// What an embedded server answers, since there is no `Arc` to start a thread
/// behind and no socket for a replica to have arrived on either.
fn embedded() -> Error {
    Error::new(
        Code::Invalid,
        "FAILOVER is not available on an embedded server",
    )
}

// ---------------------------------------------------------------- the wait

/// Watch until a replica catches up, or until the time runs out.
///
/// A thread of its own for the same reason the replica link has one: what it
/// does is wait, and the threads that serve connections are measured in
/// nanoseconds per command. It lives for as long as one failover's wait, which
/// is usually a few milliseconds.
fn watch(server: &Arc<Server>, epoch: u64) {
    while server.failover.epoch.load(Relaxed) == epoch
        && server.failover_stage() == Stage::WaitForSync
        && !server.stopping()
    {
        step(server);
        std::thread::sleep(WATCH);
    }
}

/// One look at whether the handover can happen yet.
fn step(server: &Arc<Server>) {
    let end = server.failover.end_ms.load(Relaxed);
    if end != 0 && end <= server.clock.now_ms() {
        if server.failover.force.load(Relaxed) {
            hand_over(server);
        } else {
            abort(server);
        }
        return;
    }
    let named = { server.failover.target.lock().clone() };
    match named {
        Some((host, port)) => {
            if server.replica_caught_up(&host, port) {
                hand_over(server);
            }
        }
        // Nobody was named, so the first replica to have everything gets it, and
        // is written down before the handover so that everything from here on is
        // about one server rather than about whoever happens to be ahead.
        None => {
            if let Some(at) = server.first_caught_up() {
                {
                    let mut held = server.failover.target.lock();
                    yo_alloc::allow(|| *held = Some(at));
                }
                hand_over(server);
            }
        }
    }
}

/// Become a replica of the target, which is where the command stops being about
/// waiting and starts being about doing.
///
/// The stage is moved first, because the link thread this starts reads it to
/// find out whether to put `FAILOVER` on the end of its `PSYNC`, and a link that
/// dialled before the stage moved would ask for an ordinary resynchronisation
/// and get one, leaving two masters.
fn hand_over(server: &Arc<Server>) {
    let Some((host, port)) = server.failover.target.lock().clone() else {
        return;
    };
    server
        .failover
        .stage
        .store(Stage::InProgress as u8, Relaxed);
    server.follow_for_failover(&host, port);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_three_stages_have_the_words_redis_prints() {
        assert_eq!(Stage::None.word(), "no-failover");
        assert_eq!(Stage::WaitForSync.word(), "waiting-for-sync");
        assert_eq!(Stage::InProgress.word(), "failover-in-progress");
    }

    #[test]
    fn a_stage_that_is_not_one_of_the_three_reads_as_no_failover() {
        assert!(Stage::from(0) == Stage::None);
        assert!(Stage::from(1) == Stage::WaitForSync);
        assert!(Stage::from(2) == Stage::InProgress);
        assert!(Stage::from(9) == Stage::None);
    }

    #[test]
    fn a_server_nobody_is_failing_over_says_so() {
        let server = Server::new();
        assert!(!server.failing_over());
        assert_eq!(server.failover_word(), "no-failover");
    }
}
