//! Being a replica: the link out to a master and the stream that comes back.
//!
//! The other half of `repl`. That module is what a master does to feed somebody
//! else, and this is what a server does to be fed. The two never run against
//! each other on one server unless somebody has built a chain, and a chain is
//! the case this file is careful about rather than the case it is written for.
//!
//! # What the link is
//!
//! One thread, one socket, and a loop that never ends until somebody says
//! `REPLICAOF NO ONE` or the server stops. It dials the master, walks the
//! handshake, takes the snapshot, and then reads commands off the socket and
//! runs them against this server's own keyspace forever. A link that breaks is
//! not an error to report to anybody, because there is nobody to report it to:
//! the client that said `REPLICAOF` was answered `OK` the moment the intent was
//! recorded, which is what a real server does and is the only thing it can do
//! when the dial has not happened yet. So a broken link waits a second and dials
//! again, and `INFO` is where an operator finds out.
//!
//! A thread of its own rather than a slot on the reactor. The reactor's threads
//! are woken by clients and this has no client, the handshake is a handful of
//! blocking round trips and the snapshot is one very large read, and all three
//! of those are the wrong shape for an event loop that is measured in
//! nanoseconds per command. One thread that is asleep on a socket for most of
//! its life costs a stack.
//!
//! # Why the commands go through the front door
//!
//! What arrives is a stream of ordinary commands, so what runs them is the
//! ordinary dispatcher, through a [`Session`] the link owns. That is not a
//! shortcut, it is the point: a replica that applied writes through some second
//! path would be a second implementation of every command, and the first bug in
//! it would be a replica that quietly disagrees with its master. Going through
//! the front door also means keyspace notifications fire on the replica, the
//! search indexes are kept up, and `WATCH` on the replica notices, all of which
//! a real replica does and none of which had to be written twice.
//!
//! Three things about that session are not ordinary. It is past the password and
//! past the access control list, because the master is not a user and there is
//! nobody to authenticate. It is exempt from the read only refusal, which is the
//! whole point of the refusal. And it is exempt from `CLIENT PAUSE`, because a
//! pause is a thing an operator does to clients and a master is not one, and a
//! paused replica that stopped reading its socket would make the master's output
//! buffer grow until the master dropped the link.
//!
//! # The offset
//!
//! The replica counts the bytes it has applied and tells the master about them,
//! and the master compares that number with its own to answer `WAIT` and to fill
//! in the lag in `INFO`. So the count has to be of the bytes as they arrived and
//! not of anything this server decided: what is added is exactly what the
//! decoder said it consumed, including the commands that did nothing, including
//! the `PING`s the master sends to keep the link warm, and including the
//! `REPLCONF GETACK` that asks for the number itself, which is why the answer to
//! a `GETACK` is sent after its own bytes have been counted.
//!
//! # A replica with replicas
//!
//! A chain works by passing the bytes on rather than by propagating what the
//! commands did. The two are not the same thing and only one of them can be:
//! the offset a sub-replica acknowledges has to be a position in the master's
//! stream, and a middle server that made up its own stream would be handing out
//! positions in a history nobody else is writing. So while this link is
//! applying, the ordinary propagation is turned off for the thread and the bytes
//! that arrived are put on this server's stream unchanged, under the master's
//! own replication id and at the master's own offsets.

use core::cell::Cell;
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::Arc;
use std::sync::atomic::Ordering::{Relaxed, Release};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64};
use std::time::{Duration, Instant};

use yo_common::lock::Lock;
use yo_common::{Code, Error, Result};

use crate::proto::{Limits, Proto};
use crate::reply::Out;
use crate::request::{Argv, Step};

use super::args::{self, Args};
use super::repl::{self, ID_LEN};
use super::{Flow, Server, Session};

/// How long a dial is given before it is called a failure.
const DIAL_TIMEOUT: Duration = Duration::from_secs(5);

/// How long a read waits before the loop looks around at other things.
///
/// Short, because this is also how often the link notices it has been called off
/// and how close to the second an acknowledgement lands. A tenth of a second on
/// a socket that is usually idle is a syscall ten times a second, which is
/// nothing next to a thread that is otherwise asleep.
const POLL: Duration = Duration::from_millis(100);

/// How often the replica tells the master where it has got to.
///
/// A second, which is Redis's `REPLCONF ACK` period. The master turns the gap
/// between acknowledgements into the lag it reports, so a longer period would
/// make every replica look worse than it is.
const ACK_EVERY: Duration = Duration::from_secs(1);

/// How long to wait before dialling again after a link went down.
const RETRY: Duration = Duration::from_millis(500);

/// How long the whole handshake is given, per attempt.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// How long a master is given to start answering a `PSYNC`, and how long a gap
/// in the snapshot is allowed to be before the link is called broken.
///
/// Much longer than the handshake, because what happens between the `PSYNC` and
/// the first byte of the snapshot is a fork and a save on a machine holding a
/// dataset this one has not seen yet. Redis's own replica gives it `repl-timeout`
/// and that is a minute by default, so this is a minute.
const SYNC_TIMEOUT: Duration = Duration::from_secs(60);

/// The longest line the handshake will read before calling the peer broken.
///
/// A handshake reply is a word and at most an id and a number. Anything past
/// this is a peer that is not answering the question that was asked, which is
/// the same rule and the same reasoning `MIGRATE` uses.
const LINE_MAX: usize = 1024;

/// What `master_link_status` says, which is also what the link is doing.
///
/// Redis reports two words where there are four states, so `Connect` and `Sync`
/// both read as `down`, and the one that tells them apart is
/// `master_sync_in_progress` beside it.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum State {
    /// Not following anybody. This server is a master.
    None = 0,
    /// Following somebody and not currently connected to them.
    Connect = 1,
    /// Connected and loading the snapshot.
    Sync = 2,
    /// Connected and applying the stream.
    Up = 3,
}

impl State {
    fn from(n: u8) -> State {
        match n {
            1 => State::Connect,
            2 => State::Sync,
            3 => State::Up,
            _ => State::None,
        }
    }
}

/// Where this server has been told to follow.
#[derive(Clone)]
struct Upstream {
    host: String,
    port: u16,
}

/// Everything about being a replica, all of it idle on a server that is nobody's.
pub(crate) struct Follower {
    /// Who this server follows, and none when it is a master.
    ///
    /// Written by `REPLICAOF` and read by the link thread, by `INFO` and by
    /// `ROLE`. A lock and not an atomic because it is a host name, and it is
    /// touched once per link rather than once per command.
    upstream: Lock<Option<Upstream>>,
    /// Which link is the current one.
    ///
    /// Every `REPLICAOF` bumps this and the thread it starts carries the number
    /// it was started with. A thread whose number is no longer the current one
    /// has been replaced and lets itself go at the next thing it does, which is
    /// how a link is called off without anything having to reach into a blocking
    /// socket read.
    epoch: AtomicU64,
    /// What the link is doing, one of [`State`].
    state: AtomicU8,
    /// Whether an ordinary client's write is refused, which is Redis's
    /// `replica-read-only` and is on by default.
    ///
    /// Read once per write command on a server that is not a replica, next to
    /// the pause check and the freeze check that are already there, and the load
    /// it reads is of a bool that is false.
    read_only: AtomicBool,
    /// Whether this server is following anybody at all, so the read only check
    /// above is one load rather than a lock.
    on: AtomicBool,
    /// The user and password the link authenticates with, both empty when the
    /// master wants neither.
    auth: Lock<(Vec<u8>, Vec<u8>)>,
    /// The port to tell the master this server listens on, which is what the
    /// master reports in its own `INFO` and in `ROLE`.
    ///
    /// Zero when nobody has said, which is every embedded caller and every test,
    /// and is what a master shows for a replica that did not say either.
    port: AtomicU64,
    /// When the link last had anything out of the master, for
    /// `master_last_io_seconds_ago`.
    last_io_ms: AtomicU64,
    /// When the link last went down, for `master_link_down_since_seconds`.
    down_ms: AtomicU64,
    /// Whether this server's replication id and offset came from a master, so
    /// the next `PSYNC` may ask to carry on rather than starting again.
    ///
    /// The position itself is not kept here on purpose. It is the server's own
    /// replication offset, which `adopt` sets to the master's and which every
    /// applied byte moves along, so there is one number rather than two that
    /// could disagree. A second copy updated at the end of the stream loop would
    /// be a copy that is behind by whatever the link died in the middle of, and
    /// asking to carry on from behind is asking for the same bytes twice.
    ///
    /// Kept across a broken link, which is the whole reason a partial resync is
    /// possible at all, and kept across a change of master too, which is what
    /// lets a replica be handed to a promoted one without a snapshot. Thrown
    /// away by `REPLICAOF NO ONE`, because that takes a new id and asking a
    /// master about a history it has never heard of is a full resync with an
    /// extra round trip in front of it.
    resume: AtomicBool,
}

impl Default for Follower {
    fn default() -> Follower {
        Follower {
            upstream: Lock::new(None),
            epoch: AtomicU64::new(0),
            state: AtomicU8::new(State::None as u8),
            read_only: AtomicBool::new(true),
            on: AtomicBool::new(false),
            auth: Lock::new((Vec::new(), Vec::new())),
            port: AtomicU64::new(0),
            last_io_ms: AtomicU64::new(0),
            down_ms: AtomicU64::new(0),
            resume: AtomicBool::new(false),
        }
    }
}

thread_local! {
    /// Whether this thread is applying a master's stream rather than running a
    /// client's command.
    ///
    /// Read by the propagation site, which has no other way to tell the two
    /// apart and has to, because what a replica passes on is the bytes it was
    /// given and not what running them turned out to do. See the module header.
    static APPLYING: Cell<bool> = const { Cell::new(false) };
}

/// Whether what is running arrived from a master.
#[must_use]
pub(crate) fn applying() -> bool {
    APPLYING.get()
}

impl Server {
    /// Whether this server is following somebody, which is the whole cost of
    /// this file on a server that is not.
    #[must_use]
    pub(crate) fn following(&self) -> bool {
        self.follow.on.load(Relaxed)
    }

    /// Whether an ordinary client's write is refused here.
    ///
    /// Both halves, because a server that is a replica and has been told it is
    /// writable takes writes, and a server that is not a replica at all is not
    /// made read only by the setting sitting there at its default.
    #[must_use]
    pub(crate) fn read_only_replica(&self) -> bool {
        self.follow.on.load(Relaxed) && self.follow.read_only.load(Relaxed)
    }

    /// Say what port to announce to a master, which is what it reports back.
    ///
    /// Called once by whoever bound the socket, which is the only place that
    /// knows. A server nobody tells announces nothing, which is what a real
    /// master shows for a replica that did not say.
    pub fn announce_port(&self, port: u16) {
        self.follow.port.store(u64::from(port), Relaxed);
    }

    /// The port whoever bound the socket said this server is on, which `INFO`
    /// reports and a cluster node writes into its config file.
    ///
    /// Nought on an embedded caller that never opened a socket, which is what a
    /// reader should see rather than a guess.
    #[must_use]
    pub(crate) fn announced_port(&self) -> u16 {
        self.follow.port.load(Relaxed) as u16
    }

    /// Say what the link should authenticate with, which is Redis's
    /// `masteruser` and `masterauth`.
    pub fn master_auth(&self, user: &[u8], pass: &[u8]) {
        let mut auth = self.follow.auth.lock();
        yo_alloc::allow(|| *auth = (user.to_vec(), pass.to_vec()));
    }

    /// Read back what the link would authenticate with, for `CONFIG GET`.
    pub(crate) fn with_master_auth<T>(&self, each: impl FnOnce(&[u8], &[u8]) -> T) -> T {
        let auth = self.follow.auth.lock();
        each(&auth.0, &auth.1)
    }

    /// Whether an ordinary client's write is refused while this server is a
    /// replica, which is Redis's `replica-read-only`.
    ///
    /// Writable on a running server, and a change takes effect on the next
    /// command rather than on the next link, because it is a rule about clients
    /// and not about the master.
    pub fn set_replica_read_only(&self, yes: bool) {
        self.follow.read_only.store(yes, Relaxed);
    }

    /// The setting on its own, which is what `CONFIG GET` answers whether or not
    /// this server is a replica.
    pub(crate) fn replica_read_only_setting(&self) -> bool {
        self.follow.read_only.load(Relaxed)
    }

    /// Start following a master, for a server that was told to at startup.
    ///
    /// The same thing `REPLICAOF host port` does, reachable before anything has
    /// connected. It is a separate entry point rather than a command run against
    /// the server, because the caller has the `Arc` in its hand and a command
    /// body does not.
    pub fn follow_master(self: &Arc<Server>, host: &str, port: u16) {
        self.is_behind();
        let host = yo_alloc::allow(|| host.to_owned());
        self.follow_now(Some(Upstream { host, port }));
    }

    /// Whether the link to the master is up and applying, which is the question
    /// a `PSYNC` from somebody else asks before it trusts what we would send.
    #[must_use]
    pub(crate) fn master_link_up(&self) -> bool {
        State::from(self.follow.state.load(Relaxed)) == State::Up
    }

    /// Stop following anybody, which is `REPLICAOF NO ONE` and the two ways a
    /// failover ends up back where it started.
    ///
    /// The promotion is the part that matters: the history that was being
    /// followed is kept as the second id and a new one is taken, so a replica
    /// that was following this server through the old master can be handed over
    /// without a snapshot. See `repl::promote`.
    pub(super) fn stop_following(self: &Arc<Server>) {
        if self.following() {
            self.promote();
        }
        self.follow_now(None);
    }

    /// Start following the server a failover picked.
    ///
    /// Two things make this different from `REPLICAOF host port`. There is no
    /// promotion, because this server is handing its history over rather than
    /// starting a new one, and the next `PSYNC` asks to carry on from where this
    /// server has got to rather than starting again, because the target is
    /// caught up to exactly there and a snapshot would be a copy of what it
    /// already holds.
    pub(super) fn follow_for_failover(self: &Arc<Server>, host: &str, port: u16) {
        self.follow.resume.store(true, Relaxed);
        let host = yo_alloc::allow(|| host.to_owned());
        self.follow_now(Some(Upstream { host, port }));
    }

    /// Start following, or stop.
    ///
    /// The intent is recorded and a thread is started, and the answer goes back
    /// before the dial has been tried, which is what a real server does: there
    /// is no reply to hold open while a socket is opened to somewhere that might
    /// not answer for five seconds.
    fn follow_now(self: &Arc<Server>, to: Option<Upstream>) {
        let epoch = self.follow.epoch.fetch_add(1, Relaxed) + 1;
        {
            let mut upstream = self.follow.upstream.lock();
            yo_alloc::allow(|| *upstream = to.clone());
        }
        let Some(to) = to else {
            self.follow.on.store(false, Release);
            self.follow.state.store(State::None as u8, Relaxed);
            self.follow.resume.store(false, Relaxed);
            return;
        };
        self.follow.on.store(true, Release);
        self.follow.state.store(State::Connect as u8, Relaxed);
        self.follow.down_ms.store(self.clock.now_ms(), Relaxed);
        let server = Arc::clone(self);
        yo_alloc::allow(|| {
            let _ = std::thread::Builder::new()
                .name(String::from("yo-replica"))
                .spawn(move || link(&server, epoch, &to));
        });
    }
}

#[cfg(test)]
impl Server {
    /// Say this server follows somebody, without a socket and without a thread.
    ///
    /// The same idea as `repl::pretend_replica` and for the same reason. What
    /// the tests below look at is the refusal an ordinary client gets, what
    /// `INFO` and `ROLE` say, and what a master's own session is let past, and
    /// every one of those reads the flags rather than the link. So a server told
    /// this is a replica in every way a test can see, and no port anywhere has
    /// to be listening.
    pub(super) fn pretend_following(&self, host: &str, port: u16, up: bool) {
        {
            let mut upstream = self.follow.upstream.lock();
            *upstream = Some(Upstream {
                host: host.to_owned(),
                port,
            });
        }
        self.follow.on.store(true, Release);
        self.follow
            .state
            .store(if up { State::Up } else { State::Connect } as u8, Relaxed);
        self.follow.last_io_ms.store(self.clock.now_ms(), Relaxed);
        self.follow.down_ms.store(self.clock.now_ms(), Relaxed);
    }

    /// Stop pretending, without the promotion `REPLICAOF NO ONE` does.
    pub(super) fn pretend_master(&self) {
        self.follow.on.store(false, Release);
        self.follow.state.store(State::None as u8, Relaxed);
    }
}

// ------------------------------------------------------------- the command

/// `REPLICAOF host port` and `REPLICAOF NO ONE`, and `SLAVEOF` for the same.
///
/// The two words are the same command under two names, which is Redis's own
/// arrangement: `SLAVEOF` is what it was called and answering to both is what
/// stops a decade of scripts breaking. Nothing here reads which name was used.
pub(super) fn replicaof(server: &Server, args: Args<'_>, out: &mut Out) -> Result<()> {
    let name = if args.name().eq_ignore_ascii_case(b"slaveof") {
        "slaveof"
    } else {
        "replicaof"
    };
    if args.len() != 3 {
        return Err(args::wrong_arity(name));
    }
    // A failover is already deciding who the master is, and two commands that
    // both decide that are two commands that can disagree. `FAILOVER ABORT` is
    // the way out, which is why it is the only one.
    if server.failing_over() {
        return Err(Error::new(
            Code::Invalid,
            "REPLICAOF not allowed while failing over.",
        ));
    }
    let host = args.get(1);
    let port = args.get(2);
    // Both words are read before anything is looked up, so a caller who got the
    // command wrong hears what was wrong with it rather than hearing about the
    // server it was sent to.
    let told = if host.eq_ignore_ascii_case(b"no") && port.eq_ignore_ascii_case(b"one") {
        None
    } else {
        // The reference's own sentence, and it is the answer to a port that is
        // not a number as well as to one that is out of range. So this reads the
        // digits itself rather than going through `args.int`, whose sentence is
        // about integers and is the wrong one here.
        let Some(port) = core::str::from_utf8(port)
            .ok()
            .and_then(|w| w.parse::<u16>().ok())
        else {
            return Err(Error::new(Code::Invalid, "Invalid master port"));
        };
        Some(port)
    };
    let Some(shared) = server.myself() else {
        return Err(Error::new(
            Code::Invalid,
            "REPLICAOF is not available on an embedded server",
        ));
    };
    let Some(port) = told else {
        shared.stop_following();
        out.ok();
        return Ok(());
    };
    let host = yo_alloc::allow(|| String::from_utf8_lossy(host).into_owned());
    // Told to follow the master it is already following, and there is nothing to
    // do. Starting a link anyway would drop the one that is up and take a
    // snapshot of a server this one is already in step with, which is what a real
    // server refuses to do and says so in the same words.
    {
        let upstream = server.follow.upstream.lock();
        let same = upstream
            .as_ref()
            .is_some_and(|at| at.port == port && at.host == host);
        if same && server.following() {
            out.simple(
                b"OK REPLICAOF would result into synchronization with the master we are already connected with. No operation performed.",
            );
            return Ok(());
        }
    }
    shared.follow_now(Some(Upstream { host, port }));
    out.ok();
    Ok(())
}

/// The refusal an ordinary client's write gets on a read only replica.
pub(super) const READONLY: &str = "READONLY You can't write against a read only replica.";

// ------------------------------------------------------------- the reporting

/// The replica's half of the `Replication` section of `INFO`.
///
/// Written between `connected_slaves` and the identity lines, which is where
/// Redis puts it, so a tool that reads the section in order sees the same shape.
pub(super) fn info(server: &Server, s: &mut String) {
    use core::fmt::Write as _;
    let state = State::from(server.follow.state.load(Relaxed));
    if state == State::None {
        return;
    }
    let (host, port) = {
        let upstream = server.follow.upstream.lock();
        match upstream.as_ref() {
            Some(up) => (up.host.clone(), up.port),
            None => (String::new(), 0),
        }
    };
    let now = server.clock.now_ms();
    let up = state == State::Up;
    let last = now.saturating_sub(server.follow.last_io_ms.load(Relaxed)) / 1000;
    let offset = server.repl_offset();
    let _ = write!(
        s,
        "master_host:{host}\r\nmaster_port:{port}\r\n\
         master_link_status:{}\r\nmaster_last_io_seconds_ago:{}\r\n\
         master_sync_in_progress:{}\r\n\
         slave_read_repl_offset:{offset}\r\nslave_repl_offset:{offset}\r\n",
        if up { "up" } else { "down" },
        if up { last as i64 } else { -1 },
        usize::from(state == State::Sync),
    );
    if !up {
        let down = now.saturating_sub(server.follow.down_ms.load(Relaxed)) / 1000;
        let _ = write!(s, "master_link_down_since_seconds:{down}\r\n");
    }
    // Priority and announcement are settings a failover reads and nothing here
    // acts on, so they are reported at the values a server that was never
    // configured has. Read only is the one of the three that is real.
    let _ = write!(
        s,
        "slave_priority:100\r\nslave_read_only:{}\r\nreplica_announced:1\r\n",
        usize::from(server.follow.read_only.load(Relaxed)),
    );
}

/// The word `INFO` and `ROLE` lead with, which is the only thing on this server
/// that two clients could disagree about if they asked at the wrong moment.
#[must_use]
pub(super) fn role_word(server: &Server) -> &'static str {
    if server.following() {
        "slave"
    } else {
        "master"
    }
}

/// What `ROLE` answers on a replica.
///
/// Five fields: the word, the master's host and port, the link state as one of
/// Redis's five words, and how much of the stream has been applied. The state
/// words are not the two `INFO` uses, which is not a tidy arrangement and is the
/// one every client library already reads.
pub(super) fn role(server: &Server, out: &mut Out) {
    let (host, port) = {
        let upstream = server.follow.upstream.lock();
        match upstream.as_ref() {
            Some(up) => (up.host.clone(), up.port),
            None => (String::new(), 0),
        }
    };
    out.array(5);
    out.bulk(b"slave");
    out.bulk(host.as_bytes());
    out.int(i64::from(port));
    out.bulk(match State::from(server.follow.state.load(Relaxed)) {
        State::Up => b"connected".as_slice(),
        State::Sync => b"sync".as_slice(),
        _ => b"connect".as_slice(),
    });
    out.int(server.repl_offset() as i64);
}

// ---------------------------------------------------------------- the link

/// The link thread: dial, hand shake, load, follow, and do it again.
///
/// Every failure lands in the same place, which is a wait and another dial. That
/// is the right shape for this because there is nothing else it could do: the
/// operator asked for this server to follow that one, and a master that is not
/// answering yet is the ordinary case at startup rather than an error.
fn link(server: &Arc<Server>, epoch: u64, to: &Upstream) {
    while server.follow.epoch.load(Relaxed) == epoch && !server.stopping() {
        let _ = once(server, epoch, to);
        if server.follow.epoch.load(Relaxed) != epoch {
            return;
        }
        if State::from(server.follow.state.load(Relaxed)) != State::Connect {
            server.follow.state.store(State::Connect as u8, Relaxed);
            server.follow.down_ms.store(server.clock.now_ms(), Relaxed);
        }
        std::thread::sleep(RETRY);
    }
}

/// One attempt: connect, hand shake, take what is offered, follow until it
/// breaks.
fn once(server: &Arc<Server>, epoch: u64, to: &Upstream) -> std::io::Result<()> {
    let mut wire = dial(to)?;
    handshake(server, &mut wire)?;
    // Written and not sent through `command`, because what comes back is not one
    // line the way every other answer in the handshake is: a full resync answers
    // with a line and then a snapshot, and reading the line here is the same read
    // either way.
    // A fourth word on a server that is handing its job over, which is what
    // tells the other end to stop being a replica and start being the master.
    // See the `failover` module.
    let handing_over = server.failover_stage() == super::failover::Stage::InProgress;
    if server.follow.resume.load(Relaxed) {
        // The server's own id and offset, which are the master's, because the
        // last thing applied moved them and nothing else writes them while a
        // link is up. One past the end, because the number a master reads is the
        // position of the first byte wanted counted from one. The other side of
        // the step `repl::psync` takes coming the other way.
        let id = server.repl_id();
        let from = (server.repl_offset() + 1).to_string();
        if handing_over {
            wire.write(&[b"PSYNC", &id, from.as_bytes(), b"FAILOVER"])?;
        } else {
            wire.write(&[b"PSYNC", &id, from.as_bytes()])?;
        }
    } else {
        wire.write(&[b"PSYNC", b"?", b"-1"])?;
    }
    // A master that has to fork before it can answer takes as long as the fork
    // takes, so this is the long wait and not the handshake's short one.
    let head = wire.line(SYNC_TIMEOUT)?;
    if head.starts_with(b"+FULLRESYNC ") {
        full_resync(server, &mut wire, &head[12..])?;
        server.follow.state.store(State::Up as u8, Relaxed);
        server.follow.resume.store(true, Relaxed);
    } else if head.starts_with(b"+CONTINUE") {
        // A master that has changed its id since we last spoke says the new one
        // here, and everything from this point is under that id.
        if let Some(id) = head.get(10..).and_then(fixed_id) {
            server.adopt(id, server.repl_offset());
        }
        server.follow.state.store(State::Up as u8, Relaxed);
        server.follow.resume.store(true, Relaxed);
    } else {
        // A target that would not take the handover, which is a failover that
        // cannot happen. This server takes its job back and lets the writes it
        // has been holding through, rather than sitting paused forever waiting
        // for a server that has already said no.
        if handing_over {
            super::failover::abort(server);
        }
        return Err(broken("the master would not resynchronise"));
    }
    // Answered either way, so the handover is done and the pause lifts.
    super::failover::landed(server);
    stream(server, epoch, &mut wire)
}

/// Read the snapshot and become it.
///
/// The header names the history and the position this image is an image as of,
/// and both are taken as ours: from here on this server's stream is the master's
/// stream, at the master's offsets, which is what makes a chain underneath it
/// hand out positions anybody else can honour.
fn full_resync(server: &Arc<Server>, wire: &mut Link, head: &[u8]) -> std::io::Result<()> {
    let mut words = head.split(|&b| b == b' ');
    let id = words
        .next()
        .and_then(fixed_id)
        .ok_or_else(|| broken("the master named no replication id"))?;
    let offset = words
        .next()
        .and_then(|w| core::str::from_utf8(w).ok())
        .and_then(|w| w.trim().parse::<u64>().ok())
        .ok_or_else(|| broken("the master named no offset"))?;
    server.follow.state.store(State::Sync as u8, Relaxed);
    let image = wire.payload()?;
    server
        .load_image(&image, true)
        .map_err(|e| broken(&format!("the snapshot would not load: {e}")))?;
    server.adopt(id, offset);
    Ok(())
}

/// Follow the stream until it breaks or the link is called off.
///
/// Everything the loop does other than run a command is on a clock: it looks at
/// the epoch to find out whether it is still wanted and it sends an
/// acknowledgement once a second. Both of those are why the read has a timeout
/// on it rather than blocking forever on a socket that is quiet.
fn stream(server: &Arc<Server>, epoch: u64, wire: &mut Link) -> std::io::Result<()> {
    let mut session = Session::new(server.next_client());
    session.admit(true);
    session.serve_master(true);
    let mut out = Out::new(Proto::Resp2);
    let mut argv = Argv::new();
    let limits = Limits::default();
    let mut acked = Instant::now();
    let mut sent = 0u64;
    APPLYING.set(true);
    let ended = loop {
        if server.follow.epoch.load(Relaxed) != epoch || server.stopping() {
            break Ok(());
        }
        match argv.decode(wire.held(), &limits) {
            Err(_) => break Err(broken("the master sent something that is not a command")),
            Ok(Step::Incomplete) => {
                if let Err(e) = wire.fill(POLL) {
                    break Err(e);
                }
            }
            Ok(Step::Command { consumed }) => {
                let getack = is_getack(&argv, wire.held());
                if !getack {
                    apply(server, &mut session, &argv, wire.held(), &mut out);
                }
                // Counted before the acknowledgement is written, because the
                // number the master is asking about includes the question.
                let bytes = wire.take(consumed);
                repl::relayed(server, bytes, session.db());
                server
                    .follow
                    .last_io_ms
                    .store(server.clock.now_ms(), Relaxed);
                if getack {
                    acked = Instant::now();
                    sent = server.repl_offset();
                    if let Err(e) = wire.ack(sent) {
                        break Err(e);
                    }
                }
                continue;
            }
        }
        let now = server.repl_offset();
        if acked.elapsed() >= ACK_EVERY || now != sent {
            acked = Instant::now();
            sent = now;
            if let Err(e) = wire.ack(now) {
                break Err(e);
            }
        }
    };
    APPLYING.set(false);
    super::forget_session(server, &mut session);
    ended
}

/// Whether the command sitting at the front of the buffer is the master asking
/// where we have got to, which is answered rather than run.
fn is_getack(argv: &Argv, buf: &[u8]) -> bool {
    argv.len() == 3
        && argv
            .arg(buf, 0)
            .is_some_and(|w| w.eq_ignore_ascii_case(b"replconf"))
        && argv
            .arg(buf, 1)
            .is_some_and(|w| w.eq_ignore_ascii_case(b"getack"))
}

/// Run one command from the master against this server.
///
/// A command the freeze is holding is run again rather than dropped, because a
/// full resync for a sub-replica is a pause of this server and not a reason to
/// lose a byte of the master's stream. It is the one place in the tree that
/// spins, and what it spins on is a snapshot being built, which finishes.
fn apply(server: &Server, session: &mut Session, argv: &Argv, buf: &[u8], out: &mut Out) {
    loop {
        out.clear();
        let args = Args::new(argv, buf);
        if super::execute(server, session, args, out) != Flow::Hold {
            return;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

// --------------------------------------------------------------- the socket

/// The socket and whatever has arrived on it that has not been used yet.
struct Link {
    sock: TcpStream,
    buf: Vec<u8>,
}

impl Link {
    /// What has arrived and not been used.
    fn held(&self) -> &[u8] {
        &self.buf
    }

    /// Take the first `n` bytes off the front and answer with them.
    fn take(&mut self, n: usize) -> Vec<u8> {
        self.buf.drain(..n).collect()
    }

    /// Read once, waiting at most `wait`.
    ///
    /// A timeout is not a failure and answers with nothing added, which is what
    /// lets the caller look around between reads. End of file is a failure,
    /// because a master that closed the socket is a link that has to be dialled
    /// again.
    fn fill(&mut self, wait: Duration) -> std::io::Result<()> {
        self.sock.set_read_timeout(Some(wait))?;
        let mut chunk = [0u8; 16 * 1024];
        match self.sock.read(&mut chunk) {
            Ok(0) => Err(broken("the master closed the link")),
            Ok(n) => {
                self.buf.extend_from_slice(&chunk[..n]);
                Ok(())
            }
            Err(e) if soft(&e) => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// One line, without its newline, waiting at most `wait` in total.
    fn line(&mut self, wait: Duration) -> std::io::Result<Vec<u8>> {
        let until = Instant::now() + wait;
        loop {
            if let Some(at) = self.buf.iter().position(|&b| b == b'\n') {
                let mut line = self.take(at + 1);
                while line.last().is_some_and(|&b| b == b'\n' || b == b'\r') {
                    line.pop();
                }
                // A master preparing a snapshot sends bare newlines to keep the
                // link warm. They are not a reply and are not counted.
                if line.is_empty() {
                    continue;
                }
                return Ok(line);
            }
            if self.buf.len() > LINE_MAX {
                return Err(broken("the master sent a line with no end to it"));
            }
            if Instant::now() >= until {
                return Err(broken("the master did not answer"));
            }
            self.fill(POLL)?;
        }
    }

    /// Send a command and read the one line it is answered with.
    fn command(&mut self, parts: &[&[u8]]) -> std::io::Result<Vec<u8>> {
        self.write(parts)?;
        self.line(HANDSHAKE_TIMEOUT)
    }

    /// Send a command and do not wait for anything.
    fn write(&mut self, parts: &[&[u8]]) -> std::io::Result<()> {
        let mut wire = Vec::with_capacity(32);
        wire.extend_from_slice(b"*");
        wire.extend_from_slice(parts.len().to_string().as_bytes());
        wire.extend_from_slice(b"\r\n");
        for part in parts {
            wire.extend_from_slice(b"$");
            wire.extend_from_slice(part.len().to_string().as_bytes());
            wire.extend_from_slice(b"\r\n");
            wire.extend_from_slice(part);
            wire.extend_from_slice(b"\r\n");
        }
        self.sock.write_all(&wire)
    }

    /// Tell the master how far we have got.
    fn ack(&mut self, offset: u64) -> std::io::Result<()> {
        self.write(&[b"REPLCONF", b"ACK", offset.to_string().as_bytes()])
    }

    /// The snapshot, which is a bulk string with nothing after it.
    ///
    /// The one place in the protocol where a bulk string has no newline behind
    /// it, because everything after the last byte of it is the stream. A master
    /// that was given `capa eof` would send a different shape here, which is why
    /// the handshake does not offer it.
    fn payload(&mut self) -> std::io::Result<Vec<u8>> {
        let head = self.line(SYNC_TIMEOUT)?;
        let want = core::str::from_utf8(head.get(1..).unwrap_or_default())
            .ok()
            .and_then(|n| n.parse::<usize>().ok())
            .filter(|_| head.first() == Some(&b'$'))
            .ok_or_else(|| broken("the master did not say how long the snapshot is"))?;
        // The clock is on the gap between reads and not on the whole transfer,
        // because a snapshot that is genuinely large is a link that is working
        // and a link that stopped mid snapshot is one nothing will ever finish.
        let mut last = Instant::now();
        while self.buf.len() < want {
            let had = self.buf.len();
            self.fill(POLL)?;
            if self.buf.len() > had {
                last = Instant::now();
            } else if last.elapsed() >= SYNC_TIMEOUT {
                return Err(broken("the master stopped part way through the snapshot"));
            }
        }
        Ok(self.take(want))
    }
}

/// Whether an error means nothing arrived rather than that the link is gone.
fn soft(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        ErrorKind::WouldBlock | ErrorKind::TimedOut | ErrorKind::Interrupted
    )
}

/// A link failure with a sentence on it, which nobody reads and which is worth
/// writing anyway: the moment one of these needs a log line, the sentence is
/// already there.
fn broken(why: &str) -> std::io::Error {
    std::io::Error::other(String::from(why))
}

/// Forty hex characters, or nothing.
fn fixed_id(word: &[u8]) -> Option<[u8; ID_LEN]> {
    let word = word.strip_suffix(b"\r").unwrap_or(word);
    let word = word.get(..ID_LEN)?;
    word.iter()
        .all(u8::is_ascii_hexdigit)
        .then(|| <[u8; ID_LEN]>::try_from(word).ok())
        .flatten()
}

/// Open the socket.
fn dial(to: &Upstream) -> std::io::Result<Link> {
    let at = (to.host.as_str(), to.port)
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| broken("the master's address does not resolve"))?;
    let sock = TcpStream::connect_timeout(&at, DIAL_TIMEOUT)?;
    sock.set_nodelay(true)?;
    Ok(Link {
        sock,
        buf: Vec::new(),
    })
}

/// Everything before `PSYNC`.
///
/// The capabilities are the two a master has to be told about and no more.
/// `psync2` says this replica understands a partial resync across a promotion,
/// which it does. `eof` is deliberately not offered: it tells a master it may
/// send the snapshot without knowing its length first, and the shape that
/// arrives then is different enough that not asking for it is cheaper than
/// reading it.
fn handshake(server: &Server, wire: &mut Link) -> std::io::Result<()> {
    wire.command(&[b"PING"])?;
    let (user, pass) = {
        let auth = server.follow.auth.lock();
        auth.clone()
    };
    if !pass.is_empty() {
        let said = if user.is_empty() {
            wire.command(&[b"AUTH", &pass])?
        } else {
            wire.command(&[b"AUTH", &user, &pass])?
        };
        if said.first() == Some(&b'-') {
            return Err(broken("the master would not take the password"));
        }
    }
    let port = server.follow.port.load(Relaxed).to_string();
    wire.command(&[b"REPLCONF", b"listening-port", port.as_bytes()])?;
    wire.command(&[b"REPLCONF", b"capa", b"psync2"])?;
    server
        .follow
        .last_io_ms
        .store(server.clock.now_ms(), Relaxed);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ID_LEN, State, fixed_id, soft};

    #[test]
    fn a_replication_id_is_forty_hex_characters_and_nothing_else() {
        let good = b"0123456789abcdef0123456789abcdef01234567";
        assert_eq!(fixed_id(good).unwrap(), *good);
        // The trailing carriage return of the line it was read out of comes off,
        // because the caller splits on the space and not on the line ending.
        let mut with_cr = good.to_vec();
        with_cr.push(b'\r');
        assert_eq!(fixed_id(&with_cr).unwrap(), *good);
        // Anything longer is read as the first forty, which is what a master
        // that appends something we do not know about would send.
        let mut longer = good.to_vec();
        longer.extend_from_slice(b"more");
        assert_eq!(fixed_id(&longer).unwrap(), *good);
        // Too short is nothing, and so is the right length with a character in
        // it that is not hex.
        assert!(fixed_id(&good[..ID_LEN - 1]).is_none());
        let mut wrong = good.to_vec();
        wrong[7] = b'z';
        assert!(fixed_id(&wrong).is_none());
        assert!(fixed_id(b"").is_none());
    }

    #[test]
    fn a_state_that_is_not_one_of_the_four_reads_as_no_master() {
        assert!(State::from(1) == State::Connect);
        assert!(State::from(2) == State::Sync);
        assert!(State::from(3) == State::Up);
        assert!(State::from(0) == State::None);
        assert!(State::from(99) == State::None);
    }

    #[test]
    fn a_read_that_timed_out_is_not_a_broken_link() {
        use std::io::{Error, ErrorKind};
        assert!(soft(&Error::from(ErrorKind::WouldBlock)));
        assert!(soft(&Error::from(ErrorKind::TimedOut)));
        assert!(soft(&Error::from(ErrorKind::Interrupted)));
        // Everything else is, including the one that means the master hung up.
        assert!(!soft(&Error::from(ErrorKind::ConnectionReset)));
        assert!(!soft(&Error::from(ErrorKind::UnexpectedEof)));
    }
}
