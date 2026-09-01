//! `MIGRATE`, the one command that talks to another server.
//!
//! Everything else in this crate answers out of memory the process already
//! owns. This opens a socket to somebody else, writes a pipeline down it and
//! waits for the replies, and it does that on the thread that is running the
//! event loop, so a slow peer stops every other client on the shard. That is
//! what Redis does too and it is not an accident: the command deletes the local
//! key only once the far side has said it has the value, so there is a moment
//! where the value exists in two places and no moment where it exists in
//! neither, and holding the thread is how that ordering is bought.
//!
//! The whole of it is a `DUMP` and a `RESTORE` with a wire in between, which is
//! why it landed after those two and not before. What it adds is the pipeline,
//! the socket cache and a long list of things a client can get wrong.
//!
//! # What goes down the socket
//!
//! One write, not one per key. `AUTH` if a password was given, then `SELECT` if
//! the socket is not already on the database being migrated into, then one
//! `RESTORE` per key that is actually here. The replies come back in that order
//! and are read in that order, and every one of them is a single line.
//!
//! Pipelining the lot has a consequence worth writing down, because it looks
//! like a bug and it is what a real server does: a `SELECT` that fails does not
//! stop the `RESTORE`s behind it. They were already on the wire, so the peer
//! runs them against whatever database it was on, and the client is told the
//! `SELECT` error while the values quietly land somewhere else. Migrating into a
//! database index the peer does not have is exactly this case.
//!
//! # The socket cache
//!
//! A socket is kept after the command finishes and the next `MIGRATE` to the
//! same host and port reuses it, which is what makes moving a keyspace across
//! one key at a time not cost a handshake per key. Sixty four of them at most
//! and ten seconds of idleness each, both Redis's numbers. A socket that has
//! seen an error is dropped rather than kept, because the one thing this cache
//! cannot afford is to hand out a connection with half a reply still in it.
//!
//! # Failure
//!
//! Three kinds, and a client can tell them apart from the reply. `IOERR` means
//! the conversation broke and nothing can be said about what the peer did with
//! what it had already been sent. `ERR Target instance replied with error:`
//! means the conversation worked and the peer refused something, and then the
//! local key is left alone. `NOKEY` means none of the named keys were here,
//! which is not an error at all, because a key expiring between the client
//! deciding to move it and this running is normal.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

use super::Server;
use super::args::{self, Args};
use crate::reply::Out;
use yo_common::num::push_i64;
use yo_common::{Code, Error, Result};
use yo_kv::Ask;

/// What `MIGRATE` says when `KEYS` is given and the key argument is not empty.
const KEYS_NEEDS_EMPTY: &str =
    "When using MIGRATE KEYS option, the key argument must be set to the empty string";

/// The prefix on an error the peer sent back, which the peer's own message
/// follows with its leading dash taken off.
const TARGET_ERROR: &[u8] = b"ERR Target instance replied with error: ";

/// What a connect that ran out of time says.
///
/// The wording is Redis's and so is the fact that it names the client rather
/// than the target, which every other message in this file does. It is only
/// reachable when the connect itself hangs, which on a routable address that
/// answers nothing is what happens. A refused connect and a name that does not
/// resolve both come out as the write error instead, because that is where a
/// real server notices them, and a client that has learned to match on these
/// strings should keep getting the ones it has always got.
const CONNECT_FAILED: &[u8] = b"IOERR error or timeout connecting to the client";

/// What a write that failed says.
const WRITE_FAILED: &[u8] = b"IOERR error or timeout writing to target instance";

/// And a read. The grammar is Redis's, one format string for both halves.
const READ_FAILED: &[u8] = b"IOERR error or timeout reading to target instance";

/// What a timeout of zero or less means, which is one second and not no wait.
///
/// Redis does not reject a negative timeout, it replaces it, so `MIGRATE host
/// port key 0 -5` is a working command and not a syntax error.
const DEFAULT_TIMEOUT_MS: i64 = 1000;

/// How many peer sockets are held open at once.
const CACHE_MAX: usize = 64;

/// How long a peer socket is held after the command that last used it.
const CACHE_TTL_MS: u64 = 10_000;

/// The longest reply line that is read before giving up on the peer.
///
/// A reply here is `+OK` or an error message, and an error message long enough
/// to reach this is a peer that is not answering the question that was asked.
const LINE_MAX: usize = 1024;

/// Sockets held open to the servers this one has migrated keys to.
///
/// Lives on the [`Server`] because it outlives the command, and it is a vector
/// rather than a map because it holds sixty four entries at the very most and a
/// linear walk over sixty four short strings is not something a command that
/// just opened a socket can measure.
#[derive(Default)]
pub(super) struct Peers {
    open: Vec<Peer>,
}

/// One held socket.
struct Peer {
    /// The host and port as the client wrote them, joined by a colon.
    ///
    /// As the client wrote them, so two spellings of the same address are two
    /// entries. That is Redis's key too and it is the right one: this is a cache
    /// and a miss costs a handshake, not a wrong answer.
    at: String,
    sock: TcpStream,
    /// The database the last `SELECT` on this socket chose.
    ///
    /// Minus one when nothing is known, which is the state a fresh socket is in
    /// and the state one is put back into after the peer reports an error.
    last_db: i64,
    /// The millisecond the command that last used this socket ran on.
    used_ms: u64,
}

impl Peers {
    /// Drop the sockets that have been idle too long.
    ///
    /// Called at the top of every `MIGRATE` rather than on a timer, because this
    /// is the only thing that knows the cache exists and a server that has
    /// stopped migrating has nothing to hurry about. What it does mean is that a
    /// server that migrated once and never again keeps one socket open until it
    /// migrates again or stops, which is what a real server does as well.
    fn expire(&mut self, now: u64) {
        self.open
            .retain(|p| now.saturating_sub(p.used_ms) < CACHE_TTL_MS);
    }

    /// Where the socket for `at` is, if it is held.
    fn find(&self, at: &str) -> Option<usize> {
        self.open.iter().position(|p| p.at == at)
    }

    /// Forget the socket for `at`, which closes it.
    fn close(&mut self, at: &str) {
        self.open.retain(|p| p.at != at);
    }
}

/// What the arguments asked for, once they have all been read.
struct Plan<'a> {
    host: &'a [u8],
    port: &'a [u8],
    /// The database on the peer, unchecked here because this server does not
    /// know how many databases the peer has and the peer will say.
    db: i64,
    timeout: Duration,
    copy: bool,
    replace: bool,
    user: Option<&'a [u8]>,
    pass: Option<&'a [u8]>,
    /// Where the keys start and how many there are, which is argument three and
    /// one key in the ordinary form and everything after `KEYS` in the other.
    first_key: usize,
    num_keys: usize,
}

/// One key that is here and the bytes and deadline it is going over with.
struct Going<'a> {
    key: &'a [u8],
    /// Milliseconds from now, and zero for a key with no deadline. Relative and
    /// not absolute, because `RESTORE` reads it as relative unless it is told
    /// otherwise and `MIGRATE` does not tell it otherwise.
    ttl: i64,
    payload: Vec<u8>,
}

/// `MIGRATE host port key destination-db timeout [COPY] [REPLACE]
/// [AUTH password | AUTH2 username password] [KEYS key [key ...]]`.
///
/// Allocating throughout, which is the one place in the command layer that is
/// true, and it is not something to fix. The payloads are values serialised out
/// of the store and the socket is a socket. There is no arrangement of this that
/// does not allocate and no workload where it matters, because a command that
/// opens a TCP connection is not one whose cost is in a `malloc`.
pub(super) fn execute(server: &mut Server, at: usize, args: Args<'_>, out: &mut Out) -> Result<()> {
    yo_alloc::allow(|| run(server, at, args, out))
}

fn run(server: &mut Server, at: usize, args: Args<'_>, out: &mut Out) -> Result<()> {
    let plan = parse(args)?;

    // Everything that has to touch the store happens here, and the borrow ends
    // before the socket work starts. What comes out is owned, so the peer cache
    // and the databases are never borrowed at the same time.
    let db = server.db(at);
    let now = db.clock().now_ms();
    let mut going: Vec<Going<'_>> = Vec::new();
    for i in 0..plan.num_keys {
        let key = args.get(plan.first_key + i);
        // Redis looks the keys up once to decide whether to say `NOKEY` and
        // again while building the payloads, and skips a key that expired in
        // between. Here the clock is read once per batch and both readings would
        // be the same millisecond, so the second pass cannot find anything the
        // first one missed and there is only one pass.
        let ttl = match db.deadline_of(key) {
            Ask::Missing => continue,
            Ask::NoDeadline => 0,
            Ask::At(when) if when <= now => continue,
            // At least one, because a zero ttl in a `RESTORE` means no deadline
            // at all and the key would arrive immortal.
            Ask::At(when) => (when - now).max(1) as i64,
        };
        let Some(payload) = db.dump(key) else {
            continue;
        };
        going.push(Going { key, ttl, payload });
    }
    if going.is_empty() {
        // Not an error, and this is the reply for a key that was never there as
        // well as for one that expired a moment ago. A client that cares which
        // asks before it migrates.
        out.simple(b"NOKEY");
        return Ok(());
    }

    let mut name = String::new();
    name.push_str(&String::from_utf8_lossy(plan.host));
    name.push(':');
    name.push_str(&String::from_utf8_lossy(plan.port));

    server.peers.expire(now);

    // One retry, and only for the errors a stale cached socket produces. A peer
    // that closed the connection while it was sitting in the cache looks exactly
    // like a peer that has gone away, and the difference is that the first one
    // works on the second attempt. A timeout is not retried, because waiting the
    // whole timeout twice is not a retry, it is twice the wait.
    let mut retry = true;
    loop {
        match attempt(server, &name, &plan, &going, now) {
            Ok(replies) => {
                finish(server, at, &name, &plan, &going, &replies, out);
                return Ok(());
            }
            Err(Broke { failed, timed_out }) => {
                server.peers.close(&name);
                if retry && !timed_out {
                    retry = false;
                    continue;
                }
                out.error(failed);
                return Ok(());
            }
        }
    }
}

/// The socket half of the command broke.
struct Broke {
    /// Which of the three `IOERR` lines to send.
    failed: &'static [u8],
    /// Whether it broke by running out of time, which is the one kind that is
    /// not worth trying again.
    timed_out: bool,
}

/// What came back, one line per thing that was sent.
struct Replies {
    /// The `AUTH` reply, when a password was sent.
    auth: Option<Vec<u8>>,
    /// The `SELECT` reply, when one was sent.
    select: Option<Vec<u8>>,
    /// One per key, in the order the keys went out.
    keys: Vec<Vec<u8>>,
}

/// Connect if there is nothing cached, write the pipeline, read the replies.
///
/// Every error out of here closes the socket, because a socket that failed
/// halfway through a pipeline has an unknown number of unread replies in it and
/// there is no way to get back in step with it.
fn attempt(
    server: &mut Server,
    name: &str,
    plan: &Plan<'_>,
    going: &[Going<'_>],
    now: u64,
) -> std::result::Result<Replies, Broke> {
    let held = server.peers.find(name);
    let last_db = held.map_or(-1, |i| server.peers.open[i].last_db);
    let select = last_db != plan.db;

    if held.is_none() {
        let sock = connect(plan)?;
        // Room is made by dropping the oldest, which on a cache this size is the
        // one that has gone longest without being wanted.
        if server.peers.open.len() >= CACHE_MAX {
            let oldest = server
                .peers
                .open
                .iter()
                .enumerate()
                .min_by_key(|(_, p)| p.used_ms)
                .map_or(0, |(i, _)| i);
            server.peers.open.swap_remove(oldest);
        }
        server.peers.open.push(Peer {
            at: name.to_string(),
            sock,
            last_db: -1,
            used_ms: now,
        });
    }
    let i = server
        .peers
        .find(name)
        .expect("the socket was just put here");
    let peer = &mut server.peers.open[i];
    peer.used_ms = now;

    let mut cmd =
        Vec::with_capacity(64 + going.iter().map(|g| g.payload.len() + 64).sum::<usize>());
    if let Some(pass) = plan.pass {
        array(&mut cmd, if plan.user.is_some() { 3 } else { 2 });
        bulk(&mut cmd, b"AUTH");
        if let Some(user) = plan.user {
            bulk(&mut cmd, user);
        }
        bulk(&mut cmd, pass);
    }
    if select {
        array(&mut cmd, 2);
        bulk(&mut cmd, b"SELECT");
        int(&mut cmd, plan.db);
    }
    for g in going {
        array(&mut cmd, if plan.replace { 5 } else { 4 });
        bulk(&mut cmd, b"RESTORE");
        bulk(&mut cmd, g.key);
        int(&mut cmd, g.ttl);
        bulk(&mut cmd, &g.payload);
        if plan.replace {
            bulk(&mut cmd, b"REPLACE");
        }
    }

    peer.sock.write_all(&cmd).map_err(broke(WRITE_FAILED))?;
    peer.sock.flush().map_err(broke(WRITE_FAILED))?;

    let auth = match plan.pass {
        Some(_) => Some(line(&mut peer.sock)?),
        None => None,
    };
    let select = if select {
        Some(line(&mut peer.sock)?)
    } else {
        None
    };
    let mut keys = Vec::with_capacity(going.len());
    for _ in going {
        keys.push(line(&mut peer.sock)?);
    }
    Ok(Replies { auth, select, keys })
}

/// Read the replies, delete the keys the peer took, and write the answer.
///
/// A failed `AUTH` or `SELECT` fails every key, because the peer was not in the
/// state the `RESTORE`s were written for. What it does not do is stop them
/// running on the peer, and this is where that shows: the local keys stay and
/// the peer may well have taken copies of them anyway.
fn finish(
    server: &mut Server,
    at: usize,
    name: &str,
    plan: &Plan<'_>,
    going: &[Going<'_>],
    replies: &Replies,
    out: &mut Out,
) {
    fn bad(r: &Option<Vec<u8>>) -> Option<&[u8]> {
        r.as_deref().filter(|line| line.first() == Some(&b'-'))
    }
    let before = bad(&replies.auth).or_else(|| bad(&replies.select));

    let mut told: Option<&[u8]> = None;
    for (g, reply) in going.iter().zip(&replies.keys) {
        let failed = before.or_else(|| {
            if reply.first() == Some(&b'-') {
                Some(reply.as_slice())
            } else {
                None
            }
        });
        match failed {
            // Only the first one is reported, which is Redis's choice. A client
            // that migrated ten keys and got one error learns nothing about the
            // other nine from the reply and has to look.
            Some(line) => told = told.or(Some(line)),
            None if !plan.copy => {
                server.db(at).del(g.key);
            }
            None => {}
        }
    }

    // The socket is fine either way and stays cached. What changes is whether
    // the database it is selected on is still known: an error may have been the
    // `SELECT` itself, and after one there is nothing to be sure of.
    if let Some(i) = server.peers.find(name) {
        server.peers.open[i].last_db = if told.is_some() { -1 } else { plan.db };
    }
    match told {
        Some(line) => out.error_line(TARGET_ERROR, &line[1..]),
        None => out.ok(),
    }
}

/// Open a socket to the peer with the timeout the client asked for.
///
/// A name that does not resolve is reported as a write failure and not as a
/// connect failure, which is not a mistake. A real server hands the name to a
/// non blocking connect and only notices on the first write, so that is the
/// message clients have always got for it, and the same goes for a connection
/// that is refused outright. What is left for the connect message is the case
/// where the handshake never completes, which is what a host sees when nothing
/// answers and no reset comes back either.
fn connect(plan: &Plan<'_>) -> std::result::Result<TcpStream, Broke> {
    let host = std::str::from_utf8(plan.host).map_err(|_| Broke {
        failed: WRITE_FAILED,
        timed_out: false,
    })?;
    let mut addrs = (host, port_of(plan.port))
        .to_socket_addrs()
        .map_err(|_| Broke {
            failed: WRITE_FAILED,
            timed_out: false,
        })?;
    let addr = addrs.next().ok_or(Broke {
        failed: WRITE_FAILED,
        timed_out: false,
    })?;
    let sock = TcpStream::connect_timeout(&addr, plan.timeout).map_err(|e| {
        let timed_out = is_timeout(&e);
        Broke {
            failed: if timed_out {
                CONNECT_FAILED
            } else {
                WRITE_FAILED
            },
            timed_out,
        }
    })?;
    sock.set_read_timeout(Some(plan.timeout))
        .and_then(|()| sock.set_write_timeout(Some(plan.timeout)))
        .and_then(|()| sock.set_nodelay(true))
        .map_err(|_| Broke {
            failed: WRITE_FAILED,
            timed_out: false,
        })?;
    Ok(sock)
}

/// Turn an IO error into a [`Broke`] carrying one of the two messages.
fn broke(failed: &'static [u8]) -> impl Fn(std::io::Error) -> Broke {
    move |e| Broke {
        failed,
        timed_out: is_timeout(&e),
    }
}

/// Whether an IO error is the socket timeout running out.
///
/// Two kinds, because a socket with a receive timeout on it reports one of them
/// on Linux and the other on the BSDs, and which one is not something worth
/// caring about anywhere else.
fn is_timeout(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
    )
}

/// Read one reply line, without the break on the end.
///
/// A byte at a time, which is what a real server does here and is right for the
/// same reason: the socket is cached and read again by the next `MIGRATE`, so
/// reading further than the line would leave bytes in a buffer that is about to
/// be dropped. Five syscalls per key on a command that just opened a TCP
/// connection is not the thing to optimise.
fn line(sock: &mut TcpStream) -> std::result::Result<Vec<u8>, Broke> {
    let mut out = Vec::with_capacity(32);
    let mut b = [0u8; 1];
    loop {
        match sock.read(&mut b) {
            // The peer closed with the line half read, which is a broken
            // conversation and not a short reply.
            Ok(0) => {
                return Err(Broke {
                    failed: READ_FAILED,
                    timed_out: false,
                });
            }
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(broke(READ_FAILED)(e)),
        }
        if b[0] == b'\n' {
            if out.last() == Some(&b'\r') {
                out.pop();
            }
            return Ok(out);
        }
        if out.len() >= LINE_MAX {
            return Err(Broke {
                failed: READ_FAILED,
                timed_out: false,
            });
        }
        out.push(b[0]);
    }
}

/// Read the arguments.
///
/// The options are read before the two numbers, which is Redis's order and is
/// visible: `MIGRATE host port key notanumber alsonot BOGUS` is a syntax error
/// and not an integer error. The two numbers themselves give the same message
/// either way round, so the order between them is not observable and this reads
/// them in the order they were written.
fn parse(args: Args<'_>) -> Result<Plan<'_>> {
    let mut plan = Plan {
        host: args.get(1),
        port: args.get(2),
        db: 0,
        timeout: Duration::from_millis(DEFAULT_TIMEOUT_MS as u64),
        copy: false,
        replace: false,
        user: None,
        pass: None,
        first_key: 3,
        num_keys: 1,
    };
    let mut i = 6;
    while i < args.len() {
        let arg = args.get(i);
        let more = args.len() - i - 1;
        if args::is(arg, b"copy") {
            plan.copy = true;
        } else if args::is(arg, b"replace") {
            plan.replace = true;
        } else if args::is(arg, b"auth") {
            if more < 1 {
                return Err(args::syntax());
            }
            plan.pass = Some(args.get(i + 1));
            i += 1;
        } else if args::is(arg, b"auth2") {
            if more < 2 {
                return Err(args::syntax());
            }
            plan.user = Some(args.get(i + 1));
            plan.pass = Some(args.get(i + 2));
            i += 2;
        } else if args::is(arg, b"keys") {
            // The empty string is the only thing the key argument may be once
            // `KEYS` has been given, and it is checked here rather than up front
            // because a `MIGRATE` with no `KEYS` migrates a key called the empty
            // string quite happily.
            if !args.get(3).is_empty() {
                return Err(Error::new(Code::Invalid, KEYS_NEEDS_EMPTY));
            }
            plan.first_key = i + 1;
            // Everything after the word, so an option written after `KEYS` is a
            // key and not an option. `KEYS a COPY` migrates two keys, one of
            // which is called `COPY`.
            plan.num_keys = args.len() - i - 1;
            break;
        } else {
            return Err(args::syntax());
        }
        i += 1;
    }
    plan.db = args.int(4)?;
    let ms = args.int(5)?;
    plan.timeout = Duration::from_millis(if ms <= 0 {
        DEFAULT_TIMEOUT_MS as u64
    } else {
        ms as u64
    });
    Ok(plan)
}

/// The port, read the way C's `atoi` reads it.
///
/// So a port with a word on the end is the number at the front of it, a port
/// with no digits at all is zero, and a port past sixty five thousand keeps its
/// low sixteen bits. None of that is good behaviour and all of it is what a
/// client gets today, and each of the three fails on the connect with the same
/// message rather than in the parser with a better one.
fn port_of(s: &[u8]) -> u16 {
    let mut i = 0;
    while i < s.len() && s[i].is_ascii_whitespace() {
        i += 1;
    }
    let negative = i < s.len() && s[i] == b'-';
    if i < s.len() && (s[i] == b'-' || s[i] == b'+') {
        i += 1;
    }
    let mut n: i64 = 0;
    while i < s.len() && s[i].is_ascii_digit() {
        n = n.saturating_mul(10).saturating_add(i64::from(s[i] - b'0'));
        i += 1;
    }
    if negative {
        n = -n;
    }
    n as u16
}

/// `*n\r\n`.
fn array(buf: &mut Vec<u8>, n: i64) {
    buf.push(b'*');
    push_i64(buf, n);
    buf.extend_from_slice(b"\r\n");
}

/// `$len\r\n...\r\n`.
fn bulk(buf: &mut Vec<u8>, s: &[u8]) {
    buf.push(b'$');
    push_i64(buf, s.len() as i64);
    buf.extend_from_slice(b"\r\n");
    buf.extend_from_slice(s);
    buf.extend_from_slice(b"\r\n");
}

/// A number as a bulk string, which is how every number goes out on a request.
fn int(buf: &mut Vec<u8>, n: i64) {
    let mut digits = Vec::with_capacity(20);
    push_i64(&mut digits, n);
    bulk(buf, &digits);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::{Args, Session, execute};
    use crate::proto::{Limits, Proto};
    use crate::request::Argv;
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    /// A server on the other end of the socket that answers from a script.
    ///
    /// One entry in `rounds` per `MIGRATE` the test is going to run, and one
    /// string in each entry per reply that migration expects. It waits until it
    /// has the whole round before answering any of it, which is what makes the
    /// test deterministic: the pipeline goes out in one write and the replies
    /// come back in one batch, so there is no arrangement of the two that can
    /// race.
    struct Fake {
        port: String,
        seen: Arc<Mutex<Vec<u8>>>,
    }

    impl Fake {
        /// What the peer has been sent so far, split into commands.
        fn seen(&self) -> Vec<Vec<Vec<u8>>> {
            commands(&self.seen.lock().expect("the peer thread has not panicked"))
        }

        /// The commands as words, which is what most of these tests compare.
        fn words(&self) -> Vec<Vec<String>> {
            self.seen()
                .iter()
                .map(|cmd| {
                    cmd.iter()
                        .map(|a| String::from_utf8_lossy(a).into_owned())
                        .collect()
                })
                .collect()
        }
    }

    fn fake(rounds: Vec<Vec<&'static str>>) -> Fake {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a free port");
        let port = listener
            .local_addr()
            .expect("the port it bound")
            .port()
            .to_string();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mine = Arc::clone(&seen);
        std::thread::spawn(move || {
            let Ok((mut sock, _)) = listener.accept() else {
                return;
            };
            let mut buf = Vec::new();
            let mut chunk = [0u8; 8192];
            let mut done = 0usize;
            for round in rounds {
                let want = done + round.len();
                while commands(&buf).len() < want {
                    match sock.read(&mut chunk) {
                        Ok(0) | Err(_) => return,
                        Ok(n) => buf.extend_from_slice(&chunk[..n]),
                    }
                }
                *mine.lock().expect("nobody poisoned it") = buf.clone();
                for r in round {
                    if sock.write_all(r.as_bytes()).is_err() {
                        return;
                    }
                }
                done = want;
            }
            *mine.lock().expect("nobody poisoned it") = buf;
            // Held open until the other end goes away, so a socket the server
            // cached is still there for the next migration in the test.
            while sock.read(&mut chunk).is_ok_and(|n| n > 0) {}
        });
        Fake { port, seen }
    }

    /// Split a stream of requests into commands, stopping at a partial one.
    fn commands(buf: &[u8]) -> Vec<Vec<Vec<u8>>> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < buf.len() {
            let Some((n, after)) = header(buf, i, b'*') else {
                break;
            };
            let mut cmd = Vec::new();
            let mut at = after;
            let mut whole = true;
            for _ in 0..n {
                let Some((len, body)) = header(buf, at, b'$') else {
                    whole = false;
                    break;
                };
                if body + len + 2 > buf.len() {
                    whole = false;
                    break;
                }
                cmd.push(buf[body..body + len].to_vec());
                at = body + len + 2;
            }
            if !whole {
                break;
            }
            out.push(cmd);
            i = at;
        }
        out
    }

    /// The count after a type byte, and where the line after it starts.
    fn header(buf: &[u8], at: usize, kind: u8) -> Option<(usize, usize)> {
        if buf.get(at) != Some(&kind) {
            return None;
        }
        let end = at + buf[at..].windows(2).position(|w| w == b"\r\n")?;
        let n = std::str::from_utf8(&buf[at + 1..end]).ok()?.parse().ok()?;
        Some((n, end + 2))
    }

    /// A server and one connection on it, driven the way the reactor will.
    struct At {
        server: Server,
        session: Session,
        argv: Argv,
        out: Out,
    }

    impl At {
        fn new() -> At {
            At {
                server: Server::new(),
                // The client id, not the database. Every one of these runs on
                // database zero, which is where a connection starts.
                session: Session::new(3),
                argv: Argv::new(),
                out: Out::new(Proto::Resp2),
            }
        }

        fn run(&mut self, parts: &[&[u8]]) -> String {
            let mut wire = format!("*{}\r\n", parts.len()).into_bytes();
            for p in parts {
                wire.extend_from_slice(format!("${}\r\n", p.len()).as_bytes());
                wire.extend_from_slice(p);
                wire.extend_from_slice(b"\r\n");
            }
            self.argv
                .decode(&wire, &Limits::default())
                .expect("the test wrote a legal command");
            self.out.clear();
            execute(
                &mut self.server,
                &mut self.session,
                Args::new(&self.argv, &wire),
                &mut self.out,
            );
            String::from_utf8_lossy(self.out.as_slice()).into_owned()
        }

        /// Move the clock the session's database is on.
        fn advance(&mut self, ms: u64) {
            self.server.db(0).clock_mut().advance(ms);
        }
    }

    #[test]
    fn a_key_goes_over_and_leaves_nothing_behind() {
        let peer = fake(vec![vec!["+OK\r\n", "+OK\r\n"]]);
        let mut at = At::new();
        at.run(&[b"set", b"k", b"v"]);
        let reply = at.run(&[
            b"migrate",
            b"127.0.0.1",
            peer.port.as_bytes(),
            b"k",
            b"0",
            b"1000",
        ]);
        assert_eq!(reply, "+OK\r\n");
        assert_eq!(at.run(&[b"exists", b"k"]), ":0\r\n");

        let sent = peer.words();
        assert_eq!(sent[0], ["SELECT", "0"]);
        assert_eq!(sent[1][0], "RESTORE");
        assert_eq!(sent[1][1], "k");
        assert_eq!(sent[1][2], "0", "no deadline goes over as a zero ttl");
        assert_eq!(sent[1].len(), 4, "and no REPLACE was asked for");

        // The payload is a real one, so a server that is handed it back gets the
        // value. That is the whole claim this command makes.
        let mut other = At::new();
        let payload = peer.seen()[1][3].clone();
        assert_eq!(
            other.run(&[b"restore", b"k", b"0", &payload]),
            "+OK\r\n",
            "the bytes that went over are a payload RESTORE takes"
        );
        assert_eq!(other.run(&[b"get", b"k"]), "$1\r\nv\r\n");
    }

    #[test]
    fn copy_leaves_the_key_where_it_is_and_replace_is_passed_on() {
        let peer = fake(vec![vec!["+OK\r\n", "+OK\r\n"]]);
        let mut at = At::new();
        at.run(&[b"set", b"k", b"v"]);
        let reply = at.run(&[
            b"migrate",
            b"127.0.0.1",
            peer.port.as_bytes(),
            b"k",
            b"0",
            b"1000",
            b"COPY",
            b"REPLACE",
        ]);
        assert_eq!(reply, "+OK\r\n");
        assert_eq!(at.run(&[b"exists", b"k"]), ":1\r\n");
        let sent = peer.words();
        assert_eq!(sent[1].len(), 5);
        assert_eq!(sent[1][4], "REPLACE");
    }

    #[test]
    fn a_key_that_is_not_here_is_not_an_error() {
        let peer = fake(vec![]);
        let mut at = At::new();
        assert_eq!(
            at.run(&[
                b"migrate",
                b"127.0.0.1",
                peer.port.as_bytes(),
                b"gone",
                b"0",
                b"1000"
            ]),
            "+NOKEY\r\n"
        );
        assert!(
            peer.seen().is_empty(),
            "and nothing was said to the other server"
        );
    }

    #[test]
    fn a_key_that_has_expired_is_the_same_nokey() {
        let peer = fake(vec![]);
        let mut at = At::new();
        at.run(&[b"set", b"k", b"v", b"PX", b"50"]);
        at.advance(60);
        assert_eq!(
            at.run(&[
                b"migrate",
                b"127.0.0.1",
                peer.port.as_bytes(),
                b"k",
                b"0",
                b"1000"
            ]),
            "+NOKEY\r\n"
        );
    }

    #[test]
    fn a_deadline_goes_over_as_what_is_left_of_it() {
        let peer = fake(vec![vec!["+OK\r\n", "+OK\r\n"]]);
        let mut at = At::new();
        at.run(&[b"set", b"k", b"v", b"PX", b"5000"]);
        at.advance(1000);
        at.run(&[
            b"migrate",
            b"127.0.0.1",
            peer.port.as_bytes(),
            b"k",
            b"0",
            b"1000",
        ]);
        assert_eq!(peer.words()[1][2], "4000");
    }

    #[test]
    fn the_keys_form_takes_the_ones_that_are_here() {
        let peer = fake(vec![vec!["+OK\r\n", "+OK\r\n", "+OK\r\n"]]);
        let mut at = At::new();
        at.run(&[b"mset", b"a", b"1", b"b", b"2"]);
        let reply = at.run(&[
            b"migrate",
            b"127.0.0.1",
            peer.port.as_bytes(),
            b"",
            b"0",
            b"1000",
            b"KEYS",
            b"a",
            b"missing",
            b"b",
        ]);
        assert_eq!(reply, "+OK\r\n");
        let sent = peer.words();
        assert_eq!(sent.len(), 3, "one SELECT and two RESTOREs");
        assert_eq!(sent[1][1], "a");
        assert_eq!(sent[2][1], "b");
        assert_eq!(at.run(&[b"exists", b"a", b"b"]), ":0\r\n");
    }

    #[test]
    fn the_keys_form_wants_the_key_argument_empty() {
        let mut at = At::new();
        assert_eq!(
            at.run(&[
                b"migrate",
                b"127.0.0.1",
                b"1",
                b"k",
                b"0",
                b"1000",
                b"KEYS",
                b"k"
            ]),
            format!("-ERR {KEYS_NEEDS_EMPTY}\r\n")
        );
    }

    /// And a `MIGRATE` with no `KEYS` moves a key called the empty string, so
    /// the check above cannot be done up front.
    #[test]
    fn the_empty_string_is_a_key_like_any_other() {
        let peer = fake(vec![vec!["+OK\r\n", "+OK\r\n"]]);
        let mut at = At::new();
        at.run(&[b"set", b"", b"v"]);
        assert_eq!(
            at.run(&[
                b"migrate",
                b"127.0.0.1",
                peer.port.as_bytes(),
                b"",
                b"0",
                b"1000"
            ]),
            "+OK\r\n"
        );
        assert_eq!(peer.words()[1][1], "");
    }

    #[test]
    fn an_option_written_after_keys_is_a_key() {
        let peer = fake(vec![vec!["+OK\r\n", "+OK\r\n"]]);
        let mut at = At::new();
        at.run(&[b"set", b"COPY", b"v"]);
        at.run(&[
            b"migrate",
            b"127.0.0.1",
            peer.port.as_bytes(),
            b"",
            b"0",
            b"1000",
            b"KEYS",
            b"COPY",
        ]);
        assert_eq!(peer.words()[1][1], "COPY");
        assert_eq!(
            at.run(&[b"exists", b"COPY"]),
            ":0\r\n",
            "so it was migrated and not read as an option"
        );
    }

    /// The options are read before the two numbers, so a command with both a bad
    /// option and a bad number is told about the option.
    #[test]
    fn the_options_are_read_before_the_numbers() {
        let mut at = At::new();
        assert_eq!(
            at.run(&[
                b"migrate",
                b"127.0.0.1",
                b"1",
                b"k",
                b"notanum",
                b"alsonot",
                b"BOGUS"
            ]),
            "-ERR syntax error\r\n"
        );
        assert_eq!(
            at.run(&[b"migrate", b"127.0.0.1", b"1", b"k", b"notanum", b"1000"]),
            "-ERR value is not an integer or out of range\r\n"
        );
    }

    #[test]
    fn auth_and_auth2_go_out_in_front() {
        let peer = fake(vec![vec!["+OK\r\n", "+OK\r\n", "+OK\r\n"]]);
        let mut at = At::new();
        at.run(&[b"set", b"k", b"v"]);
        at.run(&[
            b"migrate",
            b"127.0.0.1",
            peer.port.as_bytes(),
            b"k",
            b"0",
            b"1000",
            b"AUTH2",
            b"someone",
            b"hunter2",
        ]);
        let sent = peer.words();
        assert_eq!(sent[0], ["AUTH", "someone", "hunter2"]);
        assert_eq!(sent[1], ["SELECT", "0"]);
        assert_eq!(sent[2][0], "RESTORE");
    }

    #[test]
    fn auth_and_auth2_want_their_arguments() {
        let mut at = At::new();
        for bad in [
            &[b"migrate".as_slice(), b"h", b"1", b"k", b"0", b"1", b"AUTH"][..],
            &[b"migrate", b"h", b"1", b"k", b"0", b"1", b"AUTH2"],
            &[b"migrate", b"h", b"1", b"k", b"0", b"1", b"AUTH2", b"user"],
        ] {
            assert_eq!(at.run(bad), "-ERR syntax error\r\n");
        }
    }

    #[test]
    fn a_key_the_peer_refuses_stays_here() {
        let peer = fake(vec![vec![
            "+OK\r\n",
            "-BUSYKEY Target key name already exists.\r\n",
        ]]);
        let mut at = At::new();
        at.run(&[b"set", b"k", b"v"]);
        let reply = at.run(&[
            b"migrate",
            b"127.0.0.1",
            peer.port.as_bytes(),
            b"k",
            b"0",
            b"1000",
        ]);
        assert_eq!(
            reply,
            "-ERR Target instance replied with error: BUSYKEY Target key name already exists.\r\n"
        );
        assert_eq!(at.run(&[b"exists", b"k"]), ":1\r\n");
    }

    /// Each key stands on its own, and only the first complaint is reported.
    #[test]
    fn one_key_refused_does_not_hold_the_others_back() {
        let peer = fake(vec![vec!["+OK\r\n", "-BUSYKEY no\r\n", "+OK\r\n"]]);
        let mut at = At::new();
        at.run(&[b"mset", b"a", b"1", b"b", b"2"]);
        let reply = at.run(&[
            b"migrate",
            b"127.0.0.1",
            peer.port.as_bytes(),
            b"",
            b"0",
            b"1000",
            b"KEYS",
            b"a",
            b"b",
        ]);
        assert_eq!(
            reply,
            "-ERR Target instance replied with error: BUSYKEY no\r\n"
        );
        assert_eq!(
            at.run(&[b"exists", b"a"]),
            ":1\r\n",
            "the refused one stays"
        );
        assert_eq!(at.run(&[b"exists", b"b"]), ":0\r\n", "the taken one goes");
    }

    /// A `SELECT` that failed means every `RESTORE` behind it went to the wrong
    /// place, so nothing is deleted here whatever the peer said about the keys.
    #[test]
    fn a_select_that_fails_fails_every_key() {
        let peer = fake(vec![vec![
            "-ERR DB index is out of range\r\n",
            "+OK\r\n",
            "+OK\r\n",
        ]]);
        let mut at = At::new();
        at.run(&[b"mset", b"a", b"1", b"b", b"2"]);
        let reply = at.run(&[
            b"migrate",
            b"127.0.0.1",
            peer.port.as_bytes(),
            b"",
            b"9",
            b"1000",
            b"KEYS",
            b"a",
            b"b",
        ]);
        assert_eq!(
            reply,
            "-ERR Target instance replied with error: ERR DB index is out of range\r\n"
        );
        assert_eq!(at.run(&[b"exists", b"a", b"b"]), ":2\r\n");
    }

    /// The socket is kept, so the second migration to the same place does not
    /// say `SELECT` again. After a failure it does, because what the socket is
    /// selected on is no longer known.
    #[test]
    fn the_socket_is_kept_and_so_is_what_it_is_selected_on() {
        let peer = fake(vec![
            vec!["+OK\r\n", "+OK\r\n"],
            vec!["+OK\r\n"],
            vec!["-BUSYKEY no\r\n"],
            vec!["+OK\r\n", "+OK\r\n"],
        ]);
        let mut at = At::new();
        at.run(&[b"mset", b"a", b"1", b"b", b"2", b"c", b"3", b"d", b"4"]);
        let port = peer.port.clone();
        let go = |at: &mut At, key: &[u8]| {
            at.run(&[
                b"migrate",
                b"127.0.0.1",
                port.as_bytes(),
                key,
                b"0",
                b"1000",
            ])
        };
        assert_eq!(go(&mut at, b"a"), "+OK\r\n");
        assert_eq!(go(&mut at, b"b"), "+OK\r\n");
        assert_eq!(
            go(&mut at, b"c"),
            "-ERR Target instance replied with error: BUSYKEY no\r\n"
        );
        assert_eq!(go(&mut at, b"d"), "+OK\r\n");

        let sent = peer.words();
        let selects: Vec<&Vec<String>> = sent.iter().filter(|c| c[0] == "SELECT").collect();
        assert_eq!(selects.len(), 2, "one at the start and one after the error");
        assert_eq!(sent[0][0], "SELECT");
        assert_eq!(sent[1][0], "RESTORE");
        assert_eq!(sent[2][0], "RESTORE", "the second one did not select again");
        assert_eq!(sent[3][0], "RESTORE");
        assert_eq!(sent[4][0], "SELECT", "and the fourth one had to");
    }

    /// A peer that is not there is an `IOERR`, and the key stays put.
    ///
    /// Which of the two `IOERR` lines comes back is not this code's decision and
    /// the test does not pin it. A machine that answers a connection to a closed
    /// port with a reset gets the write line, because that is what [`connect`]
    /// says a refusal is. A machine whose firewall drops the packet instead
    /// never gets an answer at all and times out, which is the connect line. The
    /// Windows runners do the second one and every other machine we build on
    /// does the first, so asserting either line would be asserting a property of
    /// the host and not of `MIGRATE`.
    ///
    /// What is worth pinning is what the client can act on: it is an `IOERR` and
    /// not some other error, and the key is still here to try again with.
    #[test]
    fn a_peer_that_is_not_listening_is_an_io_error() {
        // Bound and then dropped, so the port is one nothing answers on.
        let port = TcpListener::bind("127.0.0.1:0")
            .expect("a free port")
            .local_addr()
            .expect("the port it bound")
            .port()
            .to_string();
        let mut at = At::new();
        at.run(&[b"set", b"k", b"v"]);
        let reply = at.run(&[
            b"migrate",
            b"127.0.0.1",
            port.as_bytes(),
            b"k",
            b"0",
            b"200",
        ]);
        let either = [CONNECT_FAILED, WRITE_FAILED]
            .map(|line| format!("-{}\r\n", String::from_utf8_lossy(line)));
        assert!(either.contains(&reply), "got {reply:?}");
        assert_eq!(at.run(&[b"exists", b"k"]), ":1\r\n", "and the key stays");
    }

    #[test]
    fn a_peer_that_never_answers_is_a_read_error() {
        let peer = fake(vec![]);
        let mut at = At::new();
        at.run(&[b"set", b"k", b"v"]);
        assert_eq!(
            at.run(&[
                b"migrate",
                b"127.0.0.1",
                peer.port.as_bytes(),
                b"k",
                b"0",
                b"200"
            ]),
            format!("-{}\r\n", String::from_utf8_lossy(READ_FAILED))
        );
        assert_eq!(at.run(&[b"exists", b"k"]), ":1\r\n");
    }

    /// A timeout of zero or less is one second, not no wait and not an error.
    #[test]
    fn a_timeout_that_is_not_positive_is_one_second() {
        let peer = fake(vec![vec!["+OK\r\n", "+OK\r\n"]]);
        let mut at = At::new();
        at.run(&[b"set", b"k", b"v"]);
        assert_eq!(
            at.run(&[
                b"migrate",
                b"127.0.0.1",
                peer.port.as_bytes(),
                b"k",
                b"0",
                b"-5"
            ]),
            "+OK\r\n"
        );
    }

    #[test]
    fn a_port_is_read_the_way_atoi_reads_one() {
        assert_eq!(port_of(b"6379"), 6379);
        assert_eq!(port_of(b" 6379"), 6379);
        assert_eq!(port_of(b"63rubbish"), 63);
        assert_eq!(port_of(b"rubbish"), 0);
        assert_eq!(port_of(b""), 0);
        assert_eq!(
            port_of(b"-1"),
            65535,
            "the low sixteen bits, as htons sees it"
        );
        assert_eq!(port_of(b"65536"), 0);
    }
}
