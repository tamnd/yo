//! `yodb serve`: a socket in front of the engine.
//!
//! `yo_resp::engine` framed commands and wrote replies into a sink, and a sink
//! that keeps its bytes in a `Vec` is enough to test with and not enough to
//! point `redis-benchmark` at. This is the sink that is a socket, plus the
//! accept loop around it, which is what makes the M2 exit gate runnable at all.
//!
//! # Why `std::net` and not the ring
//!
//! Because the gate is measured on four machines and three of them are not
//! Linux. `04` section 7 puts the network on io_uring, and that is still where
//! this ends up, but a server that only exists on Linux cannot produce the
//! macOS and Windows rows the milestone asks for. So the loop here is
//! non blocking sockets and a readiness scan, which is the same shape with a
//! worse multiplexer: accept what is waiting, read what is readable, run one
//! batch, write once per connection. When the ring lands it replaces the scan
//! and nothing above this file changes, because the engine already talks to a
//! [`Sink`] rather than to a socket.
//!
//! # Asking instead of guessing
//!
//! One turn used to walk every open connection and try to read from each one,
//! which is a syscall per idle connection per turn. A profile of the gate run
//! said what that costs: 2.26 `recvfrom` per command, most of them returning
//! `EWOULDBLOCK`, and no waiting call anywhere in the trace. With 50 busy
//! connections and one request in flight on each, about half the reads were the
//! kernel being asked a question it had already answered.
//!
//! So the loop asks once per turn instead, through [`Poller`]: `epoll` on
//! Linux, `kqueue` on macOS, and the old scan everywhere else. The listener is
//! registered like any other source, which also takes the wasted `accept` off
//! every turn.
//!
//! An idle turn waits in the kernel rather than sleeping on a timer, so a quiet
//! server costs nothing and the first command after a quiet period is not
//! waiting on a sleep to finish. The wait is kept short while any reply is
//! still owed, because a socket that was full is retried on a timer and not on
//! an event.

//! # Two doors into the same loop
//!
//! A TCP port and, on Unix, a socket file. Same engine, same batch, same
//! everything above the descriptor: the only difference is which listener
//! accepted the connection, and by the time it is a `ConnId` nothing further up
//! can tell. The socket file is there because the loopback round trip is what
//! bounds every wire number this project publishes (`bench/00` section 4.2) and
//! a Unix socket does not pay for the TCP stack, so it is the cheapest thing
//! that moves the ceiling rather than the engine.

use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

#[cfg(unix)]
use std::os::unix::net::{UnixListener, UnixStream};

use yo_reactor::Reactor;
use yo_resp::dispatch::Server as Shared;
use yo_resp::dispatch::{Loaded, Refused};
use yo_resp::engine::{Cmd, ConnId, Sink, Wire, pump};

use crate::poll::Poller;
use crate::store::Store;

/// How much is read off one connection at a time.
///
/// A pipeline of 64 `SET`s with sixteen byte keys and values is about four
/// kilobytes, so this holds a full batch from a benchmark client and the loop
/// does not go round again for the tail of one.
const READ_CHUNK: usize = 16 * 1024;

/// Turns with nothing to do before the loop starts waiting in the kernel.
///
/// A short spin first, because a request response client sends the next command
/// as soon as it has the answer to the last one, and the answer left this
/// process microseconds ago.
const SPIN_TURNS: u32 = 256;

/// How long an idle loop waits for something to arrive.
///
/// It comes back the moment anything does, so this is only how often a server
/// with nothing to do wakes up to check the stop flag.
const IDLE_WAIT: Duration = Duration::from_millis(20);

/// The longest wait while a reply is still owed to a full socket.
///
/// Writability is not registered, so nothing arriving will wake the loop up to
/// retry that write, and this is the timer it is retried on instead.
///
/// A thread holding a blocked client waits the same millisecond, for the same
/// reason: what answers a `BLPOP` is a push, and on a server with more than one
/// thread that push lands on whichever thread its client is on. Nothing arrives
/// on this thread's poller to say so, so a thread that slept the idle wait would
/// leave its client blocked for up to twenty milliseconds after the list it is
/// waiting on already had something in it. One millisecond is also finer than
/// the ten a second Redis checks its own blocked clients at.
const OWED_WAIT: Duration = Duration::from_millis(1);

/// The token the listener is registered under.
///
/// Connections are registered under their own id, and ids come from a free list
/// that starts at zero, so the top of the range is the one value that is never
/// a connection.
const LISTENER: u64 = u64::MAX;

/// The token the socket file listener is registered under.
///
/// One below the other one, for the same reason: ids come from a free list that
/// starts at zero and there are not four billion connections.
const UNIX_LISTENER: u64 = u64::MAX - 1;

/// How many stripes each database is cut into, for a given thread count.
///
/// Four to a thread rather than one, because a stripe is taken for the length of
/// a command and threads do not take turns: with one stripe each, two of eight
/// threads are on the same stripe about a third of the time, and with four each
/// that falls to under a tenth. The database rounds this up to a power of two
/// and clamps it, so what comes back is not always what goes in.
///
/// The multiplier is a number to settle with a benchmark rather than by
/// argument, and this is the starting point rather than the answer.
fn stripes(threads: usize) -> usize {
    threads * 4
}

/// One accepted connection, whichever door it came in through.
///
/// An enum and not a boxed trait object, because the read and the write are on
/// the hot path and this way both stay direct calls. On Windows there is one
/// variant, which the compiler is welcome to notice.
enum Sock {
    Tcp(TcpStream),
    #[cfg(unix)]
    Unix(UnixStream),
}

impl Read for Sock {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Sock::Tcp(s) => s.read(buf),
            #[cfg(unix)]
            Sock::Unix(s) => s.read(buf),
        }
    }
}

impl Write for Sock {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Sock::Tcp(s) => s.write(buf),
            #[cfg(unix)]
            Sock::Unix(s) => s.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Sock::Tcp(s) => s.flush(),
            #[cfg(unix)]
            Sock::Unix(s) => s.flush(),
        }
    }
}

#[cfg(unix)]
impl std::os::fd::AsRawFd for Sock {
    fn as_raw_fd(&self) -> std::os::fd::RawFd {
        match self {
            Sock::Tcp(s) => s.as_raw_fd(),
            Sock::Unix(s) => s.as_raw_fd(),
        }
    }
}

/// The same handle for the poller on Windows, where a socket is its own kind of
/// number and not a file handle.
#[cfg(windows)]
impl std::os::windows::io::AsRawSocket for Sock {
    fn as_raw_socket(&self) -> std::os::windows::io::RawSocket {
        match self {
            Sock::Tcp(s) => s.as_raw_socket(),
        }
    }
}

impl Sock {
    /// The two ends of this connection, as `CLIENT INFO` spells them.
    ///
    /// A TCP connection is `ip:port` on both sides, and IPv6 keeps the brackets
    /// Rust's formatting gives it, which is what Redis reports too. A Unix
    /// connection is the socket file's path with `:0` after it, on both sides,
    /// because the accepted end of a Unix socket has no name of its own and the
    /// path is what an operator is looking for. An address the system will not
    /// give us leaves the empty string rather than an error, since a connection
    /// is not worth refusing over a field only a report reads.
    fn ends(&self) -> (String, String) {
        match self {
            Sock::Tcp(s) => {
                let peer = s.peer_addr().map(|a| a.to_string()).unwrap_or_default();
                let local = s.local_addr().map(|a| a.to_string()).unwrap_or_default();
                (peer, local)
            }
            #[cfg(unix)]
            Sock::Unix(s) => {
                let path = s
                    .local_addr()
                    .ok()
                    .and_then(|a| a.as_pathname().map(|p| p.display().to_string()))
                    .unwrap_or_default();
                let both = format!("{path}:0");
                (both.clone(), both)
            }
        }
    }

    /// Whether this is a Unix socket, which `CLIENT INFO` reports as a flag.
    fn is_unix(&self) -> bool {
        #[cfg(unix)]
        {
            matches!(self, Sock::Unix(_))
        }
        #[cfg(not(unix))]
        {
            false
        }
    }

    /// The descriptor number, or minus one on a platform where a socket is not
    /// one.
    fn fd(&self) -> i32 {
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            self.as_raw_fd()
        }
        #[cfg(not(unix))]
        {
            -1
        }
    }
}

/// A door the server is listening at.
enum Door {
    Tcp(TcpListener),
    #[cfg(unix)]
    /// The listener and the path it has to unlink on the way out, because a
    /// socket file outlives the process that made it and the next start would
    /// find the address in use.
    Unix(UnixListener, PathBuf),
}

impl Door {
    /// The listener behind this door, when the door is a port and not a socket
    /// file.
    ///
    /// This is a method and not a `match` at the one place that needs it,
    /// because on a platform with no unix sockets `Door` has a single variant
    /// and every shape of that `match` written inline is a lint: an `if let` is
    /// irrefutable, a loop around it never loops, and a closure that only ever
    /// answers `Some` is a `map` wearing a `find_map`. Behind a call the caller
    /// reads the same on every platform and clippy has nothing to say.
    fn tcp(&self) -> Option<&TcpListener> {
        #[cfg(unix)]
        {
            match self {
                Door::Tcp(l) => Some(l),
                Door::Unix(..) => None,
            }
        }
        #[cfg(not(unix))]
        {
            let Door::Tcp(l) = self;
            Some(l)
        }
    }

    /// Take one waiting connection, already set up the way the loop wants it.
    fn accept(&self) -> io::Result<Sock> {
        match self {
            Door::Tcp(l) => {
                let (stream, _) = l.accept()?;
                stream.set_nonblocking(true)?;
                // Redis sets this and so does everything that talks to it.
                // Without it a reply waits for the next packet's worth of data
                // that a request response client is never going to send, which
                // turns a 50 microsecond round trip into a 40 millisecond one.
                let _ = stream.set_nodelay(true);
                Ok(Sock::Tcp(stream))
            }
            #[cfg(unix)]
            Door::Unix(l, _) => {
                let (stream, _) = l.accept()?;
                stream.set_nonblocking(true)?;
                // No Nagle on a Unix socket, because there is no TCP under it.
                Ok(Sock::Unix(stream))
            }
        }
    }

    /// The token this door is reported under.
    fn token(&self) -> u64 {
        match self {
            Door::Tcp(_) => LISTENER,
            #[cfg(unix)]
            Door::Unix(..) => UNIX_LISTENER,
        }
    }
}

#[cfg(unix)]
impl std::os::fd::AsRawFd for Door {
    fn as_raw_fd(&self) -> std::os::fd::RawFd {
        match self {
            Door::Tcp(l) => l.as_raw_fd(),
            Door::Unix(l, _) => l.as_raw_fd(),
        }
    }
}

#[cfg(windows)]
impl std::os::windows::io::AsRawSocket for Door {
    fn as_raw_socket(&self) -> std::os::windows::io::RawSocket {
        match self {
            Door::Tcp(l) => l.as_raw_socket(),
        }
    }
}

impl Drop for Door {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Door::Unix(_, path) = self {
            // Ours to remove, because we made it. A failure here means somebody
            // else already did, which is the outcome this wanted anyway.
            let _ = std::fs::remove_file(path);
        }
    }
}

/// The sockets, indexed by the connection id the engine handed out.
#[derive(Default)]
struct Net {
    streams: Vec<Option<Sock>>,
    /// Connections whose socket failed, to be told to the engine after the
    /// batch rather than in the middle of it.
    dead: Vec<ConnId>,
    /// Connections whose socket has just been dropped, to be taken out of the
    /// poller after the batch for the same reason.
    gone: Vec<ConnId>,
    /// How many of the slots above have a socket in them.
    ///
    /// Kept rather than counted because it is read once a turn by the code that
    /// decides whether this worker should still be in the queue for the doors,
    /// and the slot vector only ever grows, so counting it would get slower for
    /// the rest of the run every time a connection with a high id arrives.
    open: usize,
}

impl Net {
    /// Put a freshly accepted socket at the id the engine gave it.
    fn attach(&mut self, conn: ConnId, stream: Sock) {
        if self.streams.len() <= conn as usize {
            self.streams.resize_with(conn as usize + 1, || None);
        }
        if self.streams[conn as usize].replace(stream).is_none() {
            self.open += 1;
        }
    }

    /// Whether this id currently has a socket.
    fn is_open(&self, conn: ConnId) -> bool {
        self.streams.get(conn as usize).is_some_and(Option::is_some)
    }

    /// How many sockets this worker is holding, which is how many connections
    /// it took.
    fn held(&self) -> usize {
        self.open
    }

    /// Read whatever is waiting, or `None` if the peer has gone or the socket
    /// failed.
    fn read(&mut self, conn: ConnId, buf: &mut [u8]) -> Option<usize> {
        let stream = self.streams.get_mut(conn as usize)?.as_mut()?;
        match stream.read(buf) {
            // A read of zero on a socket is the peer closing, not an empty
            // read. The distinction matters: one is a hangup and the other is
            // the ordinary case below.
            Ok(0) => None,
            Ok(n) => Some(n),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => Some(0),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => Some(0),
            Err(_) => None,
        }
    }
}

impl Sink for Net {
    fn write(&mut self, conn: ConnId, bytes: &[u8]) -> usize {
        let Some(stream) = self.streams.get_mut(conn as usize).and_then(Option::as_mut) else {
            // The socket has already gone. Say the bytes were taken so the
            // engine drops them instead of holding a reply nobody will read.
            return bytes.len();
        };
        match stream.write(bytes) {
            Ok(n) => n,
            // The socket is full. The engine keeps the rest and offers it
            // again next turn, which is the whole of the backpressure story.
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => 0,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => 0,
            Err(_) => {
                self.dead.push(conn);
                bytes.len()
            }
        }
    }

    fn closed(&mut self, conn: ConnId) {
        // The one place a socket is dropped. The engine calls this when the
        // last command holding that connection's buffer has run, so a client
        // that hangs up mid batch does not free a buffer still being read.
        if let Some(slot) = self.streams.get_mut(conn as usize)
            && slot.take().is_some()
        {
            self.open -= 1;
        }
        self.gone.push(conn);
    }
}

/// The doors, with the engine behind them.
pub struct Server {
    doors: Vec<Door>,
    /// The keyspace and the numbers, which is the one thing the threads share.
    ///
    /// Built here rather than in [`Server::run`] because everything a server is
    /// told at startup, which is the directory, the limit and the file, is told
    /// to this and has to be told before a thread is looking at it.
    shared: Arc<Shared>,
    /// How many threads will serve out of it.
    threads: usize,
}

impl Server {
    /// Bind whichever doors were asked for, and hand back a server that has
    /// accepted nothing yet.
    ///
    /// The thread count is taken here and not in [`Server::run`] because it
    /// decides how many stripes each database is cut into, and that is fixed
    /// when the databases are made. Zero is not a count and is read as one,
    /// which is what the caller should already have resolved.
    ///
    /// # Errors
    ///
    /// Whatever `bind` says, and an error of its own when neither a port nor a
    /// path was given, because a server nobody can reach is not a server.
    pub fn open(
        addr: Option<SocketAddr>,
        path: Option<PathBuf>,
        threads: usize,
    ) -> io::Result<Server> {
        let mut doors = Vec::new();
        if let Some(addr) = addr {
            let listener = TcpListener::bind(addr)?;
            listener.set_nonblocking(true)?;
            doors.push(Door::Tcp(listener));
        }
        if let Some(path) = path {
            doors.push(unix_door(&path)?);
        }
        if doors.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "nothing to listen on: give a port, a socket file, or both",
            ));
        }
        let threads = threads.max(1);
        let mut shared = if threads == 1 {
            Shared::new()
        } else {
            Shared::with_width(stripes(threads))
        };
        shared.set_threads(threads);
        Ok(Server {
            doors,
            shared: Arc::new(shared),
            threads,
        })
    }

    /// The server every thread will be serving out of, while it is still this
    /// one's to change.
    ///
    /// # Panics
    ///
    /// Never from here. Nothing else holds a handle until [`Server::run`] makes
    /// the workers, and everything that calls this runs before that.
    fn setup(&mut self) -> &mut Shared {
        Arc::get_mut(&mut self.shared).expect("nothing is serving out of it yet")
    }

    /// Where the server writes, which today is where `BACKUP` puts its files.
    ///
    /// It is what `CONFIG GET dir` answers and what `BACKUP LIST` builds its
    /// absolute paths out of, so it is taken here at startup and cannot be
    /// changed afterwards, the same as on a real server without protected
    /// configs turned on.
    pub fn set_dir(&mut self, dir: PathBuf) {
        self.setup().set_dir(dir);
    }

    /// How much memory the server may use before something has to go.
    pub fn set_maxmemory(&mut self, bytes: u64) {
        self.setup().set_maxmemory(bytes);
    }

    /// Point the server at an ACL file, which is where its users come from.
    ///
    /// Taken here at startup and not readable afterwards, the same as on a real
    /// server, where `aclfile` is an immutable config. Giving one does not read
    /// it, [`Server::load_acl`] does, so the caller decides what a file that
    /// will not parse means.
    pub fn set_aclfile(&mut self, path: PathBuf) {
        self.setup().set_aclfile(path);
    }

    /// Read the ACL file and make it the server's users.
    ///
    /// # Errors
    ///
    /// Everything the file got wrong, in one sentence, and the users are left
    /// exactly as they were. Nothing on an ACL file that will not open, because
    /// a server that was not given one has nothing to read.
    pub fn load_acl(&self) -> Result<(), String> {
        self.shared.load_acl()
    }

    /// The password every connection has to send `AUTH` with.
    ///
    /// Taken here at startup so that the first connection after the port opens
    /// is already asked for it. An empty one is no password, the same as
    /// `CONFIG SET requirepass ""`, and no password is the default.
    pub fn set_password(&mut self, password: &[u8]) {
        self.setup().set_password(password);
    }

    /// Give the engine a file to move cold values into when it hits that limit.
    ///
    /// With a file under it a memory limit stops meaning "delete keys" and
    /// starts meaning "keep the working set in memory", which is `14` section
    /// 4.1 and is the whole point of the thing. Without one the same limit
    /// evicts, which is Redis and is what every server did before this existed.
    ///
    /// A log is only opened for a database that actually comes under pressure,
    /// so a server that is given a file and never fills memory writes nothing to
    /// it and pays nothing for having been offered one.
    pub fn use_store(&mut self, store: Store) {
        self.setup().set_store_source(store.source());
    }

    /// Tell the engine which port to announce to a master.
    ///
    /// Whoever bound the socket is the only one who knows it, and a port of zero
    /// means the number a master would be told is not the number it could dial
    /// back on. So this is called with what the listener actually got rather than
    /// with what was asked for.
    pub fn announce_port(&self, port: u16) {
        self.shared.announce_port(port);
    }

    /// What the link to a master authenticates with, if anything.
    ///
    /// An empty password is no password and the link sends no `AUTH` at all,
    /// which is what a master that asks for nothing wants.
    pub fn set_master_auth(&self, user: &[u8], pass: &[u8]) {
        self.shared.master_auth(user, pass);
    }

    /// Whether an ordinary client's write is refused while this server follows.
    pub fn set_replica_read_only(&self, yes: bool) {
        self.shared.set_replica_read_only(yes);
    }

    /// Start following a master, the same as `REPLICAOF host port` would.
    ///
    /// Called after everything else at startup, because the link starts applying
    /// the moment it is up and every other setting decides what applying means.
    /// It returns as soon as the intent is recorded: the dial, the snapshot and
    /// the stream all happen on a thread of their own, so a master that is not
    /// up yet is a replica that keeps trying rather than a server that will not
    /// start.
    pub fn follow(&self, host: &str, port: u16) {
        self.shared.follow_master(host, port);
    }

    /// Build the dataset out of a Redis RDB image before anybody connects.
    ///
    /// The doors are already bound at this point, because the keyspace being
    /// filled lives behind them, but nothing has been accepted: a client that
    /// connects while this is running is sitting in the backlog and gets a whole
    /// dataset the moment [`Server::run`] starts. So there is no window in which
    /// a connection can read half a file.
    ///
    /// The flush is asked for and there is nothing to flush, since this runs on a
    /// server that has served nothing. It is passed anyway rather than left to
    /// chance, because what this means is "the dataset is the file" and that is
    /// the same sentence whether or not there was something in the way.
    ///
    /// # Errors
    ///
    /// [`Refused`] for a file that will not parse or that wants a database this
    /// server has not got. The caller ends the process, because a server that
    /// came up holding half of the file it was pointed at is worse than one that
    /// did not come up.
    pub fn restore(&self, image: &[u8]) -> Result<Loaded, Refused> {
        yo_alloc::allow(|| self.shared.load_image(image, true))
    }

    /// Where it actually landed, which is the only way to find out when the
    /// port asked for was zero.
    ///
    /// # Errors
    ///
    /// Whatever the socket says.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        match self.doors.iter().find_map(Door::tcp) {
            Some(l) => l.local_addr(),
            None => Err(io::Error::new(
                io::ErrorKind::NotFound,
                "this server has no port, only a socket file",
            )),
        }
    }

    /// Turn the loop until `stop` is set, or until a client says `SHUTDOWN`.
    ///
    /// Two doors out and they are the same door. `stop` is what a signal
    /// handler sets and the engine's flag is what the command sets, and both
    /// are read by every thread, so everything that happens after this returns,
    /// which is the socket file going away and the file being closed, happens
    /// once and in one place regardless of which one asked and regardless of
    /// which thread heard it.
    ///
    /// Every worker is built before any of them starts, so a poller that will
    /// not register is an error out of here rather than a thread that quietly
    /// dies on its own while the rest carry on serving.
    ///
    /// # Errors
    ///
    /// Only an accept failing for a reason that is not "nothing waiting", or a
    /// worker that could not be set up. A connection failing is that
    /// connection's problem and closes it.
    ///
    /// # Panics
    ///
    /// If a worker thread panicked, which is carried out to this thread rather
    /// than swallowed.
    pub fn run(&mut self, stop: &AtomicBool) -> io::Result<()> {
        let mut workers = Vec::with_capacity(self.threads);
        let split = Arc::new(Split::new(self.threads));
        for me in 0..self.threads {
            workers.push(Worker::new(
                &self.doors,
                Wire::over(Arc::clone(&self.shared), Net::default()),
                Arc::clone(&split),
                me,
            )?);
        }
        // One thread and there is nothing to spawn: this thread is it, which
        // keeps a single threaded server exactly as it was and keeps the stack
        // a panic unwinds through the same one it used to be.
        if workers.len() == 1 {
            return workers.remove(0).run(stop);
        }
        let mut failed = Ok(());
        std::thread::scope(|scope| {
            let mut running = Vec::with_capacity(workers.len());
            for mut worker in workers {
                running.push(scope.spawn(move || worker.run(stop)));
            }
            for handle in running {
                match handle.join() {
                    Ok(Ok(())) => {}
                    // The first one wins. The ones behind it are usually the
                    // same failure seen from another thread, and a listener
                    // that stopped listening stops every worker anyway.
                    Ok(Err(e)) if failed.is_ok() => failed = Err(e),
                    Ok(Err(_)) => {}
                    Err(panic) => std::panic::resume_unwind(panic),
                }
            }
        });
        failed
    }
}

/// Who is in the queue for the doors.
///
/// Every worker publishes how many connections it is holding and reads what the
/// others published, and a worker keeps the doors in its poller only while it is
/// holding no more than the fewest of them. A worker that has just accepted is
/// above the rest by one and steps out, so the next connection wakes somebody
/// who has not, and when everybody has taken one they are level again and all of
/// them step back in. That is round robin without anybody having to keep a turn
/// counter, and unlike a turn counter it is still right when the connections do
/// not close at the same rate: a worker whose clients went away drops back to the
/// fewest and starts taking again.
///
/// A count and not a rate, because the number that was wrong is the number of
/// connections a thread ends up owning for the life of the benchmark, and
/// because a count is the one thing a worker can publish without a lock.
///
/// # Why the counts are atomics and the per thread `Stats` are not
///
/// A worker's own statistics are single writer cells on purpose, so nothing else
/// may look at them. These are written by one thread and read by all of them, so
/// they are `AtomicU32` and every access is relaxed: a stale count means a
/// connection goes to the second least loaded worker instead of the least, which
/// is not worth a fence.
struct Split {
    /// How many connections each worker is holding, indexed by worker.
    live: Vec<AtomicU32>,
    /// How many workers have the doors in their poller.
    ///
    /// The one thing here that has to be exactly right. A worker will not step
    /// out if it is the last one in, because a moment with nobody watching the
    /// doors is a client waiting on a connection that nothing is going to
    /// accept, and level triggered means nothing wakes anybody to fix it.
    ///
    /// The price is that the last worker in the queue keeps taking connections
    /// while it is ahead, until one of the others notices that it is now the one
    /// with the fewest and comes back. That is a handful of connections in a
    /// burst, because a worker with a connection on it is running rather than
    /// waiting and comes back round within microseconds, and it is the same
    /// direction of error as before rather than the old one: a few extra on one
    /// thread instead of the whole backlog. The alternative, letting the count
    /// reach nought and having whoever notices put it right, trades that for a
    /// client that waits a whole idle wait to be accepted, and a connection that
    /// is not accepted for twenty milliseconds is worse than a thread that has
    /// two more than its neighbours.
    watchers: AtomicU32,
    /// Moved on whenever any of the above changes.
    ///
    /// A worker reads this once a turn and does nothing further if it has not
    /// moved, which is what keeps the ordinary turn at one relaxed load rather
    /// than a walk of every worker's count.
    epoch: AtomicU64,
}

impl Split {
    /// Everybody holding nothing and everybody in the queue, which is the state
    /// [`Worker::new`] leaves behind it.
    fn new(threads: usize) -> Split {
        Split {
            live: (0..threads).map(|_| AtomicU32::new(0)).collect(),
            watchers: AtomicU32::new(u32::try_from(threads).unwrap_or(u32::MAX)),
            epoch: AtomicU64::new(0),
        }
    }

    /// Say how many connections a worker is holding now.
    fn publish(&self, me: usize, held: u32) {
        self.live[me].store(held, Ordering::Relaxed);
        self.stir();
    }

    /// The fewest connections any worker is holding.
    fn fewest(&self) -> u32 {
        self.live
            .iter()
            .map(|n| n.load(Ordering::Relaxed))
            .min()
            .unwrap_or(0)
    }

    /// Take a worker out of the queue for the doors, or refuse if it is the
    /// last one in it.
    ///
    /// The decrement happens first and is put back if it went too far, so two
    /// workers leaving at once cannot both read the count as safe: one of them
    /// sees the value the other already took away.
    fn step_out(&self) -> bool {
        if self.watchers.fetch_sub(1, Ordering::Relaxed) > 1 {
            self.stir();
            return true;
        }
        self.watchers.fetch_add(1, Ordering::Relaxed);
        false
    }

    /// Put a worker back in the queue for the doors.
    fn step_in(&self) {
        self.watchers.fetch_add(1, Ordering::Relaxed);
        self.stir();
    }

    /// Tell every worker to look again.
    fn stir(&self) {
        self.epoch.fetch_add(1, Ordering::Relaxed);
    }

    /// What a worker compares against what it saw last turn.
    fn turn(&self) -> u64 {
        self.epoch.load(Ordering::Relaxed)
    }
}

/// One thread's loop, with its own connections in front of the shared server.
///
/// Everything here belongs to the thread that made it: the poller, the sockets,
/// the read buffer and every connection accepted through it. The doors are
/// borrowed rather than owned, because the listeners are the one thing on this
/// side that all the threads look at.
///
/// # Why every thread accepts, and why it takes one at a time
///
/// The listeners go into every worker's poller, so a connection waiting at a
/// door wakes the threads that are in the queue for it and the first one to call
/// `accept` takes it. The rest get "nothing waiting", which they already handle,
/// since that is what a listener with no backlog says on an ordinary turn. There
/// is no handoff between threads at any point: a connection belongs to the thread
/// that accepted it for as long as it is open.
///
/// A worker takes one connection per ready event and goes back to the poller
/// rather than draining the door. Draining looks like the obvious thing and is
/// what this did first. It is wrong here because clients do not arrive one at a
/// time: a benchmark opens its two hundred and fifty six connections at once,
/// every worker is idle at that moment, and whichever one wins the wakeup drains
/// the entire backlog into itself while the rest find nothing.
///
/// Taking one at a time is not enough on its own, which is what the numbers from
/// the sweep said next. Whoever wins the wakeup wins it again straight away,
/// because it is already awake and running and the others are coming back from a
/// wait, so the connections still pile up on the threads that got there first.
/// Measured at sixteen threads, one thread accepted 215 of 771 connections and
/// another accepted 22, and since the work follows the connections exactly, the
/// busiest thread ran two and a half times the commands of the quietest.
///
/// So a worker steps out of the queue for the doors once it is holding more than
/// the fewest of them, and steps back in when it is level again. That is
/// [`Split`], and it turns the burst above into round robin: each worker takes
/// one, drops out, and the door wakes somebody who has not taken one yet. It
/// costs one relaxed load a turn in the ordinary case and one `epoll_ctl` per
/// accept in the busy one.
///
/// The doors are level triggered, so a door with more waiting is ready again
/// straight away and the next worker in the queue takes the next one. The cost is
/// one extra wakeup per connection, which is paid once per connection rather than
/// once per command.
///
/// `SO_REUSEPORT` would let the kernel do the split and does not help here: it is
/// a TCP and UDP option, and the sweep that found this runs every server over a
/// unix socket.
struct Worker<'a> {
    doors: &'a [Door],
    reactor: Reactor<Wire<Net>>,
    poller: Poller,
    /// The batch the reactor runs, kept across turns so no turn allocates.
    batch: Vec<Cmd>,
    /// The tokens the poller said were ready, kept for the same reason.
    ready: Vec<u64>,
    buf: Vec<u8>,
    /// Which worker this is, which is the slot it publishes its count into.
    me: usize,
    split: Arc<Split>,
    /// Whether the doors are in this worker's poller right now.
    watching: bool,
    /// The count this worker last published, so an unchanged turn writes
    /// nothing that another thread has to fetch the line for.
    held: u32,
    /// The epoch this worker last looked at.
    seen: u64,
}

impl<'a> Worker<'a> {
    /// A worker with the doors registered and nothing accepted yet.
    fn new(
        doors: &'a [Door],
        engine: Wire<Net>,
        split: Arc<Split>,
        me: usize,
    ) -> io::Result<Worker<'a>> {
        let mut poller = Poller::new()?;
        for door in doors {
            poller.add(door, door.token())?;
        }
        Ok(Worker {
            doors,
            reactor: Reactor::inline(engine),
            poller,
            batch: Vec::with_capacity(64),
            ready: Vec::with_capacity(64),
            buf: vec![0; READ_CHUNK],
            me,
            // Everybody starts in the queue for the doors, which is what
            // `Split::new` counted, and holding nothing, which is what it put in
            // every slot. So the first turn has nothing to publish and nothing
            // to decide.
            split,
            watching: true,
            held: 0,
            seen: 0,
        })
    }

    /// Turn this thread's loop until somebody says to stop.
    ///
    /// The engine is asked at the top of a turn rather than the moment the
    /// command runs, so the batch that carried the `SHUTDOWN` finishes and its
    /// replies go out first. There are none for the `SHUTDOWN` itself, and
    /// there may well be some for the commands that shared its batch.
    fn run(&mut self, stop: &AtomicBool) -> io::Result<()> {
        let mut idle = 0u32;
        while !stop.load(Ordering::Relaxed) && !self.reactor.engine().stopping() {
            let engine = self.reactor.engine();
            let wait = if idle <= SPIN_TURNS {
                Duration::ZERO
            } else if engine.owed() > 0 || engine.waiting() > 0 || engine.posted() > 0 {
                OWED_WAIT
            } else {
                IDLE_WAIT
            };
            self.poller.wait(&mut self.ready, wait)?;

            let mut worked = false;
            for at in 0..self.ready.len() {
                match self.ready[at] {
                    LISTENER => self.accept_ready(LISTENER)?,
                    UNIX_LISTENER => self.accept_ready(UNIX_LISTENER)?,
                    token => self.read_conn(token as ConnId),
                }
                worked = true;
            }

            if pump(&mut self.reactor, &mut self.batch) > 0 {
                worked = true;
            }
            self.bury_dead();
            self.forget_closed();
            self.share_the_doors()?;

            if worked {
                idle = 0;
            } else {
                idle = idle.saturating_add(1);
            }
        }
        Ok(())
    }

    /// Take one connection waiting at one door, and leave the rest for the
    /// other workers.
    ///
    /// One rather than all of them, for the reason on the struct. The door is
    /// level triggered, so anything still queued behind this one makes the door
    /// ready again immediately and the next worker to look takes it.
    fn accept_ready(&mut self, token: u64) -> io::Result<()> {
        let Some(at) = self.doors.iter().position(|d| d.token() == token) else {
            return Ok(());
        };
        loop {
            match self.doors[at].accept() {
                Ok(stream) => {
                    let (peer, local) = stream.ends();
                    let conn = self.reactor.engine_mut().accept_from(
                        &peer,
                        &local,
                        stream.fd(),
                        stream.is_unix(),
                    );
                    // Registered before the socket is handed over, because
                    // after that the sink owns it and this is the last look.
                    self.poller.add(&stream, u64::from(conn))?;
                    self.reactor.engine_mut().sink_mut().attach(conn, stream);
                    return Ok(());
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => return Ok(()),
                // A signal arrived before anything was accepted, so this has
                // taken nothing yet and asks again.
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
    }

    /// Step out of the queue for the doors, or back into it.
    ///
    /// Run at the end of every turn, because both halves of the decision can
    /// have changed during it: this worker may have accepted or lost a
    /// connection, and so may everybody else.
    ///
    /// The rule is on [`Split`]: be in the queue while holding no more than the
    /// fewest anybody is holding. The one case that does not follow it is the
    /// last worker in the queue, which stays whatever its count says, because
    /// the alternative is a door nobody is listening at.
    ///
    /// # Errors
    ///
    /// Whatever the kernel says about changing the poller. A door that will not
    /// come out of a set is a worker that is about to take connections it should
    /// not, and it is a thing that does not happen, so it stops the server
    /// rather than being carried on from.
    fn share_the_doors(&mut self) -> io::Result<()> {
        let held = u32::try_from(self.reactor.engine().sink().held()).unwrap_or(u32::MAX);
        if held != self.held {
            self.held = held;
            self.split.publish(self.me, held);
        }
        let turn = self.split.turn();
        if turn == self.seen {
            return Ok(());
        }
        let wanted = held <= self.split.fewest();
        if wanted == self.watching {
            self.seen = turn;
            return Ok(());
        }
        if wanted {
            for door in self.doors {
                self.poller.add(door, door.token())?;
            }
            self.watching = true;
            self.split.step_in();
            self.seen = turn;
        } else if self.split.step_out() {
            for door in self.doors {
                self.poller.unwatch(door, door.token())?;
            }
            self.watching = false;
            self.seen = turn;
        }
        // Otherwise this is the only worker left in the queue and it stays in
        // it. `seen` is deliberately not moved on, so the next turn asks again
        // rather than waiting for somebody else to change something.
        Ok(())
    }

    /// Read everything waiting on one connection.
    fn read_conn(&mut self, conn: ConnId) {
        // A token for a connection that closed earlier in this same turn, which
        // the poller reported before it knew.
        if !self.reactor.engine().sink().is_open(conn) {
            return;
        }
        loop {
            let read = self
                .reactor
                .engine_mut()
                .sink_mut()
                .read(conn, &mut self.buf);
            match read {
                Some(0) => break,
                Some(n) => {
                    self.reactor.engine_mut().feed(conn, &self.buf[..n]);
                    // A short read means the socket is empty, so going round
                    // again would only buy an extra `EWOULDBLOCK`.
                    if n < self.buf.len() {
                        break;
                    }
                }
                None => {
                    self.reactor.engine_mut().hangup(conn);
                    break;
                }
            }
        }
    }

    /// One turn of the loop, for the tests that are about how connections are
    /// shared out.
    ///
    /// The body of [`Worker::run`] without the stop check and without the wait,
    /// which is asked for with no timeout here so that a turn with nothing on it
    /// returns rather than sleeping. A test that starts a server cannot say
    /// which thread woke first and so cannot assert anything about the split, so
    /// the way to pin the split down is to drive the workers a turn at a time
    /// and choose the order.
    #[cfg(all(test, unix, not(miri)))]
    fn turn_once(&mut self) -> io::Result<()> {
        self.poller.wait(&mut self.ready, Duration::ZERO)?;
        for at in 0..self.ready.len() {
            match self.ready[at] {
                LISTENER => self.accept_ready(LISTENER)?,
                UNIX_LISTENER => self.accept_ready(UNIX_LISTENER)?,
                token => self.read_conn(token as ConnId),
            }
        }
        pump(&mut self.reactor, &mut self.batch);
        self.bury_dead();
        self.forget_closed();
        self.share_the_doors()
    }

    /// Tell the engine about the sockets that failed under a write.
    fn bury_dead(&mut self) {
        while let Some(conn) = self.reactor.engine_mut().sink_mut().dead.pop() {
            self.reactor.engine_mut().hangup(conn);
        }
    }

    /// Take the connections that closed this turn out of the poller.
    ///
    /// On Linux and macOS closing the descriptor has already done it and this
    /// is bookkeeping for the fallback, which has no kernel to keep the list
    /// for it. An id that closed and was handed straight back out to a new
    /// socket in the same turn is still open and is left alone, because what is
    /// registered under it now is the new socket.
    fn forget_closed(&mut self) {
        while let Some(conn) = self.reactor.engine_mut().sink_mut().gone.pop() {
            if !self.reactor.engine().sink().is_open(conn) {
                self.poller.remove(u64::from(conn));
            }
        }
    }
}

/// Bind a socket file, clearing one left behind by a process that is gone.
///
/// A socket file outlives the process that made it, so a server that was killed
/// leaves a path that `bind` refuses. Removing it blind would let a second
/// server steal a running one's socket, so the stale case is told from the live
/// one by connecting: something that answers is somebody else's and the error
/// stands, and something that does not is a leftover and is removed.
#[cfg(unix)]
fn unix_door(path: &Path) -> io::Result<Door> {
    let listener = match UnixListener::bind(path) {
        Ok(l) => l,
        Err(e) if e.kind() == io::ErrorKind::AddrInUse => {
            if UnixStream::connect(path).is_ok() {
                return Err(io::Error::new(
                    io::ErrorKind::AddrInUse,
                    format!(
                        "{} is a live socket, something is already serving on it",
                        path.display()
                    ),
                ));
            }
            std::fs::remove_file(path)?;
            UnixListener::bind(path)?
        }
        Err(e) => return Err(e),
    };
    listener.set_nonblocking(true)?;
    Ok(Door::Unix(listener, path.to_path_buf()))
}

/// There are no socket files here, so asking for one is an error and not a
/// silent fallback to a port nobody asked for.
#[cfg(not(unix))]
fn unix_door(path: &Path) -> io::Result<Door> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        format!(
            "{}: this platform has no unix sockets, so serve on a port",
            path.display()
        ),
    ))
}

// Every test below stands a real server on a real socket and talks to it from
// another thread, which is not something Miri can be asked to do. Unix sockets
// it refuses outright, since it only knows `AF_INET` and `AF_INET6`. The port
// ones it will run, but an interpreter is slow enough that a client waiting on
// a reply hits the ten second read timeout and fails a test that has nothing
// wrong with it: two of them took over three and a half minutes each in CI
// before giving up. Raising the timeout would only trade a false failure for a
// job that runs out of its twenty five minutes.
//
// Nothing is lost by leaving these out. Miri is here for unsafe code, and this
// file has none: it is a listener, a poller and some bookkeeping over the
// engine, and the engine's own crates are interpreted in full. What these tests
// are really watching for is threads getting in each other's way, and that is
// what the `tsan` job runs them for, natively and at full speed with the
// thread sanitizer underneath.
#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    /// Sets the stop flag however the client thread ends, panic included, so a
    /// failing assertion is a failing test rather than a hanging one.
    struct Stopper(Arc<AtomicBool>);

    impl Drop for Stopper {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Relaxed);
        }
    }

    /// Run a server on a port the operating system picked, and talk to it from
    /// another thread.
    ///
    /// The client is the one that moves, because `run` is the calling thread's
    /// loop when there is one worker and the thread that waits for the rest
    /// when there are more.
    fn served(client: impl FnOnce(SocketAddr) + Send + 'static) {
        served_on(1, client);
    }

    /// The same, on however many threads.
    fn served_on(threads: usize, client: impl FnOnce(SocketAddr) + Send + 'static) {
        let mut server = Server::open(
            Some("127.0.0.1:0".parse().expect("a literal address")),
            None,
            threads,
        )
        .expect("a free port");
        let addr = server.local_addr().expect("bound");
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);

        let thread = std::thread::spawn(move || {
            let _stopper = Stopper(flag);
            client(addr);
        });

        server.run(&stop).expect("the listener stays up");
        if let Err(panic) = thread.join() {
            std::panic::resume_unwind(panic);
        }
    }

    /// The same harness, for a server that was given a file and a limit.
    ///
    /// The path goes away with the test whichever way it ends, because a leftover
    /// from a failed run is what makes the next run fail for a different reason.
    fn served_with_store(name: &str, client: impl FnOnce(SocketAddr) + Send + 'static) {
        struct Tmp(PathBuf);
        impl Drop for Tmp {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        let mut path = std::env::temp_dir();
        path.push(format!("yodb-test-{name}-{}.yo", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let path = Tmp(path);

        let mut server = Server::open(
            Some("127.0.0.1:0".parse().expect("a literal address")),
            None,
            1,
        )
        .expect("a free port");
        // Small enough that the test reaches it by writing rather than by
        // waiting, and large enough to be a limit this store can hold to: space
        // comes back a two megabyte segment at a time, so a limit of one or two
        // is a limit nothing can get under.
        server.set_maxmemory(8 * 1024 * 1024);
        server.use_store(Store::create(&path.0).expect("a fresh file"));

        let addr = server.local_addr().expect("bound");
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);

        let thread = std::thread::spawn(move || {
            let _stopper = Stopper(flag);
            client(addr);
        });

        server.run(&stop).expect("the listener stays up");
        if let Err(panic) = thread.join() {
            std::panic::resume_unwind(panic);
        }
    }

    /// A client with a timeout on it, so a reply that never comes fails the
    /// test instead of hanging it.
    fn connect(addr: SocketAddr) -> TcpStream {
        let s = TcpStream::connect(addr).expect("the server is listening");
        s.set_read_timeout(Some(Duration::from_secs(10)))
            .expect("a timeout the platform accepts");
        s
    }

    /// Read exactly `want` bytes, which is what a test knows and a client does
    /// not.
    fn read_exact(stream: &mut impl Read, want: usize) -> Vec<u8> {
        let mut got = vec![0; want];
        stream.read_exact(&mut got).expect("the reply arrives");
        got
    }

    #[test]
    fn a_client_gets_its_replies_over_a_real_socket() {
        served(|addr| {
            let mut client = connect(addr);

            client.write_all(b"*1\r\n$4\r\nPING\r\n").expect("sent");
            assert_eq!(read_exact(&mut client, 7), b"+PONG\r\n");

            client
                .write_all(b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$5\r\nvalue\r\n")
                .expect("sent");
            assert_eq!(read_exact(&mut client, 5), b"+OK\r\n");

            client
                .write_all(b"*2\r\n$3\r\nGET\r\n$1\r\nk\r\n")
                .expect("sent");
            assert_eq!(read_exact(&mut client, 11), b"$5\r\nvalue\r\n");
        });
    }

    #[test]
    fn a_pipeline_comes_back_in_one_piece_and_in_order() {
        served(|addr| {
            let mut client = connect(addr);
            let mut sent = Vec::new();
            for _ in 0..64 {
                sent.extend_from_slice(b"*2\r\n$4\r\nINCR\r\n$1\r\nn\r\n");
            }
            client.write_all(&sent).expect("sent");

            let mut want = Vec::new();
            for i in 1..=64 {
                want.extend_from_slice(format!(":{i}\r\n").as_bytes());
            }
            assert_eq!(read_exact(&mut client, want.len()), want);
        });
    }

    /// Two clients, two sessions, one server. The `SELECT` on one of them is
    /// not the other one's business.
    #[test]
    fn two_clients_have_their_own_database_and_share_the_store() {
        served(|addr| {
            let mut a = connect(addr);
            let mut b = connect(addr);

            a.write_all(b"*2\r\n$6\r\nSELECT\r\n$1\r\n3\r\n")
                .expect("sent");
            assert_eq!(read_exact(&mut a, 5), b"+OK\r\n");

            a.write_all(b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\na\r\n")
                .expect("sent");
            assert_eq!(read_exact(&mut a, 5), b"+OK\r\n");

            // Database zero has never been written, so this is a miss and not
            // what `a` wrote into database three.
            b.write_all(b"*2\r\n$3\r\nGET\r\n$1\r\nk\r\n")
                .expect("sent");
            assert_eq!(read_exact(&mut b, 5), b"$-1\r\n");

            b.write_all(b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\nb\r\n")
                .expect("sent");
            assert_eq!(read_exact(&mut b, 5), b"+OK\r\n");

            a.write_all(b"*2\r\n$3\r\nGET\r\n$1\r\nk\r\n")
                .expect("sent");
            assert_eq!(read_exact(&mut a, 7), b"$1\r\na\r\n");
        });
    }

    #[test]
    fn quit_is_answered_and_then_the_socket_closes() {
        served(|addr| {
            let mut client = connect(addr);
            client.write_all(b"*1\r\n$4\r\nQUIT\r\n").expect("sent");

            let mut rest = Vec::new();
            client
                .read_to_end(&mut rest)
                .expect("the server closes rather than leaving it open");
            assert_eq!(rest, b"+OK\r\n");
        });
    }

    /// A command split across two packets, which is the case a framing bug
    /// hides in and which a fast local client will not produce on its own.
    #[test]
    fn a_command_arriving_in_two_packets_is_one_command() {
        served(|addr| {
            let mut client = connect(addr);
            client
                .write_all(b"*3\r\n$3\r\nSET\r\n$3\r\nkey\r\n$5\r\nv")
                .expect("sent");
            std::thread::sleep(Duration::from_millis(20));
            client.write_all(b"alue\r\n").expect("sent");
            assert_eq!(read_exact(&mut client, 5), b"+OK\r\n");

            client
                .write_all(b"*2\r\n$3\r\nGET\r\n$3\r\nkey\r\n")
                .expect("sent");
            assert_eq!(read_exact(&mut client, 11), b"$5\r\nvalue\r\n");
        });
    }

    /// A server with a file under it, doing the thing the file is for.
    ///
    /// Thirty two megabytes of values into an eight megabyte server, over a real
    /// socket. Redis answers that by deleting most of them and this answers it by
    /// moving them into the file, so the test is that every key is still there
    /// and that the file is not empty. It is the only end to end check of `14`
    /// section 4.1 there is, because every layer under it can be tested with a
    /// store made of a vector and none of that proves a `.yo` file was ever
    /// opened.
    ///
    /// Four thousand keys of eight kilobytes and not forty thousand of one,
    /// because what goes to the file is the value and what stays behind is the
    /// key, its index entry and the record that says where the value went. That
    /// floor is per key and it does not move, so a key count whose floor is
    /// already over the limit is a server that cannot get under it however much
    /// it demotes, and the honest answer to that is the OOM it gives.
    #[test]
    fn a_server_with_a_file_moves_values_into_it_rather_than_losing_them() {
        const KEYS: usize = 4_000;
        const LEN: usize = 8 * 1024;
        const BATCH: usize = 100;

        served_with_store("demote", |addr| {
            let mut client = connect(addr);
            let value = vec![b'v'; LEN];

            // Pipelined in batches rather than one at a time, because four
            // thousand round trips is four thousand times the socket latency
            // and this test is not about the socket.
            let mut at = 0;
            while at < KEYS {
                let upto = (at + BATCH).min(KEYS);
                let mut sent = Vec::new();
                for i in at..upto {
                    sent.extend_from_slice(&cmd(&[b"SET", format!("k{i}").as_bytes(), &value]));
                }
                client.write_all(&sent).expect("sent");
                assert_eq!(
                    read_exact(&mut client, 5 * (upto - at)),
                    b"+OK\r\n".repeat(upto - at),
                    "a write was refused, so the file did not make room for it"
                );
                at = upto;
            }

            // Every one of them, not a sample, because the failure this is
            // looking for is a handful of keys that quietly went away.
            let one = {
                let mut w = format!("${LEN}\r\n").into_bytes();
                w.extend_from_slice(&value);
                w.extend_from_slice(b"\r\n");
                w
            };
            let mut at = 0;
            while at < KEYS {
                let upto = (at + BATCH).min(KEYS);
                let mut sent = Vec::new();
                for i in at..upto {
                    sent.extend_from_slice(&cmd(&[b"GET", format!("k{i}").as_bytes()]));
                }
                client.write_all(&sent).expect("sent");
                assert_eq!(
                    read_exact(&mut client, one.len() * (upto - at)),
                    one.repeat(upto - at),
                    "a key between k{at} and k{upto} did not come back"
                );
                at = upto;
            }

            let memory = info(&mut client, "memory");
            let held: u64 = field(&memory, "yo_store_bytes").parse().expect("a number");
            assert!(held > 0, "nothing reached the file\n{memory}");
            assert_eq!(field(&memory, "yo_memory_regime"), "migrate", "{memory}");

            // The count as well as the reads, because a key that answers is one
            // key and this says none of the other ones were dropped on the way.
            let stats = info(&mut client, "stats");
            assert_eq!(field(&stats, "evicted_keys"), "0", "keys were thrown away");

            // G9 in miniature. The working set here is four times memory rather
            // than the ten the gate asks for, so this is not the gate, but the
            // shape of the number is the same: a point read off the file should
            // cost about one fault and not several. A read that faults twice is
            // a chain being walked or a value being promoted and demoted again
            // in the same pass, and both of those show up here first.
            let count = |name| field(&stats, name).parse::<u64>().expect("a number");
            let (demoted, faults) = (count("yo_cold_demoted"), count("yo_cold_faults"));
            assert!(demoted > 0, "nothing was demoted\n{stats}");
            assert!(
                faults <= KEYS as u64 * 105 / 100,
                "{faults} faults for {KEYS} reads\n{stats}"
            );
            client.write_all(&cmd(&[b"DBSIZE"])).expect("sent");
            assert_eq!(
                read_exact(&mut client, format!(":{KEYS}\r\n").len()),
                format!(":{KEYS}\r\n").into_bytes()
            );
        });
    }

    /// One command, encoded the way a client sends it.
    fn cmd(parts: &[&[u8]]) -> Vec<u8> {
        let mut out = format!("*{}\r\n", parts.len()).into_bytes();
        for p in parts {
            out.extend_from_slice(format!("${}\r\n", p.len()).as_bytes());
            out.extend_from_slice(p);
            out.extend_from_slice(b"\r\n");
        }
        out
    }

    /// `INFO section`, read back as the text of the bulk string.
    fn info(stream: &mut TcpStream, section: &str) -> String {
        stream
            .write_all(&cmd(&[b"INFO", section.as_bytes()]))
            .expect("sent");
        let mut header = Vec::new();
        loop {
            let mut b = [0u8; 1];
            stream.read_exact(&mut b).expect("the reply arrives");
            if b[0] == b'\n' {
                break;
            }
            header.push(b[0]);
        }
        let len: usize = String::from_utf8_lossy(&header[1..header.len() - 1])
            .parse()
            .expect("a bulk length");
        let body = read_exact(stream, len + 2);
        String::from_utf8_lossy(&body[..len]).into_owned()
    }

    /// One `name:value` line out of an `INFO` section.
    fn field<'a>(info: &'a str, name: &str) -> &'a str {
        info.lines()
            .find_map(|l| l.strip_prefix(name)?.strip_prefix(':'))
            .unwrap_or_else(|| panic!("no {name} in\n{info}"))
            .trim_end()
    }

    /// Several clients on a server with several threads, all counting into one
    /// key.
    ///
    /// The count is the point. Every `INCR` is a read and a write of the same
    /// value from whichever thread accepted the client that sent it, so a
    /// number that comes out short is two threads that were in the same stripe
    /// at the same time and a number that comes out right is the stripe lock
    /// doing its job. Eight clients over four threads, so at least two clients
    /// share a thread and at least two threads share the key.
    #[test]
    fn several_threads_count_into_one_key_without_losing_any() {
        const CLIENTS: usize = 8;
        const EACH: usize = 500;

        served_on(4, |addr| {
            let mut sent = Vec::new();
            for _ in 0..EACH {
                sent.extend_from_slice(&cmd(&[b"INCR", b"n"]));
            }
            let sent = Arc::new(sent);

            let mut clients = Vec::new();
            for _ in 0..CLIENTS {
                let sent = Arc::clone(&sent);
                clients.push(std::thread::spawn(move || {
                    let mut client = connect(addr);
                    client.write_all(&sent).expect("sent");
                    // Read the whole pipeline back, so the client is not gone
                    // before the server has answered it.
                    let mut got = 0;
                    let mut buf = [0u8; 4096];
                    while got < EACH {
                        let n = client.read(&mut buf).expect("a reply");
                        assert!(n > 0, "the server closed the connection");
                        got += buf[..n].iter().filter(|b| **b == b'\n').count();
                    }
                }));
            }
            for c in clients {
                c.join().expect("the client thread");
            }

            let mut last = connect(addr);
            last.write_all(&cmd(&[b"GET", b"n"])).expect("sent");
            let want = (CLIENTS * EACH).to_string();
            let reply = format!("${}\r\n{want}\r\n", want.len());
            assert_eq!(
                read_exact(&mut last, reply.len()),
                reply.as_bytes(),
                "every INCR should be in there"
            );
        });
    }

    /// A push on one thread answers a client blocked on another.
    ///
    /// Eight clients on four threads, so several of them are certainly not on
    /// the thread the pusher landed on. What could go wrong is not the answer
    /// but the waiting: nothing arrives on a blocked client's thread to say the
    /// key it wants now has something in it, so a thread that slept its full
    /// idle wait would leave its client hanging. The read timeout the harness
    /// puts on a client is what turns that into a failure instead of a test that
    /// never finishes.
    #[test]
    fn a_push_on_one_thread_answers_a_client_blocked_on_another() {
        const CLIENTS: usize = 8;

        served_on(4, |addr| {
            let mut blocked = Vec::new();
            for i in 0..CLIENTS {
                let key = format!("q{i}");
                let mut client = connect(addr);
                client
                    .write_all(&cmd(&[b"BLPOP", key.as_bytes(), b"0"]))
                    .expect("sent");
                blocked.push(client);
            }

            // One connection pushing to all eight keys, so whichever thread it
            // is on is the only thread that hears about any of them.
            let mut pusher = connect(addr);
            for i in 0..CLIENTS {
                let key = format!("q{i}");
                pusher
                    .write_all(&cmd(&[b"RPUSH", key.as_bytes(), b"v"]))
                    .expect("sent");
            }
            let mut ack = [0u8; 64];
            let mut lines = 0;
            while lines < CLIENTS {
                let n = pusher.read(&mut ack).expect("a reply to every push");
                lines += ack[..n].iter().filter(|b| **b == b'\n').count();
            }

            for (i, client) in blocked.iter_mut().enumerate() {
                let key = format!("q{i}");
                let want = format!("*2\r\n${}\r\n{key}\r\n$1\r\nv\r\n", key.len());
                assert_eq!(
                    read_exact(client, want.len()),
                    want.as_bytes(),
                    "the client blocked on {key} should have been woken"
                );
            }
        });
    }

    /// A `SHUTDOWN` heard by one thread stops all of them.
    ///
    /// The flag is on the shared server and every worker reads it at the top of
    /// its turn, so this is a test that the loop asks and not a test that the
    /// command works. Without it a `SHUTDOWN` would stop the thread that ran it
    /// and leave the others serving, which is the worst of both answers.
    #[test]
    fn shutdown_on_one_thread_stops_the_others() {
        let mut server = Server::open(
            Some("127.0.0.1:0".parse().expect("a literal address")),
            None,
            4,
        )
        .expect("a free port");
        let addr = server.local_addr().expect("bound");
        // Never set, so the only way out of `run` is the command.
        let stop = AtomicBool::new(false);

        let thread = std::thread::spawn(move || {
            let mut client = connect(addr);
            client.write_all(&cmd(&[b"PING"])).expect("sent");
            assert_eq!(read_exact(&mut client, 7), b"+PONG\r\n");
            client.write_all(&cmd(&[b"SHUTDOWN"])).expect("sent");
        });

        server.run(&stop).expect("the listener stays up");
        thread.join().expect("the client thread");
    }

    /// A client that goes away without saying `QUIT`, which is what every
    /// benchmark client does at the end of a run.
    #[test]
    fn a_client_that_drops_frees_its_slot() {
        served(|addr| {
            for _ in 0..8 {
                let mut client = connect(addr);
                client
                    .write_all(b"*2\r\n$4\r\nINCR\r\n$1\r\nn\r\n")
                    .expect("sent");
                let mut reply = [0; 16];
                let n = client.read(&mut reply).expect("a reply");
                assert!(reply[..n].starts_with(b":"), "{:?}", &reply[..n]);
            }

            // Nine clients over however many slots, and the counter has seen
            // all eight of the ones that went away, so the slots came back
            // rather than the server running out of them.
            let mut last = connect(addr);
            last.write_all(b"*2\r\n$3\r\nGET\r\n$1\r\nn\r\n")
                .expect("sent");
            assert_eq!(read_exact(&mut last, 7), b"$1\r\n8\r\n");
        });
    }

    #[cfg(unix)]
    mod unix {
        use super::*;
        use std::os::unix::net::UnixStream;

        /// A path in the temporary directory that no other test is using.
        ///
        /// Named after the test rather than after a random number, because a
        /// leftover from a crashed run should be recognisable and should be
        /// reused rather than accumulating.
        fn socket_path(name: &str) -> PathBuf {
            let mut p = std::env::temp_dir();
            p.push(format!("yodb-test-{name}-{}.sock", std::process::id()));
            let _ = std::fs::remove_file(&p);
            p
        }

        /// The same harness as `served`, over a socket file.
        fn served_unix(name: &str, client: impl FnOnce(PathBuf) + Send + 'static) {
            let path = socket_path(name);
            let mut server = Server::open(None, Some(path.clone()), 1).expect("a fresh path");
            let stop = Arc::new(AtomicBool::new(false));
            let flag = Arc::clone(&stop);
            let theirs = path.clone();

            let thread = std::thread::spawn(move || {
                let _stopper = Stopper(flag);
                client(theirs);
            });

            server.run(&stop).expect("the listener stays up");
            if let Err(panic) = thread.join() {
                std::panic::resume_unwind(panic);
            }
        }

        #[test]
        fn a_client_gets_its_replies_over_a_socket_file() {
            served_unix("replies", |path| {
                let mut client = UnixStream::connect(&path).expect("the server is listening");
                client
                    .set_read_timeout(Some(Duration::from_secs(10)))
                    .expect("a timeout the platform accepts");

                client.write_all(b"*1\r\n$4\r\nPING\r\n").expect("sent");
                assert_eq!(read_exact(&mut client, 7), b"+PONG\r\n");

                client
                    .write_all(b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$5\r\nvalue\r\n")
                    .expect("sent");
                assert_eq!(read_exact(&mut client, 5), b"+OK\r\n");

                client
                    .write_all(b"*2\r\n$3\r\nGET\r\n$1\r\nk\r\n")
                    .expect("sent");
                assert_eq!(read_exact(&mut client, 11), b"$5\r\nvalue\r\n");
            });
        }

        /// Both doors, one keyspace. A client on the port and a client on the
        /// socket file are talking to the same engine, which is the thing that
        /// would be easy to get wrong by running two of anything.
        #[test]
        fn the_port_and_the_socket_file_are_the_same_server() {
            let path = socket_path("both");
            let mut server = Server::open(
                Some("127.0.0.1:0".parse().expect("a literal address")),
                Some(path.clone()),
                1,
            )
            .expect("a free port and a fresh path");
            let addr = server.local_addr().expect("bound");
            let stop = Arc::new(AtomicBool::new(false));
            let flag = Arc::clone(&stop);

            let thread = std::thread::spawn(move || {
                let _stopper = Stopper(flag);
                let mut over_tcp = connect(addr);
                over_tcp
                    .write_all(b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$4\r\nboth\r\n")
                    .expect("sent");
                assert_eq!(read_exact(&mut over_tcp, 5), b"+OK\r\n");

                let mut over_file = UnixStream::connect(&path).expect("listening there too");
                over_file
                    .set_read_timeout(Some(Duration::from_secs(10)))
                    .expect("a timeout the platform accepts");
                over_file
                    .write_all(b"*2\r\n$3\r\nGET\r\n$1\r\nk\r\n")
                    .expect("sent");
                assert_eq!(read_exact(&mut over_file, 10), b"$4\r\nboth\r\n");
            });

            server.run(&stop).expect("the listener stays up");
            if let Err(panic) = thread.join() {
                std::panic::resume_unwind(panic);
            }
        }

        /// A socket file left behind by a process that is gone is not a reason
        /// to refuse to start.
        #[test]
        fn a_leftover_socket_file_is_cleared() {
            let path = socket_path("leftover");
            {
                let _first = Server::open(None, Some(path.clone()), 1).expect("a fresh path");
            }
            // The first server is dropped, which unlinks it, so put a file
            // back by hand. Dropping a UnixListener closes the descriptor and
            // leaves the path, which is exactly the state a killed process
            // leaves behind.
            drop(std::os::unix::net::UnixListener::bind(&path).expect("bound"));
            assert!(path.exists(), "the leftover is there");

            let second = Server::open(None, Some(path.clone()), 1);
            assert!(second.is_ok(), "{:?}", second.err());
        }

        /// A socket file with a server on it is somebody else's.
        #[test]
        fn a_live_socket_file_is_not_stolen() {
            let path = socket_path("live");
            let _first = Server::open(None, Some(path.clone()), 1).expect("a fresh path");
            let e = match Server::open(None, Some(path.clone()), 1) {
                Ok(_) => panic!("something is already serving there"),
                Err(e) => e,
            };
            assert_eq!(e.kind(), io::ErrorKind::AddrInUse, "{e}");
        }

        /// The path goes away with the server that made it.
        #[test]
        fn the_socket_file_is_removed_on_the_way_out() {
            let path = socket_path("cleanup");
            {
                let _server = Server::open(None, Some(path.clone()), 1).expect("a fresh path");
                assert!(path.exists(), "it is there while the server is");
            }
            assert!(!path.exists(), "and gone once the server is dropped");
        }

        #[test]
        fn a_server_with_no_door_is_refused() {
            let e = match Server::open(None, None, 1) {
                Ok(_) => panic!("nothing to listen on"),
                Err(e) => e,
            };
            assert_eq!(e.kind(), io::ErrorKind::InvalidInput);
        }

        /// A burst of connections is shared out, and this is the test the
        /// change to `accept_ready` exists for.
        ///
        /// Clients do not arrive one at a time. A benchmark opens all of them
        /// at once, every worker is idle at that moment, and a worker that
        /// drained the door would take the lot and keep them for as long as
        /// they stayed open. Four workers are given a door with sixteen
        /// connections queued at it and asked in turn, and the count each ends
        /// up with is the whole assertion. Before, it was sixteen, zero, zero,
        /// zero.
        ///
        /// The workers are driven by hand rather than by starting a server,
        /// because the point is what one call to `accept_ready` takes, and a
        /// running server would make that a race and this test a coin toss.
        #[test]
        fn a_burst_of_connections_is_shared_out_between_the_workers() {
            let path = socket_path("accept_share");
            let server = Server::open(None, Some(path.clone()), 4).expect("a fresh path");
            let mut workers = crowd(&server, 4);

            // Held open for the length of the test, because a connection the
            // client dropped is one the worker would bury before it is counted.
            let _clients: Vec<UnixStream> = (0..16)
                .map(|_| UnixStream::connect(&path).expect("the door is open"))
                .collect();

            for _ in 0..16 {
                for worker in &mut workers {
                    worker
                        .accept_ready(UNIX_LISTENER)
                        .expect("the door is open");
                }
            }

            assert_eq!(spread(&workers), vec![4, 4, 4, 4]);
        }

        /// A worker that is ahead of the others is not asked again until they
        /// have caught up, however many chances it gets.
        ///
        /// Taking one connection per ready event shares a burst out only if the
        /// workers get their chances in turn, and in the real thing they do not.
        /// A worker that just accepted is awake and running and comes back to
        /// its poller first, while the ones that were waiting are still on their
        /// way, so the same thread wins the door over and over. At sixteen
        /// threads that put 215 of 771 connections on one thread and 22 on
        /// another.
        ///
        /// So this driver is as unfair as a driver can be: worker 0 gets sixteen
        /// turns in a row with four clients queued at the door, before worker 1
        /// gets a single one. That is the worst case the race can produce, and
        /// the answer is still one each, because a worker that has taken one
        /// steps out of the queue for the door and its next fifteen turns find
        /// nothing ready.
        #[test]
        fn a_worker_that_is_ahead_stops_being_offered_connections() {
            let path = socket_path("accept_ahead");
            let server = Server::open(None, Some(path.clone()), 4).expect("a fresh path");
            let mut workers = crowd(&server, 4);
            let _clients: Vec<UnixStream> = (0..4)
                .map(|_| UnixStream::connect(&path).expect("the door is open"))
                .collect();

            for worker in &mut workers {
                for _ in 0..16 {
                    worker.turn_once().expect("the door is open");
                }
            }

            assert_eq!(spread(&workers), vec![1, 1, 1, 1]);
        }

        /// The last worker in the queue for the doors stays in it however far
        /// ahead it is.
        ///
        /// A worker steps out because somebody else will take the next
        /// connection, and there is a window where that is not true: a worker
        /// that has just dropped below the rest has published its count but has
        /// not registered the doors again yet. If the one still at the door were
        /// allowed to leave during it, nobody would be listening, and a level
        /// triggered door does not wake anybody to say so. The client just
        /// waits.
        ///
        /// The window is a race in the running server, so it is made by hand
        /// here: worker 1 gets ahead and leaves, then worker 0 gets further
        /// ahead than worker 1 and asks to leave too, and is refused.
        #[test]
        fn the_last_worker_in_the_queue_does_not_step_out() {
            let path = socket_path("accept_last");
            let server = Server::open(None, Some(path.clone()), 2).expect("a fresh path");
            let mut workers = crowd(&server, 2);
            let _clients: Vec<UnixStream> = (0..6)
                .map(|_| UnixStream::connect(&path).expect("the door is open"))
                .collect();

            // Worker 1 takes two while worker 0 has none, so it is ahead and
            // leaves the queue. Two workers, so worker 0 is now the only one in
            // it.
            for _ in 0..2 {
                workers[1].accept_ready(UNIX_LISTENER).expect("the door");
            }
            workers[1]
                .share_the_doors()
                .expect("the poller lets it out");
            assert!(!workers[1].watching, "it is ahead and has stepped out");

            // Worker 0 takes three, which puts it ahead of worker 1. It asks to
            // leave and does not, because there would be nobody left.
            for _ in 0..3 {
                workers[0].accept_ready(UNIX_LISTENER).expect("the door");
            }
            workers[0].share_the_doors().expect("nothing to change");
            assert_eq!(spread(&workers), vec![3, 2]);
            assert!(workers[0].watching, "somebody has to be at the door");

            // And it is a real registration and not just a flag: the sixth
            // client is still accepted.
            workers[0].turn_once().expect("the door is open");
            assert_eq!(spread(&workers), vec![4, 2]);
        }

        /// A worker whose clients went away comes back into the queue.
        ///
        /// What is published is a live count and not a record of whose turn it
        /// was, so a thread that emptied out is the one with the fewest and
        /// starts taking again while the ones that are still busy stay out. A
        /// scheme that handed connections round in order has no way to see that,
        /// and connections do not close at the same rate they arrive.
        #[test]
        fn a_worker_that_loses_its_clients_starts_taking_again() {
            let path = socket_path("accept_back");
            let server = Server::open(None, Some(path.clone()), 2).expect("a fresh path");
            let mut workers = crowd(&server, 2);
            let early: Vec<UnixStream> = (0..3)
                .map(|_| UnixStream::connect(&path).expect("the door is open"))
                .collect();
            let _kept = UnixStream::connect(&path).expect("the door is open");

            // Worker 0 is made to take the first three, which is not what the
            // rule would have done and is the point: the test needs one worker
            // holding more than the other. Worker 1 then takes the fourth.
            for _ in 0..3 {
                workers[0].accept_ready(UNIX_LISTENER).expect("the door");
            }
            workers[0]
                .share_the_doors()
                .expect("the poller lets it out");
            workers[1].turn_once().expect("the door is open");
            assert_eq!(spread(&workers), vec![3, 1]);
            assert!(!workers[0].watching, "it is ahead");
            assert!(workers[1].watching, "it has the fewest");

            // Worker 0's three clients hang up. It reads the ends of them, the
            // engine closes them, and it is the one with the fewest now.
            drop(early);
            for _ in 0..4 {
                workers[0].turn_once().expect("the door is open");
            }
            assert_eq!(spread(&workers), vec![0, 1]);
            assert!(workers[0].watching, "it has the fewest now");

            // Worker 1 finds that out on its next turn and steps out, with
            // nothing waiting at the door yet for either of them.
            workers[1].turn_once().expect("the door is open");
            assert!(!workers[1].watching, "it is the one ahead now");

            // So the next connection goes to worker 0 even though worker 1 is
            // the one that is asked first.
            let _late = UnixStream::connect(&path).expect("the door is open");
            workers[1].turn_once().expect("the door is open");
            assert_eq!(spread(&workers), vec![0, 1], "worker 1 is out of the queue");
            workers[0].turn_once().expect("the door is open");
            assert_eq!(spread(&workers), vec![1, 1], "worker 0 took it");
        }

        /// The workers `run` would build, not started, so that a test can drive
        /// them a turn at a time and choose the order.
        ///
        /// The server is the caller's because the workers borrow its doors, and
        /// a function that returned both would be handing back a value that
        /// borrows itself.
        fn crowd(server: &Server, threads: usize) -> Vec<Worker<'_>> {
            let split = Arc::new(Split::new(threads));
            (0..threads)
                .map(|me| {
                    Worker::new(
                        &server.doors,
                        Wire::over(Arc::clone(&server.shared), Net::default()),
                        Arc::clone(&split),
                        me,
                    )
                    .expect("a poller")
                })
                .collect()
        }

        /// How many connections each worker ended up holding.
        fn spread(workers: &[Worker<'_>]) -> Vec<usize> {
            workers
                .iter()
                .map(|w| w.reactor.engine().sink().held())
                .collect()
        }
    }
}
