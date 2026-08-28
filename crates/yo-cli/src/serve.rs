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

use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use yo_reactor::Reactor;
use yo_resp::engine::{Cmd, ConnId, Sink, Wire, pump};

use crate::poll::Poller;

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
const OWED_WAIT: Duration = Duration::from_millis(1);

/// The token the listener is registered under.
///
/// Connections are registered under their own id, and ids come from a free list
/// that starts at zero, so the top of the range is the one value that is never
/// a connection.
const LISTENER: u64 = u64::MAX;

/// The sockets, indexed by the connection id the engine handed out.
#[derive(Default)]
struct Net {
    streams: Vec<Option<TcpStream>>,
    /// Connections whose socket failed, to be told to the engine after the
    /// batch rather than in the middle of it.
    dead: Vec<ConnId>,
    /// Connections whose socket has just been dropped, to be taken out of the
    /// poller after the batch for the same reason.
    gone: Vec<ConnId>,
}

impl Net {
    /// Put a freshly accepted socket at the id the engine gave it.
    fn attach(&mut self, conn: ConnId, stream: TcpStream) {
        if self.streams.len() <= conn as usize {
            self.streams.resize_with(conn as usize + 1, || None);
        }
        self.streams[conn as usize] = Some(stream);
    }

    /// Whether this id currently has a socket.
    fn is_open(&self, conn: ConnId) -> bool {
        self.streams.get(conn as usize).is_some_and(Option::is_some)
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
        if let Some(slot) = self.streams.get_mut(conn as usize) {
            *slot = None;
        }
        self.gone.push(conn);
    }
}

/// A bound listener with the engine behind it.
pub struct Server {
    listener: TcpListener,
    reactor: Reactor<Wire<Net>>,
    poller: Poller,
    /// The batch the reactor runs, kept across turns so no turn allocates.
    batch: Vec<Cmd>,
    /// The tokens the poller said were ready, kept for the same reason.
    ready: Vec<u64>,
    buf: Vec<u8>,
}

impl Server {
    /// Bind, and hand back a server that has accepted nothing yet.
    ///
    /// # Errors
    ///
    /// Whatever `bind` says, which is usually the port being taken.
    pub fn bind(addr: SocketAddr) -> io::Result<Server> {
        let listener = TcpListener::bind(addr)?;
        listener.set_nonblocking(true)?;
        let mut poller = Poller::new()?;
        poller.add(&listener, LISTENER)?;
        Ok(Server {
            listener,
            reactor: Reactor::inline(Wire::new(Net::default())),
            poller,
            batch: Vec::with_capacity(64),
            ready: Vec::with_capacity(64),
            buf: vec![0; READ_CHUNK],
        })
    }

    /// Where it actually landed, which is the only way to find out when the
    /// port asked for was zero.
    ///
    /// # Errors
    ///
    /// Whatever the socket says.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.listener.local_addr()
    }

    /// Turn the loop until `stop` is set.
    ///
    /// # Errors
    ///
    /// Only an accept failing for a reason that is not "nothing waiting". A
    /// connection failing is that connection's problem and closes it.
    pub fn run(&mut self, stop: &AtomicBool) -> io::Result<()> {
        let mut idle = 0u32;
        while !stop.load(Ordering::Relaxed) {
            let wait = if idle <= SPIN_TURNS {
                Duration::ZERO
            } else if self.reactor.engine().owed() > 0 {
                OWED_WAIT
            } else {
                IDLE_WAIT
            };
            self.poller.wait(&mut self.ready, wait)?;

            let mut worked = false;
            for at in 0..self.ready.len() {
                if self.ready[at] == LISTENER {
                    self.accept_ready()?;
                } else {
                    self.read_conn(self.ready[at] as ConnId);
                }
                worked = true;
            }

            if pump(&mut self.reactor, &mut self.batch) > 0 {
                worked = true;
            }
            self.bury_dead();
            self.forget_closed();

            // Housekeeping goes after the batch and not before it, so a turn
            // that has commands waiting answers them first. It does nothing at
            // all unless a segment has gone half dead, and asking costs a
            // comparison, so this is not a reason to stay awake.
            self.reactor.engine_mut().maintain();

            if worked {
                idle = 0;
            } else {
                idle = idle.saturating_add(1);
            }
        }
        Ok(())
    }

    /// Take every connection that is waiting.
    fn accept_ready(&mut self) -> io::Result<()> {
        loop {
            match self.listener.accept() {
                Ok((stream, _)) => {
                    stream.set_nonblocking(true)?;
                    // Redis sets this and so does everything that talks to it.
                    // Without it a reply waits for the next packet's worth of
                    // data that a request response client is never going to
                    // send, which turns a 50 microsecond round trip into a 40
                    // millisecond one.
                    let _ = stream.set_nodelay(true);
                    let conn = self.reactor.engine_mut().accept();
                    // Registered before the socket is handed over, because
                    // after that the sink owns it and this is the last look.
                    self.poller.add(&stream, u64::from(conn))?;
                    self.reactor.engine_mut().sink_mut().attach(conn, stream);
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => return Ok(()),
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
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

#[cfg(test)]
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
    /// The client is the one that moves, because the engine and everything
    /// under it belong to the thread that made them: one shard, one thread, no
    /// locks (Y1). That is the design and not a limitation of the test.
    fn served(client: impl FnOnce(SocketAddr) + Send + 'static) {
        let mut server =
            Server::bind("127.0.0.1:0".parse().expect("a literal address")).expect("a free port");
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
}
