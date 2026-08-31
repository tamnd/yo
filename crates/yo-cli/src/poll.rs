//! Which connections have something on them, asked once instead of guessed
//! fifty times.
//!
//! The serve loop used to walk every open connection every turn and try to read
//! from each one. That is one syscall per idle connection per turn, and a
//! profile of the gate run said so: 2.26 `recvfrom` per command, most of them
//! returning `EWOULDBLOCK`, and no waiting call anywhere. A benchmark with 50
//! connections and one request in flight on each has about half of them ready
//! at any moment, so about half of the reads were the kernel being asked a
//! question it had already answered.
//!
//! This asks the kernel once per turn instead. `epoll` on Linux, `kqueue` on
//! macOS, `WSAPoll` on Windows, and on anything else the old scan, which is
//! still correct and is nobody's platform today.
//!
//! Windows had the scan until the tests below said it could not. They assert
//! that a quiet connection is not reported, which is the whole point of the
//! module, and the scan reports everything it has ever been handed. Two tests
//! failing on one platform is not a Windows problem to leave for later, it is
//! this module keeping a promise on two platforms out of three.
//!
//! # Level triggered on purpose
//!
//! Every backend registers for readability only and every one of them is level
//! triggered, so a connection with bytes still in it is reported again next
//! turn. That is the same contract the scan had, which means the loop above did
//! not have to learn anything new: read until the socket says it is empty, and
//! come back.
//!
//! # Writes are not registered
//!
//! A reply that does not fit in the socket stays in the engine and is offered
//! again on a later turn, and the loop keeps its wait short while anything is
//! owed rather than registering for writability. A client that reads slowly is
//! retried on a timer instead of on an event, which is the same thing the scan
//! did and is worth changing when there is a workload that shows it.
//!
//! # What this is not
//!
//! The ring. `04` section 7 puts the network on io_uring and this is not a step
//! towards that, it is a better multiplexer for the portable loop that has to
//! keep existing for the machines that are not Linux.

use std::io;
use std::time::Duration;

/// Something with a handle the kernel will accept, which on Unix is a file
/// descriptor.
#[cfg(unix)]
pub trait Source: std::os::fd::AsRawFd {}
#[cfg(unix)]
impl<T: std::os::fd::AsRawFd> Source for T {}

/// Something with a handle the kernel will accept, which on Windows is a
/// socket and not a file handle, because the two are different kinds of number
/// there and only one of them can be polled.
#[cfg(windows)]
pub trait Source: std::os::windows::io::AsRawSocket {}
#[cfg(windows)]
impl<T: std::os::windows::io::AsRawSocket> Source for T {}

/// Something with a handle the kernel will accept.
#[cfg(not(any(unix, windows)))]
pub trait Source {}
#[cfg(not(any(unix, windows)))]
impl<T> Source for T {}

/// Asks the kernel which registered sources are readable.
pub struct Poller {
    inner: Inner,
}

impl Poller {
    /// A poller with nothing registered.
    ///
    /// # Errors
    ///
    /// Whatever the kernel says when it will not give us the object, which in
    /// practice is the process being out of descriptors.
    pub fn new() -> io::Result<Poller> {
        Ok(Poller {
            inner: Inner::new()?,
        })
    }

    /// Watch a source for readability, and report it as `token`.
    ///
    /// # Errors
    ///
    /// Whatever the kernel says. A descriptor that cannot be registered is a
    /// connection that cannot be served, so this is not swallowed.
    pub fn add(&mut self, src: &impl Source, token: u64) -> io::Result<()> {
        self.inner.add(src, token)
    }

    /// Stop reporting `token`.
    ///
    /// Called after the socket behind it has already been closed, because
    /// closing a descriptor takes it out of an `epoll` set and a `kqueue` on
    /// its own. The backends that keep the registration list themselves, which
    /// is Windows and the scan, are the ones with work to do here.
    pub fn remove(&mut self, token: u64) {
        self.inner.remove(token);
    }

    /// Fill `out` with the tokens that are ready, waiting up to `timeout`.
    ///
    /// A zero timeout asks and returns, which is what the loop does while it is
    /// busy. `out` is the caller's buffer and is cleared here, so a turn that
    /// finds nothing costs no allocation.
    ///
    /// # Errors
    ///
    /// Whatever the kernel says, except being interrupted, which comes back as
    /// no events rather than as a failure.
    pub fn wait(&mut self, out: &mut Vec<u64>, timeout: Duration) -> io::Result<()> {
        out.clear();
        self.inner.wait(out, timeout)
    }
}

/// How many events one `wait` will take at a time.
///
/// The loop runs a batch of 64 commands, so more ready connections than this in
/// one turn is more work than a turn wants anyway. What is left is still ready
/// and comes back on the next call.
///
/// Windows and the scan hand over their whole registration list on every call
/// and have no separate event array to size, so on those two this number would
/// be dead code.
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "ios"))]
const EVENTS: usize = 64;

#[cfg(any(target_os = "macos", target_os = "ios"))]
use bsd as backend;
#[cfg(target_os = "linux")]
use linux as backend;
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "ios", windows)))]
use scan as backend;
#[cfg(windows)]
use win as backend;

use backend::Inner;

/// `epoll`, which is the one that matters: the gate box is Linux.
#[cfg(target_os = "linux")]
mod linux {
    use super::{EVENTS, Source};
    use std::io;
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
    use std::time::Duration;

    pub struct Inner {
        epfd: OwnedFd,
        events: Vec<libc::epoll_event>,
    }

    impl Inner {
        pub fn new() -> io::Result<Inner> {
            // SAFETY: a call with no pointer arguments. `EPOLL_CLOEXEC` so a
            // child of this process does not inherit the set.
            let fd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(Inner {
                // SAFETY: `epoll_create1` returned it and nothing else owns it.
                epfd: unsafe { OwnedFd::from_raw_fd(fd) },
                events: vec![libc::epoll_event { events: 0, u64: 0 }; EVENTS],
            })
        }

        pub fn add(&mut self, src: &impl Source, token: u64) -> io::Result<()> {
            let mut ev = libc::epoll_event {
                // Level triggered, which is the default and is why there is no
                // `EPOLLET` here.
                events: libc::EPOLLIN as u32,
                u64: token,
            };
            // SAFETY: both descriptors are open and the event outlives the call.
            let rc = unsafe {
                libc::epoll_ctl(
                    self.epfd.as_raw_fd(),
                    libc::EPOLL_CTL_ADD,
                    src.as_raw_fd() as RawFd,
                    &raw mut ev,
                )
            };
            if rc < 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        }

        pub fn remove(&mut self, _token: u64) {}

        pub fn wait(&mut self, out: &mut Vec<u64>, timeout: Duration) -> io::Result<()> {
            let ms = i32::try_from(timeout.as_millis()).unwrap_or(i32::MAX);
            // SAFETY: the buffer is ours and is `EVENTS` long, which is what is
            // being promised.
            let n = unsafe {
                libc::epoll_wait(
                    self.epfd.as_raw_fd(),
                    self.events.as_mut_ptr(),
                    EVENTS as i32,
                    ms,
                )
            };
            if n < 0 {
                let e = io::Error::last_os_error();
                // A signal arrived. Nothing is ready, which is not a failure.
                if e.kind() == io::ErrorKind::Interrupted {
                    return Ok(());
                }
                return Err(e);
            }
            for ev in &self.events[..n as usize] {
                out.push(ev.u64);
            }
            Ok(())
        }
    }
}

/// `kqueue`, for the macOS rows of the gate.
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod bsd {
    use super::{EVENTS, Source};
    use std::io;
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::ptr;
    use std::time::Duration;

    pub struct Inner {
        kq: OwnedFd,
        events: Vec<libc::kevent>,
    }

    /// A `kevent` with nothing in it, to fill the buffer with.
    fn blank() -> libc::kevent {
        libc::kevent {
            ident: 0,
            filter: 0,
            flags: 0,
            fflags: 0,
            data: 0,
            udata: ptr::null_mut(),
        }
    }

    impl Inner {
        pub fn new() -> io::Result<Inner> {
            // SAFETY: a call with no arguments.
            let fd = unsafe { libc::kqueue() };
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            // A kqueue is not inherited across exec, so there is no CLOEXEC to
            // set here the way there is on the epoll side.
            Ok(Inner {
                // SAFETY: `kqueue` returned it and nothing else owns it.
                kq: unsafe { OwnedFd::from_raw_fd(fd) },
                events: vec![blank(); EVENTS],
            })
        }

        pub fn add(&mut self, src: &impl Source, token: u64) -> io::Result<()> {
            let mut change = blank();
            change.ident = src.as_raw_fd() as usize;
            change.filter = libc::EVFILT_READ;
            change.flags = libc::EV_ADD | libc::EV_ENABLE;
            // The token rides in `udata`, which is the only field a kqueue
            // hands back untouched.
            change.udata = usize::try_from(token).unwrap_or(usize::MAX) as *mut libc::c_void;
            // SAFETY: one change, no event buffer, and a null timeout, which
            // for a call with no events asked for means do not wait.
            let rc = unsafe {
                libc::kevent(
                    self.kq.as_raw_fd(),
                    &raw const change,
                    1,
                    ptr::null_mut(),
                    0,
                    ptr::null(),
                )
            };
            if rc < 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        }

        pub fn remove(&mut self, _token: u64) {}

        pub fn wait(&mut self, out: &mut Vec<u64>, timeout: Duration) -> io::Result<()> {
            let ts = libc::timespec {
                tv_sec: libc::time_t::try_from(timeout.as_secs()).unwrap_or(libc::time_t::MAX),
                tv_nsec: libc::c_long::from(timeout.subsec_nanos()),
            };
            // SAFETY: the buffer is ours and is `EVENTS` long, and the timespec
            // outlives the call.
            let n = unsafe {
                libc::kevent(
                    self.kq.as_raw_fd(),
                    ptr::null(),
                    0,
                    self.events.as_mut_ptr(),
                    EVENTS as i32,
                    &raw const ts,
                )
            };
            if n < 0 {
                let e = io::Error::last_os_error();
                if e.kind() == io::ErrorKind::Interrupted {
                    return Ok(());
                }
                return Err(e);
            }
            for ev in &self.events[..n as usize] {
                out.push(ev.udata as u64);
            }
            Ok(())
        }
    }
}

/// `WSAPoll`, for Windows.
///
/// The odd one out: `epoll` and `kqueue` are objects the kernel keeps and this
/// is an array the caller keeps and hands over on every call. So `remove` has
/// work to do here, where on the other two closing the socket is enough, and
/// the array is walked twice per turn rather than never. At the connection
/// counts this server sees that is nothing against the syscall per idle
/// connection it replaces, and if it ever is not, the answer is IOCP and a
/// different shape of loop rather than a faster walk.
///
/// A socket is a `SOCKET` and not a file handle. They are different numbers on
/// this platform and only one of them can be polled, which is why [`Source`]
/// asks for `AsRawSocket` here.
#[cfg(windows)]
mod win {
    use super::Source;
    use std::io;
    use std::time::Duration;
    use windows_sys::Win32::Networking::WinSock::{
        POLLRDNORM, WSAGetLastError, WSAPOLLFD, WSAPoll,
    };

    pub struct Inner {
        /// The array `WSAPoll` is handed, and the token for each row beside it.
        ///
        /// Two vectors and not one of pairs, because the first has to be one
        /// contiguous run of `WSAPOLLFD` for the call and a vector of pairs is
        /// not that.
        fds: Vec<WSAPOLLFD>,
        tokens: Vec<u64>,
    }

    impl Inner {
        pub fn new() -> io::Result<Inner> {
            Ok(Inner {
                fds: Vec::new(),
                tokens: Vec::new(),
            })
        }

        pub fn add(&mut self, src: &impl Source, token: u64) -> io::Result<()> {
            // `POLLRDNORM` and not `POLLIN`, because `POLLIN` here is that plus
            // `POLLRDBAND`, which is out of band data this server never reads.
            self.fds.push(WSAPOLLFD {
                fd: src.as_raw_socket() as usize,
                events: POLLRDNORM,
                revents: 0,
            });
            self.tokens.push(token);
            Ok(())
        }

        pub fn remove(&mut self, token: u64) {
            if let Some(i) = self.tokens.iter().position(|t| *t == token) {
                // Order does not matter, so the last row moves into the hole
                // rather than everything above it moving down.
                self.fds.swap_remove(i);
                self.tokens.swap_remove(i);
            }
        }

        pub fn wait(&mut self, out: &mut Vec<u64>, timeout: Duration) -> io::Result<()> {
            // `WSAPoll` refuses an empty array rather than treating it as a
            // sleep, so the sleep is here. This is the state the server is in
            // between its last connection closing and its next one arriving,
            // which for a listener that is always registered is never, and for
            // a poller with nothing in it at all is the moment before the doors
            // open.
            if self.fds.is_empty() {
                if !timeout.is_zero() {
                    std::thread::sleep(timeout);
                }
                return Ok(());
            }
            let ms = i32::try_from(timeout.as_millis()).unwrap_or(i32::MAX);
            let n = {
                let len = u32::try_from(self.fds.len()).unwrap_or(u32::MAX);
                // SAFETY: the array is ours, it is `len` rows long, and the
                // call writes only to the `revents` field of each row.
                unsafe { WSAPoll(self.fds.as_mut_ptr(), len, ms) }
            };
            if n < 0 {
                // SAFETY: a call with no arguments that reads thread local
                // state the failing call just wrote.
                return Err(io::Error::from_raw_os_error(unsafe { WSAGetLastError() }));
            }
            for (row, token) in self.fds.iter().zip(&self.tokens) {
                // Readable, and also hung up or in error, because both of those
                // are the loop's business: a read on a closed socket is how the
                // connection finds out it is closed. `revents` is reported
                // whatever was asked for, which is why the mask here is wider
                // than the one in `add`.
                if row.revents != 0 {
                    out.push(*token);
                }
            }
            Ok(())
        }
    }
}

/// Everywhere else, which today is nowhere: the scan this replaced.
///
/// It reports every registered token every time, so the loop above reads from
/// every open connection and finds out the hard way which ones had something.
/// That is what the server did before this module existed, so it is not a
/// downgrade, it is the previous behaviour kept for a platform that is neither
/// unix nor Windows and that nobody has asked for.
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "ios", windows)))]
mod scan {
    use super::Source;
    use std::io;
    use std::time::Duration;

    pub struct Inner {
        tokens: Vec<u64>,
    }

    impl Inner {
        pub fn new() -> io::Result<Inner> {
            Ok(Inner { tokens: Vec::new() })
        }

        pub fn add(&mut self, _src: &impl Source, token: u64) -> io::Result<()> {
            if !self.tokens.contains(&token) {
                self.tokens.push(token);
            }
            Ok(())
        }

        pub fn remove(&mut self, token: u64) {
            self.tokens.retain(|t| *t != token);
        }

        pub fn wait(&mut self, out: &mut Vec<u64>, timeout: Duration) -> io::Result<()> {
            // There is nothing to wait on, so the wait is the sleep the loop
            // used to do for itself.
            if !timeout.is_zero() {
                std::thread::sleep(timeout);
            }
            out.extend_from_slice(&self.tokens);
            Ok(())
        }
    }
}

/// What a poller promises, which is that a source with nothing on it is not
/// reported.
///
/// Compiled for the three platforms that have a real backend, because the scan
/// cannot make that promise and these tests are the promise written down. On a
/// platform that falls through to the scan there is nothing here to run, which
/// is the honest state of affairs rather than a test that asserts less.
#[cfg(test)]
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "ios", windows))]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};

    /// A listener with nothing connecting to it is not ready, and one with a
    /// client waiting is. That is the whole contract the accept path needs.
    #[test]
    fn a_listener_is_ready_only_when_somebody_is_waiting() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        listener.set_nonblocking(true).expect("nonblocking");
        let addr = listener.local_addr().expect("addr");

        let mut poller = Poller::new().expect("poller");
        poller.add(&listener, 7).expect("add");

        let mut ready = Vec::new();
        poller.wait(&mut ready, Duration::ZERO).expect("wait");
        assert!(ready.is_empty(), "nothing has connected yet");

        let _client = TcpStream::connect(addr).expect("connect");
        poller
            .wait(&mut ready, Duration::from_secs(2))
            .expect("wait");
        assert_eq!(ready, vec![7]);
    }

    /// The part the syscall count depends on: a connection with nothing on it
    /// is not reported, so the loop does not read from it.
    #[test]
    fn a_quiet_connection_is_not_reported_and_a_busy_one_is() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let mut client = TcpStream::connect(addr).expect("connect");
        let (server, _) = listener.accept().expect("accept");
        server.set_nonblocking(true).expect("nonblocking");

        let mut poller = Poller::new().expect("poller");
        poller.add(&server, 11).expect("add");

        let mut ready = Vec::new();
        poller.wait(&mut ready, Duration::ZERO).expect("wait");
        assert!(ready.is_empty(), "the client has not said anything");

        client.write_all(b"PING\r\n").expect("write");
        poller
            .wait(&mut ready, Duration::from_secs(2))
            .expect("wait");
        assert_eq!(ready, vec![11]);

        // Level triggered: still unread, so still ready.
        poller.wait(&mut ready, Duration::ZERO).expect("wait");
        assert_eq!(ready, vec![11]);

        let mut buf = [0u8; 16];
        let mut server = server;
        let n = server.read(&mut buf).expect("read");
        assert_eq!(&buf[..n], b"PING\r\n");

        poller.wait(&mut ready, Duration::ZERO).expect("wait");
        assert!(ready.is_empty(), "everything on it has been read");
    }
}
