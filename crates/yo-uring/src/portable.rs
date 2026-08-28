//! The same shape without a ring under it.
//!
//! `04` section 7: everywhere that is not Linux gets the same state machine with
//! synchronous storage, correct and tested and not gated. So this does the read
//! or the write on the spot and queues the completion, and a caller sees exactly
//! what it sees on Linux: it parks its state, submits a tag, and picks the state
//! back up when the completion turns up on a later turn of the loop.
//!
//! The difference is latency and nothing else. There is no thread pool here
//! pretending to be a ring, because a thread pool would change when the work
//! happens relative to the loop and that is the one thing the shape is for. A
//! macOS build is a development and test target. The gate numbers come off
//! Linux, and every benchmark row says which of the two produced it because
//! [`Features::is_uring`](crate::Features::is_uring) is on the row.

use std::collections::VecDeque;
use std::fs::File;
use std::io;
use std::mem::ManuallyDrop;

use yo_common::{Code, Error, Result};

use crate::Fd;
use crate::completion::{Completion, Stats};
use crate::config::{Features, RingConfig};
use crate::token::Token;

pub(crate) struct Backend {
    done: VecDeque<Completion>,
    features: Features,
    stats: Stats,
}

impl Backend {
    pub(crate) fn new(cfg: &RingConfig) -> Result<Backend> {
        cfg.check()?;
        Ok(Backend {
            done: VecDeque::with_capacity(64),
            features: Features {
                entries: cfg.entries,
                // None of the three is a thing that exists here. Reporting them
                // as asked for rather than as got is how a row ends up claiming
                // SQPoll on a machine that has never had a kernel thread.
                iopoll: false,
                sqpoll: false,
                registered_buffers: false,
                is_uring: false,
            },
            stats: Stats::default(),
        })
    }

    pub(crate) const fn features(&self) -> Features {
        self.features
    }

    pub(crate) fn in_flight(&self) -> u32 {
        u32::try_from(self.done.len()).unwrap_or(u32::MAX)
    }

    pub(crate) const fn stats(&self) -> Stats {
        self.stats
    }

    /// Room for one more, or the same [`Code::Full`] the ring gives, so a caller
    /// has one case to handle rather than one per platform.
    fn admit(&mut self) -> Result<()> {
        if self.done.len() >= self.features.entries as usize {
            self.stats.refused += 1;
            return Err(
                Error::new(Code::Full, "the submission queue is full").with_detail(format!(
                    "entries={} undrained={}",
                    self.features.entries,
                    self.done.len()
                )),
            );
        }
        Ok(())
    }

    fn finish(&mut self, token: Token, r: io::Result<usize>) {
        let result = match r {
            Ok(n) => i32::try_from(n).unwrap_or(i32::MAX),
            // Negated, which is the kernel's convention and is what the ring
            // hands back, so `Completion::error` is one implementation.
            Err(e) => -e.raw_os_error().unwrap_or(5),
        };
        self.done.push_back(Completion::new(token, result));
        self.stats.submitted += 1;
    }

    /// # Safety
    ///
    /// `buf` has to be readable for `len` bytes for the duration of the call,
    /// and `fd` has to be an open descriptor. The Linux contract is stricter and
    /// asks for the buffer to outlive the completion, and a caller written to
    /// that contract satisfies this one too.
    pub(crate) unsafe fn write_at(
        &mut self,
        fd: Fd,
        buf: *const u8,
        len: u32,
        offset: u64,
        token: Token,
    ) -> Result<()> {
        self.admit()?;
        // SAFETY: the caller promises `buf` is readable for `len` bytes.
        let bytes = unsafe { core::slice::from_raw_parts(buf, len as usize) };
        // SAFETY: the caller promises `fd` is open. `ManuallyDrop` means the
        // descriptor is not closed when this borrow ends, which would otherwise
        // close a file the caller still owns.
        let f = ManuallyDrop::new(unsafe { file_from(fd) });
        let r = write_at(&f, bytes, offset);
        self.finish(token, r);
        Ok(())
    }

    /// # Safety
    ///
    /// `buf` has to be writable for `len` bytes for the duration of the call,
    /// and `fd` has to be an open descriptor.
    pub(crate) unsafe fn read_at(
        &mut self,
        fd: Fd,
        buf: *mut u8,
        len: u32,
        offset: u64,
        token: Token,
    ) -> Result<()> {
        self.admit()?;
        // SAFETY: the caller promises `buf` is writable for `len` bytes.
        let bytes = unsafe { core::slice::from_raw_parts_mut(buf, len as usize) };
        // SAFETY: the caller promises `fd` is open, and `ManuallyDrop` keeps it
        // open past this borrow.
        let f = ManuallyDrop::new(unsafe { file_from(fd) });
        let r = read_at(&f, bytes, offset);
        self.finish(token, r);
        Ok(())
    }

    pub(crate) fn fsync_data(&mut self, fd: Fd, token: Token) -> Result<()> {
        self.admit()?;
        // SAFETY: the caller promises `fd` is open, and `ManuallyDrop` keeps it
        // open past this borrow.
        let f = ManuallyDrop::new(unsafe { file_from(fd) });
        let r = sync_data(&f).map(|()| 0usize);
        self.finish(token, r);
        Ok(())
    }

    /// Nothing to hand over, because the work is already done. The counter still
    /// moves so that a test asserting one submit a turn asserts the same thing
    /// on both backends.
    pub(crate) fn submit(&mut self) -> Result<u32> {
        self.stats.enters += 1;
        Ok(0)
    }

    /// There is nothing to wait for either. Everything submitted has completed,
    /// so this returns as soon as `want` is satisfiable, which it always is when
    /// that many were submitted, and never blocks.
    pub(crate) fn submit_and_wait(&mut self, want: u32) -> Result<u32> {
        self.stats.enters += 1;
        if want > 0 {
            self.stats.waits += 1;
        }
        Ok(0)
    }

    pub(crate) fn drain<F: FnMut(Completion)>(&mut self, mut f: F) -> u32 {
        let mut n = 0u32;
        while let Some(c) = self.done.pop_front() {
            f(c);
            n += 1;
        }
        self.stats.completed += u64::from(n);
        n
    }
}

/// # Safety
///
/// `fd` has to be an open descriptor, and the [`File`] this hands back has to be
/// wrapped in [`ManuallyDrop`] by the caller so that it does not close it.
#[cfg(unix)]
unsafe fn file_from(fd: Fd) -> File {
    use std::os::fd::FromRawFd;
    // SAFETY: the caller's promise, forwarded.
    unsafe { File::from_raw_fd(fd) }
}

/// # Safety
///
/// As the unix one.
#[cfg(windows)]
unsafe fn file_from(fd: Fd) -> File {
    use std::os::windows::io::FromRawHandle;
    // SAFETY: the caller's promise, forwarded.
    unsafe { File::from_raw_handle(fd) }
}

#[cfg(unix)]
fn write_at(f: &File, buf: &[u8], offset: u64) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;
    loop {
        match f.write_at(buf, offset) {
            // The ring never surfaces this, so neither does this, or a caller
            // would have a case to handle on one platform and not the other.
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            other => return other,
        }
    }
}

#[cfg(windows)]
fn write_at(f: &File, buf: &[u8], offset: u64) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;
    loop {
        match f.seek_write(buf, offset) {
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            other => return other,
        }
    }
}

#[cfg(unix)]
fn read_at(f: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;
    loop {
        match f.read_at(buf, offset) {
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            other => return other,
        }
    }
}

#[cfg(windows)]
fn read_at(f: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;
    loop {
        match f.seek_read(buf, offset) {
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            other => return other,
        }
    }
}

/// The strongest thing the platform has, which on macOS is `F_FULLFSYNC` and
/// not `fsync`.
///
/// The same choice `yo-file` makes and for the same reason: plain `fsync` there
/// returns once the bytes reached the drive, which is not the same as surviving
/// a power cut, and a durability mode that lies is worse than not having one.
#[cfg(target_os = "macos")]
fn sync_data(f: &File) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    // SAFETY: `fcntl` with `F_FULLFSYNC` takes no argument past the command and
    // reads nothing through a pointer. The descriptor is valid for as long as
    // `f` is borrowed.
    let rc = unsafe { libc::fcntl(f.as_raw_fd(), libc::F_FULLFSYNC) };
    if rc == -1 {
        // Not every filesystem has it, and the fallback is the weaker thing
        // rather than a failure, which is what `yo-file` does too.
        return f.sync_data();
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn sync_data(f: &File) -> io::Result<()> {
    f.sync_data()
}
