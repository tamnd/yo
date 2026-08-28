//! The per shard submission ring.
//!
//! `04` section 7 says one ring per shard, four thousand and ninety six
//! entries, storage submissions and network submissions on the same ring told
//! apart by the user data tag, and the knobs decided by the qualification run
//! rather than by taste. It also says the thing this crate exists to make
//! possible: execution is restructured to be asynchronous, and it is not wrapped
//! in an async runtime. There is no future here, no waker, no executor and no
//! `.await`. A submission that has not finished has its caller's state parked in
//! a slot, and the next turn of the loop picks the state up when the completion
//! arrives.
//!
//! That is the whole difference between the two ends of the io_uring ladder in
//! the VLDB 2026 study. Sixteen and a half thousand transactions a second when
//! every submission is waited on, a hundred and eighty three thousand when the
//! execution is restructured this way, five hundred and forty six thousand with
//! SQPoll on top. The parking is not an optimisation, it is the design.
//!
//! # The three parts
//!
//! [`Token`] is the eight bytes io_uring hands back, packed so that a drain can
//! route a completion and find its state without a lookup and can tell a stale
//! completion from a live one.
//!
//! [`Pending`] is where the state waits. A slab with a free list, capped at the
//! ring's depth, handing out the slots the tag names.
//!
//! [`Ring`] is the submissions themselves. On Linux it is io_uring through the
//! `io-uring` crate, which `16` section 0 picks and which has no executor in it.
//! Everywhere else it is the same state machine doing the storage synchronously,
//! per `04` section 7, which is correct and tested and never gated.
//!
//! # Using it
//!
//! ```
//! use yo_uring::{Kind, Pending, Ring, RingConfig};
//!
//! # fn main() -> yo_common::Result<()> {
//! let mut ring = Ring::new(&RingConfig::plain().with_entries(64))?;
//! let mut pending: Pending<&'static str> = Pending::with_capacity(ring.entries());
//!
//! // What the caller wants back when this finishes goes in the table, and the
//! // tag the table hands out goes on the submission.
//! let token = pending.park(Kind::Fsync, "the commit that is waiting on this")?;
//! # let _ = token;
//! # let _ = &mut ring;
//! # Ok(())
//! # }
//! ```

#![deny(missing_docs)]

mod completion;
mod config;
mod pending;
mod token;

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
use linux as imp;

#[cfg(not(target_os = "linux"))]
mod portable;
#[cfg(not(target_os = "linux"))]
use portable as imp;

pub use completion::{Completion, Stats};
pub use config::{DEFAULT_ENTRIES, Features, RingConfig, SqPoll};
pub use pending::Pending;
pub use token::{Kind, MAX_SLOT, Token};

use yo_common::Result;

/// What a submission names.
///
/// A raw descriptor rather than a borrowed [`std::fs::File`], because a
/// submission outlives the call that made it and a borrow would say otherwise.
/// Keeping the file open until the completion is drained is the caller's job and
/// is part of the contract on every method that takes one.
#[cfg(unix)]
pub type Fd = std::os::fd::RawFd;

/// What a submission names. See the unix one.
#[cfg(windows)]
pub type Fd = std::os::windows::io::RawHandle;

/// One shard's ring.
///
/// Not `Sync` and never shared. R6 puts one of these on each shard thread and
/// nothing else touches it, which is what makes the counters plain fields and
/// the pending table a plain `Vec`.
///
/// # The loop this is built for
///
/// `04` section 2 is six stages and this appears in three of them. Submit the
/// queue once, whatever went into it. Run the batch, parking anything that has
/// to wait. Drain the completions and resume what they belong to. One trip into
/// the kernel a turn, not one a submission, and under SQPoll not even that.
pub struct Ring {
    inner: imp::Backend,
}

impl Ring {
    /// Builds a ring.
    ///
    /// # Errors
    ///
    /// [`yo_common::Code::Invalid`] when the configuration cannot work, which is
    /// checked before the kernel is asked for anything so that the message names
    /// the argument. [`yo_common::Code::Unsupported`] when the kernel refuses
    /// the setup, which is what an old kernel or a container without io_uring
    /// looks like.
    pub fn new(config: &RingConfig) -> Result<Ring> {
        Ok(Ring {
            inner: imp::Backend::new(config)?,
        })
    }

    /// What the ring actually got, which is not always what was asked for.
    #[must_use]
    pub fn features(&self) -> Features {
        self.inner.features()
    }

    /// The queue depth, which the kernel may have rounded up.
    #[must_use]
    pub fn entries(&self) -> u32 {
        self.inner.features().entries
    }

    /// Whether this is a real ring, for a benchmark row that has to say so.
    #[must_use]
    pub fn is_uring(&self) -> bool {
        self.inner.features().is_uring
    }

    /// Submissions that have not been drained yet.
    #[must_use]
    pub fn in_flight(&self) -> u32 {
        self.inner.in_flight()
    }

    /// Counters since the ring was built.
    #[must_use]
    pub fn stats(&self) -> Stats {
        self.inner.stats()
    }

    /// Queues a write.
    ///
    /// # Safety
    ///
    /// `buf` has to stay alive, unmoved and readable for `len` bytes until the
    /// completion carrying `token` comes back from [`Ring::drain`], and `fd` has
    /// to stay open for just as long. This is the Linux contract, and it is the
    /// contract on both platforms even though the portable backend has finished
    /// with the buffer by the time the call returns. One contract that holds
    /// everywhere is worth more than a looser one that only holds on the
    /// platform nobody ships on.
    ///
    /// # Errors
    ///
    /// [`yo_common::Code::Full`] when the queue has no room, which is
    /// backpressure and not a failure: drain and come back.
    pub unsafe fn write_at(
        &mut self,
        fd: Fd,
        buf: *const u8,
        len: u32,
        offset: u64,
        token: Token,
    ) -> Result<()> {
        // SAFETY: the caller's promise about `buf` and `fd`, forwarded.
        unsafe { self.inner.write_at(fd, buf, len, offset, token) }
    }

    /// Queues a read.
    ///
    /// # Safety
    ///
    /// As [`Ring::write_at`], with `buf` writable rather than readable.
    ///
    /// # Errors
    ///
    /// As [`Ring::write_at`].
    pub unsafe fn read_at(
        &mut self,
        fd: Fd,
        buf: *mut u8,
        len: u32,
        offset: u64,
        token: Token,
    ) -> Result<()> {
        // SAFETY: the caller's promise about `buf` and `fd`, forwarded.
        unsafe { self.inner.read_at(fd, buf, len, offset, token) }
    }

    /// Queues a data sync, which is the group commit boundary in `06` section 3.
    ///
    /// Ordering is not implied. io_uring runs a queued sync in parallel with the
    /// writes queued before it, so a sync that has to cover them is submitted
    /// after their completions have been drained, not after their submissions.
    /// This is the single most common way to get an io_uring durability bug and
    /// it is why this is written down here rather than left to be discovered.
    ///
    /// # Errors
    ///
    /// As [`Ring::write_at`].
    pub fn fsync_data(&mut self, fd: Fd, token: Token) -> Result<()> {
        self.inner.fsync_data(fd, token)
    }

    /// Hands the queue to the kernel.
    ///
    /// Once a turn of the loop, whatever went into it. Under SQPoll with the
    /// kernel thread awake this does not enter the kernel at all, which is the
    /// zero syscall case `04` section 2 is describing.
    ///
    /// # Errors
    ///
    /// [`yo_common::Code::Io`] if the kernel refused the handover.
    pub fn submit(&mut self) -> Result<u32> {
        self.inner.submit()
    }

    /// Hands the queue over and waits for `want` completions.
    ///
    /// The loop does not call this on the hot path, because waiting is the thing
    /// the parking exists to avoid. It is for the idle case, where there is
    /// nothing to run and sleeping in the kernel beats spinning, and for tests.
    ///
    /// # Errors
    ///
    /// [`yo_common::Code::Io`] if the wait failed.
    pub fn submit_and_wait(&mut self, want: u32) -> Result<u32> {
        self.inner.submit_and_wait(want)
    }

    /// Walks everything that has finished, and returns how many that was.
    ///
    /// The callback gets each completion in the order the kernel produced them,
    /// which is not the order they were submitted in.
    pub fn drain<F: FnMut(Completion)>(&mut self, f: F) -> u32 {
        self.inner.drain(f)
    }
}

impl core::fmt::Debug for Ring {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let features = self.features();
        f.debug_struct("Ring")
            .field("uring", &features.is_uring)
            .field("entries", &features.entries)
            .field("iopoll", &features.iopoll)
            .field("sqpoll", &features.sqpoll)
            .field("in_flight", &self.in_flight())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::fs::OpenOptions;
    use std::io::Read as _;

    use super::*;

    /// A file that goes away with the test, without pulling in a crate for it.
    struct Tmp(std::path::PathBuf);

    impl Tmp {
        fn new(name: &str) -> Tmp {
            let mut p = std::env::temp_dir();
            p.push(format!("yo-uring-{}-{name}", std::process::id()));
            Tmp(p)
        }

        fn open(&self) -> std::fs::File {
            OpenOptions::new()
                .create(true)
                .truncate(true)
                .read(true)
                .write(true)
                .open(&self.0)
                .expect("a file in the temp directory")
        }

        fn reopen(&self) -> std::fs::File {
            std::fs::File::open(&self.0).expect("the file this test just wrote")
        }
    }

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[cfg(unix)]
    fn raw(f: &std::fs::File) -> Fd {
        use std::os::fd::AsRawFd;
        f.as_raw_fd()
    }

    #[cfg(windows)]
    fn raw(f: &std::fs::File) -> Fd {
        use std::os::windows::io::AsRawHandle;
        f.as_raw_handle()
    }

    /// Everything at once, because the interesting part is the round trip and
    /// not any one call in it. Park state, submit, hand the queue over, drain,
    /// and find the state the completion belongs to.
    #[test]
    #[cfg_attr(
        miri,
        ignore = "a real ring and a real file, neither of which Miri has"
    )]
    fn a_write_goes_out_and_its_state_comes_back() {
        let tmp = Tmp::new("round-trip");
        let file = tmp.open();
        let mut ring = Ring::new(&RingConfig::plain().with_entries(64)).expect("a ring");
        let mut pending: Pending<&'static str> = Pending::with_capacity(ring.entries());

        let payload = *b"the bytes that have to outlive the submission...";
        let token = pending
            .park(Kind::Write, "the commit waiting on this page")
            .expect("room in the table");
        // SAFETY: `payload` is a local that outlives the drain below, it is
        // never moved, and `file` stays open for the whole test.
        unsafe { ring.write_at(raw(&file), payload.as_ptr(), payload.len() as u32, 0, token) }
            .expect("room in the queue");
        assert_eq!(ring.in_flight(), 1);

        ring.submit_and_wait(1).expect("the kernel took it");
        let mut got = Vec::new();
        ring.drain(|c| got.push(c));
        assert_eq!(got.len(), 1, "the completion did not turn up");
        let c = got[0];
        assert!(c.is_ok(), "{:?}", c.error());
        assert_eq!(c.bytes() as usize, payload.len());
        assert_eq!(c.kind(), Kind::Write);
        assert_eq!(
            pending.take(c.token),
            Some("the commit waiting on this page")
        );
        assert_eq!(ring.in_flight(), 0);

        let mut back = Vec::new();
        tmp.reopen().read_to_end(&mut back).expect("a read");
        assert_eq!(back, payload, "the bytes on disk are not the bytes sent");
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "a real ring and a real file, neither of which Miri has"
    )]
    fn a_read_comes_back_with_what_was_written() {
        let tmp = Tmp::new("read-back");
        let file = tmp.open();
        let mut ring = Ring::new(&RingConfig::plain().with_entries(8)).expect("a ring");
        let mut pending: Pending<usize> = Pending::with_capacity(ring.entries());

        let payload = *b"0123456789abcdef";
        let w = pending.park(Kind::Write, 0).expect("room");
        // SAFETY: `payload` outlives the drain, and `file` stays open.
        unsafe { ring.write_at(raw(&file), payload.as_ptr(), 16, 0, w) }.expect("room");
        ring.submit_and_wait(1).expect("submitted");
        ring.drain(|c| assert!(c.is_ok(), "{:?}", c.error()));
        pending.take(w).expect("the write's state");

        let mut buf = [0u8; 8];
        let r = pending.park(Kind::Read, 8).expect("room");
        // SAFETY: `buf` outlives the drain, it is not moved, and `file` is open.
        unsafe { ring.read_at(raw(&file), buf.as_mut_ptr(), 8, 8, r) }.expect("room");
        ring.submit_and_wait(1).expect("submitted");
        let mut n = 0;
        ring.drain(|c| {
            assert!(c.is_ok(), "{:?}", c.error());
            assert_eq!(c.bytes(), 8);
            n += 1;
        });
        assert_eq!(n, 1);
        assert_eq!(&buf, b"89abcdef", "the read did not honour the offset");
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "a real ring and a real file, neither of which Miri has"
    )]
    fn a_sync_completes_and_says_which_commit_it_was_for() {
        let tmp = Tmp::new("sync");
        let file = tmp.open();
        let mut ring = Ring::new(&RingConfig::plain().with_entries(8)).expect("a ring");
        let mut pending: Pending<u64> = Pending::with_capacity(ring.entries());

        let token = pending.park(Kind::Fsync, 4242).expect("room");
        ring.fsync_data(raw(&file), token).expect("room");
        ring.submit_and_wait(1).expect("submitted");
        let mut seen = None;
        ring.drain(|c| seen = Some(c));
        let c = seen.expect("the sync did not complete");
        assert!(c.is_ok(), "{:?}", c.error());
        assert_eq!(c.kind(), Kind::Fsync);
        assert_eq!(pending.take(c.token), Some(4242));
    }

    /// `04` section 2, as an assertion rather than a claim: a batch costs one
    /// trip into the kernel, not one per submission. This is the whole reason
    /// the loop is shaped the way it is, and aki's `HGETALL` profile at 69.7%
    /// write syscalls is what it looks like when nobody checks.
    #[test]
    #[cfg_attr(
        miri,
        ignore = "a real ring and a real file, neither of which Miri has"
    )]
    fn a_batch_of_sixty_four_costs_one_submit() {
        let tmp = Tmp::new("one-submit");
        let file = tmp.open();
        let mut ring = Ring::new(&RingConfig::plain().with_entries(256)).expect("a ring");
        let mut pending: Pending<u32> = Pending::with_capacity(ring.entries());

        let page = [7u8; 512];
        for i in 0..64u32 {
            let token = pending.park(Kind::Write, i).expect("room in the table");
            // SAFETY: `page` outlives the drain at the end of the test and is
            // never moved. Every write goes to its own offset, so the kernel
            // reading the same buffer sixty four times is fine.
            unsafe { ring.write_at(raw(&file), page.as_ptr(), 512, u64::from(i) * 512, token) }
                .expect("room in the queue");
        }
        assert_eq!(ring.stats().submitted, 64);
        assert_eq!(ring.stats().enters, 0, "something entered the kernel early");

        ring.submit_and_wait(64).expect("submitted");
        let mut done = 0;
        let mut resumed = Vec::new();
        while done < 64 {
            done += ring.drain(|c| {
                assert!(c.is_ok(), "{:?}", c.error());
                resumed.push(pending.take(c.token).expect("the state that was parked"));
            });
        }
        assert_eq!(
            ring.stats().enters,
            1,
            "sixty four submissions cost more than one trip into the kernel"
        );
        resumed.sort_unstable();
        assert_eq!(resumed, (0..64u32).collect::<Vec<_>>());
        assert!(pending.is_empty(), "state was left parked behind");
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "a real ring and a real file, neither of which Miri has"
    )]
    fn a_ring_reports_what_it_got_rather_than_what_was_asked_for() {
        let ring = Ring::new(&RingConfig::plain().with_entries(16)).expect("a ring");
        let f = ring.features();
        assert!(f.entries >= 16, "the depth shrank");
        assert!(!f.iopoll, "iopoll appeared without being asked for");
        assert!(!f.registered_buffers);
        assert_eq!(f.is_uring, cfg!(target_os = "linux"));
        assert_eq!(ring.is_uring(), f.is_uring);
    }

    #[test]
    fn a_shape_that_cannot_work_is_refused_before_the_kernel_is_asked() {
        let e = Ring::new(&RingConfig::plain().with_entries(0)).unwrap_err();
        assert_eq!(e.code(), yo_common::Code::Invalid);
    }
}
