//! The real ring.
//!
//! `16` section 0 picks the `io-uring` crate for this and rules out everything
//! with an executor in it. What that leaves is a thin thing: build the ring with
//! the setup flags the qualification run asked for, push entries, hand them over
//! once per turn of the loop, and walk the completions. There is no task, no
//! waker and no poll, because the state a submission is waiting on lives in
//! [`crate::Pending`] and the loop picks it up itself.
//!
//! One ring per shard, storage and network on the same one, told apart by the
//! tag. That is `04` section 7 and it is why there is no separate network ring
//! anywhere in this crate.

use std::os::fd::RawFd;

use io_uring::{IoUring, opcode, squeue, types};
use yo_common::{Code, Error, Result};

use crate::completion::{Completion, Stats};
use crate::config::{Features, RingConfig};
use crate::token::Token;

pub(crate) struct Backend {
    ring: IoUring,
    features: Features,
    in_flight: u32,
    stats: Stats,
}

impl Backend {
    pub(crate) fn new(cfg: &RingConfig) -> Result<Backend> {
        cfg.check()?;
        let mut builder = IoUring::builder();
        if cfg.iopoll {
            builder.setup_iopoll();
        }
        if let Some(p) = cfg.sqpoll {
            builder.setup_sqpoll(p.idle_ms);
            if let Some(cpu) = p.cpu {
                builder.setup_sqpoll_cpu(cpu);
            }
        }
        // The ring is never touched from a second thread, which is R6 and is
        // true of everything on the data path. Telling the kernel so is free and
        // lets it skip work it would otherwise do on every submission.
        builder.setup_single_issuer();
        let ring = match builder.build(cfg.entries) {
            Ok(ring) => ring,
            Err(e) => {
                // A kernel too old for `single_issuer` refuses the whole setup,
                // and refusing to start on a kernel that has a perfectly good
                // ring would be a silly way to lose. One retry without it, and
                // nothing else changes.
                let mut plain = IoUring::builder();
                if cfg.iopoll {
                    plain.setup_iopoll();
                }
                if let Some(p) = cfg.sqpoll {
                    plain.setup_sqpoll(p.idle_ms);
                    if let Some(cpu) = p.cpu {
                        plain.setup_sqpoll_cpu(cpu);
                    }
                }
                plain.build(cfg.entries).map_err(|_| {
                    Error::new(Code::Unsupported, "the kernel refused this ring")
                        .with_detail(format!("entries={} {e}", cfg.entries))
                })?
            }
        };
        let params = ring.params();
        let features = Features {
            entries: params.sq_entries(),
            // What the kernel says, not what was asked for. A row that claims a
            // flag the kernel declined is one of the ways aki published four
            // wrong numbers.
            iopoll: params.is_setup_iopoll(),
            sqpoll: params.is_setup_sqpoll(),
            // Nothing is registered yet. `04` section 7 wants the arenas
            // registered and R10 budgets it, and that lands with the sink that
            // owns the arenas rather than here.
            registered_buffers: false,
            is_uring: true,
        };
        Ok(Backend {
            ring,
            features,
            in_flight: 0,
            stats: Stats::default(),
        })
    }

    pub(crate) const fn features(&self) -> Features {
        self.features
    }

    pub(crate) const fn in_flight(&self) -> u32 {
        self.in_flight
    }

    pub(crate) const fn stats(&self) -> Stats {
        self.stats
    }

    /// # Safety
    ///
    /// `entry` may name a buffer, and if it does the buffer has to stay alive
    /// and unmoved until the completion is drained.
    unsafe fn push(&mut self, entry: squeue::Entry) -> Result<()> {
        // SAFETY: the caller's promise about the buffer, forwarded. Nothing
        // else in an entry built here points at anything.
        let first = unsafe { self.ring.submission().push(&entry) };
        if first.is_err() {
            // Full as far as this side can see, which is not the same as full.
            // Handing the queue over is what makes the kernel's consumption
            // visible, so one flush and one retry before giving up.
            self.submit()?;
            // SAFETY: as above.
            let again = unsafe { self.ring.submission().push(&entry) };
            if again.is_err() {
                self.stats.refused += 1;
                return Err(Error::new(Code::Full, "the submission queue is full")
                    .with_detail(format!("entries={}", self.features.entries)));
            }
        }
        self.in_flight += 1;
        self.stats.submitted += 1;
        Ok(())
    }

    /// # Safety
    ///
    /// `buf` has to stay alive and unmoved, and stay readable for `len` bytes,
    /// until the completion for `token` is drained.
    pub(crate) unsafe fn write_at(
        &mut self,
        fd: RawFd,
        buf: *const u8,
        len: u32,
        offset: u64,
        token: Token,
    ) -> Result<()> {
        let e = opcode::Write::new(types::Fd(fd), buf, len)
            .offset(offset)
            .build()
            .user_data(token.raw());
        // SAFETY: the caller's promise about `buf`, forwarded.
        unsafe { self.push(e) }
    }

    /// # Safety
    ///
    /// `buf` has to stay alive and unmoved, and stay writable for `len` bytes,
    /// until the completion for `token` is drained.
    pub(crate) unsafe fn read_at(
        &mut self,
        fd: RawFd,
        buf: *mut u8,
        len: u32,
        offset: u64,
        token: Token,
    ) -> Result<()> {
        let e = opcode::Read::new(types::Fd(fd), buf, len)
            .offset(offset)
            .build()
            .user_data(token.raw());
        // SAFETY: the caller's promise about `buf`, forwarded.
        unsafe { self.push(e) }
    }

    pub(crate) fn fsync_data(&mut self, fd: RawFd, token: Token) -> Result<()> {
        let e = opcode::Fsync::new(types::Fd(fd))
            .flags(types::FsyncFlags::DATASYNC)
            .build()
            .user_data(token.raw());
        // SAFETY: an fsync entry names no buffer, so there is nothing for the
        // caller to keep alive and the contract is satisfied trivially.
        unsafe { self.push(e) }
    }

    pub(crate) fn submit(&mut self) -> Result<u32> {
        self.stats.enters += 1;
        match self.ring.submit() {
            Ok(n) => Ok(u32::try_from(n).unwrap_or(u32::MAX)),
            Err(e) => Err(
                Error::new(Code::Io, "handing the queue to the kernel failed")
                    .with_detail(format!("{e}")),
            ),
        }
    }

    pub(crate) fn submit_and_wait(&mut self, want: u32) -> Result<u32> {
        self.stats.enters += 1;
        if want > 0 {
            self.stats.waits += 1;
        }
        match self.ring.submit_and_wait(want as usize) {
            Ok(n) => Ok(u32::try_from(n).unwrap_or(u32::MAX)),
            Err(e) => Err(Error::new(Code::Io, "waiting on the ring failed")
                .with_detail(format!("want={want} {e}"))),
        }
    }

    pub(crate) fn drain<F: FnMut(Completion)>(&mut self, mut f: F) -> u32 {
        let mut n = 0u32;
        let mut cq = self.ring.completion();
        cq.sync();
        for cqe in &mut cq {
            f(Completion::new(
                Token::from_raw(cqe.user_data()),
                cqe.result(),
            ));
            n += 1;
        }
        self.in_flight = self.in_flight.saturating_sub(n);
        self.stats.completed += u64::from(n);
        n
    }
}
