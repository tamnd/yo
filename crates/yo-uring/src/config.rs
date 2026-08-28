//! How the ring is set up, and who decides.
//!
//! `04` section 7 is explicit that the knobs are decided by the qualification
//! run on the box and not by taste. So this type carries the knobs and nothing
//! in this crate picks values for them beyond a default that is safe on any
//! kernel. The qualification run writes the row, the row picks the setting, and
//! the setting is recorded next to every benchmark number the box produces.

use yo_common::{Code, Error, Result};

use crate::token::MAX_SLOT;

/// The submission queue depth `04` section 7 specifies.
pub const DEFAULT_ENTRIES: u32 = 4096;

/// How a kernel side polling thread is set up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SqPoll {
    /// How long the kernel thread spins before it goes to sleep, in
    /// milliseconds. It is woken again by the next submission, at the cost of
    /// one `io_uring_enter`, which is the syscall SQPoll exists to avoid. Too
    /// short and the saving goes away under a bursty load, too long and an idle
    /// shard burns a core.
    pub idle_ms: u32,
    /// Which core the kernel thread is pinned to, or `None` to leave it to the
    /// scheduler. `04` section 1 reserves a core for the accept loop and for
    /// this, so on a qualified box it is set.
    pub cpu: Option<u32>,
}

impl Default for SqPoll {
    fn default() -> SqPoll {
        SqPoll {
            // Long enough to cover the gap between two batches on a busy shard
            // and short enough that an idle shard gives the core back inside a
            // human noticeable interval.
            idle_ms: 1000,
            cpu: None,
        }
    }
}

/// Everything the ring is built with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingConfig {
    /// Submission queue depth. Rounded up to a power of two by the kernel, so a
    /// value that is not one is accepted and quietly becomes the next one up.
    pub entries: u32,
    /// `IORING_SETUP_IOPOLL`. Busy polls the device's completion queue instead
    /// of taking an interrupt, which is worth having on NVMe and is worth
    /// nothing anywhere else. It also requires every file in the ring to have
    /// been opened `O_DIRECT`, so it is not a knob that can be turned on
    /// halfway.
    pub iopoll: bool,
    /// `IORING_SETUP_SQPOLL`, when a core can be spared for it.
    pub sqpoll: Option<SqPoll>,
    /// Whether to register the submission buffers up front. The arenas are
    /// already stable addresses, which is what makes this nearly free here and
    /// expensive elsewhere, and R10 budgets it at five percent of the gain.
    pub registered_buffers: bool,
}

impl Default for RingConfig {
    fn default() -> RingConfig {
        RingConfig {
            entries: DEFAULT_ENTRIES,
            // Both off by default. On is a decision the qualification run makes
            // about one box, and a default that turns them on is a default that
            // silently fails on every kernel and filesystem that does not
            // support them.
            iopoll: false,
            sqpoll: None,
            registered_buffers: false,
        }
    }
}

impl RingConfig {
    /// The default depth with everything off, which works on any kernel with a
    /// ring at all.
    #[must_use]
    pub fn plain() -> RingConfig {
        RingConfig::default()
    }

    /// A depth, with everything else left alone.
    #[must_use]
    pub fn with_entries(mut self, entries: u32) -> RingConfig {
        self.entries = entries;
        self
    }

    /// Turns on kernel side polling of the submission queue.
    #[must_use]
    pub fn with_sqpoll(mut self, sqpoll: SqPoll) -> RingConfig {
        self.sqpoll = Some(sqpoll);
        self
    }

    /// Turns on device side completion polling.
    #[must_use]
    pub const fn with_iopoll(mut self, yes: bool) -> RingConfig {
        self.iopoll = yes;
        self
    }

    /// Rejects a configuration that cannot work, at construction, with a
    /// reason.
    ///
    /// The ring calls this before it asks the kernel for anything, so a bad
    /// depth is a message about the depth rather than an `EINVAL` from a
    /// syscall that says nothing about which argument was wrong.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] with the offending value in the detail.
    pub fn check(&self) -> Result<()> {
        if self.entries == 0 {
            return Err(Error::new(Code::Invalid, "a ring with no entries in it"));
        }
        // The pending table is addressed by 24 bits of the tag and it is sized
        // from the ring, so a ring deeper than the tag can name would hand out
        // slots that come back as somebody else's.
        if self.entries > MAX_SLOT + 1 {
            return Err(
                Error::new(Code::Invalid, "a ring deeper than the tag can address")
                    .with_detail(format!("entries={} max={}", self.entries, MAX_SLOT + 1)),
            );
        }
        if let Some(p) = self.sqpoll
            && p.idle_ms == 0
        {
            // Zero means the kernel thread never sleeps, which reads as a
            // reasonable thing to ask for and is a core burned forever on an
            // idle shard.
            return Err(Error::new(
                Code::Invalid,
                "an SQPoll thread that never goes idle",
            ));
        }
        Ok(())
    }
}

/// What the ring actually got, which is not always what was asked for.
///
/// A kernel that does not support a setup flag says so, and the honest thing to
/// do with that is record it rather than pretend. Every benchmark row carries
/// this, because a number taken with SQPoll off on a box whose row says SQPoll
/// on is one of the four ways aki's published numbers turned out wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Features {
    /// Submission queue depth the kernel settled on.
    pub entries: u32,
    /// Whether `IORING_SETUP_IOPOLL` is on.
    pub iopoll: bool,
    /// Whether a kernel side submission thread is running.
    pub sqpoll: bool,
    /// Whether buffers were registered.
    pub registered_buffers: bool,
    /// Whether this is a real ring or the portable fallback.
    pub is_uring: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_is_the_depth_the_spec_names() {
        let c = RingConfig::default();
        assert_eq!(c.entries, 4096);
        assert!(!c.iopoll);
        assert!(c.sqpoll.is_none());
        c.check().unwrap();
    }

    #[test]
    fn a_shape_that_cannot_work_is_refused_with_a_reason() {
        let e = RingConfig::default().with_entries(0).check().unwrap_err();
        assert_eq!(e.code(), Code::Invalid);
        assert!(e.message().contains("no entries"), "{e}");

        let e = RingConfig::default()
            .with_entries(MAX_SLOT + 2)
            .check()
            .unwrap_err();
        assert_eq!(e.code(), Code::Invalid);

        let e = RingConfig::default()
            .with_sqpoll(SqPoll {
                idle_ms: 0,
                cpu: None,
            })
            .check()
            .unwrap_err();
        assert_eq!(e.code(), Code::Invalid);
        assert!(e.message().contains("idle"), "{e}");
    }

    #[test]
    fn the_largest_ring_the_tag_can_address_is_allowed() {
        RingConfig::default()
            .with_entries(MAX_SLOT + 1)
            .check()
            .unwrap();
    }

    #[test]
    fn the_builders_leave_everything_else_alone() {
        let c = RingConfig::default()
            .with_entries(64)
            .with_iopoll(true)
            .with_sqpoll(SqPoll {
                idle_ms: 50,
                cpu: Some(3),
            });
        assert_eq!(c.entries, 64);
        assert!(c.iopoll);
        assert_eq!(c.sqpoll.unwrap().cpu, Some(3));
        assert!(!c.registered_buffers);
        c.check().unwrap();
    }
}
