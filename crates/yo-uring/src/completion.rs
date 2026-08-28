//! What comes back, and what it cost to get there.

use core::fmt;

use yo_common::{Code, Error};

use crate::token::{Kind, Token};

/// One finished submission.
///
/// The two fields are everything io_uring hands back, and the portable backend
/// produces the same two so that a caller written against one works against the
/// other without a `cfg` in it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Completion {
    /// The tag the submission carried.
    pub token: Token,
    /// The syscall's return value, or the negated `errno` when it failed. This
    /// is the kernel's convention and it is kept rather than translated,
    /// because a short write is a positive number that is not the length asked
    /// for and turning that into a `Result` at this layer would lose it.
    pub result: i32,
}

impl Completion {
    /// A completion for `token` that returned `result`.
    #[must_use]
    pub const fn new(token: Token, result: i32) -> Completion {
        Completion { token, result }
    }

    /// Which subsystem this belongs to, without a lookup.
    #[must_use]
    pub const fn kind(self) -> Kind {
        self.token.kind()
    }

    /// Whether the operation succeeded.
    #[must_use]
    pub const fn is_ok(self) -> bool {
        self.result >= 0
    }

    /// How many bytes moved, which is zero on failure and can be short on
    /// success.
    #[must_use]
    pub const fn bytes(self) -> u32 {
        if self.result > 0 {
            self.result as u32
        } else {
            0
        }
    }

    /// The failure, if there was one.
    ///
    /// `EAGAIN` and `EINTR` come back as [`Code::Busy`], which is retryable, and
    /// everything else as [`Code::Io`]. The `errno` is in the detail either way,
    /// because the number is the only thing that identifies which of forty
    /// possible failures this was.
    #[must_use]
    pub fn error(self) -> Option<Error> {
        if self.result >= 0 {
            return None;
        }
        let errno = -self.result;
        let code = if errno == libc_eagain() || errno == libc_eintr() {
            Code::Busy
        } else {
            Code::Io
        };
        Some(
            Error::new(code, "the submission failed")
                .with_detail(format!("kind={:?} errno={errno}", self.kind())),
        )
    }
}

#[cfg(unix)]
fn libc_eagain() -> i32 {
    libc::EAGAIN
}

#[cfg(unix)]
fn libc_eintr() -> i32 {
    libc::EINTR
}

// Windows has no `errno` for these, and the portable backend there reports the
// operating system error code as it stands. Nothing maps onto retryable, which
// is the honest answer rather than a guess.
#[cfg(not(unix))]
fn libc_eagain() -> i32 {
    i32::MIN
}

#[cfg(not(unix))]
fn libc_eintr() -> i32 {
    i32::MIN
}

impl fmt::Debug for Completion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Completion")
            .field("token", &self.token)
            .field("result", &self.result)
            .finish()
    }
}

/// What the ring has done since it was built.
///
/// Counters rather than timings. A timing here would be a timing of the wrong
/// thing, since the whole point of the ring is that the submission and the work
/// are not in the same place. What these are for is the assertion that the loop
/// costs one syscall a turn no matter how many submissions went into it, which
/// is `04` section 2 and is checked by a test rather than assumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Stats {
    /// Submissions accepted into the queue.
    pub submitted: u64,
    /// Completions handed back by a drain.
    pub completed: u64,
    /// Calls into the kernel to hand over the queue. Under SQPoll this stays at
    /// zero for as long as the kernel thread is awake, which is the whole
    /// reason SQPoll is on the ladder.
    pub enters: u64,
    /// Calls into the kernel that asked to be woken rather than returning at
    /// once.
    pub waits: u64,
    /// Submissions that were refused because the queue was full.
    pub refused: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_short_write_is_a_success_that_moved_fewer_bytes() {
        let c = Completion::new(Token::new(Kind::Write, 1, 1), 4096);
        assert!(c.is_ok());
        assert_eq!(c.bytes(), 4096);
        assert!(c.error().is_none());
    }

    #[test]
    fn a_failure_carries_the_errno_rather_than_swallowing_it() {
        let c = Completion::new(Token::new(Kind::Write, 1, 1), -5);
        assert!(!c.is_ok());
        assert_eq!(c.bytes(), 0);
        let e = c.error().unwrap();
        assert_eq!(e.code(), Code::Io);
        assert!(format!("{e}").contains("errno=5"), "{e}");
    }

    #[cfg(unix)]
    #[test]
    fn the_two_that_mean_try_again_say_so() {
        let c = Completion::new(Token::new(Kind::Recv, 0, 0), -libc::EAGAIN);
        assert_eq!(c.error().unwrap().code(), Code::Busy);
        let c = Completion::new(Token::new(Kind::Recv, 0, 0), -libc::EINTR);
        assert_eq!(c.error().unwrap().code(), Code::Busy);
    }

    #[test]
    fn the_kind_comes_off_the_completion_without_touching_the_table() {
        let c = Completion::new(Token::new(Kind::Fsync, 7, 3), 0);
        assert_eq!(c.kind(), Kind::Fsync);
        assert!(c.kind().is_storage());
    }
}
