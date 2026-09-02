//! The flag a shutdown signal sets, and the handler that sets it.
//!
//! `yodb serve` turns its loop against an [`AtomicBool`] and always has. Nothing
//! set it, so the only way to stop the server was to kill it, which on Unix
//! leaves the socket file behind for the next start to clear up and on every
//! platform means the loop never gets to run its own way out. That was fine
//! while there was nothing to run out of, and it stops being fine the moment
//! there is a file to close: the same flag that ends the loop here is what a
//! flush on the way out will hang off in M5.
//!
//! # What a handler is allowed to do
//!
//! One store to a static, and nothing else. A signal arrives on whatever thread
//! the kernel picks and interrupts it wherever it happens to be, so a handler
//! that allocates can deadlock against an allocator the interrupted thread was
//! already inside. A relaxed store to an `AtomicBool` is one instruction and is
//! on the short list POSIX says is safe from a handler.
//!
//! The exception is the second signal. Somebody pressing Ctrl-C twice is saying
//! the polite way is taking too long, and answering that means leaving without
//! unwinding, so it is `_exit` and not `exit`: the first is on the safe list and
//! the second runs destructors that the interrupted thread may be halfway
//! through already.
//!
//! # Which signals
//!
//! `SIGINT` and `SIGTERM` on Unix, which are Ctrl-C and what a service manager
//! sends. Not `SIGHUP`, because Redis ignores it and a server that dies when the
//! terminal it was started from closes is a surprise nobody asked for.
//!
//! On Windows the console handler covers Ctrl-C, Ctrl-Break, the window being
//! closed and the machine shutting down. The last two come with a deadline of a
//! few seconds before the process is killed anyway, which is the same promise as
//! `SIGTERM` with a service manager's timer behind it.

use std::sync::atomic::{AtomicBool, Ordering};

/// Set once a shutdown signal has arrived.
///
/// A static rather than something handed around, because a signal handler takes
/// no argument and cannot be a closure over anything.
static STOP: AtomicBool = AtomicBool::new(false);

/// The flag the serve loop turns against.
#[must_use]
pub fn stop() -> &'static AtomicBool {
    &STOP
}

/// Whether a signal is what ended the loop, as opposed to an error.
#[must_use]
pub fn stopped() -> bool {
    STOP.load(Ordering::Relaxed)
}

/// Ask for [`stop`] to be set when the operating system says to stop.
///
/// Called once, before the loop starts. A second call installs the same handler
/// again, which is harmless and is not something any caller has a reason to do.
pub fn listen() {
    imp::listen();
}

#[cfg(unix)]
mod imp {
    use super::{Ordering, STOP};

    /// The handler, for both signals.
    ///
    /// The second one does not wait for the loop. A `SIGTERM` from a service
    /// manager is usually followed by a `SIGKILL` on a timer, and a second
    /// Ctrl-C is a person saying the same thing, so both mean stop now.
    extern "C" fn on_signal(_sig: libc::c_int) {
        if STOP.swap(true, Ordering::Relaxed) {
            // SAFETY: `_exit` is on POSIX's list of calls that are safe from a
            // signal handler, which `exit` is not, because `exit` runs
            // destructors belonging to a thread this handler interrupted.
            unsafe { libc::_exit(1) };
        }
    }

    pub fn listen() {
        for sig in [libc::SIGINT, libc::SIGTERM] {
            // SAFETY: installing a handler is the documented use of `signal`,
            // and the handler itself does one atomic store. The two step cast
            // is what `signal` asks for: the argument is an integer wide enough
            // to hold a function pointer, and going straight there from a
            // function item is a lint because it looks like arithmetic.
            unsafe {
                libc::signal(sig, on_signal as *const () as libc::sighandler_t);
            }
        }
    }
}

#[cfg(windows)]
mod imp {
    use super::{Ordering, STOP};
    use windows_sys::Win32::System::Console::SetConsoleCtrlHandler;

    /// The console handler, for every event that means the server is finishing.
    ///
    /// Windows runs this on a thread of its own rather than interrupting one, so
    /// the rules are looser than a Unix handler's. It does the same thing
    /// anyway, because there is nothing else it needs to do.
    ///
    /// A true answer says the event was handled, which is what stops the
    /// default handler ending the process before the loop has noticed.
    unsafe extern "system" fn on_ctrl(_event: u32) -> i32 {
        STOP.store(true, Ordering::Relaxed);
        1
    }

    pub fn listen() {
        // SAFETY: the handler is a plain function with the signature Windows
        // asks for, and a true second argument means add rather than remove.
        unsafe {
            SetConsoleCtrlHandler(Some(on_ctrl), 1);
        }
    }
}

/// Nowhere to hang a handler, so the flag is never set and the loop runs until
/// something kills it, which is where every platform was before this.
#[cfg(not(any(unix, windows)))]
mod imp {
    pub fn listen() {}
}

// Everything in here raises a signal and looks at what the handler did, so it
// is unix only, and the cfg goes on the module rather than on the test. On the
// test it leaves `use super::*` behind with nothing using it, which is a warning
// on Windows and a warning is an error in CI.
#[cfg(all(test, unix))]
mod tests {
    use super::*;

    /// The flag starts clear, so a server that is never signalled runs.
    ///
    /// In the same test as the one below rather than on its own, because both
    /// read one process wide static and two tests reading it in an order nobody
    /// chose is a flake waiting to happen.
    #[test]
    fn a_term_sets_the_flag_and_nothing_else_does() {
        assert!(!stopped(), "nothing has been signalled yet");
        listen();
        assert!(!stopped(), "installing a handler is not a signal");

        // SAFETY: this sends the signal to this process, where the handler
        // installed just above is the one that takes it. One raise and not two,
        // because the second would be the handler's own way out.
        unsafe {
            libc::raise(libc::SIGTERM);
        }
        assert!(stopped(), "the handler ran and set the flag");

        // Put it back, so nothing else in this binary starts life stopped.
        STOP.store(false, Ordering::Relaxed);
    }
}
