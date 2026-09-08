//! `AUTH`, the one password behind it, and the gate every other command passes.
//!
//! # One password and no users
//!
//! A real server has an ACL system with users, rules and per command
//! permissions, and `requirepass` is a thin layer over it: setting it gives the
//! user called `default` that password, and clearing it puts the `nopass` flag
//! back. There is no ACL here yet, so this file is the other half of that
//! sentence written on its own. There is one password, it belongs to a user
//! called `default`, and any other user name is refused the way a real server
//! refuses a name it has never heard of.
//!
//! That means every message a client can see is the message a real server
//! sends, and the day the ACL arrives this becomes what it already looks like:
//! the default user's password.
//!
//! # Who starts out let in
//!
//! A connection carries a flag saying whether it has authenticated, and the
//! flag is decided when the connection is accepted rather than when it sends
//! its first command. A connection accepted while no password is set is let in
//! at once, because the default user is `nopass` and there is nothing to ask
//! it for, and it stays let in if a password is set later. A connection
//! accepted while a password is set has to send `AUTH` first.
//!
//! That is a real server's rule and it is worth being clear about, because it
//! is not the rule anybody would guess. `CONFIG SET requirepass` does not lock
//! out the clients that are already connected, including the one that just set
//! it, and it does lock out every client that connects after it.
//!
//! `RESET` puts the connection back to how it was accepted, and that includes
//! this: a connection that authenticated and then sent `RESET` has to
//! authenticate again on a server with a password, and does not on a server
//! without one.
//!
//! # Why the compare is written out
//!
//! Comparing two passwords with `==` gives away how much of the guess was right
//! by how long the compare took, and a wrong guess that took longer is a wrong
//! guess that got further. So the compare here looks at every byte of both
//! whatever it finds, and folds the lengths in rather than returning early on
//! them, which is what a real server does with the hashes it keeps.
//!
//! What is not done here is the hashing. A real server keeps a SHA-256 of the
//! password and never the password, and the reason that matters is `ACL
//! GETUSER` and the config file rewrite, neither of which exists yet. It is
//! written down in D-127 rather than half done.

use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::{Acquire, Release};

use yo_common::{Code, Error, Result};

use super::args::{self, Args, is};
use super::{Server, Session};
use crate::reply::Out;

/// The one line a command refused for want of a password is answered with.
///
/// The whole line and not the part after the code, because it goes two places:
/// straight into the reply, and spliced into the `EXECABORT` an `EXEC` gets, and
/// the reference puts the code in both.
pub(super) const NOAUTH: &str = "NOAUTH Authentication required.";

/// The line `HELLO` gets instead, which says what to do about it.
///
/// A client that speaks RESP3 has to send `HELLO` before it can send `AUTH`, or
/// it would be speaking RESP2 by the time it authenticated, so the reference
/// spends a sentence here pointing at the option that solves it.
pub(super) const HELLO_NOAUTH: &str = "NOAUTH HELLO must be called with the client already authenticated, otherwise the HELLO <proto> AUTH <user> <pass> option can be used to authenticate the client and select the RESP protocol version at the same time";

/// The only user there is.
const DEFAULT_USER: &[u8] = b"default";

/// The password the default user has, if it has one.
///
/// The flag is separate from the password rather than read out of it because
/// every command on the server asks the question once, and on nearly every
/// server the answer is that there is no password. That is one relaxed load
/// against a word that is already warm, instead of a lock.
#[derive(Debug, Default)]
pub(crate) struct Access {
    /// Whether a password is set at all, which is the hot half.
    on: AtomicBool,
    /// The password itself, which only `AUTH` and `CONFIG` ever look at.
    secret: Mutex<Vec<u8>>,
}

impl Server {
    /// Whether this server asks connections for a password.
    #[must_use]
    pub(crate) fn guarded(&self) -> bool {
        self.access.on.load(Acquire)
    }

    /// Set or clear the password, where an empty one clears it.
    ///
    /// The flag is written after the password on the way in and before it on
    /// the way out, so a thread that sees the flag on always sees the password
    /// that goes with it, and a thread that catches the clear half done reads
    /// the flag as still on and asks for a password that is about to stop being
    /// needed. Being asked once more than necessary is the safe half of that
    /// race and the other order does not have a safe half.
    pub fn set_password(&self, password: &[u8]) {
        let mut held = self.access.secret.lock().unwrap_or_else(|e| e.into_inner());
        if password.is_empty() {
            self.access.on.store(false, Release);
            held.clear();
            return;
        }
        held.clear();
        held.extend_from_slice(password);
        self.access.on.store(true, Release);
    }

    /// Hand the password to `f`, which is how `CONFIG GET` writes it out.
    ///
    /// Borrowed rather than copied, because the one caller writes it straight
    /// into a reply buffer and a copy would be a second place a password lives.
    pub(crate) fn with_password<T>(&self, f: impl FnOnce(&[u8]) -> T) -> T {
        let held = self.access.secret.lock().unwrap_or_else(|e| e.into_inner());
        f(&held)
    }

    /// Whether `guess` is the password, compared without giving away how much
    /// of it was right.
    fn password_is(&self, guess: &[u8]) -> bool {
        self.with_password(|real| same(real, guess))
    }
}

/// Whether two byte strings are equal, in time that does not depend on where
/// they stop being equal.
///
/// The loop runs over the longer of the two and folds a byte that is not there
/// in as a difference, so neither the contents nor the length is readable from
/// how long this took. Returning early on the length would give away the length,
/// which on a password is a real thing to give away.
fn same(a: &[u8], b: &[u8]) -> bool {
    let mut diff = u8::from(a.len() != b.len());
    for i in 0..a.len().max(b.len()) {
        diff |= a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0xff);
    }
    diff == 0
}

/// Whether this user and password get in, without saying anything to the client.
///
/// Shared by `AUTH` and by `HELLO`'s `AUTH` option, which take the same pair and
/// make the same decision about it. The one thing they do not share is what an
/// unguarded server says to a one argument `AUTH`, which is `AUTH`'s own problem
/// because `HELLO` has no one argument form.
pub(super) fn admits(server: &Server, user: &[u8], password: &[u8]) -> bool {
    if !is(user, DEFAULT_USER) {
        // Nobody else exists, on a server with a password and on a server
        // without one. A real server answers the same thing for a user it has
        // never heard of and a user whose password was wrong, on purpose: the
        // difference between the two is a list of user names.
        return false;
    }
    // No password set means the default user is `nopass`, and a `nopass` user
    // takes any password at all rather than refusing every one of them.
    !server.guarded() || server.password_is(password)
}

/// `AUTH password` or `AUTH username password`.
pub(super) fn execute(
    server: &Server,
    session: &mut Session,
    args: Args<'_>,
    out: &mut Out,
) -> Result<()> {
    // Arity is a minimum of two, so a third argument is where this stops being
    // a command it knows and the reference calls that a syntax error rather
    // than a wrong number of arguments.
    if args.len() > 3 {
        return Err(args::syntax());
    }
    let (user, password) = if args.len() == 3 {
        (args.get(1), args.get(2))
    } else {
        (DEFAULT_USER, args.get(1))
    };
    if args.len() == 2 && !server.guarded() && is(user, DEFAULT_USER) {
        // The one message here that is not about the password being wrong. A
        // client that sends a one argument `AUTH` to a server with no password
        // has almost certainly connected to the wrong server, so the reference
        // says so at length rather than letting it through.
        return Err(Error::new(
            Code::Invalid,
            "AUTH <password> called without any password configured for the default user. Are you sure your configuration is correct?",
        ));
    }
    if !admits(server, user, password) {
        // Written straight into the buffer, because the code in front of it is
        // what a client branches on and this is the only place in the engine
        // that sends it. The same reason `NOPROTO` is written where it is
        // decided.
        out.error(b"WRONGPASS invalid username-password pair or user is disabled.");
        return Ok(());
    }
    session.admit(true);
    out.ok();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::same;

    /// The compare says the same thing `==` says, for everything `==` is asked.
    ///
    /// Constant time is the point of it and constant time is not a thing a test
    /// can assert without a clock and a lot of runs, so what is pinned here is
    /// that being careful about the timing did not change any answer. The empty
    /// pair is in because the sentinel the loop folds in for a byte that is not
    /// there has to differ from the one on the other side, and two empty strings
    /// are the case that catches getting that backwards.
    #[test]
    fn the_careful_compare_agrees_with_the_ordinary_one() {
        let words: [&[u8]; 9] = [
            b"",
            b"a",
            b"b",
            b"ab",
            b"ba",
            b"hunter2",
            b"hunter3",
            b"hunter22",
            b"\x00",
        ];
        for a in words {
            for b in words {
                assert_eq!(same(a, b), a == b, "{a:?} against {b:?}");
            }
        }
    }
}
