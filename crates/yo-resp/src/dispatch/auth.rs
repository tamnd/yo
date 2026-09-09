//! `AUTH`, and the two lines a command refused for want of it is answered with.
//!
//! The passwords themselves are not here. They belong to users and users belong
//! to the `acl` module, which is where `requirepass` lives too: setting it is a
//! way of writing one rule on the user called `default`. What is left here is
//! the command, because `AUTH` is a command like any other and the rest of the
//! ACL is not.
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
//! without one. It also puts the connection back on the default user, whatever
//! it had authenticated as.

use yo_common::{Code, Error, Result};

use super::acl;
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
        (acl::DEFAULT, args.get(1))
    };
    if args.len() == 2 && !server.guarded() && is(user, acl::DEFAULT) {
        // The one message here that is not about the password being wrong. A
        // client that sends a one argument `AUTH` to a server with no password
        // has almost certainly connected to the wrong server, so the reference
        // says so at length rather than letting it through.
        return Err(Error::new(
            Code::Invalid,
            "AUTH <password> called without any password configured for the default user. Are you sure your configuration is correct?",
        ));
    }
    if !acl::authenticate(server, session, user, password, args, out) {
        // Written straight into the buffer, because the code in front of it is
        // what a client branches on and this is the only place in the engine
        // that sends it. The same reason `NOPROTO` is written where it is
        // decided.
        out.error(b"WRONGPASS invalid username-password pair or user is disabled.");
        return Ok(());
    }
    out.ok();
    Ok(())
}
