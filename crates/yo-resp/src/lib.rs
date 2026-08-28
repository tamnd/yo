//! The RESP2 and RESP3 codec.
//!
//! Requests come in as ranges into the connection's own read buffer, and
//! replies go out as the bytes that go on the socket. Nothing in between is
//! materialised, because everything in between is what the lineage's profiles
//! kept finding at the top.
//!
//! # Reading
//!
//! [`Argv`] decodes commands. It is per connection, it remembers where it got
//! to when a command arrives in pieces, and after the first few commands it
//! stops allocating. Multibulk and inline requests both land in the same place,
//! so the command layer never learns which one a client used.
//!
//! ```
//! use yo_resp::{Argv, Limits, Step};
//!
//! let buf = b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\nv\r\n";
//! let mut argv = Argv::new();
//! match argv.decode(buf, &Limits::default())? {
//!     Step::Command { consumed } => {
//!         assert_eq!(consumed, buf.len());
//!         assert_eq!(argv.arg(buf, 0), Some(&b"SET"[..]));
//!         assert_eq!(argv.arg(buf, 2), Some(&b"v"[..]));
//!     }
//!     Step::Incomplete => unreachable!("the whole command is here"),
//! }
//! # Ok::<(), yo_resp::ProtocolError>(())
//! ```
//!
//! # Writing
//!
//! [`Out`] is the reply buffer, and it knows which protocol the connection is
//! speaking. A command writes the richer form once and the RESP2 downgrade
//! happens here rather than in the command:
//!
//! ```
//! use yo_resp::{Out, Proto};
//!
//! fn hgetall(out: &mut Out) {
//!     out.map(1);
//!     out.bulk(b"field");
//!     out.bulk(b"value");
//! }
//!
//! let mut two = Out::new(Proto::Resp2);
//! hgetall(&mut two);
//! assert_eq!(two.as_slice(), b"*2\r\n$5\r\nfield\r\n$5\r\nvalue\r\n");
//!
//! let mut three = Out::new(Proto::Resp3);
//! hgetall(&mut three);
//! assert_eq!(three.as_slice(), b"%1\r\n$5\r\nfield\r\n$5\r\nvalue\r\n");
//! ```
//!
//! # Reading replies
//!
//! [`frame`] decodes a reply into a borrowed [`Frame`]. The server has no use
//! for it. The replication client, the differential harness and this crate's
//! own round trip tests do.
//!
//! # Running a command
//!
//! [`dispatch`] is the layer above both halves. It looks a command name up,
//! checks its arity, and calls the same `yo-kv` method the embedded API calls,
//! which is the placement rule Y23 is about: one implementation of `INCR`, two
//! ways to reach it.
//!
//! ```
//! use yo_resp::{Argv, Limits, Out, Proto};
//! use yo_resp::dispatch::{Args, Server, Session, execute};
//!
//! let mut server = Server::new();
//! let mut session = Session::new(1);
//! let mut out = Out::new(Proto::Resp2);
//! let wire = b"*1\r\n$4\r\nPING\r\n";
//! let mut argv = Argv::new();
//! argv.decode(wire, &Limits::default())?;
//! execute(&mut server, &mut session, Args::new(&argv, wire), &mut out);
//! assert_eq!(out.as_slice(), b"+PONG\r\n");
//! # Ok::<(), yo_resp::ProtocolError>(())
//! ```
//!
//! # Driving it from the loop
//!
//! [`engine`] is the piece between the two: connections, read buffers, framing
//! and one write per connection per batch, put on `yo_reactor::Engine` so the
//! loop can run commands without knowing what a command is. It is where a
//! server becomes possible, and it works over anything that implements
//! [`engine::Sink`], which is a socket in production and a `Vec` in a test.
//!
//! ```
//! use yo_reactor::Reactor;
//! use yo_resp::engine::{Recorder, Wire, pump};
//!
//! let mut r = Reactor::inline(Wire::new(Recorder::new()));
//! let conn = r.engine_mut().accept();
//!
//! r.engine_mut().feed(conn, b"*1\r\n$4\r\nPING\r\n");
//! pump(&mut r, &mut Vec::new());
//! assert_eq!(r.engine().sink().sent(conn), b"+PONG\r\n");
//! ```
//!
//! # What is not here
//!
//! Sockets. This crate turns bytes into arguments, runs them, turns values into
//! bytes, and says which connection they belong to. Reading and writing the
//! bytes themselves is the ring's job, and `04` section 7 owns the ring.

#![deny(missing_docs)]

pub mod dispatch;
pub mod engine;
pub mod error;
pub mod frame;
pub mod proto;
pub mod reply;
pub mod request;

pub use engine::{Cmd, ConnId, Sink, Wire};
pub use error::ProtocolError;
pub use frame::Frame;
pub use proto::{Limits, Proto};
pub use reply::Out;
pub use request::{Argv, Step};
/// Redis's own integer and double text, shared with the string type.
///
/// This module lives in `yo-common` because the codec is not the only thing
/// that needs it: whether a string is stored int encoded is decided by the same
/// `string2ll` rules that decide whether a bulk length parses. Re-exported here
/// so that `yo_resp::num` keeps meaning what it meant.
pub use yo_common::num;

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape a connection actually runs: read some bytes, decode what is
    /// whole, reply to each, keep the remainder. Written here rather than in
    /// either half because it is the only test that exercises both against each
    /// other in the order the reactor will.
    #[test]
    fn a_connection_reads_commands_and_writes_replies() {
        // Two whole commands and the front of a third, which is what a read
        // that lands in the middle of a pipeline looks like.
        let wire = b"*1\r\n$4\r\nPING\r\n*2\r\n$3\r\nGET\r\n$7\r\nmissing\r\n*2\r\n$3\r\nGE";

        let mut argv = Argv::new();
        let mut out = Out::new(Proto::Resp2);
        let mut at = 0;
        loop {
            match argv.decode(&wire[at..], &Limits::default()).unwrap() {
                Step::Incomplete => break,
                Step::Command { consumed } => {
                    let buf = &wire[at..];
                    match argv.arg(buf, 0) {
                        Some(b"PING") => out.simple(b"PONG"),
                        Some(b"GET") => out.nil(),
                        _ => out.error(b"ERR unknown command"),
                    }
                    at += consumed;
                }
            }
        }

        assert_eq!(out.as_slice(), b"+PONG\r\n$-1\r\n");
        // The partial third command is still waiting, and it is waiting at the
        // right place: everything before it has been accounted for.
        assert_eq!(&wire[at..], b"*2\r\n$3\r\nGE");
    }

    /// The same exchange on RESP3, where only the null is spelled differently.
    /// The command bodies above did not change and that is the point.
    #[test]
    fn the_same_replies_come_out_in_resp3_spelling() {
        let mut out = Out::new(Proto::Resp3);
        out.simple(b"PONG");
        out.nil();
        assert_eq!(out.as_slice(), b"+PONG\r\n_\r\n");
    }
}
