//! The arguments a command body sees, and the errors it raises about them.
//!
//! A command never gets a `Vec<Vec<u8>>`. It gets this, which is the decoder's
//! ranges and the connection's read buffer travelling together, so an argument
//! is a slice of the bytes that came off the socket and nothing is copied
//! between the two. That is the whole reason `Argv` records ranges instead of
//! building strings, and a dispatcher that immediately materialised them would
//! have thrown the saving away at the first opportunity.
//!
//! The error helpers are here because they are the same handful of sentences
//! over and over, and because they have to be Redis's sentences exactly. A
//! client that matches on the text of `ERR value is not an integer or out of
//! range` is doing something ugly, and it is doing it against every Redis
//! deployment in the world, so the text is part of the contract.

use crate::request::Argv;
use yo_common::num::{parse_f64, parse_i64};
use yo_common::{Code, Error, Result};

/// What Redis says when an argument should have been an integer and was not.
pub const NOT_AN_INT: &str = "value is not an integer or out of range";
/// What Redis says when an argument should have been a float and was not.
pub const NOT_A_FLOAT: &str = "value is not a valid float";
/// What Redis says about an option it did not expect where it found it.
pub const SYNTAX: &str = "syntax error";

/// One command's arguments, borrowed from the connection's read buffer.
///
/// Index zero is the command name, the same as Redis's `argv`, so an argument
/// index in this file matches the argument index in Redis's source and in its
/// error messages.
#[derive(Clone, Copy)]
pub struct Args<'a> {
    argv: &'a Argv,
    buf: &'a [u8],
}

impl<'a> Args<'a> {
    /// The arguments of the command `argv` last decoded out of `buf`.
    #[must_use]
    pub fn new(argv: &'a Argv, buf: &'a [u8]) -> Args<'a> {
        Args { argv, buf }
    }

    /// How many arguments there are, counting the command name.
    #[must_use]
    pub fn len(&self) -> usize {
        self.argv.len()
    }

    /// Whether there is not even a command name.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.argv.is_empty()
    }

    /// Argument `i`, or an empty slice past the end.
    ///
    /// Past the end is empty rather than a panic because every caller has
    /// already been through the arity check, so an index past the end is a bug
    /// in the arity table rather than something a client can cause, and a
    /// wrong answer in the reply is a better way to find that bug than a
    /// process that stops answering.
    #[must_use]
    pub fn get(&self, i: usize) -> &'a [u8] {
        self.argv.arg(self.buf, i).unwrap_or(b"")
    }

    /// Argument `i`, or `None` past the end.
    #[must_use]
    pub fn opt(&self, i: usize) -> Option<&'a [u8]> {
        self.argv.arg(self.buf, i)
    }

    /// The command name, which is argument zero.
    #[must_use]
    pub fn name(&self) -> &'a [u8] {
        self.get(0)
    }

    /// Argument `i` as an integer, with Redis's message when it is not one.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] when the argument is not an integer in `i64`.
    pub fn int(&self, i: usize) -> Result<i64> {
        parse_i64(self.get(i)).ok_or_else(|| Error::new(Code::Invalid, NOT_AN_INT).at(i as u32))
    }

    /// Argument `i` as a float, with Redis's message when it is not one.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] when the argument is not a float, which includes NaN.
    pub fn float(&self, i: usize) -> Result<f64> {
        parse_f64(self.get(i)).ok_or_else(|| Error::new(Code::Invalid, NOT_A_FLOAT).at(i as u32))
    }
}

/// Whether an argument is this keyword, ignoring case the way Redis does.
///
/// Option keywords are matched case insensitively and command names are too,
/// so `set k v nx` works and always has. This compares in place rather than
/// upper casing into a buffer, because the buffer would be the only allocation
/// on the whole dispatch path.
#[must_use]
pub fn is(arg: &[u8], keyword: &[u8]) -> bool {
    arg.len() == keyword.len() && arg.eq_ignore_ascii_case(keyword)
}

/// `ERR syntax error`, which is what an option in the wrong place gets.
#[must_use]
pub fn syntax() -> Error {
    Error::new(Code::Invalid, SYNTAX)
}

/// `ERR wrong number of arguments for 'x' command`.
///
/// The name is lower cased into the message because Redis reports the name
/// from its own table and not the spelling the client used, so `GET` and `get`
/// produce the same sentence.
#[must_use]
pub fn wrong_arity(name: &str) -> Error {
    Error::fmt(
        Code::Invalid,
        format_args!("wrong number of arguments for '{name}' command"),
    )
}

/// `ERR wrong number of arguments for 'config|get' command`.
///
/// A subcommand is reported with the container in front of it and a bar
/// between, which is how Redis names them everywhere including in `COMMAND
/// INFO`.
#[must_use]
pub fn wrong_arity_sub(name: &str, sub: &str) -> Error {
    Error::fmt(
        Code::Invalid,
        format_args!("wrong number of arguments for '{name}|{sub}' command"),
    )
}

/// `ERR unknown subcommand 'X'. Try CONFIG HELP.`
///
/// The subcommand is quoted exactly as the client spelled it, which is what a
/// real server does, so `config nosuch` and `config NOSUCH` produce different
/// sentences. The container is upper case because that is how Redis writes it
/// in this message and lower case everywhere else.
#[must_use]
pub fn unknown_subcommand(sub: &[u8], container: &str) -> Error {
    yo_alloc::allow(|| {
        Error::fmt(
            Code::Unsupported,
            format_args!(
                "unknown subcommand '{}'. Try {} HELP.",
                String::from_utf8_lossy(sub),
                container
            ),
        )
    })
}

/// `ERR invalid expire time in 'x' command`.
#[must_use]
pub fn invalid_expire(name: &str) -> Error {
    Error::fmt(
        Code::Invalid,
        format_args!("invalid expire time in '{name}' command"),
    )
}

/// `ERR unknown command 'X', with args beginning with: 'a' 'b' `.
///
/// With no arguments at all the sentence stops early, at `unknown command 'X'`,
/// and that is a real difference and not a tidier way of saying the same thing.
/// Redis 8.10.1 answers a bare `NOTACOMMAND` without the second clause, and we
/// were sending it with an empty list hanging off the end.
///
/// Redis quotes each argument, separates them with a space and leaves the
/// trailing one, which looks like an oversight and is not worth diverging
/// over. It stops once the argument list reaches 128 bytes rather than echoing
/// a megabyte back at a client that sent one, and it truncates the argument
/// that crosses the line rather than dropping it, and so does this.
///
/// An argument that is not UTF-8 comes back with the replacement character
/// where Redis would send the raw bytes, because an error carries a `String`.
/// It is inside the message for a command that does not exist, so nothing can
/// be depending on it.
///
/// The whole body is inside [`yo_alloc::allow`] rather than only the final
/// constructor, because the message is built before the constructor sees it
/// and a shard thread that allocates aborts.
#[must_use]
pub fn unknown_command(args: Args<'_>) -> Error {
    yo_alloc::allow(|| {
        let mut msg = String::from("unknown command '");
        msg.push_str(&String::from_utf8_lossy(args.name()));
        msg.push('\'');
        if args.len() == 1 {
            return Error::new(Code::Unsupported, msg);
        }
        msg.push_str(", with args beginning with: ");
        let start = msg.len();
        for i in 1..args.len() {
            let used = msg.len() - start;
            if used >= 128 {
                break;
            }
            let arg = args.get(i);
            let arg = &arg[..arg.len().min(128 - used)];
            msg.push('\'');
            msg.push_str(&String::from_utf8_lossy(arg));
            msg.push_str("' ");
        }
        Error::new(Code::Unsupported, msg)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::tests::encode;
    use crate::proto::Limits;
    use crate::request::Step;

    #[test]
    fn arguments_are_slices_of_the_read_buffer() {
        let wire = encode(&[b"SET", b"k", b"v"]);
        let mut argv = Argv::new();
        assert!(matches!(
            argv.decode(&wire, &Limits::default()).unwrap(),
            Step::Command { .. }
        ));
        let args = Args::new(&argv, &wire);
        assert_eq!(args.len(), 3);
        assert_eq!(args.name(), b"SET");
        assert_eq!(args.get(2), b"v");
        // Past the end is empty and not a panic.
        assert_eq!(args.get(9), b"");
        assert_eq!(args.opt(9), None);
    }

    #[test]
    fn keywords_match_whatever_case_the_client_used() {
        assert!(is(b"nx", b"NX"));
        assert!(is(b"Nx", b"NX"));
        assert!(!is(b"nxx", b"NX"));
        assert!(!is(b"n", b"NX"));
    }

    #[test]
    fn the_unknown_command_message_is_redis_own() {
        let wire = encode(&[b"NOPE", b"a", b"b"]);
        let mut argv = Argv::new();
        argv.decode(&wire, &Limits::default()).unwrap();
        let e = unknown_command(Args::new(&argv, &wire));
        // Checked against a real 8.8, trailing space included.
        assert_eq!(
            e.message(),
            "unknown command 'NOPE', with args beginning with: 'a' 'b' "
        );
    }

    #[test]
    fn a_command_with_no_arguments_gets_the_short_sentence() {
        let wire = encode(&[b"NOPE"]);
        let mut argv = Argv::new();
        argv.decode(&wire, &Limits::default()).unwrap();
        let e = unknown_command(Args::new(&argv, &wire));
        // Checked against a real 8.10.1. The second clause is not there at all,
        // rather than being there with nothing after it.
        assert_eq!(e.message(), "unknown command 'NOPE'");
    }

    #[test]
    fn a_client_that_sends_a_megabyte_does_not_get_it_back() {
        let big = vec![b'x'; 1024];
        let wire = encode(&[b"NOPE", &big, &big]);
        let mut argv = Argv::new();
        argv.decode(&wire, &Limits::default()).unwrap();
        let e = unknown_command(Args::new(&argv, &wire));
        // The argument list is 128 bytes of `x` plus the two quotes and the
        // space Redis puts around it, and the second argument never starts.
        assert_eq!(
            e.message().len(),
            "unknown command 'NOPE', with args beginning with: ".len() + 131
        );
    }
}
