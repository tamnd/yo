//! What the codec refuses, and the exact words it refuses it in.
//!
//! The messages are Redis's messages, character for character. A client that
//! branches on the text of a protocol error is doing something questionable,
//! but the differential harness in `yo-compat` compares replies byte for byte,
//! and a protocol error is a reply. Anything invented here would be a
//! divergence that has to be registered, so nothing is invented here.

use core::fmt;
use yo_common::{Code, Error};

/// A frame the codec will not accept.
///
/// This is a value rather than a string because the connection has to do two
/// things with it: write the Redis text to the client, and close the
/// connection, which is what Redis does after any protocol error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ProtocolError {
    /// The `*` count was not an integer, was negative past what Redis allows,
    /// or was larger than the multibulk limit.
    InvalidMultibulkLength,
    /// The `$` length was not an integer, was negative, or was larger than the
    /// bulk limit.
    InvalidBulkLength,
    /// An argument did not begin with `$`. Carries the byte that was there
    /// instead, because that byte is in the message Redis sends.
    ExpectedDollar(u8),
    /// The `*` count line has not ended and there is already more pending than
    /// an inline request is allowed to be.
    TooBigMbulkCount,
    /// The `$` length line has not ended and there is already more pending than
    /// an inline request is allowed to be.
    TooBigBulkCount,
    /// An inline request has no newline and has passed the inline limit.
    TooBigInline,
    /// An inline request opened a quote it never closed.
    UnbalancedQuotes,
    /// A reply began with a byte that is not a type in either protocol.
    /// Carries the byte.
    UnknownType(u8),
    /// A reply nested deeper than the configured limit. This is only reachable
    /// from the reply decoder, which is the one part of the codec that recurses.
    TooDeep,
    /// A frame this build does not decode, currently the streamed aggregates
    /// and streamed strings. Carries the type byte.
    Unsupported(u8),
}

impl ProtocolError {
    /// Appends the error line the client should see, `-` and CRLF included.
    ///
    /// Written straight into the output buffer rather than returned as a
    /// `String`, because the failure path runs on a shard thread too and a
    /// shard thread that allocates aborts.
    pub fn write_reply(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(b"-ERR Protocol error: ");
        match *self {
            ProtocolError::InvalidMultibulkLength => {
                out.extend_from_slice(b"invalid multibulk length");
            }
            ProtocolError::InvalidBulkLength => out.extend_from_slice(b"invalid bulk length"),
            ProtocolError::ExpectedDollar(got) => {
                out.extend_from_slice(b"expected '$', got '");
                // Redis prints the offending byte with `%c` and then flattens
                // newlines to spaces, because a newline inside an error line
                // would end the line early and desynchronise the client.
                out.push(if got == b'\r' || got == b'\n' {
                    b' '
                } else {
                    got
                });
                out.push(b'\'');
            }
            ProtocolError::TooBigMbulkCount => {
                out.extend_from_slice(b"too big mbulk count string");
            }
            ProtocolError::TooBigBulkCount => out.extend_from_slice(b"too big bulk count string"),
            ProtocolError::TooBigInline => out.extend_from_slice(b"too big inline request"),
            ProtocolError::UnbalancedQuotes => {
                out.extend_from_slice(b"unbalanced quotes in request");
            }
            ProtocolError::UnknownType(got) => {
                out.extend_from_slice(b"unknown type byte '");
                out.push(if got == b'\r' || got == b'\n' {
                    b' '
                } else {
                    got
                });
                out.push(b'\'');
            }
            ProtocolError::TooDeep => out.extend_from_slice(b"nesting too deep"),
            ProtocolError::Unsupported(got) => {
                out.extend_from_slice(b"unsupported type byte '");
                out.push(if got == b'\r' || got == b'\n' {
                    b' '
                } else {
                    got
                });
                out.push(b'\'');
            }
        }
        out.extend_from_slice(b"\r\n");
    }
}

impl fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut line = Vec::new();
        self.write_reply(&mut line);
        // Strip the `-ERR ` that belongs to the wire and the trailing CRLF that
        // belongs to the frame. What is left is the sentence.
        let body = &line[5..line.len() - 2];
        f.write_str(&String::from_utf8_lossy(body))
    }
}

impl core::error::Error for ProtocolError {}

impl From<ProtocolError> for Error {
    /// A protocol error reaching the typed API is an invalid argument, because
    /// on that side of the boundary the caller handed us the bytes.
    fn from(e: ProtocolError) -> Error {
        Error::new(Code::Invalid, e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reply(e: ProtocolError) -> String {
        let mut v = Vec::new();
        e.write_reply(&mut v);
        String::from_utf8(v).unwrap()
    }

    #[test]
    fn the_messages_are_the_ones_redis_sends() {
        assert_eq!(
            reply(ProtocolError::InvalidMultibulkLength),
            "-ERR Protocol error: invalid multibulk length\r\n"
        );
        assert_eq!(
            reply(ProtocolError::InvalidBulkLength),
            "-ERR Protocol error: invalid bulk length\r\n"
        );
        assert_eq!(
            reply(ProtocolError::ExpectedDollar(b'x')),
            "-ERR Protocol error: expected '$', got 'x'\r\n"
        );
        assert_eq!(
            reply(ProtocolError::TooBigMbulkCount),
            "-ERR Protocol error: too big mbulk count string\r\n"
        );
        assert_eq!(
            reply(ProtocolError::TooBigBulkCount),
            "-ERR Protocol error: too big bulk count string\r\n"
        );
        assert_eq!(
            reply(ProtocolError::TooBigInline),
            "-ERR Protocol error: too big inline request\r\n"
        );
        assert_eq!(
            reply(ProtocolError::UnbalancedQuotes),
            "-ERR Protocol error: unbalanced quotes in request\r\n"
        );
    }

    /// The offending byte is printed, and a newline in it must not end the
    /// line, because a client reading a short line then reads the rest of the
    /// error as its next reply and every reply after that is off by one.
    #[test]
    fn a_newline_in_the_offending_byte_does_not_end_the_line() {
        let r = reply(ProtocolError::ExpectedDollar(b'\n'));
        assert_eq!(r, "-ERR Protocol error: expected '$', got ' '\r\n");
        assert_eq!(r.matches("\r\n").count(), 1);
    }

    #[test]
    fn it_carries_into_the_typed_api_as_an_invalid_argument() {
        let e: Error = ProtocolError::InvalidBulkLength.into();
        assert_eq!(e.code(), Code::Invalid);
        assert_eq!(e.message(), "Protocol error: invalid bulk length");
    }
}
