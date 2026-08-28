//! Replies in: the general decoder, for the side of the wire that reads them.
//!
//! The server never parses a reply, so nothing here is on the engine's hot
//! path. It exists for three callers that all matter:
//!
//! - the differential harness in `yo-compat`, which sends the same command
//!   stream to `yo`, Redis and Valkey and compares what comes back;
//! - the replication client, which is a Redis client wearing a different hat;
//! - these tests, which is the only way to know that what [`crate::Out`] wrote
//!   is what a client would read back.
//!
//! Frames borrow. A bulk string is a slice of the buffer it arrived in. The
//! aggregates own a `Vec` of their children, which is an allocation per
//! aggregate and is the right trade here: the alternative is a cursor API that
//! every caller would have to drive by hand, and none of these callers is
//! counting nanoseconds.
//!
//! Streamed aggregates and streamed strings, RESP3's `?` and `;` forms, are not
//! decoded. Redis does not send them and no client asks for them. They return
//! [`ProtocolError::Unsupported`] rather than being silently mis-parsed.

use crate::error::ProtocolError;
use crate::proto::Limits;
use yo_common::num::parse_i64;

/// One reply, borrowed from the buffer it arrived in.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Frame<'a> {
    /// `+`, a one line string.
    Simple(&'a [u8]),
    /// `-`, a one line error, prefix included.
    Error(&'a [u8]),
    /// `!`, RESP3's error that may span lines.
    BlobError(&'a [u8]),
    /// `:`, an integer.
    Int(i64),
    /// `$`, a length prefixed string.
    Bulk(&'a [u8]),
    /// A missing value: RESP2's `$-1` or `*-1`, or RESP3's `_`.
    ///
    /// The two RESP2 spellings decode to the same thing on purpose. A caller
    /// that needs to tell them apart is testing the protocol rather than the
    /// reply, and [`crate::Out`] is where that is tested.
    Null,
    /// `,`, a double.
    Double(f64),
    /// `#`, a boolean.
    Bool(bool),
    /// `(`, an integer too large for `i64`, as its digits.
    BigNumber(&'a [u8]),
    /// `=`, a string with a three byte format tag.
    Verbatim {
        /// The tag, such as `txt` or `mkd`.
        format: &'a [u8],
        /// Everything after the colon.
        text: &'a [u8],
    },
    /// `*`, an ordered list.
    Array(Vec<Frame<'a>>),
    /// `%`, pairs in the order they were sent.
    Map(Vec<(Frame<'a>, Frame<'a>)>),
    /// `~`, an unordered list.
    Set(Vec<Frame<'a>>),
    /// `>`, an out of band message: pub/sub delivery or a cache invalidation.
    Push(Vec<Frame<'a>>),
    /// `|`, metadata that belongs to the frame after it.
    ///
    /// Returned on its own rather than attached, because attaching it would
    /// mean every caller matching on a reply has to look through a wrapper.
    /// A caller that cares reads the next frame; a caller that does not can
    /// drop it, which is what RESP3 says clients may do.
    Attribute(Vec<(Frame<'a>, Frame<'a>)>),
}

impl Frame<'_> {
    /// Whether this is an error of either kind.
    pub fn is_error(&self) -> bool {
        matches!(self, Frame::Error(_) | Frame::BlobError(_))
    }
}

/// Reads one frame from the front of `buf`.
///
/// Returns the frame and how many bytes it used, or `None` if the frame has not
/// fully arrived. Pipelined replies are read by calling this again on what is
/// left.
///
/// # Errors
///
/// Any [`ProtocolError`]. As on the request side, there is no recovering from
/// one: the two ends disagree about where the next frame starts.
pub fn decode<'a>(
    buf: &'a [u8],
    limits: &Limits,
) -> Result<Option<(Frame<'a>, usize)>, ProtocolError> {
    decode_at(buf, 0, limits, 0)
}

fn decode_at<'a>(
    buf: &'a [u8],
    at: usize,
    limits: &Limits,
    depth: usize,
) -> Result<Option<(Frame<'a>, usize)>, ProtocolError> {
    if depth > limits.max_depth {
        return Err(ProtocolError::TooDeep);
    }
    let Some(&kind) = buf.get(at) else {
        return Ok(None);
    };
    let Some((line, after)) = line_at(buf, at + 1) else {
        return Ok(None);
    };
    match kind {
        b'+' => Ok(Some((Frame::Simple(line), after))),
        b'-' => Ok(Some((Frame::Error(line), after))),
        b':' => {
            let n = parse_i64(line).ok_or(ProtocolError::InvalidBulkLength)?;
            Ok(Some((Frame::Int(n), after)))
        }
        b'_' => Ok(Some((Frame::Null, after))),
        b'#' => match line {
            b"t" => Ok(Some((Frame::Bool(true), after))),
            b"f" => Ok(Some((Frame::Bool(false), after))),
            _ => Err(ProtocolError::UnknownType(b'#')),
        },
        b',' => Ok(Some((Frame::Double(parse_double(line)?), after))),
        b'(' => Ok(Some((Frame::BigNumber(line), after))),
        b'$' | b'!' | b'=' => {
            if line == b"?" {
                return Err(ProtocolError::Unsupported(kind));
            }
            let len = parse_i64(line).ok_or(ProtocolError::InvalidBulkLength)?;
            if len < 0 {
                // Only `$-1` is a null. `!-1` and `=-1` do not exist.
                return if kind == b'$' && len == -1 {
                    Ok(Some((Frame::Null, after)))
                } else {
                    Err(ProtocolError::InvalidBulkLength)
                };
            }
            let len = len as usize;
            if len > limits.max_bulk {
                return Err(ProtocolError::InvalidBulkLength);
            }
            if buf.len() < after + len + 2 {
                return Ok(None);
            }
            let body = &buf[after..after + len];
            let end = after + len + 2;
            match kind {
                b'$' => Ok(Some((Frame::Bulk(body), end))),
                b'!' => Ok(Some((Frame::BlobError(body), end))),
                _ => {
                    // `=` carries a three byte format, a colon, then the text.
                    if body.len() < 4 || body[3] != b':' {
                        return Err(ProtocolError::InvalidBulkLength);
                    }
                    Ok(Some((
                        Frame::Verbatim {
                            format: &body[..3],
                            text: &body[4..],
                        },
                        end,
                    )))
                }
            }
        }
        b'*' | b'~' | b'>' => {
            if line == b"?" {
                return Err(ProtocolError::Unsupported(kind));
            }
            let n = parse_i64(line).ok_or(ProtocolError::InvalidMultibulkLength)?;
            if n < 0 {
                return if kind == b'*' && n == -1 {
                    Ok(Some((Frame::Null, after)))
                } else {
                    Err(ProtocolError::InvalidMultibulkLength)
                };
            }
            let Some((items, end)) = children(buf, after, n as usize, limits, depth)? else {
                return Ok(None);
            };
            Ok(Some((
                match kind {
                    b'*' => Frame::Array(items),
                    b'~' => Frame::Set(items),
                    _ => Frame::Push(items),
                },
                end,
            )))
        }
        b'%' | b'|' => {
            if line == b"?" {
                return Err(ProtocolError::Unsupported(kind));
            }
            let n = parse_i64(line).ok_or(ProtocolError::InvalidMultibulkLength)?;
            if n < 0 {
                return Err(ProtocolError::InvalidMultibulkLength);
            }
            // A map of n pairs is 2n frames on the wire.
            let Some((items, end)) = children(buf, after, (n as usize) * 2, limits, depth)? else {
                return Ok(None);
            };
            let mut pairs = Vec::with_capacity(n as usize);
            let mut it = items.into_iter();
            while let (Some(k), Some(v)) = (it.next(), it.next()) {
                pairs.push((k, v));
            }
            Ok(Some((
                if kind == b'%' {
                    Frame::Map(pairs)
                } else {
                    Frame::Attribute(pairs)
                },
                end,
            )))
        }
        other => Err(ProtocolError::UnknownType(other)),
    }
}

/// Reads `n` frames in a row, or `None` if they have not all arrived.
///
/// The count is not used to reserve, because it comes off the wire: a `*` line
/// claiming four billion elements would otherwise be four billion frames of
/// capacity before the second byte of the array has been seen. The vector grows
/// as the frames actually arrive, which bounds it by the bytes received.
fn children<'a>(
    buf: &'a [u8],
    from: usize,
    n: usize,
    limits: &Limits,
    depth: usize,
) -> Result<Option<(Vec<Frame<'a>>, usize)>, ProtocolError> {
    let mut items = Vec::new();
    let mut at = from;
    for _ in 0..n {
        let Some((frame, next)) = decode_at(buf, at, limits, depth + 1)? else {
            return Ok(None);
        };
        items.push(frame);
        at = next;
    }
    Ok(Some((items, at)))
}

/// A double as RESP3 spells it, the three words included.
fn parse_double(line: &[u8]) -> Result<f64, ProtocolError> {
    match line {
        b"inf" | b"+inf" => return Ok(f64::INFINITY),
        b"-inf" => return Ok(f64::NEG_INFINITY),
        b"nan" => return Ok(f64::NAN),
        _ => {}
    }
    core::str::from_utf8(line)
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .ok_or(ProtocolError::UnknownType(b','))
}

/// The CRLF terminated line starting at `from`, and the offset just past it.
fn line_at(buf: &[u8], from: usize) -> Option<(&[u8], usize)> {
    let off = buf.get(from..)?.iter().position(|&b| b == b'\r')?;
    let cr = from + off;
    if buf.get(cr + 1) == Some(&b'\n') {
        Some((&buf[from..cr], cr + 2))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::Proto;
    use crate::reply::Out;

    fn whole(buf: &[u8]) -> Frame<'_> {
        let (frame, used) = decode(buf, &Limits::default())
            .expect("should not be a protocol error")
            .expect("should be a whole frame");
        assert_eq!(used, buf.len(), "the frame should use the whole buffer");
        frame
    }

    #[test]
    fn the_resp2_types_decode() {
        assert_eq!(whole(b"+OK\r\n"), Frame::Simple(b"OK"));
        assert_eq!(whole(b"-ERR nope\r\n"), Frame::Error(b"ERR nope"));
        assert_eq!(whole(b":-7\r\n"), Frame::Int(-7));
        assert_eq!(whole(b"$3\r\nabc\r\n"), Frame::Bulk(b"abc"));
        assert_eq!(whole(b"$0\r\n\r\n"), Frame::Bulk(b""));
        assert_eq!(whole(b"$-1\r\n"), Frame::Null);
        assert_eq!(whole(b"*-1\r\n"), Frame::Null);
        assert_eq!(
            whole(b"*2\r\n$1\r\na\r\n:1\r\n"),
            Frame::Array(vec![Frame::Bulk(b"a"), Frame::Int(1)])
        );
        assert_eq!(whole(b"*0\r\n"), Frame::Array(Vec::new()));
    }

    #[test]
    fn the_resp3_types_decode() {
        assert_eq!(whole(b"_\r\n"), Frame::Null);
        assert_eq!(whole(b"#t\r\n"), Frame::Bool(true));
        assert_eq!(whole(b"#f\r\n"), Frame::Bool(false));
        assert_eq!(whole(b",1.5\r\n"), Frame::Double(1.5));
        assert_eq!(whole(b",inf\r\n"), Frame::Double(f64::INFINITY));
        assert_eq!(whole(b",-inf\r\n"), Frame::Double(f64::NEG_INFINITY));
        assert_eq!(
            whole(b"(12345678901234567890\r\n"),
            Frame::BigNumber(b"12345678901234567890")
        );
        assert_eq!(whole(b"!5\r\nboom!\r\n"), Frame::BlobError(b"boom!"));
        assert_eq!(
            whole(b"=15\r\ntxt:Some string\r\n"),
            Frame::Verbatim {
                format: b"txt",
                text: b"Some string"
            }
        );
        assert_eq!(
            whole(b"%1\r\n$1\r\na\r\n:1\r\n"),
            Frame::Map(vec![(Frame::Bulk(b"a"), Frame::Int(1))])
        );
        assert_eq!(whole(b"~1\r\n:9\r\n"), Frame::Set(vec![Frame::Int(9)]));
        assert_eq!(whole(b">1\r\n:9\r\n"), Frame::Push(vec![Frame::Int(9)]));
        assert_eq!(
            whole(b"|1\r\n$3\r\nttl\r\n:60\r\n"),
            Frame::Attribute(vec![(Frame::Bulk(b"ttl"), Frame::Int(60))])
        );
    }

    #[test]
    fn a_nan_decodes_even_though_it_never_equals_itself() {
        let Frame::Double(d) = whole(b",nan\r\n") else {
            panic!("not a double")
        };
        assert!(d.is_nan());
    }

    /// Every prefix of a reply must say "not yet" rather than guess. A decoder
    /// that returns a short frame from a partial buffer hands the client half
    /// an answer, which is worse than no answer.
    #[test]
    fn every_prefix_of_a_reply_is_incomplete() {
        let replies: &[&[u8]] = &[
            b"+OK\r\n",
            b"$5\r\nhello\r\n",
            b"*2\r\n$1\r\na\r\n$1\r\nb\r\n",
            b"%1\r\n$1\r\na\r\n*2\r\n:1\r\n:2\r\n",
            b"=15\r\ntxt:Some string\r\n",
        ];
        for reply in replies {
            for n in 0..reply.len() {
                assert_eq!(
                    decode(&reply[..n], &Limits::default()),
                    Ok(None),
                    "{:?} truncated to {n} bytes",
                    core::str::from_utf8(reply).unwrap_or("?")
                );
            }
            assert!(decode(reply, &Limits::default()).unwrap().is_some());
        }
    }

    #[test]
    fn pipelined_replies_come_out_one_at_a_time() {
        let buf = b"+OK\r\n:1\r\n$3\r\nabc\r\n";
        let mut at = 0;
        let mut seen = Vec::new();
        while let Some((frame, used)) = decode(&buf[at..], &Limits::default()).unwrap() {
            seen.push(frame);
            at += used;
        }
        assert_eq!(at, buf.len());
        assert_eq!(
            seen,
            vec![Frame::Simple(b"OK"), Frame::Int(1), Frame::Bulk(b"abc")]
        );
    }

    /// The reason the depth limit exists. Without it this is a stack overflow,
    /// which on a server is a crash rather than an error.
    #[test]
    fn a_deeply_nested_reply_is_refused_rather_than_overflowing_the_stack() {
        let mut buf = Vec::new();
        for _ in 0..10_000 {
            buf.extend_from_slice(b"*1\r\n");
        }
        buf.extend_from_slice(b":1\r\n");
        assert_eq!(
            decode(&buf, &Limits::default()),
            Err(ProtocolError::TooDeep)
        );
    }

    /// A count off the wire must not become capacity. This says four billion
    /// elements and delivers none, and the decoder has to survive it.
    #[test]
    fn an_enormous_element_count_does_not_reserve_anything() {
        assert_eq!(
            decode(b"*4000000000\r\n", &Limits::default()),
            Ok(None),
            "it should be waiting for elements, not allocating for them"
        );
    }

    #[test]
    fn the_streamed_forms_say_so_rather_than_being_mis_parsed() {
        for buf in [&b"$?\r\n"[..], b"*?\r\n", b"%?\r\n", b"~?\r\n"] {
            assert!(
                matches!(
                    decode(buf, &Limits::default()),
                    Err(ProtocolError::Unsupported(_))
                ),
                "{buf:?}"
            );
        }
    }

    #[test]
    fn an_unknown_type_byte_is_named() {
        assert_eq!(
            decode(b"@1\r\n", &Limits::default()),
            Err(ProtocolError::UnknownType(b'@'))
        );
    }

    /// The round trip that makes the encoder and the decoder check each other.
    /// Everything the encoder can write is written in both protocols and read
    /// back, which is how a downgrade that produces bytes no client can parse
    /// gets caught here rather than in a client library's issue tracker.
    #[test]
    fn everything_the_encoder_writes_reads_back() {
        for proto in [Proto::Resp2, Proto::Resp3] {
            let mut out = Out::new(proto);
            out.simple(b"OK");
            out.error(b"ERR nope");
            out.int(-7);
            out.bulk(b"hello");
            out.nil();
            out.nil_array();
            out.bool(true);
            out.double(1.5);
            out.verbatim(b"txt", b"note");
            out.big_number(b"123456789012345678901234567890");
            out.array(2);
            out.bulk(b"a");
            out.int(1);
            out.map(1);
            out.bulk(b"k");
            out.bulk(b"v");
            out.set(1);
            out.bulk(b"m");
            out.push(2);
            out.bulk(b"message");
            out.bulk(b"ch");

            let buf = out.into_inner();
            let mut at = 0;
            let mut count = 0;
            while at < buf.len() {
                let (_, used) = decode(&buf[at..], &Limits::default())
                    .unwrap_or_else(|e| panic!("{proto:?} produced bytes that do not parse: {e}"))
                    .unwrap_or_else(|| panic!("{proto:?} produced a truncated frame at {at}"));
                at += used;
                count += 1;
            }
            assert_eq!(at, buf.len());
            // Fourteen top level frames either way. RESP2 spells seven of them
            // differently, and the count is what proves the downgrade did not
            // quietly drop one or split one in two.
            assert_eq!(count, 14, "{proto:?} top level frames");
        }
    }
}
