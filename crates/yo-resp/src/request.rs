//! Requests in: the multibulk decoder and the inline decoder.
//!
//! This is the read half of the hot path and it does not allocate once a
//! connection is warm. An argument is a range in the connection's own read
//! buffer, so the command layer works on the bytes the kernel delivered and
//! nothing is copied on the way. The one exception is an inline request with
//! escapes in it, which has to be unescaped somewhere, and that goes into a
//! scratch buffer on the same [`Argv`].
//!
//! # The contract
//!
//! A connection owns one [`Argv`] and one read buffer, and drives them like
//! this:
//!
//! 1. Read bytes and append them to the buffer. Never insert, never reorder.
//! 2. Call [`Argv::decode`]. On [`Step::Incomplete`], go back to 1.
//! 3. On [`Step::Command`], read the arguments, then drop `consumed` bytes from
//!    the front of the buffer, then go back to 2 in case the read carried more
//!    than one command.
//!
//! The arguments are only valid between step 3 and the moment the buffer is
//! drained, because they are ranges into it. That is the price of not copying
//! and it is the reason the buffer is passed to [`Argv::arg`] rather than held.
//!
//! Between calls the decoder remembers how far it got, so a 512 MiB value that
//! arrives in ten thousand pieces is scanned once rather than ten thousand
//! times. That is the difference between linear and quadratic on a slow link
//! and it is why the resume state exists at all.

use crate::error::ProtocolError;
use crate::proto::Limits;
use yo_common::num::parse_i64;

/// One argument: where it starts, how long it is, and which buffer it is in.
#[derive(Debug, Clone, Copy)]
struct Span {
    start: usize,
    len: u32,
    /// True for an unescaped inline argument, which lives in the scratch buffer
    /// rather than in the caller's read buffer.
    scratch: bool,
}

/// What a decode attempt produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Nothing yet. Read more and call again with the same buffer, extended.
    Incomplete,
    /// A whole command. Read the arguments, then drop `consumed` bytes from the
    /// front of the buffer.
    ///
    /// `consumed` can describe a command with no arguments at all, which is
    /// what `*0\r\n` and a blank inline line are. Redis accepts both and
    /// replies to neither, so the connection should skip them rather than
    /// treat them as an error.
    Command {
        /// Bytes at the front of the buffer that this command used up.
        consumed: usize,
    },
}

/// A command's arguments, and the decoder that fills them.
///
/// One per connection, reused for the life of the connection. After the first
/// few commands it has the capacity it needs and never allocates again.
#[derive(Debug, Default)]
pub struct Argv {
    spans: Vec<Span>,
    /// Unescaped inline arguments. Empty for every multibulk command, which is
    /// every command a real client sends.
    scratch: Vec<u8>,
    /// Where the next unparsed byte is, for a command that arrived in pieces.
    next: usize,
    /// Arguments still to come, or `None` when no command is part way through.
    want: Option<u32>,
}

impl Argv {
    /// An empty one.
    pub fn new() -> Argv {
        Argv::default()
    }

    /// An empty one with room for `n` arguments already reserved.
    ///
    /// Worth doing at accept time. The first command on a connection is the
    /// only one that would otherwise allocate.
    pub fn with_capacity(n: usize) -> Argv {
        Argv {
            spans: Vec::with_capacity(n),
            ..Argv::default()
        }
    }

    /// Forgets everything, including any half read command.
    ///
    /// The connection calls this if it discards unread bytes for any reason,
    /// because the resume state is an offset into a buffer that is about to
    /// stop being the same buffer.
    pub fn reset(&mut self) {
        self.spans.clear();
        self.scratch.clear();
        self.next = 0;
        self.want = None;
    }

    /// How many arguments the last complete command had.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Whether the last complete command had no arguments.
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Argument `i`, or `None` if there are fewer than that many.
    ///
    /// `buf` must be the same buffer that was decoded and must not have been
    /// drained since.
    pub fn arg<'a>(&'a self, buf: &'a [u8], i: usize) -> Option<&'a [u8]> {
        let s = self.spans.get(i)?;
        let src = if s.scratch { &self.scratch[..] } else { buf };
        src.get(s.start..s.start + s.len as usize)
    }

    /// Every argument in order.
    pub fn args<'a>(&'a self, buf: &'a [u8]) -> impl Iterator<Item = &'a [u8]> {
        (0..self.spans.len()).filter_map(move |i| self.arg(buf, i))
    }

    /// Reads one command from the front of `buf`.
    ///
    /// # Errors
    ///
    /// Any [`ProtocolError`]. Redis closes the connection after one of these
    /// and so should the caller: a protocol error means the two ends no longer
    /// agree on where the next frame starts, and there is no recovering from
    /// that by reading further.
    pub fn decode(&mut self, buf: &[u8], limits: &Limits) -> Result<Step, ProtocolError> {
        if self.want.is_none() {
            // A fresh command. The previous one's arguments stop being valid
            // here rather than when it finished, so that the caller has the
            // whole gap between the two calls to read them.
            self.spans.clear();
            self.scratch.clear();
            self.next = 0;

            if buf.is_empty() {
                return Ok(Step::Incomplete);
            }
            if buf[0] != b'*' {
                return self.inline(buf, limits);
            }
            // The two arms agree on a `u64` and the slow one folds a negative
            // into zero, because `*-1` and `*0` are the same command with
            // nothing in it and neither is ever above the limit.
            let (count, after) = match digits_at(buf, 1) {
                Some(got) => got,
                None => {
                    let Some((line, after)) =
                        line_at(buf, 1, ProtocolError::InvalidMultibulkLength)?
                    else {
                        // No end to the count line yet. A client that keeps
                        // sending digits and never a newline is not going to
                        // become valid, and the pending bytes are being held for
                        // it, so there is a bound.
                        return if buf.len() > limits.max_inline {
                            Err(ProtocolError::TooBigMbulkCount)
                        } else {
                            Ok(Step::Incomplete)
                        };
                    };
                    let n = parse_i64(line).ok_or(ProtocolError::InvalidMultibulkLength)?;
                    (n.max(0) as u64, after)
                }
            };
            if count > limits.max_multibulk as u64 {
                return Err(ProtocolError::InvalidMultibulkLength);
            }
            if count == 0 {
                // `*0` and `*-1` are both a command with nothing in it. Redis
                // consumes them and replies to neither.
                return Ok(Step::Command { consumed: after });
            }
            // The count is bounded above, so this reserve is bounded too. It is
            // the only reason the limit is checked before this line.
            self.spans.reserve(count as usize);
            self.next = after;
            self.want = Some(count as u32);
        }

        while self.want.is_some_and(|w| w > 0) {
            let Some(&kind) = buf.get(self.next) else {
                return Ok(Step::Incomplete);
            };
            if kind != b'$' {
                return Err(ProtocolError::ExpectedDollar(kind));
            }
            let (len, after) = match digits_at(buf, self.next + 1) {
                Some(got) => got,
                None => {
                    let Some((line, after)) =
                        line_at(buf, self.next + 1, ProtocolError::InvalidBulkLength)?
                    else {
                        return if buf.len() - self.next > limits.max_inline {
                            Err(ProtocolError::TooBigBulkCount)
                        } else {
                            Ok(Step::Incomplete)
                        };
                    };
                    let n = parse_i64(line).ok_or(ProtocolError::InvalidBulkLength)?;
                    if n < 0 {
                        return Err(ProtocolError::InvalidBulkLength);
                    }
                    (n as u64, after)
                }
            };
            if len > limits.max_bulk as u64 {
                return Err(ProtocolError::InvalidBulkLength);
            }
            // The limit above is what makes this cast safe on a 32 bit target.
            let len = len as usize;
            // The body and its trailing CRLF, which is not optional and is not
            // checked here: a client that lies about it desynchronises itself
            // and the next `expected '$'` says so.
            if buf.len() < after + len + 2 {
                return Ok(Step::Incomplete);
            }
            self.spans.push(Span {
                start: after,
                len: len as u32,
                scratch: false,
            });
            self.next = after + len + 2;
            self.want = self.want.map(|w| w - 1);
        }

        let consumed = self.next;
        self.want = None;
        self.next = 0;
        Ok(Step::Command { consumed })
    }

    /// A telnet style request: one line, split on whitespace, quotes honoured.
    ///
    /// Cold by construction. Nothing that cares about speed sends inline
    /// commands, and the unescaping copies, which is why this is the one path
    /// that touches the scratch buffer.
    fn inline(&mut self, buf: &[u8], limits: &Limits) -> Result<Step, ProtocolError> {
        let Some(nl) = buf.iter().position(|&b| b == b'\n') else {
            return if buf.len() > limits.max_inline {
                Err(ProtocolError::TooBigInline)
            } else {
                Ok(Step::Incomplete)
            };
        };
        let mut line = &buf[..nl];
        if line.last() == Some(&b'\r') {
            line = &line[..line.len() - 1];
        }
        self.split_inline(line)?;
        Ok(Step::Command { consumed: nl + 1 })
    }

    /// Redis's `sdssplitargs`, byte for byte.
    ///
    /// Reimplemented rather than approximated because `redis-cli` sends inline
    /// commands in some modes and the test suite has cases for every corner of
    /// it: `\x41` hex escapes inside double quotes, `\'` inside single quotes,
    /// and the rule that a closing quote must be followed by whitespace or by
    /// the end of the line.
    fn split_inline(&mut self, line: &[u8]) -> Result<(), ProtocolError> {
        let mut i = 0;
        loop {
            while i < line.len() && is_space(line[i]) {
                i += 1;
            }
            if i >= line.len() {
                return Ok(());
            }
            let start = self.scratch.len();
            let mut in_double = false;
            let mut in_single = false;
            let mut done = false;
            while !done {
                let c = line.get(i).copied();
                if in_double {
                    match c {
                        Some(b'\\')
                            if i + 3 < line.len()
                                && line[i + 1] == b'x'
                                && hex(line[i + 2]).is_some()
                                && hex(line[i + 3]).is_some() =>
                        {
                            let hi = hex(line[i + 2]).unwrap_or(0);
                            let lo = hex(line[i + 3]).unwrap_or(0);
                            self.scratch.push(hi * 16 + lo);
                            i += 3;
                        }
                        Some(b'\\') if i + 1 < line.len() => {
                            i += 1;
                            self.scratch.push(match line[i] {
                                b'n' => b'\n',
                                b'r' => b'\r',
                                b't' => b'\t',
                                b'b' => 0x08,
                                b'a' => 0x07,
                                other => other,
                            });
                        }
                        Some(b'"') => {
                            if line.get(i + 1).is_some_and(|&n| !is_space(n)) {
                                return Err(ProtocolError::UnbalancedQuotes);
                            }
                            done = true;
                        }
                        None => return Err(ProtocolError::UnbalancedQuotes),
                        Some(ch) => self.scratch.push(ch),
                    }
                } else if in_single {
                    match c {
                        Some(b'\\') if line.get(i + 1) == Some(&b'\'') => {
                            i += 1;
                            self.scratch.push(b'\'');
                        }
                        Some(b'\'') => {
                            if line.get(i + 1).is_some_and(|&n| !is_space(n)) {
                                return Err(ProtocolError::UnbalancedQuotes);
                            }
                            done = true;
                        }
                        None => return Err(ProtocolError::UnbalancedQuotes),
                        Some(ch) => self.scratch.push(ch),
                    }
                } else {
                    match c {
                        None | Some(b' ') | Some(b'\n') | Some(b'\r') | Some(b'\t') => done = true,
                        Some(b'"') => in_double = true,
                        Some(b'\'') => in_single = true,
                        Some(ch) => self.scratch.push(ch),
                    }
                }
                if i < line.len() {
                    i += 1;
                }
            }
            let len = self.scratch.len() - start;
            self.spans.push(Span {
                start,
                len: len as u32,
                scratch: true,
            });
        }
    }
}

/// C's `isspace`, which includes the vertical tab that Rust's does not.
#[inline]
const fn is_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

/// The value of one hex digit.
#[inline]
const fn hex(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// The CRLF terminated line starting at `from`, and the offset just past it.
///
/// `Ok(None)` means the line has not arrived yet. `bad` is the error to raise
/// for a `\r` that is not followed by `\n`, which is the one place this is
/// stricter than Redis: Redis finds the `\r`, assumes the `\n` and carries on,
/// which desynchronises a byte later with a different message. Both ends close
/// the connection either way, so what differs is the text and not the outcome.
/// The plain positive number at `from`, and where the line after it starts.
///
/// One pass over the digits instead of two. The general path finds the `\r`, and
/// then `parse_i64` walks the same one to three bytes again to turn them into a
/// number, which is two loops and two bounds checks per byte to read a length
/// that is almost always a single digit. This reads a byte and accumulates it in
/// the same step, and there are three of these per `GET` off the wire.
///
/// `None` is not an error. It means this is not the shape the fast path handles,
/// which covers a line that has not arrived yet, a negative number, a leading
/// zero and anything malformed, and the caller answers it by taking the general
/// path. That path already knows which of `Incomplete`, `TooBigBulkCount` and
/// `InvalidBulkLength` each of those is, and there is no value in writing those
/// rules down twice. Being wrong here costs the second pass this exists to
/// avoid, on a command that is about to be refused anyway.
///
/// The leading zero rule is `string2ll`'s and it is why `$007` is an error in
/// Redis and an error here. It is enforced by refusing rather than by accepting,
/// so the general path stays the only place that decides.
fn digits_at(buf: &[u8], from: usize) -> Option<(u64, usize)> {
    let mut at = from;
    let mut v: u64 = 0;
    while let Some(&c) = buf.get(at) {
        if !c.is_ascii_digit() {
            break;
        }
        // Nineteen digits is where the twentieth may not fit in a `u64`. No
        // caller here accepts a number within many orders of magnitude of that,
        // so this is an overflow guard rather than a limit, and the general path
        // is welcome to the case.
        if at - from == 19 {
            return None;
        }
        v = v * 10 + u64::from(c - b'0');
        at += 1;
    }
    if at == from || (at - from > 1 && buf[from] == b'0') {
        return None;
    }
    if buf.get(at) != Some(&b'\r') || buf.get(at + 1) != Some(&b'\n') {
        return None;
    }
    Some((v, at + 2))
}

fn line_at(
    buf: &[u8],
    from: usize,
    bad: ProtocolError,
) -> Result<Option<(&[u8], usize)>, ProtocolError> {
    let Some(off) = buf[from..].iter().position(|&b| b == b'\r') else {
        return Ok(None);
    };
    let cr = from + off;
    match buf.get(cr + 1) {
        None => Ok(None),
        Some(&b'\n') => Ok(Some((&buf[from..cr], cr + 2))),
        Some(_) => Err(bad),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decodes one command from a complete buffer and returns its arguments as
    /// owned bytes, which the tests can compare without holding the buffer.
    fn one(buf: &[u8]) -> Result<(Vec<Vec<u8>>, usize), ProtocolError> {
        let mut argv = Argv::new();
        match argv.decode(buf, &Limits::default())? {
            Step::Incomplete => panic!("expected a whole command in {buf:?}"),
            Step::Command { consumed } => Ok((
                argv.args(buf).map(<[u8]>::to_vec).collect::<Vec<_>>(),
                consumed,
            )),
        }
    }

    fn words(buf: &[u8]) -> Vec<Vec<u8>> {
        one(buf).expect("should decode").0
    }

    #[test]
    fn a_multibulk_command_comes_out_as_its_arguments() {
        let (args, consumed) = one(b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\nv\r\n").unwrap();
        assert_eq!(args, vec![b"SET".to_vec(), b"k".to_vec(), b"v".to_vec()]);
        assert_eq!(consumed, 27);
    }

    #[test]
    fn an_empty_argument_is_an_argument() {
        assert_eq!(
            words(b"*2\r\n$3\r\nGET\r\n$0\r\n\r\n"),
            vec![b"GET".to_vec(), Vec::new()]
        );
    }

    #[test]
    fn a_value_can_hold_anything_including_crlf() {
        let args = words(b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$4\r\na\r\nb\r\n");
        assert_eq!(args[2], b"a\r\nb".to_vec());
    }

    /// The pipelining case: two commands in one read, and the second one only
    /// found because the first reported what it used.
    #[test]
    fn commands_come_out_one_at_a_time_from_one_buffer() {
        let buf = b"*1\r\n$4\r\nPING\r\n*2\r\n$3\r\nGET\r\n$1\r\nk\r\n";
        let mut argv = Argv::new();
        let mut at = 0;
        let mut seen: Vec<Vec<Vec<u8>>> = Vec::new();
        loop {
            match argv.decode(&buf[at..], &Limits::default()).unwrap() {
                Step::Incomplete => break,
                Step::Command { consumed } => {
                    seen.push(argv.args(&buf[at..]).map(<[u8]>::to_vec).collect());
                    at += consumed;
                }
            }
        }
        assert_eq!(at, buf.len());
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[0], vec![b"PING".to_vec()]);
        assert_eq!(seen[1], vec![b"GET".to_vec(), b"k".to_vec()]);
    }

    /// Fed one byte at a time, which is the shape a slow link produces and the
    /// shape that finds an off by one in the resume state. Every prefix must
    /// say incomplete and the last byte must produce exactly the same command
    /// as the whole buffer at once.
    #[test]
    fn a_command_arriving_one_byte_at_a_time_decodes_once_at_the_end() {
        let whole = b"*3\r\n$3\r\nSET\r\n$5\r\nhello\r\n$5\r\nworld\r\n";
        let mut argv = Argv::new();
        for n in 0..whole.len() {
            assert_eq!(
                argv.decode(&whole[..n], &Limits::default()).unwrap(),
                Step::Incomplete,
                "the first {n} bytes should not be a command"
            );
        }
        let step = argv.decode(whole, &Limits::default()).unwrap();
        assert_eq!(
            step,
            Step::Command {
                consumed: whole.len()
            }
        );
        assert_eq!(
            argv.args(whole).map(<[u8]>::to_vec).collect::<Vec<_>>(),
            vec![b"SET".to_vec(), b"hello".to_vec(), b"world".to_vec()]
        );
    }

    /// The resume state is what stops a value that arrives in pieces from being
    /// rescanned once per piece. This checks the state actually moves, because
    /// a decoder that quietly restarted every time would pass every other test
    /// here and be quadratic on a real link.
    #[test]
    fn a_partly_arrived_command_remembers_where_it_got_to() {
        let mut argv = Argv::new();
        let head = b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$10\r\nabc";
        assert_eq!(
            argv.decode(head, &Limits::default()).unwrap(),
            Step::Incomplete
        );
        assert_eq!(argv.want, Some(1), "two of three arguments are in");
        assert_eq!(argv.next, 20, "the third argument's body starts here");
    }

    #[test]
    fn an_empty_command_is_consumed_and_has_no_arguments() {
        for buf in [&b"*0\r\n"[..], b"*-1\r\n"] {
            let (args, consumed) = one(buf).unwrap();
            assert!(args.is_empty(), "{buf:?}");
            assert_eq!(consumed, buf.len());
        }
    }

    #[test]
    fn an_inline_command_is_split_on_whitespace() {
        assert_eq!(words(b"PING\r\n"), vec![b"PING".to_vec()]);
        assert_eq!(
            words(b"SET  key   value\n"),
            vec![b"SET".to_vec(), b"key".to_vec(), b"value".to_vec()]
        );
        assert!(words(b"\r\n").is_empty());
        assert!(words(b"   \n").is_empty());
    }

    #[test]
    fn inline_quotes_and_escapes_follow_redis() {
        assert_eq!(
            words(b"SET k \"a b\"\r\n"),
            vec![b"SET".to_vec(), b"k".to_vec(), b"a b".to_vec()]
        );
        assert_eq!(words(b"ECHO \"\\x41\\x42\"\r\n")[1], b"AB".to_vec());
        assert_eq!(words(b"ECHO \"a\\nb\"\r\n")[1], b"a\nb".to_vec());
        assert_eq!(words(b"ECHO 'it\\'s'\r\n")[1], b"it's".to_vec());
        assert_eq!(words(b"ECHO \"\"\r\n")[1], Vec::<u8>::new());
    }

    #[test]
    fn an_unclosed_or_misplaced_quote_is_an_error() {
        for bad in [
            &b"ECHO \"abc\r\n"[..],
            b"ECHO 'abc\r\n",
            b"ECHO \"abc\"d\r\n",
            b"ECHO 'abc'd\r\n",
        ] {
            let mut argv = Argv::new();
            assert_eq!(
                argv.decode(bad, &Limits::default()),
                Err(ProtocolError::UnbalancedQuotes),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn the_wrong_type_byte_where_an_argument_belongs_names_the_byte() {
        let mut argv = Argv::new();
        assert_eq!(
            argv.decode(b"*1\r\n+OK\r\n", &Limits::default()),
            Err(ProtocolError::ExpectedDollar(b'+'))
        );
    }

    #[test]
    fn lengths_that_are_not_lengths_are_refused() {
        let cases: &[(&[u8], ProtocolError)] = &[
            (b"*x\r\n", ProtocolError::InvalidMultibulkLength),
            (b"*\r\n", ProtocolError::InvalidMultibulkLength),
            (b"*01\r\n", ProtocolError::InvalidMultibulkLength),
            (
                b"*99999999999999999999\r\n",
                ProtocolError::InvalidMultibulkLength,
            ),
            (b"*2\r\n$x\r\n", ProtocolError::InvalidBulkLength),
            (b"*2\r\n$-1\r\n", ProtocolError::InvalidBulkLength),
            // A leading zero is not a length, which is `string2ll`'s rule and
            // therefore Redis's. It is here because it is the one rule the fast
            // path in `digits_at` could accept by accident, and `$003\r\nGET`
            // parsing as a three byte argument would be a divergence nobody
            // would find until a client sent one.
            (b"*2\r\n$007\r\n", ProtocolError::InvalidBulkLength),
            (b"*2\r\n$00\r\n", ProtocolError::InvalidBulkLength),
        ];
        for &(buf, want) in cases {
            let mut argv = Argv::new();
            assert_eq!(argv.decode(buf, &Limits::default()), Err(want), "{buf:?}");
        }
    }

    /// The fast length scanner and the general path have to agree, byte for
    /// byte, on every number a client can send.
    ///
    /// `digits_at` is an optimisation and an optimisation that changes an answer
    /// is a bug. It answers `None` for anything it does not want, and the whole
    /// of its correctness is that `None` means the general path runs and decides
    /// exactly what it decided before. This walks a length through both.
    #[test]
    fn the_fast_length_scanner_agrees_with_the_general_path() {
        for n in [0usize, 1, 9, 10, 99, 100, 4096, 65535, 1_000_000] {
            let digits = n.to_string();
            let body = vec![b'v'; n];
            let mut buf = format!("*1\r\n${}\r\n", digits).into_bytes();
            buf.extend_from_slice(&body);
            buf.extend_from_slice(b"\r\n");

            // What the scanner says about the length line on its own.
            let at = 4 + 1;
            assert_eq!(
                digits_at(&buf, at),
                Some((n as u64, at + digits.len() + 2)),
                "the scanner refused {digits}"
            );

            let (args, consumed) = one(&buf).expect("a whole command");
            assert_eq!(args, vec![body], "{digits} came back wrong");
            assert_eq!(consumed, buf.len());
        }
    }

    /// Everything the fast scanner is supposed to hand back, and why.
    #[test]
    fn the_fast_length_scanner_refuses_what_it_does_not_understand() {
        // A line that has not arrived, a line with no newline after its return,
        // no digits at all, a negative, a leading zero, and a number too long
        // to be sure of in a `u64`.
        for buf in [
            &b"7"[..],
            b"7\r",
            b"7\rx",
            b"\r\n",
            b"-1\r\n",
            b"01\r\n",
            b"00\r\n",
            b"99999999999999999999\r\n",
        ] {
            assert_eq!(digits_at(buf, 0), None, "{buf:?}");
        }
        // And the one that looks like a leading zero and is not.
        assert_eq!(digits_at(b"0\r\n", 0), Some((0, 3)));
    }

    /// A count of two billion must be refused before anything is reserved for
    /// it. If this ever regresses the symptom is not a wrong answer, it is the
    /// process disappearing.
    #[test]
    fn an_enormous_count_is_refused_rather_than_reserved() {
        let mut argv = Argv::new();
        assert_eq!(
            argv.decode(b"*2000000000\r\n", &Limits::default()),
            Err(ProtocolError::InvalidMultibulkLength)
        );
        assert_eq!(
            argv.spans.capacity(),
            0,
            "nothing should have been reserved"
        );
    }

    #[test]
    fn a_bulk_past_the_limit_is_refused() {
        let limits = Limits {
            max_bulk: 16,
            ..Limits::default()
        };
        let mut argv = Argv::new();
        assert_eq!(
            argv.decode(b"*1\r\n$17\r\n", &limits),
            Err(ProtocolError::InvalidBulkLength)
        );
        // One under the limit is still a matter of waiting for the body.
        let mut argv = Argv::new();
        assert_eq!(argv.decode(b"*1\r\n$16\r\n", &limits), Ok(Step::Incomplete));
    }

    #[test]
    fn a_line_that_never_ends_is_refused_rather_than_buffered_forever() {
        let limits = Limits {
            max_inline: 8,
            ..Limits::default()
        };
        let mut argv = Argv::new();
        assert_eq!(
            argv.decode(b"*123456789", &limits),
            Err(ProtocolError::TooBigMbulkCount)
        );
        let mut argv = Argv::new();
        assert_eq!(
            argv.decode(b"*1\r\n$123456789", &limits),
            Err(ProtocolError::TooBigBulkCount)
        );
        let mut argv = Argv::new();
        assert_eq!(
            argv.decode(b"PING PING PING", &limits),
            Err(ProtocolError::TooBigInline)
        );
    }

    #[test]
    fn a_carriage_return_with_no_newline_after_it_is_a_protocol_error() {
        let mut argv = Argv::new();
        assert_eq!(
            argv.decode(b"*1\rx", &Limits::default()),
            Err(ProtocolError::InvalidMultibulkLength)
        );
    }

    #[test]
    fn a_reset_forgets_a_half_read_command() {
        let mut argv = Argv::new();
        assert_eq!(
            argv.decode(b"*2\r\n$3\r\nGET\r\n", &Limits::default())
                .unwrap(),
            Step::Incomplete
        );
        assert_eq!(argv.want, Some(1));
        argv.reset();
        assert_eq!(argv.want, None);
        assert_eq!(argv.next, 0);
        // And the next command on the fresh buffer decodes from the start.
        assert_eq!(
            argv.decode(b"*1\r\n$4\r\nPING\r\n", &Limits::default())
                .unwrap(),
            Step::Command { consumed: 14 }
        );
    }

    /// Once the spans have been read the connection reuses the buffer, so a
    /// decoder that kept stale spans would hand the next command's caller the
    /// previous command's bytes.
    #[test]
    fn arguments_do_not_survive_into_the_next_command() {
        let mut argv = Argv::new();
        argv.decode(b"*2\r\n$3\r\nGET\r\n$1\r\nk\r\n", &Limits::default())
            .unwrap();
        assert_eq!(argv.len(), 2);
        argv.decode(b"*1\r\n$4\r\nPING\r\n", &Limits::default())
            .unwrap();
        assert_eq!(argv.len(), 1);
        assert_eq!(argv.arg(b"*1\r\n$4\r\nPING\r\n", 1), None);
    }
}
