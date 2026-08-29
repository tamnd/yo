//! Replies out: wire bytes, written once.
//!
//! This is Y18 and it is a locked decision rather than a preference. A reply is
//! built directly as the bytes that go on the socket. There is no intermediate
//! value, no enum to match on later, no boxed trait object, and no second pass
//! that turns a structure into bytes. Redis 8.8 gained forty percent on `SCAN`
//! by fixing exactly this in its own reply path, which is the best available
//! evidence that the shape matters more than the constant factors inside it.
//!
//! The other half of Y18 is presizing. A reply whose size is known should
//! reserve once, before the first byte, from whichever side of the operation is
//! smaller. [`Out::reserve`] and the `*_len` helpers are there so that a command
//! can do that arithmetic without writing anything twice.
//!
//! # The protocol lives here
//!
//! A command writes a map. Whether that map goes out as RESP3's `%` or as
//! RESP2's flattened array is this module's problem and not the command's. Every
//! command in the engine is written once, against the richer protocol, and the
//! downgrade happens in one place where it can be tested. The alternative is a
//! protocol check in three hundred command implementations, and the failure
//! mode of that is a command that works on one protocol and not the other.

use crate::proto::Proto;
use yo_common::num::{DIGITS_MAX, i64_len, push_double, push_i64, push_u64, u64_digits};

/// A reply buffer for one connection.
///
/// Owns its bytes so that a connection can fill it across several commands and
/// hand the whole thing to one write, which is `04` section 5: one `writev` per
/// connection per batch, not one per reply.
#[derive(Debug, Clone)]
pub struct Out {
    buf: Vec<u8>,
    proto: Proto,
}

impl Out {
    /// An empty buffer speaking `proto`.
    pub fn new(proto: Proto) -> Out {
        Out {
            buf: Vec::new(),
            proto,
        }
    }

    /// An empty buffer with room already reserved.
    pub fn with_capacity(proto: Proto, cap: usize) -> Out {
        Out {
            buf: Vec::with_capacity(cap),
            proto,
        }
    }

    /// The protocol this connection is speaking.
    #[inline]
    pub const fn proto(&self) -> Proto {
        self.proto
    }

    /// Switches protocol, which is what `HELLO` does.
    ///
    /// Takes effect from the next reply written. `HELLO`'s own reply is written
    /// in the new protocol, which is why this is called before it rather than
    /// after.
    #[inline]
    pub const fn set_proto(&mut self, proto: Proto) {
        self.proto = proto;
    }

    /// The bytes written so far.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.buf
    }

    /// How many bytes are pending.
    #[inline]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// How much room it is holding, which is what it costs the process.
    ///
    /// A reply buffer keeps its capacity between batches on purpose, so `len`
    /// is what a client is owed and this is what the memory report owes.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buf.capacity()
    }

    /// Whether nothing is pending.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Drops everything written, keeping the capacity.
    ///
    /// Called after the batch has been written to the socket. The capacity is
    /// what stops a busy connection from allocating again.
    #[inline]
    pub fn clear(&mut self) {
        self.buf.clear();
    }

    /// Drops the first `n` bytes, which is what a partial write leaves behind.
    ///
    /// # Panics
    ///
    /// If `n` is past the end of what has been written.
    pub fn consume(&mut self, n: usize) {
        assert!(n <= self.buf.len(), "consumed past the end of the reply");
        self.buf.drain(..n);
    }

    /// Drops everything written after `len`, which has to be a length this
    /// buffer reported earlier.
    ///
    /// The dispatcher takes the length before it runs a command and rolls back
    /// to it when the command answers with an error, so a command that writes
    /// half a reply and then fails cannot leave the half on the wire. Every
    /// command is written to check its arguments before it writes anything,
    /// and this is what makes that a property of the dispatcher rather than a
    /// rule three hundred commands have to keep to.
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        self.buf.truncate(len);
    }

    /// Reserves room for `n` more bytes.
    ///
    /// The presize half of Y18. Call it once with the whole reply's size before
    /// writing any of it.
    #[inline]
    pub fn reserve(&mut self, n: usize) {
        self.buf.reserve(n);
    }

    /// The buffer, taken.
    pub fn into_inner(self) -> Vec<u8> {
        self.buf
    }

    /// Raw bytes, appended as they are.
    ///
    /// For a reply that was assembled elsewhere, such as a cached `COMMAND
    /// DOCS` payload or a replicated frame passing through. Nothing checks that
    /// what goes in is a valid frame, which is the point.
    #[inline]
    pub fn raw(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    // Simple strings, errors and integers. These three are the same in both
    // protocols, which is why none of them looks at `self.proto`.

    /// A simple string, `+s\r\n`. No CR or LF may appear in `s`.
    #[inline]
    pub fn simple(&mut self, s: &[u8]) {
        debug_assert!(
            !s.contains(&b'\r') && !s.contains(&b'\n'),
            "a simple string cannot carry a line ending, use a bulk string"
        );
        self.buf.reserve(s.len() + 3);
        self.buf.push(b'+');
        self.buf.extend_from_slice(s);
        self.crlf();
    }

    /// `+OK\r\n`, which is most of what a write command replies.
    #[inline]
    pub fn ok(&mut self) {
        self.buf.extend_from_slice(b"+OK\r\n");
    }

    /// An error, `-msg\r\n`.
    ///
    /// `msg` carries its own prefix, because the prefix is part of the
    /// contract: a client branches on `WRONGTYPE` or `MOVED` or `NOAUTH`, and
    /// which one applies is the command's decision and not the codec's. The
    /// full taxonomy is in `12` section 1.
    #[inline]
    pub fn error(&mut self, msg: &[u8]) {
        debug_assert!(
            !msg.contains(&b'\r') && !msg.contains(&b'\n'),
            "an error line cannot carry a line ending"
        );
        self.buf.reserve(msg.len() + 3);
        self.buf.push(b'-');
        self.buf.extend_from_slice(msg);
        self.crlf();
    }

    /// An error built from a prefix and a message that are not next to each
    /// other in memory, with any line ending in the message turned into a
    /// space.
    ///
    /// The prefix carries its own trailing space, so this is called with
    /// `b"ERR "` or `b"WRONGTYPE "`. Joining the two halves first would mean
    /// allocating a string on the failure path of a thread that is not allowed
    /// to allocate, which is the whole reason this exists.
    ///
    /// The mapping of `\r` and `\n` to spaces is Redis's, and it is not
    /// cosmetic: an error message can quote what the client sent, and a client
    /// that sends a command name with a newline in it would otherwise be
    /// writing its own frames into somebody's reply stream.
    pub fn error_line(&mut self, prefix: &[u8], msg: &[u8]) {
        self.buf.reserve(prefix.len() + msg.len() + 4);
        self.buf.push(b'-');
        self.buf.extend_from_slice(prefix);
        for &b in msg {
            self.buf
                .push(if b == b'\r' || b == b'\n' { b' ' } else { b });
        }
        self.crlf();
    }

    /// A blob error, RESP3's `!`, which may carry anything including newlines.
    ///
    /// Degrades to a normal error line in RESP2, with line endings turned into
    /// spaces, because a RESP2 error is one line by definition.
    pub fn blob_error(&mut self, msg: &[u8]) {
        if self.proto.is_resp3() {
            self.blob(b'!', msg);
        } else {
            self.buf.reserve(msg.len() + 3);
            self.buf.push(b'-');
            for &b in msg {
                self.buf
                    .push(if b == b'\r' || b == b'\n' { b' ' } else { b });
            }
            self.crlf();
        }
    }

    /// An integer, `:n\r\n`.
    #[inline]
    pub fn int(&mut self, n: i64) {
        self.buf.reserve(i64_len(n) + 3);
        self.buf.push(b':');
        push_i64(&mut self.buf, n);
        self.crlf();
    }

    // Strings.

    /// A bulk string, `$len\r\n...\r\n`.
    #[inline]
    pub fn bulk(&mut self, s: &[u8]) {
        self.blob(b'$', s);
    }

    /// A bulk string holding the decimal form of `n`.
    ///
    /// Written straight into the buffer rather than through a temporary, which
    /// is worth having as its own method because several commands reply with a
    /// number as a string and every one of them would otherwise allocate.
    pub fn bulk_int(&mut self, n: i64) {
        let digits = i64_len(n);
        self.buf.reserve(digits + 16);
        self.buf.push(b'$');
        push_u64(&mut self.buf, digits as u64);
        self.crlf();
        push_i64(&mut self.buf, n);
        self.crlf();
    }

    /// A bulk string holding the decimal form of `n`, unsigned.
    ///
    /// Not the same as [`Out::bulk_int`] for the numbers with bit 63 set, which
    /// is the only reason it exists: a scan cursor packs a partition count into
    /// the top bits, so a big enough collection hands back a number that the
    /// signed path would report as negative and no client would send back.
    pub fn bulk_u64(&mut self, n: u64) {
        let mut digits = [0u8; DIGITS_MAX];
        self.bulk(u64_digits(&mut digits, n));
    }

    /// A bulk string holding a double in Redis's own formatting.
    ///
    /// `INCRBYFLOAT` replies with one of these in both protocols, so this is
    /// not the same thing as [`Out::double`] and cannot be written in terms of
    /// it.
    ///
    /// The length has to go in front of the digits and the digits cannot be
    /// counted without writing them, so they are written first, the header is
    /// appended behind them, and the two are rotated into place. A double is a
    /// couple of dozen bytes at most, so the rotate is a few words, and nothing
    /// is allocated to hold a number on its way into a buffer it is already in.
    pub fn bulk_double(&mut self, d: f64) {
        self.buf.reserve(48);
        let start = self.buf.len();
        push_double(&mut self.buf, d);
        let digits = self.buf.len() - start;
        self.buf.push(b'$');
        push_u64(&mut self.buf, digits as u64);
        self.crlf();
        let header = self.buf.len() - start - digits;
        self.buf[start..].rotate_right(header);
        self.crlf();
    }

    /// A verbatim string, RESP3's `=`, with a three byte format such as `txt`
    /// or `mkd`.
    ///
    /// RESP2 has no such type and gets a plain bulk string of the text, without
    /// the format prefix, which is what Redis does.
    pub fn verbatim(&mut self, format: &[u8; 3], text: &[u8]) {
        if !self.proto.is_resp3() {
            self.bulk(text);
            return;
        }
        let len = text.len() + 4;
        self.buf.reserve(len + 16);
        self.buf.push(b'=');
        push_u64(&mut self.buf, len as u64);
        self.crlf();
        self.buf.extend_from_slice(format);
        self.buf.push(b':');
        self.buf.extend_from_slice(text);
        self.crlf();
    }

    /// A big number, RESP3's `(`, given as its decimal digits.
    ///
    /// RESP2 gets a bulk string of the same digits, which is what Redis does
    /// and what every client already handles.
    pub fn big_number(&mut self, digits: &[u8]) {
        if self.proto.is_resp3() {
            self.buf.reserve(digits.len() + 3);
            self.buf.push(b'(');
            self.buf.extend_from_slice(digits);
            self.crlf();
        } else {
            self.bulk(digits);
        }
    }

    // The types RESP3 added.

    /// Nothing, where a string was expected.
    ///
    /// RESP3 has one null. RESP2 has two, and this is the one that stands in
    /// for a missing string, which is what `GET` on a missing key returns.
    #[inline]
    pub fn nil(&mut self) {
        self.buf.extend_from_slice(if self.proto.is_resp3() {
            b"_\r\n"
        } else {
            b"$-1\r\n"
        });
    }

    /// Nothing, where an array was expected.
    ///
    /// The other RESP2 null. `EXEC` on a dirty `WATCH` returns this one, and a
    /// client that tells the two apart will notice if the wrong one is sent.
    #[inline]
    pub fn nil_array(&mut self) {
        self.buf.extend_from_slice(if self.proto.is_resp3() {
            b"_\r\n"
        } else {
            b"*-1\r\n"
        });
    }

    /// A double, RESP3's `,`.
    ///
    /// RESP2 gets a bulk string of the same digits. The infinities and NaN are
    /// written as words in both.
    pub fn double(&mut self, d: f64) {
        if self.proto.is_resp3() {
            self.buf.reserve(32);
            self.buf.push(b',');
            push_double(&mut self.buf, d);
            self.crlf();
            return;
        }
        // RESP2 has no double and gets the digits as a bulk string, which is
        // the same thing `INCRBYFLOAT` replies with on both protocols.
        self.bulk_double(d);
    }

    /// A boolean, RESP3's `#t` or `#f`.
    ///
    /// RESP2 gets `:1` or `:0`, which is what every command that returns a
    /// boolean has always returned there.
    #[inline]
    pub fn bool(&mut self, b: bool) {
        self.buf
            .extend_from_slice(match (self.proto.is_resp3(), b) {
                (true, true) => b"#t\r\n",
                (true, false) => b"#f\r\n",
                (false, true) => b":1\r\n",
                (false, false) => b":0\r\n",
            });
    }

    // Aggregates. Each of these writes only the header; the caller then writes
    // the elements. That is what makes a reply streamable without the codec
    // needing to hold it.

    /// An array header for `n` elements. The caller writes the elements next.
    #[inline]
    pub fn array(&mut self, n: usize) {
        self.header(b'*', n);
    }

    /// Move the last `tail` bytes back to `start`, so that something written
    /// after a reply ends up in front of it.
    ///
    /// Not every reply knows how long it is before it has been written. `SSCAN`
    /// walks a window of the set and drops the members that do not match its
    /// pattern, so the count is only true once the last member has been looked
    /// at, and it answers with a cursor that the same walk produced. The
    /// alternatives are both worse: walking the window twice runs the glob
    /// twice, and collecting the members first is an allocation per call on a
    /// thread that must not allocate.
    ///
    /// Redis solves this with a linked list of reply nodes it can patch in
    /// place. There is one flat buffer here, so the piece that belongs in front
    /// is written behind and the two are rotated past each other, which is the
    /// trick [`Out::bulk_double`] already uses and costs one move of bytes that
    /// were about to be moved to a socket anyway.
    ///
    /// # Panics
    ///
    /// If `start` is past the end, or `tail` is longer than what follows it.
    pub fn hoist(&mut self, start: usize, tail: usize) {
        assert!(
            start + tail <= self.buf.len(),
            "hoisted more than was written"
        );
        self.buf[start..].rotate_right(tail);
    }

    /// An array header for the elements written since `start`, which has to be
    /// a length this buffer reported earlier.
    ///
    /// [`Out::hoist`] is why this can be called after the elements rather than
    /// before them.
    pub fn close_array(&mut self, start: usize, n: usize) {
        let body = self.buf.len() - start;
        self.buf.push(b'*');
        push_u64(&mut self.buf, n as u64);
        self.crlf();
        let header = self.buf.len() - start - body;
        self.hoist(start, header);
    }

    /// A map header for `n` pairs. The caller writes `2 * n` elements next,
    /// key then value, `n` times.
    ///
    /// RESP2 has no map and gets a flat array of twice as many elements, which
    /// is exactly what a RESP2 client already expects from `HGETALL` and
    /// `CONFIG GET`. The command does not know which one it wrote.
    #[inline]
    pub fn map(&mut self, n: usize) {
        if self.proto.is_resp3() {
            self.header(b'%', n);
        } else {
            self.header(b'*', n * 2);
        }
    }

    /// A set header for `n` elements.
    ///
    /// RESP2 has no set and gets an array, which is what `SMEMBERS` has always
    /// returned there.
    #[inline]
    pub fn set(&mut self, n: usize) {
        self.header(if self.proto.is_resp3() { b'~' } else { b'*' }, n);
    }

    /// A push header for `n` elements, RESP3's `>`.
    ///
    /// This is how pub/sub messages and client side caching invalidations are
    /// delivered. RESP2 has no out of band type, so they go out as plain
    /// arrays on the same connection, which is how RESP2 pub/sub has always
    /// worked and is why a RESP2 connection in subscribe mode can only do a
    /// handful of things.
    #[inline]
    pub fn push(&mut self, n: usize) {
        self.header(if self.proto.is_resp3() { b'>' } else { b'*' }, n);
    }

    /// An attribute header for `n` pairs, RESP3's `|`.
    ///
    /// Attributes are metadata attached to the frame that follows. RESP2 cannot
    /// carry them at all, so the caller must check [`Out::proto`] before
    /// writing one. There is no downgrade, because turning metadata into a
    /// reply element would corrupt the reply.
    ///
    /// # Panics
    ///
    /// In debug, if the connection is not speaking RESP3.
    #[inline]
    pub fn attribute(&mut self, n: usize) {
        debug_assert!(
            self.proto.is_resp3(),
            "RESP2 has no attributes, check the protocol first"
        );
        self.header(b'|', n);
    }

    // Sizes, for the presize half of Y18.

    /// The exact number of bytes [`Out::bulk`] would write for a value of this
    /// length.
    #[inline]
    pub const fn bulk_len(value_len: usize) -> usize {
        // `$`, the digits, CRLF, the body, CRLF.
        1 + digits_of(value_len as u64) + 2 + value_len + 2
    }

    /// The exact number of bytes [`Out::int`] would write.
    #[inline]
    pub const fn int_len(n: i64) -> usize {
        1 + i64_len(n) + 2
    }

    /// The exact number of bytes an aggregate header of `n` elements would
    /// write, in either protocol, since both write one byte and the count.
    #[inline]
    pub const fn header_len(n: usize) -> usize {
        1 + digits_of(n as u64) + 2
    }

    #[inline]
    fn header(&mut self, kind: u8, n: usize) {
        self.buf.reserve(24);
        self.buf.push(kind);
        push_u64(&mut self.buf, n as u64);
        self.crlf();
    }

    /// A length prefixed blob: `$`, `!` and `=` all have this shape.
    #[inline]
    fn blob(&mut self, kind: u8, s: &[u8]) {
        self.buf.reserve(Out::bulk_len(s.len()));
        self.buf.push(kind);
        push_u64(&mut self.buf, s.len() as u64);
        self.crlf();
        self.buf.extend_from_slice(s);
        self.crlf();
    }

    #[inline]
    fn crlf(&mut self) {
        self.buf.extend_from_slice(b"\r\n");
    }
}

/// How many decimal digits `n` needs.
const fn digits_of(n: u64) -> usize {
    let mut d = 1;
    let mut v = n;
    while v >= 10 {
        v /= 10;
        d += 1;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs `f` on a fresh buffer in both protocols and returns what each one
    /// produced. Every downgrade test below is written as one call, because the
    /// point being made is always that the same command wrote both.
    fn both(f: impl Fn(&mut Out)) -> (String, String) {
        let mut two = Out::new(Proto::Resp2);
        let mut three = Out::new(Proto::Resp3);
        f(&mut two);
        f(&mut three);
        (
            String::from_utf8(two.into_inner()).unwrap(),
            String::from_utf8(three.into_inner()).unwrap(),
        )
    }

    fn one(proto: Proto, f: impl Fn(&mut Out)) -> String {
        let mut out = Out::new(proto);
        f(&mut out);
        String::from_utf8(out.into_inner()).unwrap()
    }

    #[test]
    fn the_three_types_both_protocols_share_are_written_the_same_way() {
        let (two, three) = both(|o| {
            o.simple(b"PONG");
            o.error(b"WRONGTYPE Operation against a key holding the wrong kind of value");
            o.int(-42);
            o.ok();
        });
        assert_eq!(two, three);
        assert_eq!(
            two,
            "+PONG\r\n-WRONGTYPE Operation against a key holding the wrong kind of value\r\n:-42\r\n+OK\r\n"
        );
    }

    #[test]
    fn a_bulk_string_carries_its_length_and_anything_in_it() {
        let (two, three) = both(|o| {
            o.bulk(b"hello");
            o.bulk(b"");
            o.bulk(b"a\r\nb");
        });
        assert_eq!(two, three);
        assert_eq!(two, "$5\r\nhello\r\n$0\r\n\r\n$4\r\na\r\nb\r\n");
    }

    #[test]
    fn a_number_as_a_string_gets_the_right_length() {
        assert_eq!(one(Proto::Resp2, |o| o.bulk_int(0)), "$1\r\n0\r\n");
        assert_eq!(one(Proto::Resp2, |o| o.bulk_int(-1234)), "$5\r\n-1234\r\n");
        assert_eq!(
            one(Proto::Resp2, |o| o.bulk_int(i64::MIN)),
            "$20\r\n-9223372036854775808\r\n"
        );
    }

    #[test]
    fn an_array_can_be_headed_after_its_elements_are_written() {
        let (two, three) = both(|o| {
            let start = o.len();
            o.bulk(b"a");
            o.bulk(b"bb");
            o.close_array(start, 2);
        });
        assert_eq!(two, three);
        assert_eq!(two, "*2\r\n$1\r\na\r\n$2\r\nbb\r\n");

        // And it leaves whatever was already in the buffer where it was, which
        // is the part a rotate can get wrong.
        assert_eq!(
            one(Proto::Resp2, |o| {
                o.int(1);
                let start = o.len();
                o.bulk(b"x");
                o.close_array(start, 1);
            }),
            ":1\r\n*1\r\n$1\r\nx\r\n"
        );

        // An empty one, and a header of more than one digit, which is where the
        // rotate distance stops being a constant.
        assert_eq!(
            one(Proto::Resp2, |o| {
                let start = o.len();
                o.close_array(start, 0);
            }),
            "*0\r\n"
        );
        let long = one(Proto::Resp2, |o| {
            let start = o.len();
            for _ in 0..100 {
                o.int(7);
            }
            o.close_array(start, 100);
        });
        assert!(long.starts_with("*100\r\n:7\r\n"));
        assert!(long.ends_with(":7\r\n"));
        assert_eq!(long.len(), "*100\r\n".len() + 100 * ":7\r\n".len());
    }

    #[test]
    fn resp2_has_two_nulls_and_resp3_has_one() {
        let (two, three) = both(|o| {
            o.nil();
            o.nil_array();
        });
        assert_eq!(two, "$-1\r\n*-1\r\n");
        assert_eq!(three, "_\r\n_\r\n");
    }

    #[test]
    fn a_map_becomes_a_flat_array_on_resp2() {
        let (two, three) = both(|o| {
            o.map(2);
            o.bulk(b"a");
            o.bulk(b"1");
            o.bulk(b"b");
            o.bulk(b"2");
        });
        assert_eq!(two, "*4\r\n$1\r\na\r\n$1\r\n1\r\n$1\r\nb\r\n$1\r\n2\r\n");
        assert_eq!(three, "%2\r\n$1\r\na\r\n$1\r\n1\r\n$1\r\nb\r\n$1\r\n2\r\n");
    }

    #[test]
    fn a_set_and_a_push_become_arrays_on_resp2() {
        let (two, three) = both(|o| {
            o.set(1);
            o.bulk(b"x");
            o.push(2);
            o.bulk(b"message");
            o.bulk(b"ch");
        });
        assert_eq!(two, "*1\r\n$1\r\nx\r\n*2\r\n$7\r\nmessage\r\n$2\r\nch\r\n");
        assert_eq!(
            three,
            "~1\r\n$1\r\nx\r\n>2\r\n$7\r\nmessage\r\n$2\r\nch\r\n"
        );
    }

    #[test]
    fn a_boolean_is_an_integer_on_resp2() {
        let (two, three) = both(|o| {
            o.bool(true);
            o.bool(false);
        });
        assert_eq!(two, ":1\r\n:0\r\n");
        assert_eq!(three, "#t\r\n#f\r\n");
    }

    #[test]
    fn a_double_is_a_bulk_string_on_resp2() {
        let (two, three) = both(|o| {
            o.double(1.5);
            o.double(3.0);
            o.double(f64::INFINITY);
        });
        assert_eq!(two, "$3\r\n1.5\r\n$1\r\n3\r\n$3\r\ninf\r\n");
        assert_eq!(three, ",1.5\r\n,3\r\n,inf\r\n");
    }

    #[test]
    fn a_verbatim_string_loses_its_format_on_resp2() {
        let (two, three) = both(|o| o.verbatim(b"txt", b"Some string"));
        assert_eq!(two, "$11\r\nSome string\r\n");
        assert_eq!(three, "=15\r\ntxt:Some string\r\n");
    }

    #[test]
    fn a_big_number_is_a_bulk_string_on_resp2() {
        let n = b"3492890328409238509324850943850943825024385";
        let (two, three) = both(|o| o.big_number(n));
        assert_eq!(
            two,
            format!("${}\r\n{}\r\n", n.len(), str::from_utf8(n).unwrap())
        );
        assert_eq!(three, format!("({}\r\n", str::from_utf8(n).unwrap()));
    }

    #[test]
    fn a_blob_error_keeps_its_newlines_on_resp3_and_loses_them_on_resp2() {
        let (two, three) = both(|o| o.blob_error(b"SYNTAX bad\nline two"));
        assert_eq!(two, "-SYNTAX bad line two\r\n");
        assert_eq!(three, "!19\r\nSYNTAX bad\nline two\r\n");
    }

    /// The sizes are what a command presizes from, so a size that is wrong by
    /// one is a reply that reallocates on every call and nobody notices.
    #[test]
    fn the_predicted_sizes_are_the_sizes_actually_written() {
        for len in [0usize, 1, 9, 10, 99, 100, 1000, 65536] {
            let value = vec![b'x'; len];
            let written = one(Proto::Resp2, |o| o.bulk(&value));
            assert_eq!(Out::bulk_len(len), written.len(), "bulk of {len}");
        }
        for n in [0i64, 7, -7, 100, i64::MAX, i64::MIN] {
            let written = one(Proto::Resp2, |o| o.int(n));
            assert_eq!(Out::int_len(n), written.len(), "int {n}");
        }
        for n in [0usize, 5, 1234] {
            let written = one(Proto::Resp2, |o| o.array(n));
            assert_eq!(Out::header_len(n), written.len(), "array header {n}");
        }
    }

    #[test]
    fn hello_switches_the_protocol_for_everything_after_it() {
        let mut out = Out::new(Proto::Resp2);
        out.nil();
        out.set_proto(Proto::Resp3);
        out.nil();
        assert_eq!(out.as_slice(), b"$-1\r\n_\r\n");
    }

    #[test]
    fn a_partial_write_leaves_the_rest_behind() {
        let mut out = Out::new(Proto::Resp2);
        out.ok();
        out.ok();
        out.consume(5);
        assert_eq!(out.as_slice(), b"+OK\r\n");
        out.clear();
        assert!(out.is_empty());
    }
}
