//! JSON text into a [`Builder`] and back out of a [`Value`].
//!
//! The typed API never comes through here: a struct is serialized straight into
//! the encoding and read straight back out, and text would be two conversions
//! nobody asked for. This is for the other door. `JSON.SET` arrives with a
//! bulk string that is JSON text and `JSON.GET` has to hand one back, so the
//! whole `JSON.*` surface stands on these two functions and neither of them can
//! be a dependency, because a JSON parser is where a compatibility claim goes to
//! die and this one has to agree with RedisJSON down to the byte.
//!
//! ```
//! use yo_doc::{Builder, Value};
//!
//! let mut b = Builder::new();
//! b.json(br#"{"name": "a wrench", "price": 12.5, "tags": ["hand", "steel"]}"#)?;
//! let doc = b.finish()?.to_vec();
//!
//! let v = Value::new(&doc).expect("readable");
//! assert_eq!(v.get(b"price").unwrap().as_float(), Some(12.5));
//! // Key order, not the order the text had them in. See below.
//! assert_eq!(v.to_json()?, br#"{"name":"a wrench","tags":["hand","steel"],"price":12.5}"#);
//! # Ok::<(), yo_common::Error>(())
//! ```
//!
//! # What the parser accepts
//!
//! RFC 8259 and nothing else. No trailing commas, no comments, no unquoted
//! keys, no single quoted strings, no leading plus, no leading zero, no bare
//! `NaN` or `Infinity`. Every one of those is something some parser somewhere
//! allows, and accepting one of them means a document that loads here and is
//! refused by a real Redis, which is a divergence that nobody would think to
//! look for. Being strict is the only setting that can be checked.
//!
//! A number without a fraction or an exponent that fits in an `i64` is stored as
//! an integer and everything else is stored as a float, which is the same split
//! RedisJSON makes and the reason `1` comes back as `1` rather than as `1.0`.
//! An integer literal too big for an `i64` becomes a float, and loses precision
//! the way it would anywhere else. So does `-0`, which is a number an integer
//! cannot hold and a double can.
//!
//! # Two things the writer does that are worth knowing
//!
//! **An object comes out in key order**, which is by length first and then by
//! bytes, so `name` comes before `price` and not after it. Members are stored
//! sorted because that is what makes a lookup a binary search, so the order a
//! client wrote them in is not kept anywhere and cannot be handed back.
//! RedisJSON keeps it. That shows up on any document with more than one key and
//! it belongs in the divergence register rather than in a footnote.
//!
//! **A float is printed as the shortest text that reads back as the same
//! double**, with a `.0` added when it would otherwise look like an integer.
//! Without that, a document that went out and came back would change type on
//! every round trip, which is the sort of thing that only shows up three
//! services downstream.

use core::fmt::Write as _;

use yo_common::{Code, Error, Result};

use crate::build::Builder;
use crate::head::{DEPTH_MAX, Kind};
use crate::read::Value;

/// One JSON document, as the bytes it encodes to.
///
/// A caller with more than one document to read should keep a [`Builder`] and
/// call [`Builder::json`] on it instead, since that is the whole reason a
/// builder can be cleared and reused.
pub fn from_json(text: &[u8]) -> Result<Vec<u8>> {
    let mut b = Builder::new();
    b.json(text)?;
    Ok(b.finish()?.to_vec())
}

impl Builder {
    /// Write the one JSON value in `text`.
    ///
    /// This writes a value where a value goes, so it works on an empty builder
    /// and it works just as well after a [`Builder::key`] inside an open
    /// object, which is what a path update needs: the parts of the document
    /// that are not changing are copied with [`Builder::embed`] and the part
    /// that is arrives as text.
    ///
    /// `text` holds exactly one value. Anything after it, other than
    /// whitespace, is an error rather than something quietly ignored, because a
    /// client that sent two values meant something and it was not this.
    pub fn json(&mut self, text: &[u8]) -> Result<()> {
        // Checked once here so that every string literal inside can be taken as
        // UTF-8 without checking it again. Slicing at escape boundaries cannot
        // break that, since every character an escape is made of is ASCII.
        if core::str::from_utf8(text).is_err() {
            return Err(Error::new(Code::Invalid, "the JSON text is not UTF-8"));
        }
        let mut r = Reader {
            text,
            at: 0,
            scratch: Vec::new(),
        };
        r.space();
        r.value(self)?;
        r.space();
        if r.at < text.len() {
            return Err(r.bad("more text after the value the document is"));
        }
        Ok(())
    }
}

/// How a document is laid out when it is written as text.
///
/// All three are empty by default, which is one line and no spaces, and that is
/// what `JSON.GET` answers when the client did not ask for anything else. The
/// three are Redis's `INDENT`, `NEWLINE` and `SPACE`, and they are strings
/// rather than counts there, so they are strings here.
#[derive(Debug, Clone, Copy, Default)]
pub struct Format<'a> {
    /// Written once per level of nesting at the start of a line.
    pub indent: &'a [u8],
    /// Written at the end of a line.
    pub newline: &'a [u8],
    /// Written after the colon between a key and its value.
    pub space: &'a [u8],
}

impl Format<'_> {
    /// Whether this asks for anything at all.
    ///
    /// A container with nothing in it is written as `{}` either way, so the
    /// laid out form only differs from the compact one where there is something
    /// to lay out. `JSON.GET` builds its own wrapper around what a path matched
    /// and has to make the same decision about it, which is why this is public.
    #[must_use]
    pub fn is_plain(&self) -> bool {
        self.indent.is_empty() && self.newline.is_empty() && self.space.is_empty()
    }
}

impl Value<'_> {
    /// This value as JSON text, on one line.
    pub fn to_json(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.write_json(&mut out)?;
        Ok(out)
    }

    /// This value as JSON text, appended to a buffer the caller owns.
    ///
    /// The reply path has one of those per connection, so a `JSON.GET` over a
    /// thousand keys is one buffer rather than a thousand.
    pub fn write_json(&self, out: &mut Vec<u8>) -> Result<()> {
        write_value(self, &Format::default(), out, 0)
    }

    /// The same, laid out the way `f` asks for.
    pub fn write_json_with(&self, f: &Format<'_>, out: &mut Vec<u8>) -> Result<()> {
        write_value(self, f, out, 0)
    }

    /// The same again, as if this value were already `depth` levels down.
    ///
    /// `JSON.GET` wraps what a JSONPath matched in an array and wraps several
    /// paths in an object keyed by the paths, and it lays those wrappers out
    /// too, so the values inside them start one or two levels in rather than at
    /// the margin. The wrapper is built by the caller, which is the only thing
    /// that knows how deep it went.
    pub fn write_json_at(&self, f: &Format<'_>, out: &mut Vec<u8>, depth: usize) -> Result<()> {
        write_value(self, f, out, depth)
    }
}

// ------------------------------------------------------------------- the text

/// A cursor over JSON text.
struct Reader<'a> {
    text: &'a [u8],
    at: usize,
    /// Where a string with escapes in it is put back together. Kept on the
    /// reader rather than made per string, so a document full of escaped
    /// strings allocates once.
    scratch: Vec<u8>,
}

/// Where the bytes of a string literal ended up.
///
/// Most strings have no escapes in them, and those are a range of the text and
/// no copy at all. The rest are built in [`Reader::scratch`], one at a time,
/// which is why this is a marker rather than a slice: the slice would borrow
/// the reader for as long as the caller held it.
enum Str {
    Plain(usize, usize),
    Escaped,
}

impl<'a> Reader<'a> {
    /// Step over whitespace, which JSON says is these four bytes and no others.
    fn space(&mut self) {
        while let Some(&c) = self.text.get(self.at) {
            if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' {
                self.at += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.text.get(self.at).copied()
    }

    /// Step over `word` if that is what is here.
    fn word(&mut self, word: &[u8]) -> bool {
        if self.text[self.at..].starts_with(word) {
            self.at += word.len();
            true
        } else {
            false
        }
    }

    /// One value, and whatever is under it.
    ///
    /// The recursion is bounded by the builder rather than here: opening the
    /// hundred and twenty ninth container is an error, and the error comes
    /// straight back up, so there is no depth to count in this function.
    fn value(&mut self, b: &mut Builder) -> Result<()> {
        match self.peek() {
            None => Err(self.bad("the text ends where a value should be")),
            Some(b'n') if self.word(b"null") => b.null(),
            Some(b't') if self.word(b"true") => b.bool(true),
            Some(b'f') if self.word(b"false") => b.bool(false),
            Some(b'"') => {
                let s = self.string()?;
                let bytes = self.bytes_of(&s);
                b.text_bytes(bytes)
            }
            Some(b'[') => self.array(b),
            Some(b'{') => self.object(b),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(b),
            Some(_) => Err(self.bad("this is not the start of a value")),
        }
    }

    fn array(&mut self, b: &mut Builder) -> Result<()> {
        self.at += 1;
        b.begin_array()?;
        self.space();
        if self.peek() == Some(b']') {
            self.at += 1;
            return b.end_array();
        }
        loop {
            self.space();
            self.value(b)?;
            self.space();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return b.end_array();
                }
                _ => return Err(self.bad("an array element is followed by `,` or by `]`")),
            }
        }
    }

    fn object(&mut self, b: &mut Builder) -> Result<()> {
        self.at += 1;
        b.begin_object()?;
        self.space();
        if self.peek() == Some(b'}') {
            self.at += 1;
            return b.end_object();
        }
        loop {
            self.space();
            if self.peek() != Some(b'"') {
                return Err(self.bad("an object key is a string"));
            }
            let s = self.string()?;
            b.key(self.bytes_of(&s))?;
            self.space();
            if self.peek() != Some(b':') {
                return Err(self.bad("an object key is followed by `:`"));
            }
            self.at += 1;
            self.space();
            self.value(b)?;
            self.space();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return b.end_object();
                }
                _ => return Err(self.bad("an object member is followed by `,` or by `}`")),
            }
        }
    }

    /// The bytes a [`Str`] named, whichever of the two places it is in.
    fn bytes_of(&self, s: &Str) -> &[u8] {
        match *s {
            Str::Plain(from, to) => &self.text[from..to],
            Str::Escaped => &self.scratch,
        }
    }

    /// A string literal, with the opening quote still under the cursor.
    fn string(&mut self) -> Result<Str> {
        self.at += 1;
        let from = self.at;
        // The common case walked first and on its own, so a string with no
        // escapes never touches the scratch buffer and never copies a byte.
        while let Some(c) = self.peek() {
            match c {
                b'"' => {
                    let to = self.at;
                    self.at += 1;
                    return Ok(Str::Plain(from, to));
                }
                b'\\' => break,
                // A raw control byte inside a string is what RFC 8259 forbids
                // and what every other parser also refuses, so accepting it
                // would be a divergence with nothing to gain.
                0..=0x1f => return Err(self.bad("a control byte inside a string")),
                _ => self.at += 1,
            }
        }

        self.scratch.clear();
        self.scratch.extend_from_slice(&self.text[from..self.at]);
        loop {
            let Some(c) = self.peek() else {
                return Err(self.bad("the text ends inside a string"));
            };
            self.at += 1;
            match c {
                b'"' => return Ok(Str::Escaped),
                0..=0x1f => return Err(self.bad("a control byte inside a string")),
                b'\\' => self.escape()?,
                _ => self.scratch.push(c),
            }
        }
    }

    /// One escape, with the backslash already stepped over.
    fn escape(&mut self) -> Result<()> {
        let Some(c) = self.peek() else {
            return Err(self.bad("the text ends inside an escape"));
        };
        self.at += 1;
        let plain = match c {
            b'"' => b'"',
            b'\\' => b'\\',
            b'/' => b'/',
            b'b' => 0x08,
            b'f' => 0x0c,
            b'n' => b'\n',
            b'r' => b'\r',
            b't' => b'\t',
            b'u' => return self.unicode(),
            _ => return Err(self.bad("this is not an escape JSON has")),
        };
        self.scratch.push(plain);
        Ok(())
    }

    /// A `\u` escape and, if it opened a surrogate pair, the one that closes it.
    fn unicode(&mut self) -> Result<()> {
        let first = self.hex4()?;
        let ch = if (0xd800..0xdc00).contains(&first) {
            // A high surrogate on its own is not a character, so the low one
            // has to be right here. Anything else is text that claims to be
            // UTF-16 and is not, and guessing a replacement character for it
            // would store something the client never sent.
            if !(self.peek() == Some(b'\\') && self.text.get(self.at + 1) == Some(&b'u')) {
                return Err(self.bad("a high surrogate with no low surrogate after it"));
            }
            self.at += 2;
            let second = self.hex4()?;
            if !(0xdc00..0xe000).contains(&second) {
                return Err(
                    self.bad("a high surrogate followed by something that is not a low one")
                );
            }
            0x10000 + ((first - 0xd800) << 10) + (second - 0xdc00)
        } else if (0xdc00..0xe000).contains(&first) {
            return Err(self.bad("a low surrogate with no high surrogate before it"));
        } else {
            first
        };
        let ch = char::from_u32(ch).ok_or_else(|| self.bad("an escape that is not a character"))?;
        let mut buf = [0u8; 4];
        self.scratch
            .extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
        Ok(())
    }

    /// The four hex digits of a `\u` escape.
    fn hex4(&mut self) -> Result<u32> {
        let Some(digits) = self.text.get(self.at..self.at + 4) else {
            return Err(self.bad("an escape with fewer than four hex digits"));
        };
        let mut v = 0u32;
        for &d in digits {
            let n = match d {
                b'0'..=b'9' => u32::from(d - b'0'),
                b'a'..=b'f' => u32::from(d - b'a') + 10,
                b'A'..=b'F' => u32::from(d - b'A') + 10,
                _ => return Err(self.bad("an escape with something that is not a hex digit")),
            };
            v = v * 16 + n;
        }
        self.at += 4;
        Ok(v)
    }

    /// A number, written out by the grammar rather than handed to a parser and
    /// checked afterwards, because the shapes Rust's own `parse` accepts and the
    /// shapes JSON allows are not the same list.
    fn number(&mut self, b: &mut Builder) -> Result<()> {
        let from = self.at;
        if self.peek() == Some(b'-') {
            self.at += 1;
        }
        match self.peek() {
            // A leading zero is a whole number on its own, which is what stops
            // `0123` from being read as `123` here and as an octal somewhere
            // else.
            Some(b'0') => self.at += 1,
            Some(c) if c.is_ascii_digit() => self.digits(),
            _ => return Err(self.bad("a number with no digits in it")),
        }
        let mut whole = true;
        if self.peek() == Some(b'.') {
            self.at += 1;
            if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
                return Err(self.bad("a decimal point with no digits after it"));
            }
            self.digits();
            whole = false;
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.at += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.at += 1;
            }
            if !self.peek().is_some_and(|c| c.is_ascii_digit()) {
                return Err(self.bad("an exponent with no digits in it"));
            }
            self.digits();
            whole = false;
        }

        let text = core::str::from_utf8(&self.text[from..self.at])
            .expect("a number is the ASCII this function just walked over");
        // An integer that does not fit falls through to the float, which is the
        // only thing that can be done with it and is what everyone else does.
        // So does a negative zero, which is a number an integer cannot hold and
        // a double can, and which every other JSON parser reads as a double for
        // that reason. It matters because a document that went through here
        // would otherwise come back out with the sign gone.
        if whole
            && text != "-0"
            && let Ok(i) = text.parse::<i64>()
        {
            return b.int(i);
        }
        let f: f64 = text
            .parse()
            .map_err(|_| self.bad("a number that does not fit in a double"))?;
        b.float(f)
    }

    fn digits(&mut self) {
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.at += 1;
        }
    }

    /// An error that says where it happened, because a client that sent four
    /// kilobytes of JSON needs the offset more than it needs the adjective.
    fn bad(&self, what: &str) -> Error {
        Error::fmt(
            Code::Invalid,
            format_args!("{what}, at byte {} of the JSON text", self.at),
        )
    }
}

// ----------------------------------------------------------------- the writing

fn write_value(v: &Value<'_>, f: &Format<'_>, out: &mut Vec<u8>, depth: usize) -> Result<()> {
    match v.kind() {
        Kind::Null => out.extend_from_slice(b"null"),
        Kind::Bool => out.extend_from_slice(if v.as_bool() == Some(true) {
            b"true".as_slice()
        } else {
            b"false".as_slice()
        }),
        Kind::Int => {
            let i = v.as_int().ok_or_else(unreadable)?;
            write!(Sink(out), "{i}").expect("a Vec never fails a write");
        }
        Kind::Float => write_float(v.as_float().ok_or_else(unreadable)?, out)?,
        Kind::Text => write_string(v.text_bytes().ok_or_else(unreadable)?, out),
        Kind::Array => {
            deeper(depth)?;
            let laid_out = !f.is_plain() && !v.is_empty();
            out.push(b'[');
            for (i, e) in v.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                if laid_out {
                    line(f, out, depth + 1);
                }
                write_value(&e, f, out, depth + 1)?;
            }
            if laid_out {
                line(f, out, depth);
            }
            out.push(b']');
        }
        Kind::Object => {
            deeper(depth)?;
            if v.is_interned() {
                return Err(Error::new(
                    Code::Invalid,
                    "an object whose keys are interned needs the collection's key table to be written as text",
                ));
            }
            let laid_out = !f.is_plain() && !v.is_empty();
            out.push(b'{');
            for (i, (key, e)) in v.members().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                if laid_out {
                    line(f, out, depth + 1);
                }
                write_string(key, out);
                out.push(b':');
                out.extend_from_slice(f.space);
                write_value(&e, f, out, depth + 1)?;
            }
            if laid_out {
                line(f, out, depth);
            }
            out.push(b'}');
        }
    }
    Ok(())
}

/// End the line and indent the next one to `depth`.
fn line(f: &Format<'_>, out: &mut Vec<u8>, depth: usize) {
    out.extend_from_slice(f.newline);
    for _ in 0..depth {
        out.extend_from_slice(f.indent);
    }
}

/// A document this crate wrote never nests past [`DEPTH_MAX`], so this only
/// fires on one that arrived damaged, and it fires before the recursion does
/// rather than after the stack has gone.
fn deeper(depth: usize) -> Result<()> {
    if depth >= DEPTH_MAX {
        return Err(Error::fmt(
            Code::Corrupt,
            format_args!("the document nests past {DEPTH_MAX} levels"),
        ));
    }
    Ok(())
}

fn write_float(f: f64, out: &mut Vec<u8>) -> Result<()> {
    if !f.is_finite() {
        return Err(Error::new(
            Code::Invalid,
            "JSON has no way to write an infinity or a NaN",
        ));
    }
    let from = out.len();
    write!(Sink(out), "{f}").expect("a Vec never fails a write");
    // Rust prints a whole double as `1` rather than as `1.0`, and reading that
    // back gives an integer, so a document left alone would change type once
    // per round trip.
    if !out[from..].iter().any(|&c| matches!(c, b'.' | b'e' | b'E')) {
        out.extend_from_slice(b".0");
    }
    Ok(())
}

/// A string, quoted and escaped.
///
/// The bytes above `0x7f` are copied as they are, which keeps UTF-8 as UTF-8
/// and is what every JSON writer worth using does. A string that was not UTF-8
/// going in, which only [`Builder::text_bytes`] can produce, comes out the same
/// way it went in and the result is not valid JSON. That is the caller's doing
/// and re-encoding it would be inventing bytes.
fn write_string(s: &[u8], out: &mut Vec<u8>) {
    out.push(b'"');
    for &c in s {
        match c {
            b'"' => out.extend_from_slice(b"\\\""),
            b'\\' => out.extend_from_slice(b"\\\\"),
            0x08 => out.extend_from_slice(b"\\b"),
            0x0c => out.extend_from_slice(b"\\f"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\t' => out.extend_from_slice(b"\\t"),
            0..=0x1f => write!(Sink(out), "\\u{c:04x}").expect("a Vec never fails a write"),
            _ => out.push(c),
        }
    }
    out.push(b'"');
}

/// A `Vec<u8>` that `write!` can be pointed at, so that formatting a number
/// lands in the caller's buffer instead of in a `String` on the way there.
struct Sink<'a>(&'a mut Vec<u8>);

impl core::fmt::Write for Sink<'_> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        self.0.extend_from_slice(s.as_bytes());
        Ok(())
    }
}

fn unreadable() -> Error {
    Error::new(Code::Corrupt, "a value whose header and payload disagree")
}

#[cfg(test)]
mod tests {
    use super::*;
    use yo_common::Rng;

    /// The text, through the encoding, and back to text.
    fn round(text: &str) -> String {
        let bytes = from_json(text.as_bytes()).expect("the text parses");
        let v = Value::new(&bytes).expect("readable");
        assert!(
            v.validate(),
            "the encoding this produced does not check out"
        );
        String::from_utf8(v.to_json().expect("writable")).expect("UTF-8")
    }

    fn why(text: &str) -> String {
        from_json(text.as_bytes())
            .expect_err("this should not parse")
            .message()
            .to_string()
    }

    /// The text, through the encoding, and back out laid out the way `f` asks.
    fn laid_out(text: &str, f: &Format<'_>) -> String {
        let bytes = from_json(text.as_bytes()).expect("the text parses");
        let v = Value::new(&bytes).expect("readable");
        let mut out = Vec::new();
        v.write_json_with(f, &mut out).expect("writable");
        String::from_utf8(out).expect("UTF-8")
    }

    #[test]
    fn a_document_is_laid_out_the_way_json_get_asks_for() {
        let f = Format {
            indent: b"  ",
            newline: b"\n",
            space: b" ",
        };
        assert_eq!(
            laid_out(r#"{"a":1,"bb":[2,3]}"#, &f),
            "{\n  \"a\": 1,\n  \"bb\": [\n    2,\n    3\n  ]\n}"
        );
        // A container with nothing in it has nothing to lay out, so it stays on
        // the one line either way.
        assert_eq!(
            laid_out(r#"{"a":{},"bb":[]}"#, &f),
            "{\n  \"a\": {},\n  \"bb\": []\n}"
        );
        // A scalar is a scalar whatever was asked for.
        assert_eq!(laid_out("1.5", &f), "1.5");
        // Asking for nothing is the compact form, byte for byte.
        assert_eq!(
            laid_out(r#"{"a":1,"bb":[2,3]}"#, &Format::default()),
            round(r#"{"a":1,"bb":[2,3]}"#)
        );
        // The three are separate, so a client that asked for one gets one.
        let only_space = Format {
            space: b" ",
            ..Format::default()
        };
        assert_eq!(
            laid_out(r#"{"a":1,"bb":2}"#, &only_space),
            r#"{"a": 1,"bb": 2}"#
        );
    }

    #[test]
    fn a_document_comes_back_as_the_text_it_went_in_as() {
        assert_eq!(round("null"), "null");
        assert_eq!(round("true"), "true");
        assert_eq!(round("false"), "false");
        assert_eq!(round("0"), "0");
        assert_eq!(round("-17"), "-17");
        assert_eq!(round(r#""hello""#), r#""hello""#);
        assert_eq!(round("[]"), "[]");
        assert_eq!(round("{}"), "{}");
        assert_eq!(round(r#"[1,[2,[3]]]"#), "[1,[2,[3]]]");
        assert_eq!(round(r#"{"a":{"b":[1,2,3]}}"#), r#"{"a":{"b":[1,2,3]}}"#);
    }

    #[test]
    fn whitespace_is_allowed_where_json_allows_it_and_is_not_kept() {
        assert_eq!(round("  \t\r\n [ 1 , 2 ]  \n"), "[1,2]");
        assert_eq!(round("{ \"a\" : 1 , \"b\" : 2 }"), r#"{"a":1,"b":2}"#);
    }

    #[test]
    fn a_whole_number_stays_whole_and_the_rest_do_not() {
        assert_eq!(round("1"), "1");
        assert_eq!(round("1.0"), "1.0");
        assert_eq!(round("1e2"), "100.0");
        assert_eq!(round("-0.5"), "-0.5");
        assert_eq!(round("9223372036854775807"), "9223372036854775807");
        // One past an i64, which is the point the split has to happen at.
        assert_eq!(round("9223372036854775808"), "9223372036854776000.0");
        let bytes = from_json(b"1").expect("parses");
        assert_eq!(Value::new(&bytes).expect("readable").kind(), Kind::Int);
        let bytes = from_json(b"1.0").expect("parses");
        assert_eq!(Value::new(&bytes).expect("readable").kind(), Kind::Float);
        // A negative zero is the one whole number that is a double, because an
        // integer cannot hold the sign and losing it would change the document.
        assert_eq!(round("-0"), "-0.0");
        assert_eq!(round("0"), "0");
        let bytes = from_json(b"-0").expect("parses");
        assert_eq!(Value::new(&bytes).expect("readable").kind(), Kind::Float);
    }

    #[test]
    fn an_escape_is_read_and_only_written_back_when_it_has_to_be() {
        assert_eq!(round(r#""a\"b""#), r#""a\"b""#);
        assert_eq!(round(r#""a\\b""#), r#""a\\b""#);
        assert_eq!(round(r#""a\nb""#), r#""a\nb""#);
        assert_eq!(round(r#""a\tb""#), r#""a\tb""#);
        assert_eq!(round(r#""a b""#), r#""a b""#);
        // A solidus may be escaped and does not have to be, so it goes in one
        // way and comes out the other.
        assert_eq!(round(r#""a\/b""#), r#""a/b""#);
        // And anything that is already a character comes back as that
        // character rather than as an escape.
        assert_eq!(round(r#""é""#), "\"\u{e9}\"");
        assert_eq!(round(r#""😀""#), "\"\u{1f600}\"");
        assert_eq!(round("\"caf\u{e9}\""), "\"caf\u{e9}\"");
    }

    #[test]
    fn a_string_with_no_escapes_in_it_is_not_copied_through_the_scratch() {
        let mut r = Reader {
            text: br#""plain" "esc\n""#,
            at: 0,
            scratch: Vec::new(),
        };
        assert!(matches!(r.string().expect("parses"), Str::Plain(1, 6)));
        r.space();
        assert!(matches!(r.string().expect("parses"), Str::Escaped));
        assert_eq!(r.scratch, b"esc\n");
    }

    #[test]
    fn an_object_comes_back_in_key_order_and_the_last_of_a_repeated_key_wins() {
        assert_eq!(round(r#"{"b":1,"a":2}"#), r#"{"a":2,"b":1}"#);
        assert_eq!(round(r#"{"a":1,"a":2}"#), r#"{"a":2}"#);
    }

    #[test]
    fn the_parser_refuses_what_is_not_json() {
        assert!(why("").contains("ends where a value should be"));
        assert!(why("[1,]").contains("not the start of a value"));
        assert!(why("[1 2]").contains("`,` or by `]`"));
        assert!(why("{a:1}").contains("key is a string"));
        assert!(why(r#"{"a" 1}"#).contains("followed by `:`"));
        assert!(why(r#"{"a":1,}"#).contains("key is a string"));
        assert!(why("'a'").contains("not the start of a value"));
        assert!(why("01").contains("more text after the value"));
        assert!(why("+1").contains("not the start of a value"));
        assert!(why("1.").contains("no digits after it"));
        assert!(why(".5").contains("not the start of a value"));
        assert!(why("1e").contains("exponent with no digits"));
        assert!(why("NaN").contains("not the start of a value"));
        assert!(why("Infinity").contains("not the start of a value"));
        assert!(why("nul").contains("not the start of a value"));
        assert!(why("1 2").contains("more text after the value"));
        assert!(why("// a comment\n1").contains("not the start of a value"));
        assert!(why("\"a\nb\"").contains("control byte inside a string"));
        assert!(why(r#""a"#).contains("ends inside a string"));
        assert!(why(r#""\x""#).contains("not an escape JSON has"));
        assert!(why(r#""\u00"#).contains("fewer than four hex digits"));
        assert!(why(r#""\uzzzz""#).contains("not a hex digit"));
        assert!(why(r#""\ud83d""#).contains("no low surrogate after it"));
        assert!(why(r#""\ude00""#).contains("no high surrogate before it"));
        assert!(why(r#""\ud83da""#).contains("no low surrogate after it"));
        assert!(why(r#""\ud83d\u0041""#).contains("something that is not a low one"));
        assert!(why("[").contains("ends where a value should be"));
        assert!(why("{").contains("key is a string"));
    }

    #[test]
    fn an_error_says_where_it_was() {
        assert!(why("[1, 2, x]").contains("at byte 7"));
    }

    /// The two halves against each other over documents nobody chose.
    ///
    /// A hand written table of cases tests the cases somebody thought of, and
    /// the way a parser and a writer disagree is almost always over something
    /// neither author thought of. So this builds documents at random, writes
    /// them out and reads them back, and asks for the encoding to be the same
    /// bytes both times. That is a stronger claim than the text matching:
    /// identical bytes means every type, every key and every number survived,
    /// and it catches a writer that loses a distinction the encoding was
    /// keeping.
    #[test]
    fn a_document_survives_being_written_out_and_read_back() {
        let mut rng = Rng::new(0x0d0c);
        for _ in 0..500 {
            let mut b = Builder::new();
            grow(&mut b, &mut rng, 0);
            let first = b.finish().expect("finished").to_vec();

            let v = Value::new(&first).expect("readable");
            let text = v.to_json().expect("writable");
            let again = from_json(&text)
                .unwrap_or_else(|e| panic!("{}: {}", String::from_utf8_lossy(&text), e.message()));
            assert_eq!(
                first,
                again,
                "{} did not come back as itself",
                String::from_utf8_lossy(&text)
            );
        }
    }

    /// One random value, and whatever it decides to nest under itself.
    ///
    /// The scalars are the ones JSON can carry, so no infinity, no NaN and no
    /// string that is not UTF-8, since those are cases the writer refuses on
    /// purpose and they have their own tests. The strings are drawn from bytes
    /// that need escaping and bytes that do not, in both planes, because the
    /// escaping is the half of this most likely to be wrong.
    fn grow(b: &mut Builder, rng: &mut Rng, depth: usize) {
        const CHARS: [char; 12] = [
            'a',
            'z',
            '"',
            '\\',
            '\n',
            '\t',
            '\u{0}',
            '\u{1f}',
            '/',
            '\u{e9}',
            '\u{4e2d}',
            '\u{1f600}',
        ];
        let pick = rng.next_u64() % if depth >= 4 { 6 } else { 8 };
        match pick {
            0 => b.null().expect("value"),
            1 => b.bool(rng.next_u64() & 1 == 0).expect("value"),
            2 => b.int(rng.next_u64() as i64).expect("value"),
            3 => b
                .float(f64::from_bits(rng.next_u64()).clamp(-1e300, 1e300))
                .expect("value"),
            4 => b.int(i64::from(rng.next_u64() as u8) - 128).expect("value"),
            5 => {
                let n = rng.next_u64() as usize % 8;
                let s: String = (0..n)
                    .map(|_| CHARS[rng.next_u64() as usize % CHARS.len()])
                    .collect();
                b.text(&s).expect("value");
            }
            6 => {
                b.begin_array().expect("open");
                for _ in 0..rng.next_u64() % 4 {
                    grow(b, rng, depth + 1);
                }
                b.end_array().expect("close");
            }
            _ => {
                b.begin_object().expect("open");
                for i in 0..rng.next_u64() % 4 {
                    // Keys of more than one length, so that the length first
                    // ordering is exercised and not only the byte ordering.
                    let key = "k".repeat(1 + i as usize % 3) + &i.to_string();
                    b.key(key.as_bytes()).expect("key");
                    grow(b, rng, depth + 1);
                }
                b.end_object().expect("close");
            }
        }
    }

    #[test]
    fn text_that_is_not_utf8_is_refused_before_anything_is_parsed() {
        let e = from_json(&[b'"', 0xff, b'"']).expect_err("not UTF-8");
        assert!(e.message().contains("not UTF-8"));
    }

    #[test]
    fn a_document_deeper_than_the_limit_is_refused_rather_than_recursed_into() {
        let deep = format!("{}1{}", "[".repeat(200), "]".repeat(200));
        assert!(why(&deep).contains("nests at most"));
        // And one exactly at the limit is fine, so the refusal is the limit and
        // not something one short of it.
        let ok = format!("{}1{}", "[".repeat(DEPTH_MAX), "]".repeat(DEPTH_MAX));
        assert_eq!(round(&ok), ok);
    }

    #[test]
    fn json_writes_a_value_where_a_value_goes_and_not_only_at_the_root() {
        let mut b = Builder::new();
        b.begin_object().expect("open");
        b.key(b"meta").expect("key");
        b.json(br#"{"seen":2,"tags":["a"]}"#).expect("parses");
        b.key(b"id").expect("key");
        b.int(7).expect("value");
        b.end_object().expect("close");
        let bytes = b.finish().expect("finished").to_vec();
        let v = Value::new(&bytes).expect("readable");
        assert_eq!(
            v.to_json().expect("writable"),
            br#"{"id":7,"meta":{"seen":2,"tags":["a"]}}"#
        );
    }

    #[test]
    fn a_float_that_json_cannot_write_says_so_rather_than_writing_something_else() {
        let mut b = Builder::new();
        b.float(f64::INFINITY).expect("value");
        let bytes = b.finish().expect("finished").to_vec();
        let v = Value::new(&bytes).expect("readable");
        let e = v.to_json().expect_err("infinity is not JSON");
        assert!(e.message().contains("infinity or a NaN"));
    }

    #[test]
    fn an_interned_object_needs_the_key_table_and_says_so() {
        let mut b = Builder::new();
        b.begin_object_interned().expect("open");
        b.key_id(3).expect("key");
        b.int(1).expect("value");
        b.end_object().expect("close");
        let bytes = b.finish().expect("finished").to_vec();
        let v = Value::new(&bytes).expect("readable");
        let e = v.to_json().expect_err("there is no table here");
        assert!(e.message().contains("key table"));
    }

    #[test]
    fn the_writer_appends_rather_than_replacing_what_is_in_the_buffer() {
        let bytes = from_json(b"[1,2]").expect("parses");
        let mut out = b"before ".to_vec();
        Value::new(&bytes)
            .expect("readable")
            .write_json(&mut out)
            .expect("writable");
        assert_eq!(out, b"before [1,2]");
    }
}
