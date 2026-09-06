//! `cjson`, the library a script gets when it has to speak JSON.
//!
//! This is Mark Pulford's lua-cjson 2.1.0 as Redis carries it, which is the
//! only JSON a script written against Redis can call. It is not a general JSON
//! library that happens to be reachable under that name: the shape of a decoded
//! document, the text an encoded one comes out as and the sentence a failure
//! raises are all part of what a script reads, and all three are what the C
//! does rather than what a modern encoder would choose.
//!
//! # The three things it does differently from anything else
//!
//! A table becomes an array only when every key it has is a whole number of one
//! or more. An empty table has no keys, so it becomes an object and
//! `cjson.encode({})` is `{}` and never `[]`. That is the single most surprising
//! thing about this library and it is load bearing, because a script that round
//! trips an empty array through it gets an empty object back.
//!
//! A number is written with `%.14g`, so fourteen significant digits and no
//! more. `2^53` comes out as `9.007199254741e+15` rather than as the sixteen
//! digits it actually has, and one third comes out as `0.33333333333333`. That
//! is a lossy encoder by design and a script that round trips a large integer
//! through it does not get the same integer back.
//!
//! A `null` in the text does not decode to `nil`, because `t[k] = nil` in Lua
//! removes the key and a decoded object would silently lose every null field.
//! It decodes to a light userdata holding a null pointer, which is the same
//! value `cjson.null` is, so `cjson.decode('null') == cjson.null` is true and
//! comparing it against `nil` is false.
//!
//! # Where the argument checking is
//!
//! Not here, for the same reason it is not in `bit.rs`. Everything in this file
//! takes values that have already been checked and answers either the result or
//! the sentence to raise, and the prelude does the checking and the raising,
//! because a message a script reads has to be raised from Lua to carry the
//! script's own line in front of it.
//!
//! # How deep it will go
//!
//! Both halves are written as plain recursion, so the depth a document can have
//! is bounded by the thread's stack as well as by the configured limit. A
//! ceiling well above the default of a thousand stops the stack ever being the
//! thing that runs out, and past it the answer is the same sentence the
//! configured limit gives. That is D-106: a script that raises the limit above
//! the ceiling and then hands over a document deeper than the ceiling is told it
//! is too deep where a real server would have carried on.

use mlua::{Lua, Table, Value};

/// How deep the recursion here will go whatever the configured limit says.
///
/// The default limit is a thousand either way, so this only ever comes up for a
/// script that raised the limit on purpose.
const CEILING: i64 = 2000;

/// The knobs both halves read, as the prelude keeps them.
///
/// Every one of them is a number rather than a boolean, because that is what
/// the C holds and because `encode_invalid_numbers` has three states rather than
/// two: refuse, write `nan` and `inf`, or write `null`.
struct Config {
    /// Turn a too sparse array into an object rather than refusing it.
    sparse_convert: bool,
    /// How much bigger than the number of items the largest index may be.
    /// Zero turns the check off.
    sparse_ratio: i64,
    /// An index at or under this is never too sparse whatever the ratio says.
    sparse_safe: i64,
    /// How many tables deep `encode` will go.
    encode_max_depth: i64,
    /// How many tables deep `decode` will go.
    decode_max_depth: i64,
    /// 0 refuse, 1 write `nan` and `inf`, 2 write `null`.
    invalid_encode: i64,
    /// Whether `nan`, `inf`, `0x10` and a leading plus are numbers on the way
    /// in. This starts on, which is why the decoder is not a strict one.
    invalid_decode: bool,
    /// Significant digits, one to fourteen.
    precision: usize,
    /// Whether a decoded array carries a metatable saying it is one.
    array_mt: bool,
}

impl Config {
    /// Read the table the prelude keeps one instance's settings in.
    fn read(t: &Table) -> mlua::Result<Config> {
        let n = |name: &str| -> mlua::Result<i64> { t.raw_get(name) };
        Ok(Config {
            sparse_convert: n("encode_sparse_convert")? != 0,
            sparse_ratio: n("encode_sparse_ratio")?,
            sparse_safe: n("encode_sparse_safe")?,
            encode_max_depth: n("encode_max_depth")?,
            decode_max_depth: n("decode_max_depth")?,
            invalid_encode: n("encode_invalid_numbers")?,
            invalid_decode: n("decode_invalid_numbers")? != 0,
            precision: n("encode_number_precision")?.clamp(1, 14) as usize,
            array_mt: n("decode_array_with_array_mt")? != 0,
        })
    }
}

/// The name Lua would give a value, which is what a failure sentence names.
pub(super) fn typename(v: &Value) -> &'static str {
    match v {
        Value::Nil => "nil",
        Value::Boolean(_) => "boolean",
        Value::Integer(_) | Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Table(_) => "table",
        Value::Function(_) => "function",
        Value::Thread(_) => "thread",
        _ => "userdata",
    }
}

/// A number as `%.*g` would print it.
///
/// Written out rather than reached for, because Rust has no `%g` and the exact
/// text is what a script reads. The rule is the one C states: work out the
/// exponent the value would have in scientific notation with this many
/// significant digits, use scientific notation when that exponent is under
/// minus four or is at least the digit count, use plain notation otherwise, and
/// either way drop the trailing zeros and the trailing point.
fn g_fmt(x: f64, precision: usize) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    let precision = precision.max(1);
    // The exponent has to come from a rendering rather than from a logarithm,
    // because a value that rounds up to the next power of ten at this many
    // digits has the exponent of the rounded form and not of its own.
    let scientific = format!("{:.*e}", precision - 1, x);
    let (mantissa, exponent) = match scientific.split_once('e') {
        Some(parts) => parts,
        None => return scientific,
    };
    let exponent: i32 = exponent.parse().unwrap_or(0);
    if exponent < -4 || exponent >= precision as i32 {
        let sign = if exponent < 0 { '-' } else { '+' };
        return format!("{}e{}{:02}", trim(mantissa), sign, exponent.abs());
    }
    let places = (precision as i32 - 1 - exponent).max(0) as usize;
    trim(&format!("{x:.places$}"))
}

/// A rendering with its trailing zeros and its trailing point taken off.
fn trim(text: &str) -> String {
    if !text.contains('.') {
        return text.to_string();
    }
    text.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// What every byte of a string turns into, or nothing when it goes out as it is.
///
/// The forward slash is escaped, which no other JSON encoder bothers with and
/// which this one has always done. Everything from `0x80` up goes out raw, so
/// the encoder never checks that a string is UTF-8 and never fails because it
/// is not.
fn escape(byte: u8) -> Option<&'static str> {
    Some(match byte {
        0x08 => "\\b",
        0x09 => "\\t",
        0x0a => "\\n",
        0x0c => "\\f",
        0x0d => "\\r",
        b'"' => "\\\"",
        b'/' => "\\/",
        b'\\' => "\\\\",
        0x00..=0x1f | 0x7f => {
            /// The sixteen renderings of a byte that has no short escape, laid
            /// out so that one does not have to be built per byte.
            const LONG: [&str; 33] = [
                "\\u0000", "\\u0001", "\\u0002", "\\u0003", "\\u0004", "\\u0005", "\\u0006",
                "\\u0007", "\\u0008", "\\u0009", "\\u000a", "\\u000b", "\\u000c", "\\u000d",
                "\\u000e", "\\u000f", "\\u0010", "\\u0011", "\\u0012", "\\u0013", "\\u0014",
                "\\u0015", "\\u0016", "\\u0017", "\\u0018", "\\u0019", "\\u001a", "\\u001b",
                "\\u001c", "\\u001d", "\\u001e", "\\u001f", "\\u007f",
            ];
            LONG[if byte == 0x7f { 32 } else { byte as usize }]
        }
        _ => return None,
    })
}

/// The encoder, which holds the text as it grows and the settings it grew under.
struct Encoder<'a> {
    cfg: &'a Config,
    out: Vec<u8>,
}

impl Encoder<'_> {
    /// One value, at a depth counted in tables entered so far.
    fn data(&mut self, value: &Value, depth: i64) -> Result<(), String> {
        match value {
            Value::String(s) => {
                self.string(&s.as_bytes());
                Ok(())
            }
            Value::Integer(n) => self.number(*n as f64),
            Value::Number(n) => self.number(*n),
            Value::Boolean(yes) => {
                self.out
                    .extend_from_slice(if *yes { b"true" } else { b"false" });
                Ok(())
            }
            Value::Nil => {
                self.out.extend_from_slice(b"null");
                Ok(())
            }
            // The one userdata that means something, which is `cjson.null` and
            // is what a decoded null came back as.
            Value::LightUserData(u) if u.0.is_null() => {
                self.out.extend_from_slice(b"null");
                Ok(())
            }
            Value::Table(t) => self.table(t, depth + 1),
            other => Err(format!(
                "Cannot serialise {}: type not supported",
                typename(other)
            )),
        }
    }

    /// A table as either an array or an object, whichever its keys make it.
    fn table(&mut self, t: &Table, depth: i64) -> Result<(), String> {
        if depth > self.cfg.encode_max_depth || depth > CEILING {
            return Err(format!("Cannot serialise, excessive nesting ({depth})"));
        }
        // A table the decoder marked is an array whatever its keys say, which is
        // the only way an empty one comes back out as `[]`.
        if marked(t) {
            let len = t.raw_len() as i64;
            return self.array(t, len, depth);
        }
        match self.length(t)? {
            len if len > 0 => self.array(t, len, depth),
            _ => self.object(t, depth),
        }
    }

    /// How long an array this table is, or minus one when it is not one.
    ///
    /// Every key has to be a whole number of one or more. A key of zero falls
    /// out here rather than at the test for one or more, because the C reads the
    /// key as a truthy value first and zero is not one, and the two paths differ
    /// for nothing else so the reason only shows up in the reading.
    fn length(&self, t: &Table) -> Result<i64, String> {
        let mut max = 0i64;
        let mut items = 0i64;
        for pair in t.pairs::<Value, Value>() {
            let (key, _) = pair.map_err(|e| e.to_string())?;
            let k = match key {
                Value::Integer(n) => n as f64,
                Value::Number(n) => n,
                _ => return Ok(-1),
            };
            if k == 0.0 || k.floor() != k || k < 1.0 {
                return Ok(-1);
            }
            let k = k.min(i64::MAX as f64) as i64;
            max = max.max(k);
            items += 1;
        }
        if self.cfg.sparse_ratio > 0
            && max > items.saturating_mul(self.cfg.sparse_ratio)
            && max > self.cfg.sparse_safe
        {
            if !self.cfg.sparse_convert {
                return Err("Cannot serialise table: excessively sparse array".to_string());
            }
            return Ok(-1);
        }
        Ok(max)
    }

    /// The indexes one to `len`, with a hole written out as a null.
    fn array(&mut self, t: &Table, len: i64, depth: i64) -> Result<(), String> {
        self.out.push(b'[');
        for i in 1..=len {
            if i > 1 {
                self.out.push(b',');
            }
            let value: Value = t.raw_get(i).unwrap_or(Value::Nil);
            self.data(&value, depth)?;
        }
        self.out.push(b']');
        Ok(())
    }

    /// Every key in the order the table hands them over.
    ///
    /// A number key is written as the digits it would print, in quotes, so
    /// `{[1.5] = 1}` comes out as `{"1.5":1}` and a key that is a boolean stops
    /// the whole encode.
    fn object(&mut self, t: &Table, depth: i64) -> Result<(), String> {
        self.out.push(b'{');
        let mut first = true;
        for pair in t.pairs::<Value, Value>() {
            let (key, value) = pair.map_err(|e| e.to_string())?;
            if !first {
                self.out.push(b',');
            }
            first = false;
            match &key {
                Value::Integer(n) => self.number_key(*n as f64)?,
                Value::Number(n) => self.number_key(*n)?,
                Value::String(s) => {
                    self.string(&s.as_bytes());
                    self.out.push(b':');
                }
                other => {
                    return Err(format!(
                        "Cannot serialise {}: table key must be a number or string",
                        typename(other)
                    ));
                }
            }
            self.data(&value, depth)?;
        }
        self.out.push(b'}');
        Ok(())
    }

    /// A number used as a key, which is the number in quotes and then a colon.
    fn number_key(&mut self, x: f64) -> Result<(), String> {
        self.out.push(b'"');
        self.number(x)?;
        self.out.extend_from_slice(b"\":");
        Ok(())
    }

    /// A number, refused or written out plainly depending on the setting.
    fn number(&mut self, x: f64) -> Result<(), String> {
        match self.cfg.invalid_encode {
            0 if !x.is_finite() => {
                return Err("Cannot serialise number: must not be NaN or Inf".to_string());
            }
            // Not a number is written out here rather than left to the
            // formatter, because a platform is allowed to print it as `-nan`
            // and a script reads this.
            1 if x.is_nan() => {
                self.out.extend_from_slice(b"nan");
                return Ok(());
            }
            2 if !x.is_finite() => {
                self.out.extend_from_slice(b"null");
                return Ok(());
            }
            _ => {}
        }
        self.out
            .extend_from_slice(g_fmt(x, self.cfg.precision).as_bytes());
        Ok(())
    }

    /// A string, in quotes, with the bytes that need it escaped.
    fn string(&mut self, bytes: &[u8]) {
        self.out.push(b'"');
        for &byte in bytes {
            match escape(byte) {
                Some(text) => self.out.extend_from_slice(text.as_bytes()),
                None => self.out.push(byte),
            }
        }
        self.out.push(b'"');
    }
}

/// Whether a table's metatable says the decoder built it as an array.
fn marked(t: &Table) -> bool {
    t.metatable()
        .and_then(|mt| mt.raw_get::<Value>("__is_cjson_array").ok())
        .is_some_and(|v| !matches!(v, Value::Nil | Value::Boolean(false)))
}

/// What the scanner found, in the order the C names them.
///
/// The names matter because a parse failure prints the one it found, so a
/// script that catches the failure reads `T_STRING` and not `string`.
#[derive(Debug, PartialEq)]
enum Tok {
    ObjBegin,
    ObjEnd,
    ArrBegin,
    ArrEnd,
    Str(Vec<u8>),
    Num(f64),
    Bool(bool),
    Null,
    Colon,
    Comma,
    End,
    /// A scanner failure, which prints its own message instead of a name.
    Bad(&'static str),
}

impl Tok {
    /// What a failure says it found here.
    fn found(&self) -> &str {
        match self {
            Tok::ObjBegin => "T_OBJ_BEGIN",
            Tok::ObjEnd => "T_OBJ_END",
            Tok::ArrBegin => "T_ARR_BEGIN",
            Tok::ArrEnd => "T_ARR_END",
            Tok::Str(_) => "T_STRING",
            Tok::Num(_) => "T_NUMBER",
            Tok::Bool(_) => "T_BOOLEAN",
            Tok::Null => "T_NULL",
            Tok::Colon => "T_COLON",
            Tok::Comma => "T_COMMA",
            Tok::End => "T_END",
            Tok::Bad(why) => why,
        }
    }
}

/// One token with where it started, which is what a failure counts from.
struct Found {
    tok: Tok,
    at: usize,
}

/// The decoder, walking the bytes it was handed.
struct Parser<'a> {
    data: &'a [u8],
    at: usize,
    cfg: &'a Config,
    depth: i64,
}

impl<'a> Parser<'a> {
    /// The byte at an offset, or the terminator past the end.
    ///
    /// The C reads a null terminated buffer and treats a zero byte as the end
    /// of the document wherever it is, so a zero inside the string ends it just
    /// as the one past the end does.
    fn byte(&self, at: usize) -> u8 {
        self.data.get(at).copied().unwrap_or(0)
    }

    /// The next token, with the whitespace in front of it eaten.
    fn next(&mut self) -> Found {
        while matches!(self.byte(self.at), b' ' | b'\t' | b'\n' | b'\r') {
            self.at += 1;
        }
        let at = self.at;
        let ch = self.byte(at);
        let tok = match ch {
            b'{' => {
                self.at += 1;
                Tok::ObjBegin
            }
            b'}' => {
                self.at += 1;
                Tok::ObjEnd
            }
            b'[' => {
                self.at += 1;
                Tok::ArrBegin
            }
            b']' => {
                self.at += 1;
                Tok::ArrEnd
            }
            b',' => {
                self.at += 1;
                Tok::Comma
            }
            b':' => {
                self.at += 1;
                Tok::Colon
            }
            0 => Tok::End,
            b'"' => return self.text(at),
            b'-' | b'0'..=b'9' => {
                if !self.cfg.invalid_decode && self.loose() {
                    return Found {
                        tok: Tok::Bad("invalid number"),
                        at,
                    };
                }
                return self.number();
            }
            b't' if self.word(b"true") => {
                self.at += 4;
                Tok::Bool(true)
            }
            b'f' if self.word(b"false") => {
                self.at += 5;
                Tok::Bool(false)
            }
            b'n' if self.word(b"null") => {
                self.at += 4;
                Tok::Null
            }
            // What is left starts with one of the letters a loose number can
            // start with, or with a plus. Anything the loose check does not
            // claim is not a token at all.
            b'f' | b'i' | b'I' | b'n' | b'N' | b't' | b'+' => {
                if self.cfg.invalid_decode && self.loose() {
                    return self.number();
                }
                Tok::Bad("invalid token")
            }
            _ => Tok::Bad("invalid token"),
        };
        Found { tok, at }
    }

    /// Whether the bytes here are this word.
    fn word(&self, want: &[u8]) -> bool {
        want.iter()
            .enumerate()
            .all(|(i, &b)| self.byte(self.at + i) == b)
    }

    /// Whether what starts here is a number JSON itself would not allow.
    ///
    /// A leading plus, a leading zero, hex, and the two spellings that are not
    /// numbers at all. The decoder takes all of them by default, and this is
    /// what tells it which of the letters it stopped on start one.
    fn loose(&self) -> bool {
        let mut at = self.at;
        if self.byte(at) == b'+' {
            return true;
        }
        if self.byte(at) == b'-' {
            at += 1;
        }
        if self.byte(at) == b'0' {
            let next = self.byte(at + 1);
            return next | 0x20 == b'x' || next.is_ascii_digit();
        }
        if self.byte(at) <= b'9' {
            return false;
        }
        let rest = &self.data[at.min(self.data.len())..];
        starts_ci(rest, b"inf") || starts_ci(rest, b"nan")
    }

    /// A number, taken the way the C library's `strtod` takes one.
    fn number(&mut self) -> Found {
        let at = self.at;
        match strtod(self.data, at) {
            Some((value, end)) => {
                self.at = end;
                Found {
                    tok: Tok::Num(value),
                    at,
                }
            }
            None => Found {
                tok: Tok::Bad("invalid number"),
                at,
            },
        }
    }

    /// A quoted string, with the escapes turned back into the bytes they stand
    /// for.
    ///
    /// A failure inside one reports where inside it the reader stopped rather
    /// than where the string started, which is why this sets the position
    /// itself rather than letting the caller do it. A string that reads cleanly
    /// still reports the opening quote, which is the position the caller took
    /// before handing over.
    fn text(&mut self, from: usize) -> Found {
        self.at += 1;
        let mut out = Vec::new();
        loop {
            let ch = self.byte(self.at);
            if ch == b'"' {
                self.at += 1;
                return Found {
                    tok: Tok::Str(out),
                    at: from,
                };
            }
            if ch == 0 {
                return Found {
                    tok: Tok::Bad("unexpected end of string"),
                    at: self.at,
                };
            }
            if ch != b'\\' {
                out.push(ch);
                self.at += 1;
                continue;
            }
            let what = self.byte(self.at + 1);
            if what == b'u' {
                match self.unicode(&mut out) {
                    true => continue,
                    false => {
                        return Found {
                            tok: Tok::Bad("invalid unicode escape code"),
                            at: self.at,
                        };
                    }
                }
            }
            let plain = match what {
                b'"' => b'"',
                b'\\' => b'\\',
                b'/' => b'/',
                b'b' => 0x08,
                b't' => 0x09,
                b'n' => 0x0a,
                b'f' => 0x0c,
                b'r' => 0x0d,
                _ => {
                    return Found {
                        tok: Tok::Bad("invalid escape code"),
                        at: self.at,
                    };
                }
            };
            out.push(plain);
            self.at += 2;
        }
    }

    /// One `\uXXXX`, or the pair of them a codepoint past the basic plane needs.
    ///
    /// Answers whether it worked and leaves the position on the byte after the
    /// escape when it did.
    fn unicode(&mut self, out: &mut Vec<u8>) -> bool {
        let Some(mut point) = self.hex4(self.at + 2) else {
            return false;
        };
        let mut len = 6;
        if point & 0xf800 == 0xd800 {
            // The first half of a pair has to be the high one and has to be
            // followed by the low one, spelled as another escape.
            if point & 0x400 != 0 {
                return false;
            }
            if self.byte(self.at + len) != b'\\' || self.byte(self.at + len + 1) != b'u' {
                return false;
            }
            let Some(low) = self.hex4(self.at + 2 + len) else {
                return false;
            };
            if low & 0xfc00 != 0xdc00 {
                return false;
            }
            point = (((point & 0x3ff) << 10) | (low & 0x3ff)) + 0x10000;
            len = 12;
        }
        let Some(ch) = char::from_u32(point) else {
            // A lone half of a pair is the only way to get here, and the C
            // rejects the same values for the same reason.
            return false;
        };
        let mut buf = [0u8; 4];
        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
        self.at += len;
        true
    }

    /// Four hex digits as the number they spell.
    fn hex4(&self, at: usize) -> Option<u32> {
        let mut value = 0u32;
        for i in 0..4 {
            value = value * 16 + (self.byte(at + i) as char).to_digit(16)?;
        }
        Some(value)
    }
}

/// Whether these bytes start with this word, ignoring case.
fn starts_ci(data: &[u8], want: &[u8]) -> bool {
    data.len() >= want.len() && data[..want.len()].eq_ignore_ascii_case(want)
}

/// The longest number the C library's `strtod` would read from here.
///
/// Answers the value and where it stopped, or nothing when there was no number
/// at all. This has to be `strtod` and not a JSON number reader, because the
/// decoder hands it anything that could start one and takes whatever comes
/// back, which is how `nan`, `0x10`, `+1`, `01` and `1.` all decode.
fn strtod(data: &[u8], from: usize) -> Option<(f64, usize)> {
    let byte = |i: usize| data.get(i).copied().unwrap_or(0);
    let mut at = from;
    while matches!(byte(at), b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r') {
        at += 1;
    }
    let start = at;
    let negative = byte(at) == b'-';
    if negative || byte(at) == b'+' {
        at += 1;
    }
    let sign = if negative { -1.0 } else { 1.0 };

    if byte(at) == b'0' && byte(at + 1) | 0x20 == b'x' {
        return Some(hex(data, at, sign));
    }
    if starts_ci(&data[at.min(data.len())..], b"infinity") {
        return Some((sign * f64::INFINITY, at + 8));
    }
    if starts_ci(&data[at.min(data.len())..], b"inf") {
        return Some((sign * f64::INFINITY, at + 3));
    }
    if starts_ci(&data[at.min(data.len())..], b"nan") {
        // A parenthesised tail is part of the spelling and is thrown away.
        let mut end = at + 3;
        if byte(end) == b'(' {
            let mut scan = end + 1;
            while byte(scan).is_ascii_alphanumeric() || byte(scan) == b'_' {
                scan += 1;
            }
            if byte(scan) == b')' {
                end = scan + 1;
            }
        }
        return Some((sign * f64::NAN, end));
    }

    let mut digits = 0;
    while byte(at).is_ascii_digit() {
        at += 1;
        digits += 1;
    }
    if byte(at) == b'.' {
        at += 1;
        while byte(at).is_ascii_digit() {
            at += 1;
            digits += 1;
        }
    }
    if digits == 0 {
        return None;
    }
    // The exponent only counts when it has digits, so `1e` is the number one
    // and stops before the letter.
    let mut end = at;
    if byte(at) | 0x20 == b'e' {
        let mut scan = at + 1;
        if matches!(byte(scan), b'+' | b'-') {
            scan += 1;
        }
        if byte(scan).is_ascii_digit() {
            while byte(scan).is_ascii_digit() {
                scan += 1;
            }
            end = scan;
        }
    }
    // Every byte between here is a digit, a point, a sign or an exponent
    // marker, so this is ASCII and the parse is the one Rust does correctly.
    let text = std::str::from_utf8(&data[start..end]).ok()?;
    text.parse::<f64>().ok().map(|value| (value, end))
}

/// The hex spelling, which the C library takes and JSON does not.
///
/// The position is on the zero of the `0x`. A `0x` with no digits after it is
/// the number zero and stops after the zero, which is what the C library does
/// and is why `0xz` decodes rather than failing. A `p` after the digits is a
/// power of two written in decimal, so `0x1p4` is sixteen, and it only counts
/// when there is at least one digit behind it.
fn hex(data: &[u8], at: usize, sign: f64) -> (f64, usize) {
    let byte = |i: usize| data.get(i).copied().unwrap_or(0);
    let mut scan = at + 2;
    let mut mantissa = 0u128;
    let mut shift = 0i32;
    let mut digits = 0;
    let mut fraction = false;
    loop {
        let ch = byte(scan);
        if ch == b'.' && !fraction {
            fraction = true;
            scan += 1;
            continue;
        }
        let Some(value) = (ch as char).to_digit(16) else {
            break;
        };
        // Once the mantissa is full the digits still count toward the exponent,
        // and a digit after the point does not when it was dropped.
        if mantissa <= (u128::MAX - 15) / 16 {
            mantissa = mantissa * 16 + u128::from(value);
            if fraction {
                shift -= 4;
            }
        } else if !fraction {
            shift += 4;
        }
        scan += 1;
        digits += 1;
    }
    if digits == 0 {
        return (sign * 0.0, at + 1);
    }
    if byte(scan) | 0x20 == b'p' {
        let mut walk = scan + 1;
        let negative = matches!(byte(walk), b'+' | b'-');
        let minus = byte(walk) == b'-';
        if negative {
            walk += 1;
        }
        let mut power = 0i32;
        let mut counted = 0;
        while let Some(value) = (byte(walk) as char).to_digit(10) {
            // Held well short of an overflow, because anything this large is
            // already an infinity or a zero by the time it is applied.
            power = (power * 10 + value as i32).min(100_000);
            walk += 1;
            counted += 1;
        }
        if counted > 0 {
            shift += if minus { -power } else { power };
            scan = walk;
        }
    }
    (sign * (mantissa as f64) * 2f64.powi(shift), scan)
}

/// Turn the tokens into Lua values.
impl<'a> Parser<'a> {
    /// One value, whatever it turned out to be.
    fn value(&mut self, lua: &Lua, found: Found) -> Result<Value, String> {
        match found.tok {
            Tok::Str(bytes) => Ok(Value::String(
                lua.create_string(&bytes).map_err(|e| e.to_string())?,
            )),
            Tok::Num(n) => Ok(Value::Number(n)),
            Tok::Bool(b) => Ok(Value::Boolean(b)),
            Tok::Null => Ok(null()),
            Tok::ObjBegin => self.object(lua),
            Tok::ArrBegin => self.array(lua),
            other => Err(complain(
                "value",
                &Found {
                    tok: other,
                    at: found.at,
                },
            )),
        }
    }

    /// Count one level down, or say the document is too deep.
    fn descend(&mut self) -> Result<(), String> {
        self.depth += 1;
        if self.depth <= self.cfg.decode_max_depth && self.depth <= CEILING {
            return Ok(());
        }
        Err(format!(
            "Found too many nested data structures ({}) at character {}",
            self.depth, self.at
        ))
    }

    /// Everything between a brace and its partner.
    fn object(&mut self, lua: &Lua) -> Result<Value, String> {
        self.descend()?;
        let t = lua.create_table().map_err(|e| e.to_string())?;
        let mut found = self.next();
        if found.tok == Tok::ObjEnd {
            self.depth -= 1;
            return Ok(Value::Table(t));
        }
        loop {
            let Tok::Str(name) = found.tok else {
                return Err(complain("object key string", &found));
            };
            let key = lua.create_string(&name).map_err(|e| e.to_string())?;
            let colon = self.next();
            if colon.tok != Tok::Colon {
                return Err(complain("colon", &colon));
            }
            let next = self.next();
            let value = self.value(lua, next)?;
            t.raw_set(key, value).map_err(|e| e.to_string())?;
            found = self.next();
            if found.tok == Tok::ObjEnd {
                self.depth -= 1;
                return Ok(Value::Table(t));
            }
            if found.tok != Tok::Comma {
                return Err(complain("comma or object end", &found));
            }
            found = self.next();
        }
    }

    /// Everything between a bracket and its partner.
    fn array(&mut self, lua: &Lua) -> Result<Value, String> {
        self.descend()?;
        let t = lua.create_table().map_err(|e| e.to_string())?;
        if self.cfg.array_mt {
            // A fresh one per array, the way the C makes it, so a script that
            // compares two of them finds two tables and not one.
            let mt = lua.create_table().map_err(|e| e.to_string())?;
            mt.raw_set("__is_cjson_array", true)
                .map_err(|e| e.to_string())?;
            t.set_metatable(Some(mt)).map_err(|e| e.to_string())?;
        }
        let mut found = self.next();
        if found.tok == Tok::ArrEnd {
            self.depth -= 1;
            return Ok(Value::Table(t));
        }
        let mut i = 1i64;
        loop {
            let value = self.value(lua, found)?;
            t.raw_set(i, value).map_err(|e| e.to_string())?;
            i += 1;
            found = self.next();
            if found.tok == Tok::ArrEnd {
                self.depth -= 1;
                return Ok(Value::Table(t));
            }
            if found.tok != Tok::Comma {
                return Err(complain("comma or array end", &found));
            }
            found = self.next();
        }
    }
}

/// What a parse failure reads like.
///
/// The position counts from one where everything inside the parser counts from
/// zero, which is the one adjustment the C makes on the way out.
fn complain(wanted: &str, found: &Found) -> String {
    format!(
        "Expected {} but found {} at character {}",
        wanted,
        found.tok.found(),
        found.at + 1
    )
}

/// The value a null in the text decodes to, and the value `cjson.null` is.
///
/// A light userdata holding a null pointer, so that two of them compare equal
/// and neither compares equal to anything else a script can make. It is handed
/// to the prelude as an argument rather than left on the private table, because
/// the prelude builds the module the moment it runs and the private table is
/// not filled in until afterwards.
pub(super) fn null() -> Value {
    Value::LightUserData(mlua::LightUserData(std::ptr::null_mut()))
}

/// Put the two halves on the private table the prelude reaches us through.
pub(super) fn statics(lua: &Lua, raw: &Table) -> mlua::Result<()> {
    raw.raw_set(
        "cjson_encode",
        lua.create_function(|lua, (value, settings): (Value, Table)| {
            let cfg = Config::read(&settings)?;
            let mut encoder = Encoder {
                cfg: &cfg,
                out: Vec::with_capacity(64),
            };
            match encoder.data(&value, 0) {
                Ok(()) => Ok((true, Value::String(lua.create_string(&encoder.out)?))),
                Err(why) => Ok((false, Value::String(lua.create_string(&why)?))),
            }
        })?,
    )?;
    raw.raw_set(
        "cjson_decode",
        lua.create_function(|lua, (text, settings): (mlua::LuaString, Table)| {
            let cfg = Config::read(&settings)?;
            let bytes = text.as_bytes();
            // Two zero bytes at the front is how the C spots a document that
            // was handed over in one of the wider encodings, and it is the one
            // failure that is not a parse failure.
            if bytes.len() >= 2 && (bytes[0] == 0 || bytes[1] == 0) {
                let why = "JSON parser does not support UTF-16 or UTF-32";
                return Ok((false, Value::String(lua.create_string(why)?)));
            }
            let mut parser = Parser {
                data: &bytes,
                at: 0,
                cfg: &cfg,
                depth: 0,
            };
            let first = parser.next();
            let answer = parser.value(lua, first).and_then(|value| {
                let rest = parser.next();
                match rest.tok {
                    Tok::End => Ok(value),
                    _ => Err(complain("the end", &rest)),
                }
            });
            match answer {
                Ok(value) => Ok((true, value)),
                Err(why) => Ok((false, Value::String(lua.create_string(&why)?))),
            }
        })?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{g_fmt, strtod};

    /// Fourteen significant digits and the two notations, which between them
    /// are most of what a script reads back out of this library.
    #[test]
    fn a_number_is_written_the_way_the_c_library_writes_one() {
        for (given, want) in [
            (100.0, "100"),
            (0.0, "0"),
            (-0.0, "-0"),
            (1e300, "1e+300"),
            (1e-7, "1e-07"),
            (1.0 / 3.0, "0.33333333333333"),
            (std::f64::consts::PI, "3.1415926535898"),
            (9_007_199_254_740_992.0, "9.007199254741e+15"),
            (1e14, "1e+14"),
            (1e13, "10000000000000"),
            (-1.5, "-1.5"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
        ] {
            assert_eq!(g_fmt(given, 14), want, "{given}");
        }
        assert_eq!(g_fmt(std::f64::consts::PI, 3), "3.14");
        assert_eq!(g_fmt(1234.0, 3), "1.23e+03");
    }

    /// The spellings the decoder takes because the C library's reader takes
    /// them, and where each one stops.
    #[test]
    fn a_number_is_read_the_way_the_c_library_reads_one() {
        for (given, value, end) in [
            ("1", 1.0, 1),
            ("01", 1.0, 2),
            ("+1", 1.0, 2),
            ("1.", 1.0, 2),
            ("-2.5e3", -2500.0, 6),
            ("1e", 1.0, 1),
            ("1e+", 1.0, 1),
            ("0x10", 16.0, 4),
            ("0X1f", 31.0, 4),
            ("-0x10,", -16.0, 5),
            ("0xz", 0.0, 1),
            ("1e999", f64::INFINITY, 5),
            ("inf", f64::INFINITY, 3),
            ("-Infinity", f64::NEG_INFINITY, 9),
        ] {
            let (got, at) = strtod(given.as_bytes(), 0).expect(given);
            assert_eq!(got, value, "{given}");
            assert_eq!(at, end, "{given}");
        }
        let (nan, at) = strtod(b"nan", 0).expect("nan");
        assert!(nan.is_nan());
        assert_eq!(at, 3);
        assert!(strtod(b"abc", 0).is_none());
    }
}
