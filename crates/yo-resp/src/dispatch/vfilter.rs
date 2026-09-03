//! The expression language `VSIM ... FILTER` reads over an element's attributes
//! (`10` section 9).
//!
//! `VSIM key ELE x FILTER '.year > 1980 and .genre == "action"'` is a filtered
//! vector search, and the whole difficulty of it is that the filter has to be
//! decided while the search is still choosing what to rank. Ten answers filtered
//! afterwards is nothing at all when the filter matches one element in a
//! thousand, and the client cannot tell that from a genuine no match.
//!
//! ```text
//! .year >= 1980 and .genre in ["action", "sci-fi"] and !.hidden
//! ```
//!
//! # Two tests, cheap then exact
//!
//! An attribute is a JSON string on the element and an expression over it is not
//! something a scan can afford to run on every code it touches. So the filter is
//! in two parts, which is what [`yo_vector::Filter`] is shaped for.
//!
//! The cheap one is a [`Signature`]: one bit per field and string value pair,
//! built for an element when its attribute is written and built for a query out
//! of the equality tests the expression insists on. It is a subset test on a word
//! the scan has already loaded. It only ever passes elements it should not, never
//! rejects one it should have kept, so it is safe to run first and it costs
//! nothing when the expression has no equality test in it at all.
//!
//! The exact one is this expression, run against the element's attribute string.
//! It sees only elements the signature let through that are also near enough to
//! be worth ranking, which is few enough that parsing a short JSON object for
//! each of them is not what the query is spending its time on.
//!
//! Only the equality tests an expression *requires* go into the signature, which
//! means the top level `and` chain and nothing under an `or` or a `not`. An
//! expression that requires nothing gets an empty signature, which every element
//! covers, and then the exact test does all of the work.
//!
//! # What the language is
//!
//! Redis's, because a client that already writes these should not have to write
//! them differently here.
//!
//! `.field` reads a top level field of the attribute object. Numbers, strings in
//! either kind of quote, `true` and `false` are literals, and `[a, b, c]` is a
//! list, which is only ever the right hand side of `in`.
//!
//! `and or not` spelled either as words or as `&& || !`, the six comparisons,
//! `in`, and the arithmetic `+ - * / % **` on numbers.
//!
//! # A field that is not there
//!
//! Is not an error and does not match. Reading a field the attribute does not
//! have, or one holding an object or an array rather than a scalar, gives
//! nothing, arithmetic on nothing gives nothing, and every comparison against
//! nothing is false. So `.year > 1980` is false for an element with no year, and
//! `.year > 1980 or .rating > 8` is still true for one that has no year and a
//! rating of 9.
//!
//! Comparing across types is false the same way, so `.year == "1980"` does not
//! match a numeric year. That is the one place this can surprise somebody, and
//! the alternative, coercing a string to a number, would make `.id == 0` match an
//! element whose id is the word `abc`.

use yo_common::{Code, Error, Result};
use yo_vector::Signature;

/// What an expression that does not parse gets.
const BAD_FILTER: &str = "invalid FILTER expression";

/// How deep a nesting the parser will follow.
///
/// The tree is walked by recursion on both sides, so this is what stops a
/// client's thousand open brackets from being a stack overflow. Nothing anybody
/// writes by hand comes close.
const DEPTH: usize = 32;

/// A parsed `FILTER` expression.
#[derive(Debug)]
pub(super) struct Filter {
    root: Node,
}

impl Filter {
    /// Read an expression, or say what is wrong with it.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] for anything that is not one of these.
    pub(super) fn parse(src: &[u8]) -> Result<Filter> {
        let toks = lex(src)?;
        let mut p = Parser { toks: &toks, at: 0 };
        let root = p.expr(0)?;
        if p.at != toks.len() {
            return Err(bad());
        }
        Ok(Filter { root })
    }

    /// The bits an element has to have for this expression to have any chance.
    ///
    /// The top level `and` chain and nothing else, because those are the only
    /// tests every match has to pass. See the module header.
    pub(super) fn signature(&self) -> Signature {
        let mut sig = Signature::default();
        require(&self.root, &mut sig);
        sig
    }

    /// Whether an element carrying this attribute string matches.
    ///
    /// An element with no attribute is one whose every field is missing, which
    /// is why there is no separate answer for it.
    pub(super) fn matches(&self, attr: Option<&[u8]>) -> bool {
        truth(eval(&self.root, attr.unwrap_or(b"")))
    }
}

/// The bits an element with this attribute string carries.
///
/// One per field holding a string, which is the half of the language a 64 bit
/// summary can carry: a number is compared by ranges as often as by equality and
/// a range is not something a bit can answer. Numbers are left to the exact
/// test, which is where they can be answered properly.
pub(super) fn tag(attr: &[u8]) -> u64 {
    let mut sig = Signature::default();
    let mut name = Vec::new();
    let mut value = Vec::new();
    let mut scan = Json::new(attr);
    while let Some(()) = scan.next_field(&mut name) {
        if scan.string_value(&mut value) {
            sig.insert_bytes(&name, &value);
        }
    }
    sig.bits()
}

/// Collect the equality tests every match has to pass.
fn require(node: &Node, sig: &mut Signature) {
    match node {
        Node::Bin(Op::And, pair) => {
            require(&pair.0, sig);
            require(&pair.1, sig);
        }
        Node::Bin(Op::Eq, pair) => match (&pair.0, &pair.1) {
            (Node::Field(f), Node::Text(t)) | (Node::Text(t), Node::Field(f)) => {
                sig.insert_bytes(f, t);
            }
            _ => {}
        },
        _ => {}
    }
}

// ---- the tree

/// One node of a parsed expression.
#[derive(Debug)]
enum Node {
    /// `.name`, read from the attribute object.
    Field(Box<[u8]>),
    /// A number literal.
    Num(f64),
    /// A string literal, with its escapes already resolved.
    Text(Box<[u8]>),
    /// `true` or `false`.
    Bool(bool),
    /// `[a, b, c]`, which is only ever the right of `in`.
    List(Box<[Node]>),
    /// `!x` or `not x`.
    Not(Box<Node>),
    /// `-x`.
    Neg(Box<Node>),
    /// Everything with two sides, boxed as a pair so a node is one allocation
    /// rather than two.
    Bin(Op, Box<(Node, Node)>),
}

/// An operator with two sides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Or,
    And,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    In,
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
}

// ---- values

/// What a piece of an expression came out as.
#[derive(Debug, Clone, Copy)]
enum Val<'a> {
    /// A field that is not there, or arithmetic on one.
    Null,
    Bool(bool),
    Num(f64),
    Text(Text<'a>),
}

/// A string, either a literal from the expression or one still sitting in the
/// JSON it was read from.
///
/// The second kind is left where it is rather than copied out, because most
/// attribute strings have no escape in them and comparing one against a literal
/// is then a comparison of two byte slices. When there is an escape the bytes
/// are decoded as they are compared, so a long value with a `\n` in it still
/// costs no allocation.
#[derive(Debug, Clone, Copy)]
struct Text<'a> {
    raw: &'a [u8],
    escaped: bool,
}

/// Compare two strings by the bytes they stand for.
fn text_cmp(a: Text<'_>, b: Text<'_>) -> core::cmp::Ordering {
    match (a.escaped, b.escaped) {
        (false, false) => a.raw.cmp(b.raw),
        _ => Decode::new(a).cmp(Decode::new(b)),
    }
}

/// The decoded bytes of a string, one at a time.
struct Decode<'a> {
    raw: &'a [u8],
    at: usize,
    /// Bytes produced by a `\u` escape that have not been handed out yet.
    held: [u8; 4],
    /// How many of `held` are waiting, and where in it the next one is.
    have: usize,
    took: usize,
}

impl<'a> Decode<'a> {
    fn new(t: Text<'a>) -> Decode<'a> {
        Decode {
            raw: t.raw,
            at: 0,
            held: [0; 4],
            have: 0,
            took: 0,
        }
    }
}

impl Iterator for Decode<'_> {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        if self.took < self.have {
            let byte = self.held[self.took];
            self.took += 1;
            return Some(byte);
        }
        let byte = *self.raw.get(self.at)?;
        self.at += 1;
        if byte != b'\\' {
            return Some(byte);
        }
        let esc = *self.raw.get(self.at)?;
        self.at += 1;
        let plain = match esc {
            b'n' => b'\n',
            b'r' => b'\r',
            b't' => b'\t',
            b'b' => 0x08,
            b'f' => 0x0c,
            b'u' => {
                let c = self.unicode()?;
                let mut buf = [0u8; 4];
                let s = c.encode_utf8(&mut buf);
                self.held = [0; 4];
                self.held[..s.len()].copy_from_slice(s.as_bytes());
                self.have = s.len();
                self.took = 1;
                return Some(self.held[0]);
            }
            other => other,
        };
        Some(plain)
    }
}

impl Decode<'_> {
    /// The character a `\u` escape stands for, having already eaten the `u`.
    ///
    /// A high surrogate takes the low one that follows it, which is how JSON
    /// spells anything past the basic plane. Anything malformed comes out as the
    /// replacement character rather than stopping the comparison, because this
    /// is a comparison and not a parser: a value nobody can spell is a value
    /// nothing will equal.
    fn unicode(&mut self) -> Option<char> {
        let first = self.hex4()?;
        let point = if (0xd800..0xdc00).contains(&first) {
            match self.pair() {
                Some(low) => 0x1_0000 + ((first - 0xd800) << 10) + (low - 0xdc00),
                None => return Some(char::REPLACEMENT_CHARACTER),
            }
        } else {
            first
        };
        Some(char::from_u32(point).unwrap_or(char::REPLACEMENT_CHARACTER))
    }

    /// The low half of a surrogate pair, if that is what comes next.
    fn pair(&mut self) -> Option<u32> {
        if self.raw.get(self.at..self.at + 2)? != b"\\u" {
            return None;
        }
        let mark = self.at;
        self.at += 2;
        match self.hex4() {
            Some(low) if (0xdc00..0xe000).contains(&low) => Some(low),
            _ => {
                self.at = mark;
                None
            }
        }
    }

    /// Four hex digits as a number.
    fn hex4(&mut self) -> Option<u32> {
        let digits = self.raw.get(self.at..self.at + 4)?;
        self.at += 4;
        let mut got = 0u32;
        for d in digits {
            got = got * 16 + char::from(*d).to_digit(16)?;
        }
        Some(got)
    }
}

/// Whether a value counts as a match.
///
/// A number is true when it is not zero, which is what makes `.count` a filter
/// on its own. A string is not a truth value and neither is a missing field, so
/// both are false.
fn truth(v: Val<'_>) -> bool {
    match v {
        Val::Bool(b) => b,
        Val::Num(n) => n != 0.0,
        Val::Null | Val::Text(_) => false,
    }
}

// ---- evaluation

/// What one node comes out as, over this attribute string.
fn eval<'a>(node: &'a Node, attr: &'a [u8]) -> Val<'a> {
    match node {
        Node::Field(name) => field(attr, name),
        Node::Num(n) => Val::Num(*n),
        Node::Text(t) => Val::Text(Text {
            raw: t,
            escaped: false,
        }),
        Node::Bool(b) => Val::Bool(*b),
        // A list is never a value on its own. `in` reads its right hand side
        // without evaluating it, so getting here means somebody wrote a list
        // somewhere a list cannot be, and nothing matches it.
        Node::List(_) => Val::Null,
        Node::Not(inner) => Val::Bool(!truth(eval(inner, attr))),
        Node::Neg(inner) => match eval(inner, attr) {
            Val::Num(n) => Val::Num(-n),
            _ => Val::Null,
        },
        Node::Bin(Op::And, pair) => {
            // Short circuit, so that the right hand side of a filter that has
            // already failed is not parsed out of the JSON at all.
            Val::Bool(truth(eval(&pair.0, attr)) && truth(eval(&pair.1, attr)))
        }
        Node::Bin(Op::Or, pair) => {
            Val::Bool(truth(eval(&pair.0, attr)) || truth(eval(&pair.1, attr)))
        }
        Node::Bin(Op::In, pair) => Val::Bool(contains(&pair.1, eval(&pair.0, attr), attr)),
        Node::Bin(op, pair) => binary(*op, eval(&pair.0, attr), eval(&pair.1, attr)),
    }
}

/// Whether the left hand value is one of the list on the right.
fn contains<'a>(list: &'a Node, want: Val<'a>, attr: &'a [u8]) -> bool {
    let Node::List(items) = list else {
        return false;
    };
    items
        .iter()
        .any(|item| matches!(binary(Op::Eq, want, eval(item, attr)), Val::Bool(true)))
}

/// An operator that is not `and`, `or` or `in`, on two values that are already
/// worked out.
fn binary<'a>(op: Op, a: Val<'a>, b: Val<'a>) -> Val<'a> {
    use core::cmp::Ordering;
    let order = match (a, b) {
        (Val::Num(x), Val::Num(y)) => Some(x.total_cmp(&y)),
        (Val::Text(x), Val::Text(y)) => Some(text_cmp(x, y)),
        (Val::Bool(x), Val::Bool(y)) => Some(x.cmp(&y)),
        // A missing field, or two sides of different types. Neither is an error
        // and neither matches anything.
        _ => None,
    };
    match op {
        // Equality is answered for every pair of types, because "the type is
        // wrong so they are not equal" is the right answer and "so I do not
        // know" is not.
        Op::Eq => return Val::Bool(order == Some(Ordering::Equal)),
        Op::Ne => return Val::Bool(order != Some(Ordering::Equal)),
        _ => {}
    }
    let Some(order) = order else {
        return Val::Bool(false);
    };
    match op {
        Op::Lt => return Val::Bool(order == Ordering::Less),
        Op::Le => return Val::Bool(order != Ordering::Greater),
        Op::Gt => return Val::Bool(order == Ordering::Greater),
        Op::Ge => return Val::Bool(order != Ordering::Less),
        _ => {}
    }
    let (Val::Num(x), Val::Num(y)) = (a, b) else {
        return Val::Null;
    };
    let got = match op {
        Op::Add => x + y,
        Op::Sub => x - y,
        Op::Mul => x * y,
        Op::Div => x / y,
        Op::Rem => x % y,
        Op::Pow => x.powf(y),
        Op::Or | Op::And | Op::In | Op::Eq | Op::Ne | Op::Lt | Op::Le | Op::Gt | Op::Ge => {
            unreachable!("answered above")
        }
    };
    // A division by zero is an infinity and not an error, which is what the
    // floats say and what every comparison against it then does correctly.
    Val::Num(got)
}

// ---- reading the attribute JSON

/// The value of one top level field, or nothing.
fn field<'a>(attr: &'a [u8], name: &[u8]) -> Val<'a> {
    let mut scan = Json::new(attr);
    let mut key = Vec::new();
    while scan.next_field(&mut key).is_some() {
        if key == name {
            return scan.value();
        }
        scan.skip_value();
    }
    Val::Null
}

/// A cursor over an attribute string, which is a JSON object or is nothing.
///
/// Nothing is what a malformed attribute is too. The alternative is a search
/// that fails because an element somewhere has a broken attribute, and an
/// element whose attribute cannot be read is an element that matches no filter,
/// which is the same answer without the failure.
struct Json<'a> {
    src: &'a [u8],
    at: usize,
    /// Set once the opening brace has been read, so the first field looks for a
    /// key and the ones after it look for a comma first.
    started: bool,
}

impl<'a> Json<'a> {
    fn new(src: &'a [u8]) -> Json<'a> {
        Json {
            src,
            at: 0,
            started: false,
        }
    }

    /// Move to the value of the next field, putting its name in `into`.
    fn next_field(&mut self, into: &mut Vec<u8>) -> Option<()> {
        if !self.started {
            self.started = true;
            self.space();
            self.eat(b'{')?;
        } else {
            self.space();
            match self.peek()? {
                b',' => self.at += 1,
                _ => return None,
            }
        }
        self.space();
        if self.peek()? == b'}' {
            return None;
        }
        let key = self.text()?;
        into.clear();
        into.extend(Decode::new(key));
        self.space();
        self.eat(b':')?;
        self.space();
        Some(())
    }

    /// The scalar the cursor is on, having read past it.
    fn value(&mut self) -> Val<'a> {
        match self.peek() {
            Some(b'"') => match self.text() {
                Some(t) => Val::Text(t),
                None => Val::Null,
            },
            Some(b't') if self.word(b"true") => Val::Bool(true),
            Some(b'f') if self.word(b"false") => Val::Bool(false),
            Some(b'n') if self.word(b"null") => Val::Null,
            Some(c) if c == b'-' || c.is_ascii_digit() => match self.number() {
                Some(n) => Val::Num(n),
                None => Val::Null,
            },
            // An object or an array. A filter selects scalars, so this is the
            // same as the field not being there.
            _ => Val::Null,
        }
    }

    /// The string the cursor is on, if it is on one, put into `into` decoded.
    ///
    /// Says whether there was one, which is what makes it the test for "this
    /// field holds a string" as well as the way to read it.
    fn string_value(&mut self, into: &mut Vec<u8>) -> bool {
        if self.peek() != Some(b'"') {
            self.skip_value();
            return false;
        }
        let Some(text) = self.text() else {
            return false;
        };
        into.clear();
        into.extend(Decode::new(text));
        true
    }

    /// Read past whatever value the cursor is on, however nested.
    fn skip_value(&mut self) {
        self.space();
        match self.peek() {
            Some(b'"') => {
                let _ = self.text();
            }
            Some(b'{' | b'[') => {
                let mut depth = 0usize;
                loop {
                    self.space();
                    let Some(c) = self.peek() else { return };
                    match c {
                        // A brace inside a string is not a brace, which is the
                        // whole reason this cannot be a bracket count.
                        b'"' => {
                            if self.text().is_none() {
                                return;
                            }
                        }
                        b'{' | b'[' => {
                            depth += 1;
                            self.at += 1;
                        }
                        b'}' | b']' => {
                            depth -= 1;
                            self.at += 1;
                            if depth == 0 {
                                return;
                            }
                        }
                        _ => self.at += 1,
                    }
                }
            }
            // A number, a boolean or a null, which runs to whatever ends it.
            _ => {
                while let Some(c) = self.peek() {
                    if matches!(c, b',' | b'}' | b']') {
                        return;
                    }
                    self.at += 1;
                }
            }
        }
    }

    /// The string starting at the cursor, left where it is.
    fn text(&mut self) -> Option<Text<'a>> {
        self.eat(b'"')?;
        let from = self.at;
        let mut escaped = false;
        loop {
            let c = *self.src.get(self.at)?;
            self.at += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    escaped = true;
                    self.at += 1;
                }
                _ => {}
            }
        }
        Some(Text {
            raw: self.src.get(from..self.at - 1)?,
            escaped,
        })
    }

    /// The number starting at the cursor.
    fn number(&mut self) -> Option<f64> {
        let from = self.at;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || matches!(c, b'-' | b'+' | b'.' | b'e' | b'E') {
                self.at += 1;
            } else {
                break;
            }
        }
        core::str::from_utf8(&self.src[from..self.at])
            .ok()?
            .parse()
            .ok()
    }

    /// Read past a bare word if that is what is here.
    fn word(&mut self, want: &[u8]) -> bool {
        if self.src[self.at..].starts_with(want) {
            self.at += want.len();
            return true;
        }
        false
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.at).copied()
    }

    fn eat(&mut self, want: u8) -> Option<()> {
        if self.peek()? != want {
            return None;
        }
        self.at += 1;
        Some(())
    }

    fn space(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.at += 1;
        }
    }
}

// ---- reading the expression

/// One piece of an expression as the lexer found it.
#[derive(Debug, PartialEq)]
enum Tok {
    Field(Box<[u8]>),
    Num(f64),
    Text(Box<[u8]>),
    Bool(bool),
    Open,
    Close,
    OpenList,
    CloseList,
    Comma,
    Not,
    Op(Op),
}

/// What is wrong with an expression, in the one message a client gets.
fn bad() -> Error {
    Error::new(Code::Invalid, BAD_FILTER)
}

/// Take an expression apart into its pieces.
fn lex(src: &[u8]) -> Result<Vec<Tok>> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < src.len() {
        let c = src[at];
        match c {
            b' ' | b'\t' | b'\n' | b'\r' => at += 1,
            b'(' => {
                out.push(Tok::Open);
                at += 1;
            }
            b')' => {
                out.push(Tok::Close);
                at += 1;
            }
            b'[' => {
                out.push(Tok::OpenList);
                at += 1;
            }
            b']' => {
                out.push(Tok::CloseList);
                at += 1;
            }
            b',' => {
                out.push(Tok::Comma);
                at += 1;
            }
            b'.' => {
                let from = at + 1;
                at = from;
                while at < src.len() && (src[at].is_ascii_alphanumeric() || src[at] == b'_') {
                    at += 1;
                }
                if at == from {
                    return Err(bad());
                }
                out.push(Tok::Field(src[from..at].into()));
            }
            b'"' | b'\'' => {
                let (text, next) = literal(src, at)?;
                out.push(Tok::Text(text));
                at = next;
            }
            b'0'..=b'9' => {
                let from = at;
                while at < src.len()
                    && (src[at].is_ascii_digit() || matches!(src[at], b'.' | b'e' | b'E'))
                {
                    // An `e` only continues the number when it is an exponent,
                    // so that `1 in [1]` does not read the `in` as part of the
                    // number.
                    if matches!(src[at], b'e' | b'E') {
                        let sign = at + 1 < src.len() && matches!(src[at + 1], b'+' | b'-');
                        let digit = at + 1 + usize::from(sign);
                        if digit >= src.len() || !src[digit].is_ascii_digit() {
                            break;
                        }
                        at += 1 + usize::from(sign);
                    }
                    at += 1;
                }
                let n: f64 = core::str::from_utf8(&src[from..at])
                    .map_err(|_| bad())?
                    .parse()
                    .map_err(|_| bad())?;
                out.push(Tok::Num(n));
            }
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                let from = at;
                while at < src.len() && (src[at].is_ascii_alphanumeric() || src[at] == b'_') {
                    at += 1;
                }
                out.push(match &src[from..at] {
                    b"and" => Tok::Op(Op::And),
                    b"or" => Tok::Op(Op::Or),
                    b"not" => Tok::Not,
                    b"in" => Tok::Op(Op::In),
                    b"true" => Tok::Bool(true),
                    b"false" => Tok::Bool(false),
                    _ => return Err(bad()),
                });
            }
            _ => {
                let (tok, next) = symbol(src, at)?;
                out.push(tok);
                at = next;
            }
        }
    }
    Ok(out)
}

/// A string literal, with its escapes resolved as it is read.
///
/// Either quote, because Redis takes either and an expression full of `\"` is
/// worse to write than one in single quotes.
fn literal(src: &[u8], from: usize) -> Result<(Box<[u8]>, usize)> {
    let quote = src[from];
    let mut at = from + 1;
    let body = loop {
        let c = *src.get(at).ok_or_else(bad)?;
        if c == quote {
            break &src[from + 1..at];
        }
        at += if c == b'\\' { 2 } else { 1 };
    };
    let text = Text {
        raw: body,
        escaped: body.contains(&b'\\'),
    };
    Ok((Decode::new(text).collect::<Vec<u8>>().into(), at + 1))
}

/// One of the operators written in symbols.
fn symbol(src: &[u8], at: usize) -> Result<(Tok, usize)> {
    let two = src.get(at..at + 2);
    let tok = match two {
        Some(b"==") => Some(Tok::Op(Op::Eq)),
        Some(b"!=") => Some(Tok::Op(Op::Ne)),
        Some(b"<=") => Some(Tok::Op(Op::Le)),
        Some(b">=") => Some(Tok::Op(Op::Ge)),
        Some(b"&&") => Some(Tok::Op(Op::And)),
        Some(b"||") => Some(Tok::Op(Op::Or)),
        Some(b"**") => Some(Tok::Op(Op::Pow)),
        _ => None,
    };
    if let Some(tok) = tok {
        return Ok((tok, at + 2));
    }
    let one = match src[at] {
        b'<' => Tok::Op(Op::Lt),
        b'>' => Tok::Op(Op::Gt),
        b'!' => Tok::Not,
        b'+' => Tok::Op(Op::Add),
        b'-' => Tok::Op(Op::Sub),
        b'*' => Tok::Op(Op::Mul),
        b'/' => Tok::Op(Op::Div),
        b'%' => Tok::Op(Op::Rem),
        // A single `=` is the mistake everybody makes once, and it gets the same
        // message as anything else rather than a special one, because the
        // message already says the expression is not one.
        _ => return Err(bad()),
    };
    Ok((one, at + 1))
}

/// Where in the tokens the parser is.
struct Parser<'a> {
    toks: &'a [Tok],
    at: usize,
}

/// How tightly an operator binds, loosest first.
///
/// Comparison sits between the logic and the arithmetic, which is what makes
/// `.a + 1 > 2 and .b` read the way it looks.
fn power(op: Op) -> (u8, u8) {
    match op {
        Op::Or => (1, 2),
        Op::And => (3, 4),
        Op::Eq | Op::Ne | Op::Lt | Op::Le | Op::Gt | Op::Ge | Op::In => (5, 6),
        Op::Add | Op::Sub => (7, 8),
        Op::Mul | Op::Div | Op::Rem => (9, 10),
        // Right associative, so `2 ** 3 ** 2` is 2 to the ninth, which is what
        // it is everywhere else it is written.
        Op::Pow => (12, 11),
    }
}

impl Parser<'_> {
    /// Everything binding at least this tightly, which is the whole expression
    /// at zero.
    fn expr(&mut self, least: u8) -> Result<Node> {
        self.expr_at(least, 0)
    }

    fn expr_at(&mut self, least: u8, depth: usize) -> Result<Node> {
        if depth > DEPTH {
            return Err(bad());
        }
        let mut left = self.unary(depth)?;
        while let Some(Tok::Op(op)) = self.toks.get(self.at) {
            let (lhs, rhs) = power(*op);
            if lhs < least {
                break;
            }
            let op = *op;
            self.at += 1;
            let right = if op == Op::In {
                self.list(depth)?
            } else {
                self.expr_at(rhs, depth + 1)?
            };
            left = Node::Bin(op, Box::new((left, right)));
        }
        Ok(left)
    }

    /// A prefix operator and what it applies to, or just a value.
    fn unary(&mut self, depth: usize) -> Result<Node> {
        if depth > DEPTH {
            return Err(bad());
        }
        match self.toks.get(self.at) {
            Some(Tok::Not) => {
                self.at += 1;
                // Tighter than the comparisons, so `!.a == .b` is `(!.a) == .b`,
                // which is what `!` does in the languages this borrows from.
                Ok(Node::Not(Box::new(self.unary(depth + 1)?)))
            }
            Some(Tok::Op(Op::Sub)) => {
                self.at += 1;
                Ok(Node::Neg(Box::new(self.unary(depth + 1)?)))
            }
            _ => self.value(depth),
        }
    }

    /// One literal, one field, or a bracketed expression.
    fn value(&mut self, depth: usize) -> Result<Node> {
        let tok = self.toks.get(self.at).ok_or_else(bad)?;
        self.at += 1;
        match tok {
            Tok::Field(name) => Ok(Node::Field(name.clone())),
            Tok::Num(n) => Ok(Node::Num(*n)),
            Tok::Text(t) => Ok(Node::Text(t.clone())),
            Tok::Bool(b) => Ok(Node::Bool(*b)),
            Tok::Open => {
                let inner = self.expr_at(0, depth + 1)?;
                match self.toks.get(self.at) {
                    Some(Tok::Close) => {
                        self.at += 1;
                        Ok(inner)
                    }
                    _ => Err(bad()),
                }
            }
            Tok::OpenList => {
                self.at -= 1;
                self.list(depth)
            }
            _ => Err(bad()),
        }
    }

    /// The list on the right of `in`.
    ///
    /// It has to be written out, because a list is the one thing here that is
    /// not a value: an attribute field holding an array is not selectable and
    /// there is nothing else a list could have come from.
    fn list(&mut self, depth: usize) -> Result<Node> {
        if depth > DEPTH {
            return Err(bad());
        }
        match self.toks.get(self.at) {
            Some(Tok::OpenList) => self.at += 1,
            _ => return Err(bad()),
        }
        let mut items = Vec::new();
        if self.toks.get(self.at) == Some(&Tok::CloseList) {
            self.at += 1;
            return Ok(Node::List(items.into()));
        }
        loop {
            items.push(self.expr_at(1, depth + 1)?);
            match self.toks.get(self.at) {
                Some(Tok::Comma) => self.at += 1,
                Some(Tok::CloseList) => {
                    self.at += 1;
                    return Ok(Node::List(items.into()));
                }
                _ => return Err(bad()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whether an attribute string matches an expression.
    fn hit(expr: &str, attr: &str) -> bool {
        Filter::parse(expr.as_bytes())
            .expect("the expression is one")
            .matches(Some(attr.as_bytes()))
    }

    /// What an expression that is not one gets, which is one message.
    fn refused(expr: &str) -> bool {
        Filter::parse(expr.as_bytes()).is_err()
    }

    /// The operators bind the way they look, which is the only thing anybody
    /// checks by reading an expression back.
    #[test]
    fn an_expression_reads_the_way_it_looks() {
        let doc = r#"{"a":2,"b":3,"lang":"en"}"#;
        assert!(hit(".a + 1 == 3", doc));
        assert!(hit(".a * .b == 6", doc));
        // Times before plus, and the comparison after both of them.
        assert!(hit(".a + .b * 2 == 8", doc));
        assert!(hit("(.a + .b) * 2 == 10", doc));
        // Right associative, so this is two to the ninth and not eight squared.
        assert!(hit("2 ** 3 ** 2 == 512", doc));
        assert!(hit("-.a == 0 - 2", doc));
        assert!(hit(".b % .a == 1", doc));
        // And before or, so this is (en and 2) or (never), not en and (2 or
        // never).
        assert!(hit(".lang == 'en' and .a == 2 or .a == 99", doc));
        assert!(!hit(".lang == 'de' and (.a == 2 or .a == 99)", doc));
        // The words and the symbols are the same operators.
        assert!(hit(".a == 2 && .b == 3", doc));
        assert!(hit(".a == 9 || .b == 3", doc));
        assert!(hit("!(.a == 9)", doc));
        assert!(hit("not (.a == 9)", doc));
        // `not` is the same operator as `!` and binds as tightly, so this is
        // `(not .a) == 9` and not `not (.a == 9)`, which is what it would be in
        // a language where `not` is the loose one.
        assert!(!hit("not .a == 9", doc));
        assert!(hit("!.missing", doc));
        assert!(hit(".lang in ['fr', 'en']", doc));
        assert!(!hit(".lang in []", doc));
    }

    /// A field the attribute does not have is not an error, does not match, and
    /// does not take the rest of the expression down with it.
    #[test]
    fn a_field_that_is_not_there_matches_nothing() {
        let doc = r#"{"year":1999,"nested":{"x":1},"list":[1,2]}"#;
        assert!(!hit(".missing > 1980", doc));
        assert!(!hit(".missing < 1980", doc));
        assert!(!hit(".missing == 1980", doc));
        // Not equal to nothing is still true, which is what makes `.a != 1` a
        // filter for "does not say 1" rather than for "says something else".
        assert!(hit(".missing != 1980", doc));
        // An object and an array are not scalars, so selecting one is the same
        // as selecting nothing.
        assert!(!hit(".nested == 1", doc));
        assert!(!hit(".list == 1", doc));
        // The rest of the expression is unaffected.
        assert!(hit(".missing > 1980 or .year > 1980", doc));
        assert!(!hit(".missing > 1980 and .year > 1980", doc));
        // Arithmetic on nothing is nothing, and every comparison against it is
        // false rather than an error.
        assert!(!hit(".missing + 1 == 1", doc));
        // An attribute that is not JSON at all is one whose fields are all
        // missing, which is the same answer.
        assert!(!hit(".year == 1999", "not json"));
        assert!(!hit(".year == 1999", "{\"year\":"));
        assert!(!Filter::parse(b".year == 1999").unwrap().matches(None));
    }

    /// Nothing is coerced, because coercing would make `.id == 0` match an id
    /// that is a word.
    #[test]
    fn the_types_do_not_coerce() {
        let doc = r#"{"year":1999,"open":true,"tag":"7"}"#;
        assert!(!hit(".year == '1999'", doc));
        assert!(!hit(".tag == 7", doc));
        assert!(hit(".tag == '7'", doc));
        assert!(hit(".open == true", doc));
        assert!(hit(".open", doc));
        assert!(!hit(".open == 1", doc));
        // A number is a truth value when it is not zero, which is what makes a
        // count a filter on its own.
        assert!(hit(".year", doc));
        assert!(!hit(".tag", doc));
    }

    /// A string is compared by what it stands for and not by how it was
    /// spelled, on both sides.
    #[test]
    fn an_escape_is_compared_by_what_it_stands_for() {
        assert!(hit(r".a == 'x\ny'", r#"{"a":"x\ny"}"#));
        assert!(hit(".a == 'x\ny'", r#"{"a":"x\ny"}"#));
        assert!(!hit(r".a == 'x\ny'", r#"{"a":"xny"}"#));
        // A code point spelled either way is the same string.
        assert!(hit(".a == 'é'", r#"{"a":"é"}"#));
        assert!(hit(r".a == 'é'", r#"{"a":"é"}"#));
        // Past the basic plane, which is a surrogate pair in JSON.
        assert!(hit(".a == '😀'", r#"{"a":"😀"}"#));
        // A key can be escaped too, and it is the same field either way.
        assert!(hit(".ab == 1", r#"{"ab":1}"#));
        // Ordering on strings is the bytes, which is what a sorted client
        // expects and is why it is not an error.
        assert!(hit(".a > 'abb'", r#"{"a":"abc"}"#));
    }

    /// The cheap test only carries what every match has to pass, because a bit
    /// that is set on the query and not on the element throws the element away.
    #[test]
    fn the_signature_carries_only_what_every_match_needs() {
        let want = |expr: &str| {
            Filter::parse(expr.as_bytes())
                .expect("the expression is one")
                .signature()
        };
        let of = |attr: &str| Signature::from_bits(tag(attr.as_bytes()));

        let doc = of(r#"{"lang":"en","year":1999}"#);
        assert!(doc.covers(want(".lang == 'en'")));
        assert!(doc.covers(want(".lang == 'en' and .year > 1980")));
        assert!(!doc.covers(want(".lang == 'fr'")));
        // Under an `or` nothing is required, so the summary asks for nothing
        // and the expression does all of the work.
        assert_eq!(want(".lang == 'fr' or .lang == 'en'").bits(), 0);
        // Nor under a `not`, nor for anything but equality, nor for a number,
        // which is the half of the language a bit cannot answer.
        assert_eq!(want("!(.lang == 'fr')").bits(), 0);
        assert_eq!(want(".lang != 'fr'").bits(), 0);
        assert_eq!(want(".year == 1999").bits(), 0);
        assert_eq!(want(".lang in ['fr']").bits(), 0);
        // An element with no attribute at all covers nothing, which is right:
        // it matches no equality test either.
        assert_eq!(tag(b""), 0);
        assert!(!of("").covers(want(".lang == 'en'")));
        // The two sides agree whichever way round the test was written.
        assert!(doc.covers(want("'en' == .lang")));
    }

    /// An expression that is not one says so, and nothing a client can send is
    /// a stack overflow.
    #[test]
    fn a_bad_expression_is_an_error_and_not_a_crash() {
        assert!(refused(".k =="));
        assert!(refused("=="));
        assert!(refused("junk"));
        assert!(refused(".a = 1"));
        assert!(refused("."));
        assert!(refused(".a == 'unterminated"));
        assert!(refused(".a == 1)"));
        assert!(refused("(.a == 1"));
        assert!(refused(".a in 1"));
        assert!(refused(".a in ['x'"));
        assert!(refused(".a == 1 and"));
        // Deep enough to be a stack overflow if the depth were not capped, and
        // it is refused rather than answered because nobody writes this.
        let deep = format!("{}.a == 1{}", "(".repeat(2000), ")".repeat(2000));
        assert!(refused(&deep));
        let chain = format!(".a == 1{}", " and .a == 1".repeat(2000));
        assert!(
            Filter::parse(chain.as_bytes())
                .map(|f| f.matches(Some(br#"{"a":1}"#)))
                .unwrap_or(true)
        );
    }

    /// A number in an attribute is read as a number however it was spelled.
    #[test]
    fn a_number_is_read_however_it_was_spelled() {
        assert!(hit(".a == 1", r#"{"a":1.0}"#));
        assert!(hit(".a == 100", r#"{"a":1e2}"#));
        assert!(hit(".a == 0.5", r#"{"a":5e-1}"#));
        assert!(hit(".a == -3", r#"{"a":-3}"#));
        assert!(hit(".a > 1e3", r#"{"a":2000}"#));
        // A field after a nested one is still found, which is the skip working.
        let doc = r#"{"n":{"deep":{"x":[1,{"y":"}"}]}},"a":1}"#;
        assert!(hit(".a == 1", doc));
        let doc = r#"{"s":"a,b}c","a":2}"#;
        assert!(hit(".a == 2", doc));
        // And the whitespace JSON is allowed to have does not change anything.
        assert!(hit(".a == 1", " { \"a\" : 1 } "));
    }
}
