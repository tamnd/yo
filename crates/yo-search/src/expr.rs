//! The little language `APPLY` and `FILTER` are written in.
//!
//! An aggregation step is one string holding one expression, which reads
//! properties off the row in front of it and answers one value. There is no
//! index and no wire in here: a row goes in as a list of [`Value`] and a value
//! comes out, so the whole language can be read and tested on its own.
//!
//! Everything about it was measured against Redis 8.10.1 rather than reasoned
//! about, and the parts nobody would guess are called out where they are built:
//! a number literal carries its own sign so `1+2` is two numbers in a row and
//! not a sum, there is no minus that takes one operand, `!` binds looser than a
//! comparison and tighter than `&&`, and a comparison cannot be chained.
//!
//! Comparison is the corner with the most in it. Two values of different sorts
//! are ordered by their sort with nothing at the bottom and a list above it,
//! two numbers that cannot be ordered are equal rather than neither, so
//! `0/0 == 1` is one, and text that will not read as a number beside one is an
//! error for the four orderings and never for the two equalities. Both of the
//! two word operators stop as soon as their left hand side settles the answer.

use core::fmt::Write as _;

/// One property of a row as it moves through the pipeline.
///
/// Four of these are values a client can see and the fifth is the absence of
/// one, which is not the same thing as a null: a group key that nothing filled
/// in is a null and is answered as one, where a `LOAD` of a field the key does
/// not hold leaves nothing on the row at all. Reading the first is fine and
/// reading the second is an error, which is the whole reason they are told
/// apart.
#[derive(Clone, Debug, Default)]
pub enum Value {
    /// The property is not on this row.
    #[default]
    Missing,
    /// The property is on this row and holds nothing.
    Nil,
    /// A number, which is what a numeric field and every fold answer is.
    Number(f64),
    /// A string, held as bytes because nothing here needs it to be one.
    Text(Box<[u8]>),
    /// A list, which only `split` and the terms of a match ever make.
    List(Vec<Value>),
}

impl Value {
    /// The value as a number, or nothing when it is not one and will not parse
    /// as one.
    #[must_use]
    pub fn number(&self) -> Option<f64> {
        match self {
            Value::Number(n) => Some(*n),
            Value::Text(text) => reading(text),
            _ => None,
        }
    }

    /// The value as bytes, or nothing when it is not text.
    ///
    /// A number is not turned into its digits here, because every function that
    /// wants text refuses a number rather than spelling it out.
    #[must_use]
    pub fn text(&self) -> Option<&[u8]> {
        match self {
            Value::Text(text) => Some(text),
            _ => None,
        }
    }

    /// Whether a `FILTER` keeps a row this answered.
    ///
    /// A number is true when it is not zero, text when it is not empty, a list
    /// always, and neither kind of nothing ever.
    #[must_use]
    pub fn truth(&self) -> bool {
        match self {
            Value::Missing | Value::Nil => false,
            Value::Number(n) => *n != 0.0,
            Value::Text(text) => !text.is_empty(),
            Value::List(_) => true,
        }
    }

    /// The number a type error names this value by.
    ///
    /// These are the numbers a real server prints and they are not a run,
    /// because the list of value kinds it keeps has kinds in it that an
    /// aggregation never makes. Text is in here for the sake of the match and
    /// never reaches the wire, since the only functions that check a type are
    /// the ones that want text.
    fn kind(&self) -> u8 {
        match self {
            Value::Number(_) => 1,
            Value::Text(_) => 2,
            Value::Missing | Value::Nil => 3,
            Value::List(_) => 5,
        }
    }
}

/// Reads bytes as a number the way an expression does.
///
/// Leading and trailing space is allowed and anything else left over is not, so
/// `3x` is not a number where ` 3 ` is.
fn reading(text: &[u8]) -> Option<f64> {
    let text = core::str::from_utf8(text).ok()?.trim();
    text.parse().ok()
}

/// Twelve significant digits, which is how every number an aggregation answers
/// goes on the wire.
///
/// The three that are not numbers are spelled the way C spells them rather than
/// the way Rust does, so a fold over no numbers answers `nan` and not `NaN`.
#[must_use]
pub fn twelve(d: f64) -> String {
    if d.is_nan() {
        return "nan".to_string();
    }
    if !d.is_finite() {
        return format!("{d}");
    }
    let sci = format!("{d:.11e}");
    let (mantissa, exponent) = sci.split_once('e').expect("a scientific form has an e");
    let exponent: i32 = exponent.parse().expect("and a whole number after it");
    if !(-4..12).contains(&exponent) {
        let m = mantissa.trim_end_matches('0').trim_end_matches('.');
        let sign = if exponent < 0 { '-' } else { '+' };
        return format!("{m}e{sign}{:02}", exponent.abs());
    }
    let places = (11 - exponent).max(0) as usize;
    let fixed = format!("{d:.places$}");
    match fixed.contains('.') {
        true => fixed
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string(),
        false => fixed,
    }
}

/// The two operands of a comparison, once both have been looked at.
enum Pair<'a> {
    Words(&'a [u8], &'a [u8]),
    Numbers(f64, f64),
    Lists(&'a [Value], &'a [Value]),
    Nulls,
    Broken,
}

/// What one piece of an expression does.
#[derive(Debug)]
enum Node {
    /// A literal, which is either a number or a string.
    Value(Value),
    /// A property by name, before the pipeline has said where it lives.
    Named(Box<[u8]>),
    /// A property by position in the row.
    Slot(usize, Box<[u8]>),
    /// `!`, which is the only thing here that takes one operand.
    Not(Box<Node>),
    Op(Op, Box<Node>, Box<Node>),
    Call(Func, Vec<Node>),
}

/// The operators that take two operands, in the order the parser reads them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Or,
    And,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
}

/// One function, named the way the parser found it so an error can quote the
/// client's own spelling back.
#[derive(Debug)]
struct Func {
    which: Which,
    spelled: Box<[u8]>,
}

/// Every function the language has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Which {
    Upper,
    Lower,
    Startswith,
    Contains,
    Substr,
    Format,
    MatchedTerms,
    Split,
    Strlen,
    Exists,
    ToNumber,
    Abs,
    Ceil,
    Floor,
    Log,
    Log2,
    Exp,
    Sqrt,
    Timefmt,
    Parsetime,
    Day,
    Hour,
    Minute,
    Month,
    Year,
    MonthOfYear,
    DayOfMonth,
    DayOfWeek,
    DayOfYear,
    Geodistance,
    Case,
}

/// Every function by the name it is written under and how many arguments it
/// takes, from the fewest to the most.
///
/// The widest of them is `format`, which a real server caps at 65535 and says
/// so in the line it answers when the count is wrong.
const FUNCTIONS: &[(&str, Which, usize, usize)] = &[
    ("upper", Which::Upper, 1, 1),
    ("lower", Which::Lower, 1, 1),
    ("startswith", Which::Startswith, 2, 2),
    ("contains", Which::Contains, 2, 2),
    ("substr", Which::Substr, 3, 3),
    ("format", Which::Format, 1, 65535),
    ("matched_terms", Which::MatchedTerms, 0, 1),
    ("split", Which::Split, 1, 3),
    ("strlen", Which::Strlen, 1, 1),
    ("exists", Which::Exists, 1, 1),
    ("to_number", Which::ToNumber, 1, 1),
    ("abs", Which::Abs, 1, 1),
    ("ceil", Which::Ceil, 1, 1),
    ("floor", Which::Floor, 1, 1),
    ("log", Which::Log, 1, 1),
    ("log2", Which::Log2, 1, 1),
    ("exp", Which::Exp, 1, 1),
    ("sqrt", Which::Sqrt, 1, 1),
    ("timefmt", Which::Timefmt, 1, 2),
    ("parsetime", Which::Parsetime, 2, 2),
    ("day", Which::Day, 1, 1),
    ("hour", Which::Hour, 1, 1),
    ("minute", Which::Minute, 1, 1),
    ("month", Which::Month, 1, 1),
    ("year", Which::Year, 1, 1),
    ("monthofyear", Which::MonthOfYear, 1, 1),
    ("dayofmonth", Which::DayOfMonth, 1, 1),
    ("dayofweek", Which::DayOfWeek, 1, 1),
    ("dayofyear", Which::DayOfYear, 1, 1),
    ("geodistance", Which::Geodistance, 2, 4),
    ("case", Which::Case, 3, 3),
];

/// One expression, ready to be given a row.
#[derive(Debug)]
pub struct Expr {
    node: Node,
}

/// The name of a property nothing in the pipeline answers.
#[derive(Debug)]
pub struct Unknown(pub Box<[u8]>);

impl Expr {
    /// Reads an expression, or answers the line a real server answers.
    ///
    /// # Errors
    ///
    /// When the words do not make an expression, when a bare word is not a
    /// function call, or when a function name is not one of the thirty one.
    pub fn parse(src: &[u8]) -> Result<Expr, Vec<u8>> {
        let mut lex = Lex::new(src);
        lex.step();
        let node = or(&mut lex)?;
        if lex.tok != Tok::End {
            return Err(lex.syntax());
        }
        // A word that is not a function call is not refused where it is read,
        // because a word followed by something the parser cannot use is
        // refused for the second reason and mentions the first one after it.
        // By here nothing else has gone wrong, so the word is the answer.
        match lex.pending {
            Some(pending) => Err(pending),
            None => Ok(Expr { node }),
        }
    }

    /// Tells each property in the expression where it will be read from.
    ///
    /// # Errors
    ///
    /// When the pipeline has no property under that name, which is the one
    /// error a step reports before a single row has been looked at.
    pub fn bind(&mut self, at: &mut dyn FnMut(&[u8]) -> Option<usize>) -> Result<(), Unknown> {
        bind(&mut self.node, at)
    }

    /// Works the expression out over one row.
    ///
    /// # Errors
    ///
    /// When a property the row does not hold is read, when something that is
    /// not a number is used as one, or when a function is handed the wrong kind
    /// of value.
    pub fn eval(&self, row: &[Value]) -> Result<Value, Vec<u8>> {
        eval(&self.node, row)
    }
}

fn bind(node: &mut Node, at: &mut dyn FnMut(&[u8]) -> Option<usize>) -> Result<(), Unknown> {
    match node {
        Node::Named(name) => match at(name) {
            Some(slot) => {
                *node = Node::Slot(slot, name.clone());
                Ok(())
            }
            None => Err(Unknown(name.clone())),
        },
        Node::Not(inner) => bind(inner, at),
        Node::Op(_, left, right) => {
            bind(left, at)?;
            bind(right, at)
        }
        Node::Call(_, args) => {
            for arg in args {
                bind(arg, at)?;
            }
            Ok(())
        }
        Node::Value(_) | Node::Slot(..) => Ok(()),
    }
}

/// One word of an expression.
#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Number(f64),
    Text(Box<[u8]>),
    Property(Box<[u8]>),
    Symbol(Box<[u8]>),
    Mark(&'static str),
    End,
}

/// The reader that turns bytes into words.
///
/// It carries two things beside the word itself, and both of them are only
/// there because a real server's error line carries them: where the word it is
/// holding started, and the last word that had any letters in it. A mark like
/// `+` or `)` leaves the second one alone, which is why `1 +` is refused near
/// `1` and not near the `+` that is actually wrong.
struct Lex<'a> {
    src: &'a [u8],
    at: usize,
    tok: Tok,
    spot: usize,
    near: Vec<u8>,
    pending: Option<Vec<u8>>,
}

/// The marks the reader knows, longest first so `<=` is never read as `<`.
const MARKS: &[&str] = &[
    "==", "!=", "<=", ">=", "&&", "||", "<", ">", "!", "+", "-", "*", "/", "%", "^", "(", ")", ",",
];

impl<'a> Lex<'a> {
    fn new(src: &'a [u8]) -> Lex<'a> {
        Lex {
            src,
            at: 0,
            tok: Tok::End,
            spot: 0,
            near: Vec::new(),
            pending: None,
        }
    }

    /// The line a real server answers when the words do not make an
    /// expression, with the word that was not a function call named after it
    /// when there was one.
    fn syntax(&self) -> Vec<u8> {
        let near = String::from_utf8_lossy(&self.near).into_owned();
        let mut line = format!(
            "SEARCH_EXPR Syntax error at offset {} near '{near}'",
            self.spot
        );
        if let Some(pending) = &self.pending {
            let pending = String::from_utf8_lossy(pending).into_owned();
            let pending = pending.trim_start_matches("SEARCH_EXPR ").to_string();
            line.push_str(&format!(": {pending}"));
        }
        line.into_bytes()
    }

    /// Reads the next word, stepping over anything the language has no word
    /// for, which is what makes `.5` the number five and `$x` the symbol `x`.
    fn step(&mut self) {
        loop {
            while self.at < self.src.len() && self.src[self.at].is_ascii_whitespace() {
                self.at += 1;
            }
            if self.at >= self.src.len() {
                // The offset stays where the last word started, which is what
                // makes an error at the end of the input point at the word
                // before it rather than past the end.
                self.tok = Tok::End;
                return;
            }
            let from = self.at;
            let byte = self.src[from];
            if let Some(tok) = self.number(from) {
                self.spot = from;
                self.tok = tok;
                return;
            }
            if byte == b'\'' || byte == b'"' {
                if let Some(tok) = self.string(from, byte) {
                    self.spot = from;
                    self.tok = tok;
                    return;
                }
                self.at = from + 1;
                continue;
            }
            if byte == b'@' {
                if let Some(name) = self.name(from + 1) {
                    self.spot = from;
                    self.near = name.clone();
                    self.tok = Tok::Property(name.into());
                    return;
                }
                self.at = from + 1;
                continue;
            }
            if let Some(name) = self.name(from) {
                self.spot = from;
                self.near = name.clone();
                self.tok = Tok::Symbol(name.into());
                return;
            }
            if let Some(mark) = MARKS
                .iter()
                .find(|mark| self.src[from..].starts_with(mark.as_bytes()))
            {
                self.at = from + mark.len();
                self.spot = from;
                self.tok = Tok::Mark(mark);
                return;
            }
            self.at = from + 1;
        }
    }

    /// A number, which carries its own sign.
    ///
    /// That is the one piece of the reader with a rule behind it that shows up
    /// in ordinary use: `1+2` is a one and a plus two rather than a sum, and it
    /// is refused as two numbers side by side.
    fn number(&mut self, from: usize) -> Option<Tok> {
        let mut at = from;
        if matches!(self.src.get(at), Some(b'+' | b'-')) {
            at += 1;
        }
        let digits = at;
        while matches!(self.src.get(at), Some(b) if b.is_ascii_digit()) {
            at += 1;
        }
        if at == digits {
            return None;
        }
        if self.src.get(at) == Some(&b'.') {
            let mut after = at + 1;
            while matches!(self.src.get(after), Some(b) if b.is_ascii_digit()) {
                after += 1;
            }
            if after > at + 1 {
                at = after;
            }
        }
        if matches!(self.src.get(at), Some(b'e' | b'E')) {
            let mut after = at + 1;
            if matches!(self.src.get(after), Some(b'+' | b'-')) {
                after += 1;
            }
            let start = after;
            while matches!(self.src.get(after), Some(b) if b.is_ascii_digit()) {
                after += 1;
            }
            if after > start {
                at = after;
            }
        }
        let text = &self.src[from..at];
        let value: f64 = core::str::from_utf8(text).ok()?.parse().ok()?;
        self.at = at;
        self.near = text.to_vec();
        Some(Tok::Number(value))
    }

    /// A quoted string, or nothing when the quote never closes.
    ///
    /// A backslash stands for the quote it is inside and for itself and for
    /// nothing else, so `'a\nb'` holds a backslash and an `n` rather than a new
    /// line.
    fn string(&mut self, from: usize, quote: u8) -> Option<Tok> {
        let mut at = from + 1;
        let mut held = Vec::new();
        while at < self.src.len() {
            let byte = self.src[at];
            if byte == b'\\' && matches!(self.src.get(at + 1), Some(&b) if b == quote || b == b'\\')
            {
                held.push(self.src[at + 1]);
                at += 2;
                continue;
            }
            if byte == quote {
                self.at = at + 1;
                self.near = held.clone();
                return Some(Tok::Text(held.into()));
            }
            held.push(byte);
            at += 1;
        }
        None
    }

    /// A word made of letters, digits and underscores, which has to start with
    /// one that is not a digit.
    fn name(&mut self, from: usize) -> Option<Vec<u8>> {
        let first = *self.src.get(from)?;
        if !first.is_ascii_alphabetic() && first != b'_' {
            return None;
        }
        let mut at = from;
        while matches!(self.src.get(at), Some(b) if b.is_ascii_alphanumeric() || *b == b'_') {
            at += 1;
        }
        self.at = at;
        Some(self.src[from..at].to_vec())
    }

    /// Whether the word being held is this mark, and steps over it if so.
    fn takes(&mut self, mark: &str) -> bool {
        if self.tok == Tok::Mark(known(mark)) {
            self.step();
            return true;
        }
        false
    }
}

/// Looks a mark up in the table so the word being held can be compared against
/// it by the pointer the table holds rather than byte by byte.
fn known(mark: &str) -> &'static str {
    MARKS
        .iter()
        .copied()
        .find(|held| *held == mark)
        .expect("every mark the parser asks for is in the table")
}

fn or(lex: &mut Lex<'_>) -> Result<Node, Vec<u8>> {
    let mut left = and(lex)?;
    while lex.takes("||") {
        left = Node::Op(Op::Or, Box::new(left), Box::new(and(lex)?));
    }
    Ok(left)
}

fn and(lex: &mut Lex<'_>) -> Result<Node, Vec<u8>> {
    let mut left = not(lex)?;
    while lex.takes("&&") {
        left = Node::Op(Op::And, Box::new(left), Box::new(not(lex)?));
    }
    Ok(left)
}

/// `!`, which sits between `&&` and a comparison.
///
/// That is why `!1 + 1` is nought: the `+` is read first, and the `!` is handed
/// the two rather than the one.
fn not(lex: &mut Lex<'_>) -> Result<Node, Vec<u8>> {
    if lex.takes("!") {
        return Ok(Node::Not(Box::new(not(lex)?)));
    }
    compare(lex)
}

/// A comparison, which cannot be chained: `1 < 2 < 3` is refused rather than
/// read as either half of it.
fn compare(lex: &mut Lex<'_>) -> Result<Node, Vec<u8>> {
    let left = add(lex)?;
    let op = match &lex.tok {
        Tok::Mark("==") => Op::Eq,
        Tok::Mark("!=") => Op::Ne,
        Tok::Mark("<") => Op::Lt,
        Tok::Mark("<=") => Op::Le,
        Tok::Mark(">") => Op::Gt,
        Tok::Mark(">=") => Op::Ge,
        _ => return Ok(left),
    };
    lex.step();
    Ok(Node::Op(op, Box::new(left), Box::new(add(lex)?)))
}

fn add(lex: &mut Lex<'_>) -> Result<Node, Vec<u8>> {
    let mut left = mul(lex)?;
    loop {
        let op = match &lex.tok {
            Tok::Mark("+") => Op::Add,
            Tok::Mark("-") => Op::Sub,
            _ => return Ok(left),
        };
        lex.step();
        left = Node::Op(op, Box::new(left), Box::new(mul(lex)?));
    }
}

fn mul(lex: &mut Lex<'_>) -> Result<Node, Vec<u8>> {
    let mut left = power(lex)?;
    loop {
        let op = match &lex.tok {
            Tok::Mark("*") => Op::Mul,
            Tok::Mark("/") => Op::Div,
            Tok::Mark("%") => Op::Rem,
            _ => return Ok(left),
        };
        lex.step();
        left = Node::Op(op, Box::new(left), Box::new(power(lex)?));
    }
}

/// `^`, which is the one operator that leans right, so `2 ^ 3 ^ 2` is five
/// hundred and twelve rather than sixty four.
fn power(lex: &mut Lex<'_>) -> Result<Node, Vec<u8>> {
    let left = primary(lex)?;
    if lex.takes("^") {
        return Ok(Node::Op(Op::Pow, Box::new(left), Box::new(power(lex)?)));
    }
    Ok(left)
}

fn primary(lex: &mut Lex<'_>) -> Result<Node, Vec<u8>> {
    match lex.tok.clone() {
        Tok::Number(n) => {
            lex.step();
            Ok(Node::Value(Value::Number(n)))
        }
        Tok::Text(text) => {
            lex.step();
            Ok(Node::Value(Value::Text(text)))
        }
        Tok::Property(name) => {
            lex.step();
            Ok(Node::Named(name))
        }
        Tok::Symbol(name) => {
            lex.step();
            if !lex.takes("(") {
                // Held rather than answered, because a word that is not a call
                // is only half of what is wrong when there is more after it.
                if lex.pending.is_none() {
                    let name = String::from_utf8_lossy(&name).into_owned();
                    lex.pending = Some(format!("SEARCH_EXPR Unknown symbol '{name}'").into_bytes());
                }
                return Ok(Node::Value(Value::Nil));
            }
            call(lex, &name)
        }
        Tok::Mark("(") => {
            lex.step();
            let inner = or(lex)?;
            if !lex.takes(")") {
                return Err(lex.syntax());
            }
            Ok(inner)
        }
        _ => Err(lex.syntax()),
    }
}

/// The arguments of a call, with the opening bracket already read.
fn call(lex: &mut Lex<'_>, name: &[u8]) -> Result<Node, Vec<u8>> {
    let spelled = String::from_utf8_lossy(name).into_owned();
    let mut args = Vec::new();
    if !lex.takes(")") {
        loop {
            args.push(or(lex)?);
            if lex.takes(",") {
                continue;
            }
            if !lex.takes(")") {
                return Err(lex.syntax());
            }
            break;
        }
    }
    let lower = spelled.to_ascii_lowercase();
    let Some((_, which, low, high)) = FUNCTIONS.iter().find(|(held, ..)| *held == lower).copied()
    else {
        return Err(format!("SEARCH_EXPR Unknown function name '{spelled}'").into_bytes());
    };
    if args.len() < low || args.len() > high {
        let got = args.len();
        let wants = match low == high {
            true => format!("{low} arguments"),
            false => format!("between {low} and {high} arguments"),
        };
        return Err(
            format!("SEARCH_EXPR Function '{spelled}' expects {wants}, but got {got}").into_bytes(),
        );
    }
    Ok(Node::Call(
        Func {
            which,
            spelled: spelled.into_bytes().into(),
        },
        args,
    ))
}

const NOT_A_NUMBER: &str = "SEARCH_NUMERIC_VALUE_INVALID Invalid numeric value";
const NOT_COMPARABLE: &str = "Error converting string";
const NOT_FOUND: &str = "SEARCH_VALUE_NOT_FOUND Could not find the value for a parameter name, consider using EXISTS if applicable for ";

fn eval(node: &Node, row: &[Value]) -> Result<Value, Vec<u8>> {
    match node {
        Node::Value(value) => Ok(value.clone()),
        Node::Named(name) | Node::Slot(_, name) => {
            let held = match node {
                Node::Slot(at, _) => row.get(*at).unwrap_or(&Value::Missing),
                _ => &Value::Missing,
            };
            if matches!(held, Value::Missing) {
                let name = String::from_utf8_lossy(name).into_owned();
                return Err(format!("{NOT_FOUND}{name}").into_bytes());
            }
            Ok(held.clone())
        }
        Node::Not(inner) => Ok(Value::Number(f64::from(u8::from(
            !eval(inner, row)?.truth(),
        )))),
        Node::Op(op, left, right) => operate(*op, left, right, row),
        Node::Call(func, args) => run(func, args, row),
    }
}

fn operate(op: Op, left: &Node, right: &Node, row: &[Value]) -> Result<Value, Vec<u8>> {
    if matches!(op, Op::And | Op::Or) {
        let left = eval(left, row)?.truth();
        // Both operators stop early. A true left of an `||` and a false left of
        // an `&&` settle the answer, and the right hand side is never looked at,
        // so an error waiting there never happens.
        let stops = match op {
            Op::And => !left,
            _ => left,
        };
        if stops {
            return Ok(Value::Number(f64::from(u8::from(left))));
        }
        let right = eval(right, row)?.truth();
        return Ok(Value::Number(f64::from(u8::from(right))));
    }
    let left = eval(left, row)?;
    let right = eval(right, row)?;
    if matches!(op, Op::Eq | Op::Ne | Op::Lt | Op::Le | Op::Gt | Op::Ge) {
        // Two values that cannot be compared are not equal rather than an
        // error, which is why `'a' == 1` is nought where `'a' < 1` is refused.
        let order = compared(&left, &right);
        let held = match (op, order) {
            (Op::Eq, order) => order == Some(core::cmp::Ordering::Equal),
            (Op::Ne, order) => order != Some(core::cmp::Ordering::Equal),
            (_, None) => return Err(NOT_COMPARABLE.as_bytes().to_vec()),
            (Op::Lt, Some(order)) => order == core::cmp::Ordering::Less,
            (Op::Le, Some(order)) => order != core::cmp::Ordering::Greater,
            (Op::Gt, Some(order)) => order == core::cmp::Ordering::Greater,
            (_, Some(order)) => order != core::cmp::Ordering::Less,
        };
        return Ok(Value::Number(f64::from(u8::from(held))));
    }
    let (Some(a), Some(b)) = (left.number(), right.number()) else {
        return Err(NOT_A_NUMBER.as_bytes().to_vec());
    };
    Ok(Value::Number(match op {
        Op::Add => a + b,
        Op::Sub => a - b,
        Op::Mul => a * b,
        Op::Div => a / b,
        Op::Rem => a % b,
        _ => a.powf(b),
    }))
}

/// How two values compare.
///
/// Two values of different sorts are ordered by their sort, with nothing under
/// a list and a list under everything else, so `upper(1) < split('a')` and
/// `split('a') < 'a'` are both one. Two of the same sort are compared by what
/// they hold: two pieces of text as bytes, so `'2' < '10'` is false, and
/// anything else as numbers, so `'2' < 10` is true. That is the rule behind the
/// one comparison error there is: a piece of text beside a number that will not
/// read as one has nothing to be compared with.
fn compared(left: &Value, right: &Value) -> Option<core::cmp::Ordering> {
    let sorts = sorted(left).cmp(&sorted(right));
    if sorts != core::cmp::Ordering::Equal {
        return Some(sorts);
    }
    match pairing(left, right) {
        Pair::Words(a, b) => Some(a.cmp(b)),
        // Two numbers that cannot be ordered are equal rather than an error or
        // a comparison that is false whichever way it is asked, so `0/0 == 1`
        // is one and so is `0/0 >= 0/0`.
        Pair::Numbers(a, b) => Some(match (a < b, a > b) {
            (true, _) => core::cmp::Ordering::Less,
            (_, true) => core::cmp::Ordering::Greater,
            _ => core::cmp::Ordering::Equal,
        }),
        Pair::Lists(a, b) => {
            for (one, two) in a.iter().zip(b.iter()) {
                let order = compared(one, two)?;
                if order != core::cmp::Ordering::Equal {
                    return Some(order);
                }
            }
            Some(a.len().cmp(&b.len()))
        }
        Pair::Nulls => Some(core::cmp::Ordering::Equal),
        Pair::Broken => None,
    }
}

/// Which sort a value is for the sake of a comparison, lowest first.
fn sorted(value: &Value) -> u8 {
    match value {
        Value::Missing | Value::Nil => 0,
        Value::List(_) => 1,
        Value::Number(_) | Value::Text(_) => 2,
    }
}

fn pairing<'a>(left: &'a Value, right: &'a Value) -> Pair<'a> {
    match (left, right) {
        (Value::Text(a), Value::Text(b)) => Pair::Words(a, b),
        (Value::List(a), Value::List(b)) => Pair::Lists(a, b),
        (Value::Missing | Value::Nil, Value::Missing | Value::Nil) => Pair::Nulls,
        _ => match (left.number(), right.number()) {
            (Some(a), Some(b)) => Pair::Numbers(a, b),
            _ => Pair::Broken,
        },
    }
}

fn run(func: &Func, args: &[Node], row: &[Value]) -> Result<Value, Vec<u8>> {
    if func.which == Which::Exists {
        // The one function that is handed a property rather than its value, so
        // that asking whether a row holds one is not itself an error.
        if let Node::Slot(at, _) = &args[0] {
            let held = row.get(*at).unwrap_or(&Value::Missing);
            return Ok(Value::Number(f64::from(u8::from(!matches!(
                held,
                Value::Missing
            )))));
        }
        eval(&args[0], row)?;
        return Ok(Value::Number(1.0));
    }
    let mut held = Vec::with_capacity(args.len());
    for arg in args {
        held.push(eval(arg, row)?);
    }
    apply(func, &held)
}

/// A function and its arguments, once every one of them has been worked out.
#[expect(
    clippy::too_many_lines,
    reason = "thirty one functions, each of them a line or two"
)]
fn apply(func: &Func, args: &[Value]) -> Result<Value, Vec<u8>> {
    let name = String::from_utf8_lossy(&func.spelled).into_owned();
    match func.which {
        Which::Upper => Ok(folded(&args[0], u8::to_ascii_uppercase)),
        Which::Lower => Ok(folded(&args[0], u8::to_ascii_lowercase)),
        Which::Strlen => {
            let text = words(&args[0], 0, &name)?;
            Ok(Value::Number(text.len() as f64))
        }
        Which::Startswith => {
            let whole = words(&args[0], 0, &name)?;
            let front = words(&args[1], 1, &name)?;
            Ok(Value::Number(f64::from(u8::from(whole.starts_with(front)))))
        }
        // Not a yes or a no: a real server counts how many times the second
        // string is in the first, and an empty one is found between every pair
        // of bytes and at both ends.
        Which::Contains => {
            let whole = words(&args[0], 0, &name)?;
            let part = words(&args[1], 1, &name)?;
            if part.is_empty() {
                return Ok(Value::Number(whole.len() as f64 + 1.0));
            }
            let mut found = 0.0;
            let mut at = 0;
            while at + part.len() <= whole.len() {
                if &whole[at..at + part.len()] == part {
                    found += 1.0;
                    at += part.len();
                    continue;
                }
                at += 1;
            }
            Ok(Value::Number(found))
        }
        Which::Substr => {
            // The first argument is checked on its own and named on its own,
            // where the two after it are named the way every other function
            // names an argument of the wrong kind.
            let Some(text) = args[0].text() else {
                return Err(b"SEARCH_PARSE_ARGS Invalid type for substr. Expected string".to_vec());
            };
            let from = digits(&args[1], 1, &name)?;
            let count = digits(&args[2], 2, &name)?;
            // Both of those are checked for being numbers rather than for
            // reading as one, so `substr('abc', '1', 1)` is refused where
            // `abs('3')` is not.
            Ok(Value::Text(cut(text, from as i64, count as i64).into()))
        }
        Which::Format => format(args, &name),
        Which::Split => split(args, &name),
        Which::MatchedTerms => Ok(Value::Nil),
        Which::ToNumber => match args[0].number() {
            Some(number) => Ok(Value::Number(number)),
            None => {
                let held = match args[0].text() {
                    Some(text) => String::from_utf8_lossy(text).into_owned(),
                    None => "(null)".to_string(),
                };
                Err(format!("SEARCH_PARSE_ARGS to_number: cannot convert string '{held}'").into())
            }
        },
        Which::Abs => Ok(Value::Number(count(&args[0]).abs())),
        Which::Ceil => Ok(Value::Number(count(&args[0]).ceil())),
        Which::Floor => Ok(Value::Number(count(&args[0]).floor())),
        Which::Log => Ok(Value::Number(count(&args[0]).ln())),
        Which::Log2 => Ok(Value::Number(count(&args[0]).log2())),
        Which::Exp => Ok(Value::Number(count(&args[0]).exp())),
        Which::Sqrt => Ok(Value::Number(count(&args[0]).sqrt())),
        Which::Timefmt => {
            let Some(when) = args[0].number() else {
                return Ok(Value::Nil);
            };
            // The second argument is the one place in the language where the
            // line names a function nobody can call, because a real server
            // checks it under the name it keeps the pair under rather than
            // under the name it was written with.
            let shape = match args.get(1) {
                Some(value) => value.text().map(<[u8]>::to_vec).ok_or_else(|| {
                    let kind = value.kind();
                    format!(
                        "SEARCH_PARSE_ARGS Invalid type ({kind}) for argument 1 in function 'time'. VALIDATE_ARG__TYPE(v, RSValueType_String) was false."
                    )
                    .into_bytes()
                })?,
                None => b"%FT%TZ".to_vec(),
            };
            Ok(Value::Text(stamped(when as i64, &shape).into()))
        }
        Which::Parsetime => {
            let text = words(&args[0], 0, &name)?.to_vec();
            let shape = words(&args[1], 1, &name)?;
            match read_time(&text, shape) {
                Some(when) => Ok(Value::Number(when as f64)),
                None => Ok(Value::Nil),
            }
        }
        Which::Day => Ok(cut_to(&args[0], 86400)),
        Which::Hour => Ok(cut_to(&args[0], 3600)),
        Which::Minute => Ok(cut_to(&args[0], 60)),
        Which::Month => match whole(&args[0]) {
            Some(when) => {
                let (y, m, _) = civil(when.div_euclid(86400));
                Ok(Value::Number((days(y, m, 1) * 86400) as f64))
            }
            None => Ok(Value::Nil),
        },
        Which::Year => Ok(part(&args[0], |y, _, _, _| y as f64)),
        Which::MonthOfYear => Ok(part(&args[0], |_, m, _, _| f64::from(m) - 1.0)),
        Which::DayOfMonth => Ok(part(&args[0], |_, _, d, _| f64::from(d))),
        Which::DayOfWeek => Ok(part(&args[0], |_, _, _, when| {
            ((when.div_euclid(86400) + 4).rem_euclid(7)) as f64
        })),
        Which::DayOfYear => Ok(part(&args[0], |y, m, d, _| {
            (days(y, m, d) - days(y, 1, 1)) as f64
        })),
        Which::Geodistance => apart(args),
        Which::Case => match args[0].truth() {
            true => Ok(args[1].clone()),
            false => Ok(args[2].clone()),
        },
        Which::Exists => Ok(Value::Number(1.0)),
    }
}

/// Text with every ASCII letter turned one way or the other.
///
/// A value that is not text answers nothing rather than an error, which is one
/// of the two ways this language treats a wrong argument and the reason
/// `upper(1)` is a null where `strlen(1)` is refused.
fn folded(value: &Value, each: fn(&u8) -> u8) -> Value {
    match value.text() {
        Some(text) => Value::Text(text.iter().map(each).collect()),
        None => Value::Nil,
    }
}

/// The bytes of an argument, or the line a real server answers when it is not
/// text.
///
/// The number in the brackets at the end is not the argument's position and is
/// always nought, because the check a real server writes there names a variable
/// of its own rather than the argument it is looking at. It is copied as it is
/// rather than made sensible.
fn words<'a>(value: &'a Value, at: usize, func: &str) -> Result<&'a [u8], Vec<u8>> {
    value.text().ok_or_else(|| {
        let kind = value.kind();
        format!(
            "SEARCH_PARSE_ARGS Invalid type ({kind}) for argument {at} in function '{func}'. VALIDATE_ARG__STRING(v, 0) was false."
        )
        .into_bytes()
    })
}

/// The same for an argument that has to be a number, which is spelled with the
/// name of the kind it wanted rather than with a check on the value.
///
/// This one asks whether the value is a number and not whether it reads as one,
/// so text holding digits is refused here where every arithmetic function takes
/// it.
fn digits(value: &Value, at: usize, func: &str) -> Result<f64, Vec<u8>> {
    let held = match value {
        Value::Number(number) => Some(*number),
        _ => None,
    };
    held.ok_or_else(|| {
        let kind = value.kind();
        format!(
            "SEARCH_PARSE_ARGS Invalid type ({kind}) for argument {at} in function '{func}'. VALIDATE_ARG__TYPE(v, RSValueType_Number) was false."
        )
        .into_bytes()
    })
}

/// An argument as a number, where anything that is not one is not an error but
/// a nan, which every one of the seven arithmetic functions then carries.
fn count(value: &Value) -> f64 {
    value.number().unwrap_or(f64::NAN)
}

/// A timestamp as a whole number of seconds, or nothing when the argument is
/// not a number or is before the epoch.
fn whole(value: &Value) -> Option<i64> {
    let number = value.number()?;
    (number >= 0.0 && number.is_finite()).then_some(number as i64)
}

/// A timestamp cut back to the start of the day, hour or minute it is in.
fn cut_to(value: &Value, span: i64) -> Value {
    match whole(value) {
        Some(when) => Value::Number((when - when.rem_euclid(span)) as f64),
        None => Value::Nil,
    }
}

/// One piece of the calendar a timestamp lands on.
fn part(value: &Value, each: fn(i64, u32, u32, i64) -> f64) -> Value {
    match whole(value) {
        Some(when) => {
            let (y, m, d) = civil(when.div_euclid(86400));
            Value::Number(each(y, m, d, when))
        }
        None => Value::Nil,
    }
}

/// A piece of text by offset and length, where either of them may count from
/// the far end.
fn cut(text: &[u8], from: i64, count: i64) -> Vec<u8> {
    let len = text.len() as i64;
    let from = match from < 0 {
        true => (len + from).max(0),
        false => from.min(len),
    };
    let end = match count < 0 {
        true => (len + count).max(from),
        false => (from + count).min(len),
    };
    text[from as usize..end as usize].to_vec()
}

/// `format`, which knows two things and no more: `%s` for the next argument
/// and `%%` for a per cent sign.
fn format(args: &[Value], name: &str) -> Result<Value, Vec<u8>> {
    let shape = words(&args[0], 0, name)?.to_vec();
    let mut out = Vec::new();
    let mut next = 1;
    let mut at = 0;
    while at < shape.len() {
        if shape[at] != b'%' {
            out.push(shape[at]);
            at += 1;
            continue;
        }
        match shape.get(at + 1) {
            Some(b'%') => out.push(b'%'),
            Some(b's') => {
                let Some(value) = args.get(next) else {
                    return Err(b"SEARCH_PARSE_ARGS Not enough arguments for format".to_vec());
                };
                next += 1;
                out.extend_from_slice(&spelled(value));
            }
            _ => return Err(b"SEARCH_PARSE_ARGS Unknown format specifier passed".to_vec()),
        }
        at += 2;
    }
    Ok(Value::Text(out.into()))
}

/// A value written out the way `format` writes it, which is the only place in
/// the language where a number turns into its digits.
fn spelled(value: &Value) -> Vec<u8> {
    match value {
        Value::Text(text) => text.to_vec(),
        Value::Number(number) => twelve(*number).into_bytes(),
        _ => Vec::new(),
    }
}

/// `split`, which cuts on any of the bytes it is given rather than on the whole
/// of them, drops the empty pieces, and trims the third argument's bytes off
/// both ends of what is left.
fn split(args: &[Value], name: &str) -> Result<Value, Vec<u8>> {
    let text = words(&args[0], 0, name)?;
    let marks = match args.get(1) {
        Some(value) => words(value, 1, name)?.to_vec(),
        None => b",".to_vec(),
    };
    let trim = match args.get(2) {
        Some(value) => words(value, 2, name)?.to_vec(),
        None => Vec::new(),
    };
    let marks = match marks.is_empty() {
        true => b",".to_vec(),
        false => marks,
    };
    let mut held = Vec::new();
    for piece in text.split(|byte| marks.contains(byte)) {
        let mut piece = piece;
        while matches!(piece.first(), Some(b) if trim.contains(b)) {
            piece = &piece[1..];
        }
        while matches!(piece.last(), Some(b) if trim.contains(b)) {
            piece = &piece[..piece.len() - 1];
        }
        if !piece.is_empty() {
            held.push(Value::Text(piece.to_vec().into()));
        }
    }
    Ok(Value::List(held))
}

/// The metres between two points, given as four numbers, as two `lon,lat`
/// strings, or as one of each.
fn apart(args: &[Value]) -> Result<Value, Vec<u8>> {
    let mut point = Vec::new();
    for arg in args {
        match arg {
            Value::Text(text) => {
                let held = core::str::from_utf8(text)
                    .ok()
                    .and_then(|t| t.split_once(','));
                let Some((lon, lat)) = held else {
                    return Err(b"SEARCH_PARSE_ARGS Invalid geo string: missing separator".to_vec());
                };
                let (Ok(lon), Ok(lat)) = (lon.trim().parse(), lat.trim().parse()) else {
                    return Err(b"SEARCH_PARSE_ARGS Invalid geo string: bad coordinates".to_vec());
                };
                point.push(lon);
                point.push(lat);
            }
            _ => match arg.number() {
                Some(number) => point.push(number),
                None => return Ok(Value::Nil),
            },
        }
    }
    if point.len() != 4 {
        return Ok(Value::Nil);
    }
    let metres = haversine(point[0], point[1], point[2], point[3]);
    Ok(Value::Number((metres * 100.0).round() / 100.0))
}

/// The earth's radius in metres, which is the number Redis measures a distance
/// between two points with.
const EARTH: f64 = 6_372_797.560_856;

fn haversine(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> f64 {
    let lat1 = lat1.to_radians();
    let lat2 = lat2.to_radians();
    let dlat = (lat2 - lat1) / 2.0;
    let dlon = (lon2.to_radians() - lon1.to_radians()) / 2.0;
    let a = dlat.sin().powi(2) + lat1.cos() * lat2.cos() * dlon.sin().powi(2);
    2.0 * a.sqrt().asin() * EARTH
}

/// The year, month and day a count of days since the epoch lands on.
///
/// This is the usual civil calendar walk, which holds for any year rather than
/// only for the ones a timestamp can reach.
fn civil(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = match mp < 10 {
        true => mp + 3,
        false => mp - 9,
    } as u32;
    (y + i64::from(m <= 2), m, d)
}

/// The count of days since the epoch a year, month and day lands on.
fn days(y: i64, m: u32, d: u32) -> i64 {
    let y = y - i64::from(m <= 2);
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = i64::from(m) + if m > 2 { -3 } else { 9 };
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

const DAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/// A timestamp written out, in the same specifiers C writes one with.
fn stamped(when: i64, shape: &[u8]) -> Vec<u8> {
    let day = when.div_euclid(86400);
    let rest = when.rem_euclid(86400);
    let (y, m, d) = civil(day);
    let (hh, mm, ss) = (rest / 3600, (rest / 60) % 60, rest % 60);
    let dow = ((day + 4).rem_euclid(7)) as usize;
    let doy = days(y, m, d) - days(y, 1, 1);
    let mut out = String::new();
    let mut at = 0;
    while at < shape.len() {
        if shape[at] != b'%' || at + 1 >= shape.len() {
            out.push(shape[at] as char);
            at += 1;
            continue;
        }
        match shape[at + 1] {
            b'Y' => {
                let _ = write!(out, "{y}");
            }
            b'y' => {
                let _ = write!(out, "{:02}", y.rem_euclid(100));
            }
            b'C' => {
                let _ = write!(out, "{:02}", y.div_euclid(100));
            }
            b'm' => {
                let _ = write!(out, "{m:02}");
            }
            b'd' => {
                let _ = write!(out, "{d:02}");
            }
            b'e' => {
                let _ = write!(out, "{d:2}");
            }
            b'H' => {
                let _ = write!(out, "{hh:02}");
            }
            b'I' => {
                let hour = match hh % 12 {
                    0 => 12,
                    other => other,
                };
                let _ = write!(out, "{hour:02}");
            }
            b'M' => {
                let _ = write!(out, "{mm:02}");
            }
            b'S' => {
                let _ = write!(out, "{ss:02}");
            }
            b'j' => {
                let _ = write!(out, "{:03}", doy + 1);
            }
            b'a' => out.push_str(DAYS[dow]),
            b'b' | b'h' => out.push_str(MONTHS[(m - 1) as usize]),
            b'p' => out.push_str(if hh < 12 { "AM" } else { "PM" }),
            b'w' => {
                let _ = write!(out, "{dow}");
            }
            b'u' => {
                let _ = write!(out, "{}", if dow == 0 { 7 } else { dow });
            }
            // The week of the year counted from the first Sunday and from the
            // first Monday. Whatever comes before that first one is week
            // nought.
            b'U' => {
                let _ = write!(out, "{:02}", (doy + 7 - dow as i64) / 7);
            }
            b'W' => {
                let week = (doy + 7 - (dow as i64 + 6).rem_euclid(7)) / 7;
                let _ = write!(out, "{week:02}");
            }
            b'V' | b'G' | b'g' => {
                let (year, week) = isoweek(y, doy + 1, if dow == 0 { 7 } else { dow as i64 });
                match shape[at + 1] {
                    b'V' => {
                        let _ = write!(out, "{week:02}");
                    }
                    b'G' => {
                        let _ = write!(out, "{year}");
                    }
                    _ => {
                        let _ = write!(out, "{:02}", year.rem_euclid(100));
                    }
                }
            }
            b'F' => {
                let _ = write!(out, "{y}-{m:02}-{d:02}");
            }
            b'T' => {
                let _ = write!(out, "{hh:02}:{mm:02}:{ss:02}");
            }
            b'R' => {
                let _ = write!(out, "{hh:02}:{mm:02}");
            }
            b'D' => {
                let _ = write!(out, "{m:02}/{d:02}/{:02}", y.rem_euclid(100));
            }
            b's' => {
                let _ = write!(out, "{when}");
            }
            b'Z' => out.push_str("GMT"),
            b'z' => out.push_str("+0000"),
            b'n' => out.push('\n'),
            b't' => out.push('\t'),
            b'%' => out.push('%'),
            other => {
                out.push('%');
                out.push(other as char);
            }
        }
        at += 2;
    }
    out.into_bytes()
}

/// Which ISO week a day falls in, and the year that week belongs to.
///
/// The first week of a year is the one holding its first Thursday, so the first
/// days of January can belong to the last week of the year before and the last
/// days of December to the first week of the year after.
fn isoweek(y: i64, doy: i64, dow: i64) -> (i64, i64) {
    // The weekday of the last day of a year, as a number where nought is
    // Sunday, which is all the length of a year in weeks turns on.
    let end = |y: i64| (y + y.div_euclid(4) - y.div_euclid(100) + y.div_euclid(400)).rem_euclid(7);
    let long = |y: i64| match end(y) == 4 || end(y - 1) == 3 {
        true => 53,
        false => 52,
    };
    let week = (doy - dow + 10) / 7;
    if week < 1 {
        return (y - 1, long(y - 1));
    }
    if week > long(y) {
        return (y + 1, 1);
    }
    (y, week)
}

/// Reads a timestamp back out of text, filling in whatever the shape did not
/// name from a broken down time with nothing in it.
///
/// That empty time is the first of January 1900 with a day of the month of
/// nought, which is the last day of 1899, so a shape naming only the hour reads
/// as a time in 1899 rather than one in 1970.
fn read_time(text: &[u8], shape: &[u8]) -> Option<i64> {
    let (mut y, mut m, mut d, mut hh, mut mm, mut ss) = (1900_i64, 1_u32, 0_u32, 0, 0, 0);
    let mut at = 0;
    let mut on = 0;
    let number = |at: &mut usize, wide: usize| -> Option<i64> {
        let from = *at;
        let mut held = 0_i64;
        while *at < text.len() && text[*at].is_ascii_digit() && *at - from < wide {
            held = held * 10 + i64::from(text[*at] - b'0');
            *at += 1;
        }
        (*at > from).then_some(held)
    };
    while on < shape.len() {
        if shape[on] != b'%' || on + 1 >= shape.len() {
            if shape[on].is_ascii_whitespace() {
                while at < text.len() && text[at].is_ascii_whitespace() {
                    at += 1;
                }
                on += 1;
                continue;
            }
            if text.get(at) != Some(&shape[on]) {
                return None;
            }
            at += 1;
            on += 1;
            continue;
        }
        match shape[on + 1] {
            b'Y' => y = number(&mut at, 4)?,
            b'm' => m = u32::try_from(number(&mut at, 2)?).ok()?,
            b'd' | b'e' => d = u32::try_from(number(&mut at, 2)?).ok()?,
            b'H' => hh = number(&mut at, 2)?,
            b'M' => mm = number(&mut at, 2)?,
            b'S' => ss = number(&mut at, 2)?,
            b'y' => y = 1900 + number(&mut at, 2)?,
            b'F' => {
                y = number(&mut at, 4)?;
                (text.get(at) == Some(&b'-')).then_some(())?;
                at += 1;
                m = u32::try_from(number(&mut at, 2)?).ok()?;
                (text.get(at) == Some(&b'-')).then_some(())?;
                at += 1;
                d = u32::try_from(number(&mut at, 2)?).ok()?;
            }
            b'T' => {
                hh = number(&mut at, 2)?;
                (text.get(at) == Some(&b':')).then_some(())?;
                at += 1;
                mm = number(&mut at, 2)?;
                (text.get(at) == Some(&b':')).then_some(())?;
                at += 1;
                ss = number(&mut at, 2)?;
            }
            b'%' => {
                (text.get(at) == Some(&b'%')).then_some(())?;
                at += 1;
            }
            _ => return None,
        }
        on += 2;
    }
    if !(1..=12).contains(&m) || d > 31 {
        return None;
    }
    Some(days(y, m, d) * 86400 + hh * 3600 + mm * 60 + ss)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Works an expression out over a row with nothing on it.
    fn value(src: &str) -> Result<Value, String> {
        let expr =
            Expr::parse(src.as_bytes()).map_err(|e| String::from_utf8_lossy(&e).into_owned())?;
        expr.eval(&[])
            .map_err(|e| String::from_utf8_lossy(&e).into_owned())
    }

    /// The value an expression answers, written the way the wire writes it.
    fn text(src: &str) -> String {
        match value(src) {
            Ok(Value::Number(n)) => twelve(n),
            Ok(Value::Text(t)) => String::from_utf8_lossy(&t).into_owned(),
            Ok(Value::List(l)) => format!("{:?}", l.iter().map(spelled).collect::<Vec<_>>()),
            Ok(_) => "nil".to_string(),
            Err(e) => e,
        }
    }

    #[test]
    fn a_number_carries_its_own_sign_so_two_of_them_can_stand_side_by_side() {
        // Which is why this is not a sum: `1` and `+2` are both numbers, and
        // two numbers in a row are not an expression.
        assert!(text("1+2").starts_with("SEARCH_EXPR Syntax error at offset 1 near '+2'"));
        assert_eq!(text("1 + 2"), "3");
        assert_eq!(text("2 ^ -1"), "0.5");
    }

    #[test]
    fn the_operators_bind_in_the_order_a_real_server_binds_them() {
        assert_eq!(text("1 + 2 * 3"), "7");
        assert_eq!(text("2 ^ 2 * 3"), "12");
        assert_eq!(text("2 ^ 3 ^ 2"), "512");
        assert_eq!(text("6 - 3 - 2"), "1");
        assert_eq!(text("1 || 0 && 0"), "1");
        // `!` is looser than a comparison and tighter than `&&`, so this one is
        // not what anyone reading it would expect.
        assert_eq!(text("!1 + 1"), "0");
        assert_eq!(text("!0 && 0"), "0");
    }

    #[test]
    fn a_comparison_cannot_be_chained() {
        assert!(text("1 < 2 < 3").starts_with("SEARCH_EXPR Syntax error"));
        assert_eq!(text("1 < 2"), "1");
    }

    #[test]
    fn two_pieces_of_text_are_compared_as_bytes_and_anything_else_as_numbers() {
        assert_eq!(text("'2' < '10'"), "0");
        assert_eq!(text("'2' < 10"), "1");
        assert_eq!(text("'1.0' == 1"), "1");
        assert_eq!(text("'a' < 1"), "Error converting string");
        // Text that will not read as a number is still never equal to one and
        // always unequal to one, because those two answer without comparing.
        assert_eq!(text("'a' == 1"), "0");
        assert_eq!(text("'a' != 1"), "1");
    }

    #[test]
    fn two_numbers_that_cannot_be_ordered_are_equal_rather_than_neither() {
        assert_eq!(text("0/0 == 0/0"), "1");
        assert_eq!(text("0/0 == 1"), "1");
        assert_eq!(text("0/0 != 1"), "0");
        assert_eq!(text("0/0 > 1"), "0");
        assert_eq!(text("0/0 < 1"), "0");
        assert_eq!(text("0/0 >= 1"), "1");
        // Which is the same answer text holding one of those gets.
        assert_eq!(text("'nan' == 0/0"), "1");
        assert_eq!(text("'inf' > 1"), "1");
    }

    #[test]
    fn values_of_different_sorts_are_ordered_by_their_sort() {
        assert_eq!(text("upper(1) < 1"), "1");
        assert_eq!(text("1 > upper(1)"), "1");
        assert_eq!(text("upper(1) == 1"), "0");
        assert_eq!(text("upper(1) < split('a')"), "1");
        assert_eq!(text("split('a') < 'a'"), "1");
        assert_eq!(text("split('a') == 'a'"), "0");
    }

    #[test]
    fn both_of_the_two_word_operators_stop_early() {
        // The right hand side of either one is never worked out once the left
        // has settled the answer, so the error waiting there never happens.
        assert_eq!(text("1 || ('a' > 1)"), "1");
        assert_eq!(text("0 && ('a' > 1)"), "0");
        assert_eq!(text("('a' > 1) || 1"), "Error converting string");
    }

    #[test]
    fn a_string_holds_only_the_two_escapes_a_real_server_reads() {
        assert_eq!(text(r"'a\'b'"), "a'b");
        assert_eq!(text(r"'a\\b'"), r"a\b");
        assert_eq!(text(r"'a\nb'"), r"a\nb");
    }

    #[test]
    fn the_functions_answer_what_a_real_server_answers() {
        assert_eq!(text("upper('aB')"), "AB");
        assert_eq!(text("upper(1)"), "nil");
        assert_eq!(text("contains('abc','')"), "4");
        assert_eq!(text("contains('abcabc','bc')"), "2");
        assert_eq!(text("substr('abcdef',-2,2)"), "ef");
        assert_eq!(text("substr('abcdef',1,-1)"), "bcde");
        assert_eq!(text("format('%s-%s','a',2)"), "a-2");
        assert_eq!(text("case(0,'a','b')"), "b");
        assert_eq!(text("abs('x')"), "nan");
        assert_eq!(text("1 / 0"), "inf");
        // Arithmetic reads text as a number where `substr` asks the value what
        // it is, so digits in quotes are fine in the first and refused in the
        // second.
        assert_eq!(text("abs('3')"), "3");
        assert_eq!(
            text("substr('abc','1',1)"),
            "SEARCH_PARSE_ARGS Invalid type (2) for argument 1 in function 'substr'. VALIDATE_ARG__TYPE(v, RSValueType_Number) was false."
        );
    }

    #[test]
    fn the_calendar_walks_the_way_the_c_library_walks_it() {
        assert_eq!(text("timefmt(1700000000)"), "2023-11-14T22:13:20Z");
        assert_eq!(text("day(1700000000)"), "1699920000");
        assert_eq!(text("month(1700000000)"), "1698796800");
        assert_eq!(text("year(1700000000)"), "2023");
        // Both of these count from nought, which is what the C library hands
        // back rather than what the name suggests.
        assert_eq!(text("monthofyear(1700000000)"), "10");
        assert_eq!(text("dayofyear(1700000000)"), "317");
        assert_eq!(text("dayofweek(1700000000)"), "2");
        assert_eq!(text("dayofmonth(1700000000)"), "14");
        // A time before the epoch has no calendar piece, though it still has a
        // spelling.
        assert_eq!(text("year(-1)"), "nil");
        assert_eq!(text("timefmt(-1)"), "1969-12-31T23:59:59Z");
        assert_eq!(
            text("parsetime('2023-11-14 22:13:20','%Y-%m-%d %H:%M:%S')"),
            "1700000000"
        );
        // The week fields, which the day of the week the year opened on moves
        // about. The fourteenth of November 2023 is a Tuesday in the forty
        // sixth week however the week is counted.
        assert_eq!(
            text("timefmt(1700000000,'%U %W %V %G %g')"),
            "46 46 46 2023 23"
        );
        // A shape that names none of the date reads against a time with
        // nothing in it, which is the day before the first of January 1900.
        assert_eq!(text("parsetime('5','%H')"), "-2209057200");
        assert_eq!(text("parsetime('2023','%Y')"), "1672444800");
    }

    #[test]
    fn a_function_is_named_back_the_way_the_client_spelled_it() {
        assert_eq!(
            text("UPPER()"),
            "SEARCH_EXPR Function 'UPPER' expects 1 arguments, but got 0"
        );
        assert_eq!(
            text("split()"),
            "SEARCH_EXPR Function 'split' expects between 1 and 3 arguments, but got 0"
        );
        assert_eq!(
            text("nosuch()"),
            "SEARCH_EXPR Unknown function name 'nosuch'"
        );
        assert_eq!(text("true"), "SEARCH_EXPR Unknown symbol 'true'");
    }

    #[test]
    fn a_property_the_row_does_not_hold_is_not_the_same_as_one_holding_nothing() {
        let mut expr = Expr::parse(b"@a").expect("a property on its own is an expression");
        expr.bind(&mut |_| Some(0)).expect("and it binds to a slot");
        let missing = expr.eval(&[Value::Missing]).expect_err("nothing to read");
        assert!(String::from_utf8_lossy(&missing).starts_with("SEARCH_VALUE_NOT_FOUND"));
        assert!(matches!(expr.eval(&[Value::Nil]), Ok(Value::Nil)));
    }
}
