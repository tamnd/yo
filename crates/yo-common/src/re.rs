//! A POSIX extended regular expression matcher over bytes.
//!
//! Redis has one command that takes a regular expression, `ARGREP ... RE`, and
//! it gets it from TRE, which it vendors under `deps/tre`. What it asks TRE for
//! is narrow: `REG_EXTENDED | REG_NOSUB | REG_USEBYTES`, optionally `REG_ICASE`,
//! and then a yes or no per element. No capture groups are read, no match
//! offsets are read, and a pattern with a backreference in it is refused before
//! it is ever run. So the thing that has to be built is a boolean matcher for
//! extended regular expressions over bytes, which is a much smaller object than
//! a general purpose regex crate.
//!
//! That is the whole reason this is here rather than a dependency. The
//! workspace has four third party crates in it and two of them are only for
//! tests, and adding a regex engine and its three transitive crates to the
//! engine that the C ABI and every language binding link against is a large
//! thing to pay for one command. Writing the narrow version is a few hundred
//! lines and it comes out with a property the general one cannot promise: the
//! simulation is a Thompson construction walked with a set of live states, so
//! matching is linear in the subject and cannot be made to blow up by a pattern.
//! `ARGREP` runs its predicates over every element a range touches, from a
//! pattern a client sent, so that is worth having rather than being clever.
//!
//! The syntax is TRE's, read off `deps/tre/lib/tre-parse.c` rather than off
//! POSIX, because the point is to agree with the server people are migrating
//! from. TRE has a table of macros that run before anything else, so `\n` is a
//! newline, `\d` is `[[:digit:]]` and `\w` is `[[:alnum:]_]`, and after that a
//! switch with `\b`, `\B`, `\<`, `\>` and `\xNN`. The one that surprises people
//! is that a backslash inside a bracket expression is a literal backslash and
//! not an escape, so `[\d]` is a backslash or a d.
//!
//! Bytes rather than characters, the same as `glob`, and for the same reason: an
//! array element is arbitrary bytes and deciding what a character is would mean
//! deciding what encoding it is in.
//!
//! Every rule in here was either read off a line of TRE or measured against it.
//! The measuring was done by building TRE from `deps/tre` into a small program
//! that answers the same question this does, and then generating patterns from
//! the pieces TRE's parser has cases for and comparing the two answers. That is
//! how the macro table was found, along with the way a repeated assertion stays
//! mandatory, the split between the two errors a bad bound gives, and the
//! handful of places where TRE and Redis's own fast path disagree with each
//! other. Roughly a quarter of a million comparisons agree.

use core::fmt;

/// The largest `{n,m}` repetition, which is TRE's `RE_DUP_MAX`.
pub const DUP_MAX: u32 = 255;

/// The largest program a pattern may compile to.
///
/// A bound repetition is expanded by copying, so `(a{255}){255}` is sixty five
/// thousand instructions and nesting one more level is sixteen million. The cap
/// is what stops a short pattern from turning into a long compile, and it is
/// generous enough that nothing anyone writes by hand reaches it.
const PROG_MAX: usize = 100_000;

/// The deepest a pattern may nest groups.
///
/// The parser is recursive, so this is the difference between refusing a
/// pattern and overflowing the stack on the shard thread.
const DEPTH_MAX: u32 = 64;

/// Why a pattern would not compile.
///
/// The names and the messages are TRE's, from `deps/tre/lib/regerror.c`,
/// because Redis puts the message straight into its error reply and a client
/// that matches on the text should not be able to tell the two servers apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// `REG_BADPAT`, a pattern that is not one.
    BadPattern,
    /// `REG_ECOLLATE`, a `[.` or `[=` bracket item.
    Collate,
    /// `REG_ECTYPE`, a `[:name:]` that is not a class.
    CharClass,
    /// `REG_EESCAPE`, a backslash at the very end.
    TrailingBackslash,
    /// `REG_ESUBREG`, a backreference to a group the pattern does not have.
    BackRef,
    /// A backreference to a group that does exist.
    ///
    /// Not one of TRE's codes. TRE compiles this and Redis refuses it a step
    /// later with a sentence of its own, so the message here is that sentence
    /// and a caller reporting it should not put "invalid regular expression"
    /// in front of it the way it would for the rest of these.
    Unsupported,
    /// `REG_EBRACK`, a bracket expression with no `]`.
    MissingBracket,
    /// `REG_EPAREN`, a group with no `)`.
    MissingParen,
    /// `REG_EBRACE`, a `\x{` or a `{` with no `}`.
    MissingBrace,
    /// `REG_BADBR`, a `{}` whose contents are not a bound.
    BadBrace,
    /// `REG_ERANGE`, a bracket range that runs backwards.
    BadRange,
    /// `REG_ESPACE`, a pattern that compiles to more than this engine will hold.
    Space,
    /// `REG_BADRPT`, a repetition operator with nothing in front of it.
    BadRepeat,
    /// `REG_BADMAX`, a `{n,m}` past [`DUP_MAX`].
    BadMax,
}

impl Error {
    /// TRE's message for this code, which is what the client sees.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Error::BadPattern => "Invalid regexp",
            Error::Collate => "Unknown collating element",
            Error::CharClass => "Unknown character class name",
            Error::TrailingBackslash => "Trailing backslash",
            Error::BackRef => "Invalid back reference",
            Error::Unsupported => "regular expression backreferences are not supported",
            Error::MissingBracket => "Missing ']'",
            Error::MissingParen => "Missing ')'",
            Error::MissingBrace => "Missing '}'",
            Error::BadBrace => "Invalid contents of {}",
            Error::BadRange => "Invalid character range",
            Error::Space => "Out of memory",
            Error::BadRepeat => "Invalid use of repetition operators",
            Error::BadMax => "Maximum repetition in {} larger than 255",
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A set of bytes, as one bit each.
///
/// Every consuming step in the program is one of these, so a literal, a `.`, a
/// bracket expression and a `[:alpha:]` are all the same instruction and the
/// matcher has one case rather than four.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Class([u64; 4]);

impl Class {
    const fn empty() -> Class {
        Class([0; 4])
    }

    const fn all() -> Class {
        Class([u64::MAX; 4])
    }

    fn set(&mut self, b: u8) {
        self.0[(b >> 6) as usize] |= 1 << (b & 63);
    }

    fn clear(&mut self, b: u8) {
        self.0[(b >> 6) as usize] &= !(1 << (b & 63));
    }

    fn set_range(&mut self, lo: u8, hi: u8) {
        for b in lo..=hi {
            self.set(b);
        }
    }

    const fn has(self, b: u8) -> bool {
        self.0[(b >> 6) as usize] >> (b & 63) & 1 == 1
    }

    fn negate(&mut self) {
        for w in &mut self.0 {
            *w = !*w;
        }
    }

    /// Add the other case of every letter already in the set.
    ///
    /// Done once when the class is built rather than per byte at match time,
    /// and done to the positive set before a `[^...]` is negated, so `[^a]`
    /// under NOCASE refuses `A` as well.
    fn fold_case(&mut self) {
        for b in b'a'..=b'z' {
            if self.has(b) {
                self.set(b - 32);
            }
        }
        for b in b'A'..=b'Z' {
            if self.has(b) {
                self.set(b + 32);
            }
        }
    }
}

/// A zero width test, which the closure evaluates rather than the step.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Assert {
    /// `^`, the start of the subject. Not the start of a line: Redis does not
    /// pass `REG_NEWLINE`, so a newline in the subject is an ordinary byte.
    Start,
    /// `$`, the end of the subject.
    End,
    /// `\b` when true, `\B` when false.
    ///
    /// TRE calls both of these true at the start of the subject and at a byte
    /// it reads as the end, without looking at either side, so `\b` holds and
    /// `\B` fails there whatever the neighbours are.
    Word(bool),
    /// `\<`, the first byte of a word.
    StartOfWord,
    /// `\>`, one past the last byte of a word.
    EndOfWord,
}

/// One byte as a class, whatever NOCASE says.
///
/// TRE only folds a byte it read straight out of the pattern, because the fold
/// lives in the branch of the parser that handles a plain byte. A byte that an
/// escape produced never passes through there, so `\x41` under NOCASE is `A`
/// and not `A` or `a`.
fn raw(b: u8) -> Class {
    let mut c = Class::empty();
    c.set(b);
    c
}

const fn is_word(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

impl Assert {
    fn holds(self, hay: &[u8], at: usize) -> bool {
        let before = at > 0 && is_word(hay[at - 1]);
        let after = at < hay.len() && is_word(hay[at]);
        match self {
            Assert::Start => at == 0,
            Assert::End => at == hay.len(),
            // TRE decides these by reading the byte ahead and comparing it
            // against nul, so the end of the subject and a nul byte inside it
            // look the same, and the start is not looked at either way.
            Assert::Word(want) => {
                let edge = at == 0 || at == hay.len() || hay[at] == 0;
                (edge || before != after) == want
            }
            Assert::StartOfWord => !before && after,
            Assert::EndOfWord => before && !after,
        }
    }
}

/// One step of the compiled program.
#[derive(Clone, Copy, Debug)]
enum Inst {
    /// Consume one byte in `classes[i]` and carry on at the next instruction.
    Class(u32),
    /// Be in both places at once.
    Split(u32, u32),
    /// Carry on somewhere else.
    Jump(u32),
    /// Carry on at the next instruction if the test holds.
    Assert(Assert),
    /// The pattern matched.
    Match,
}

/// What the parser builds before anything is laid out.
///
/// A tree rather than instructions with holes to patch, because `{n,m}` is
/// compiled by emitting the same subtree several times and a tree is the thing
/// that can be walked twice.
enum Ast {
    Empty,
    Class(Class),
    Assert(Assert),
    Concat(Vec<Ast>),
    Alt(Vec<Ast>),
    Repeat(Box<Ast>, u32, Option<u32>),
}

/// A compiled pattern.
///
/// Compiling allocates and matching does not, which is the split `ARGREP` wants:
/// a pattern is compiled once when the command is parsed and then run against
/// every element the range touches.
#[derive(Debug)]
pub struct Regex {
    prog: Vec<Inst>,
    classes: Vec<Class>,
}

impl Regex {
    /// Compile `pattern` as an extended regular expression over bytes.
    ///
    /// `nocase` is `REG_ICASE`, and it is applied while the byte classes are
    /// built rather than while matching, so it costs nothing per element.
    ///
    /// # Errors
    ///
    /// Any of [`Error`], carrying TRE's own message for the same mistake.
    pub fn new(pattern: &[u8], nocase: bool) -> Result<Regex, Error> {
        let mut p = Parser {
            pat: pattern,
            at: 0,
            nocase,
            newline: false,
            depth: 0,
            groups: 0,
            backref: None,
        };
        let ast = p.alternation()?;
        if p.at != pattern.len() {
            // Nothing reaches this. A `)` with no `(` is a literal and every
            // other exit from `alternation` is the end of the pattern, so this
            // is a guard against a future change rather than a live path.
            return Err(Error::BadPattern);
        }
        // Both backreference answers wait for the whole parse, because TRE
        // decides them in regcomp after tre_parse has returned, so a mistake
        // anywhere else in the pattern is reported instead of these.
        if let Some(n) = p.backref {
            if n > p.groups {
                return Err(Error::BackRef);
            }
            return Err(Error::Unsupported);
        }
        let mut c = Compiler {
            prog: Vec::new(),
            classes: Vec::new(),
        };
        c.node(&ast)?;
        c.push(Inst::Match)?;
        Ok(Regex {
            prog: c.prog,
            classes: c.classes,
        })
    }

    /// How many instructions the pattern compiled to, for the matcher's scratch.
    #[must_use]
    fn len(&self) -> usize {
        self.prog.len()
    }
}

/// The scratch a match needs, kept across calls.
///
/// `ARGREP` runs a predicate per visited element, so the three vectors are
/// allocated once for the command rather than once per element. A `Matcher` can
/// be used with any [`Regex`]; it grows to the largest one it has seen.
#[derive(Default, Debug)]
pub struct Matcher {
    /// The states live at this position, and the ones live at the next.
    now: Vec<u32>,
    next: Vec<u32>,
    /// The stamp each state was last added under, so that adding a state
    /// twice in one step is a comparison rather than a search.
    seen: Vec<u64>,
    /// The stack the epsilon closure walks with, so it is not the call stack.
    work: Vec<u32>,
    stamp: u64,
}

impl Matcher {
    /// A matcher with nothing allocated yet.
    #[must_use]
    pub fn new() -> Matcher {
        Matcher::default()
    }

    /// Whether `re` matches anywhere in `hay`.
    ///
    /// Unanchored, which is what `regexec` without an anchor does and what
    /// `ARGREP RE` means: the pattern has to match some run of bytes, not the
    /// whole element.
    ///
    /// The walk is a Thompson simulation. Every state that could be live at a
    /// position is live at once, so a byte is looked at exactly once and the
    /// cost is the subject length times the program size in the worst case,
    /// with no backtracking and therefore no pattern that makes it exponential.
    pub fn is_match(&mut self, re: &Regex, hay: &[u8]) -> bool {
        let n = re.len();
        if self.seen.len() < n {
            self.seen.resize(n, 0);
        }
        self.now.clear();
        self.next.clear();
        self.stamp += 1;
        let mut stamp = self.stamp;

        for at in 0..=hay.len() {
            // Start a fresh attempt at every position, which is what makes the
            // search unanchored without a `.*` in front of the program.
            if add(
                &mut self.now,
                &mut self.seen,
                &mut self.work,
                stamp,
                re,
                hay,
                at,
                0,
            ) {
                return true;
            }
            if at == hay.len() {
                break;
            }
            let byte = hay[at];
            self.stamp += 1;
            stamp = self.stamp;
            self.next.clear();
            for i in 0..self.now.len() {
                let pc = self.now[i];
                if let Inst::Class(c) = re.prog[pc as usize]
                    && re.classes[c as usize].has(byte)
                    && add(
                        &mut self.next,
                        &mut self.seen,
                        &mut self.work,
                        stamp,
                        re,
                        hay,
                        at + 1,
                        pc + 1,
                    )
                {
                    return true;
                }
            }
            core::mem::swap(&mut self.now, &mut self.next);
        }
        false
    }
}

/// Add `pc` and everything reachable from it without consuming a byte.
///
/// Returns whether one of them was `Match`, which is the only answer `is_match`
/// needs, so the walk stops the moment it is true rather than finishing the
/// subject.
#[allow(clippy::too_many_arguments)]
fn add(
    list: &mut Vec<u32>,
    seen: &mut [u64],
    work: &mut Vec<u32>,
    stamp: u64,
    re: &Regex,
    hay: &[u8],
    at: usize,
    pc: u32,
) -> bool {
    work.clear();
    work.push(pc);
    while let Some(pc) = work.pop() {
        let i = pc as usize;
        if seen[i] == stamp {
            continue;
        }
        seen[i] = stamp;
        match re.prog[i] {
            Inst::Class(_) => list.push(pc),
            Inst::Split(a, b) => {
                work.push(b);
                work.push(a);
            }
            Inst::Jump(a) => work.push(a),
            Inst::Assert(a) => {
                if a.holds(hay, at) {
                    work.push(pc + 1);
                }
            }
            Inst::Match => return true,
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

struct Parser<'a> {
    pat: &'a [u8],
    at: usize,
    nocase: bool,
    /// Whether `.` skips a newline, which only `(?n)` can turn on because
    /// Redis never passes `REG_NEWLINE` itself.
    newline: bool,
    depth: u32,
    /// How many groups the pattern has opened so far.
    groups: u32,
    /// The highest backreference seen, if any.
    ///
    /// Whether one is valid is not known until the whole pattern has been read,
    /// because a reference may point forwards: `\1(a)` is a reference to a
    /// group that has not been written down yet and TRE takes it.
    backref: Option<u32>,
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.pat.get(self.at).copied()
    }

    fn eat(&mut self, b: u8) -> bool {
        if self.peek() == Some(b) {
            self.at += 1;
            return true;
        }
        false
    }

    /// `branch ('|' branch)*`
    fn alternation(&mut self) -> Result<Ast, Error> {
        let mut arms = vec![self.branch()?];
        while self.eat(b'|') {
            arms.push(self.branch()?);
        }
        if arms.len() == 1 {
            return Ok(arms.pop().expect("one arm"));
        }
        Ok(Ast::Alt(arms))
    }

    /// A run of repeated atoms, up to a `|` or a closing `)` or the end.
    ///
    /// An empty branch is allowed, so `a|` matches `a` or nothing and `()` is a
    /// group that matches nothing, both of which TRE accepts.
    fn branch(&mut self) -> Result<Ast, Error> {
        let mut parts: Vec<Ast> = Vec::new();
        loop {
            match self.peek() {
                None | Some(b'|') => break,
                // A `)` only ends the branch when a `(` is waiting for it.
                // Outside a group it is an ordinary byte, which is why `a)b`
                // matches the three bytes it looks like rather than failing.
                Some(b')') if self.depth > 0 => break,
                _ => {}
            }
            // A repetition operator with nothing in front of it repeats an atom
            // that matches the empty string, which is what TRE's parser hands
            // back when it is asked for an atom and finds an operator. So `*a`
            // is `a`, and `{2}a` is `a` rather than `aa`.
            let atom = match self.peek() {
                Some(b'*' | b'+' | b'?' | b'{') => Ast::Empty,
                _ => self.atom()?,
            };
            parts.push(self.repeats(atom)?);
        }
        match parts.len() {
            0 => Ok(Ast::Empty),
            1 => Ok(parts.pop().expect("one part")),
            _ => Ok(Ast::Concat(parts)),
        }
    }

    /// Every repetition operator that follows an atom, applied outwards.
    ///
    /// A second operator straight after the first is reserved in TRE and
    /// refused, so `a**` and `a{2}+` are both errors, with one exception: a `?`
    /// asks for the shortest match rather than the longest. That changes where
    /// a match ends and not whether there is one, so for a yes or no answer it
    /// is read and dropped.
    fn repeats(&mut self, mut node: Ast) -> Result<Ast, Error> {
        loop {
            let (min, max) = match self.peek() {
                Some(b'*') => {
                    self.at += 1;
                    (0, None)
                }
                Some(b'+') => {
                    self.at += 1;
                    (1, None)
                }
                Some(b'?') => {
                    self.at += 1;
                    (0, Some(1))
                }
                Some(b'{') => {
                    self.at += 1;
                    self.bound()?
                }
                _ => return Ok(node),
            };
            match self.peek() {
                Some(b'?') => self.at += 1,
                Some(b'*' | b'+') => return Err(Error::BadRepeat),
                _ => {}
            }
            node = Ast::Repeat(Box::new(node), min, max);
        }
    }

    /// The inside of a bound, with the `{` already eaten.
    ///
    /// A count is optional on both sides of the comma, so `{,3}` is `{0,3}` and
    /// `{2,}` has no ceiling, and a missing count on both sides with no comma
    /// at all is the empty `{}`, which is an error. The three ways this can go
    /// wrong are told apart by where the parse stopped: off the end of the
    /// pattern is a missing brace, stopped without having read anything is an
    /// empty bound, and stopped on something that is not a brace is a bound
    /// with rubbish in it.
    fn bound(&mut self) -> Result<(u32, Option<u32>), Error> {
        let start = self.at;
        let mut min: i64 = self.number().map_or(-1, i64::from);
        let mut max = min;
        if self.eat(b',') {
            if min < 0 {
                min = 0;
            }
            max = self.number().map_or(-1, i64::from);
        }
        // Both of these are decided before the brace is looked for, so `{3,2`
        // is a backwards bound rather than a missing brace.
        if max >= 0 && min > max {
            return Err(Error::BadBrace);
        }
        if min > i64::from(DUP_MAX) || max > i64::from(DUP_MAX) {
            return Err(Error::BadMax);
        }
        // TRE walks past spaces and further commas on its way to the brace, so
        // `{2, }` and `{2,,}` are the same bound as `{2,}`.
        while matches!(self.peek(), Some(b' ' | b',')) {
            self.at += 1;
        }
        if self.at >= self.pat.len() {
            return Err(Error::MissingBrace);
        }
        if self.at == start {
            return Err(Error::BadBrace);
        }
        if !self.eat(b'}') {
            return Err(Error::BadBrace);
        }
        if min < 0 {
            // No count on either side and no comma, which after the parameters
            // TRE allows here would be `{~2}` and friends. Those repeat once.
            min = 1;
            max = 1;
        }
        Ok((min as u32, if max < 0 { None } else { Some(max as u32) }))
    }

    fn number(&mut self) -> Option<u32> {
        let start = self.at;
        let mut n: u32 = 0;
        while let Some(b) = self.peek() {
            if !b.is_ascii_digit() {
                break;
            }
            // A bound past DUP_MAX is refused anyway, so saturating here keeps
            // a long run of digits from wrapping into a small number.
            n = n.saturating_mul(10).saturating_add(u32::from(b - b'0'));
            self.at += 1;
        }
        if self.at == start { None } else { Some(n) }
    }

    fn atom(&mut self) -> Result<Ast, Error> {
        let b = self.peek().ok_or(Error::BadPattern)?;
        match b {
            b'(' => {
                self.at += 1;
                if self.eat(b'?') {
                    return self.extension();
                }
                self.groups += 1;
                self.group()
            }
            b'.' => {
                self.at += 1;
                let mut c = Class::all();
                if self.newline {
                    c.clear(b'\n');
                }
                Ok(Ast::Class(c))
            }
            b'^' => {
                self.at += 1;
                Ok(Ast::Assert(Assert::Start))
            }
            b'$' => {
                self.at += 1;
                Ok(Ast::Assert(Assert::End))
            }
            b'[' => {
                self.at += 1;
                self.bracket()
            }
            b'\\' => {
                self.at += 1;
                self.escape()
            }
            // Everything else is the byte, including a `)` with no `(` and a
            // `]` with no `[`.
            _ => {
                self.at += 1;
                Ok(Ast::Class(self.literal(b)))
            }
        }
    }

    /// Everything up to the `)` that closes a group whose `(` is already eaten.
    fn group(&mut self) -> Result<Ast, Error> {
        self.depth += 1;
        if self.depth > DEPTH_MAX {
            return Err(Error::Space);
        }
        let inner = self.alternation()?;
        self.depth -= 1;
        if !self.eat(b')') {
            return Err(Error::MissingParen);
        }
        Ok(inner)
    }

    /// TRE's `(?...)` extensions, with the `(?` already eaten.
    ///
    /// The letters turn compile flags on, and a `-` turns the rest of them off
    /// again. A `:` then opens a group that does not capture, while a `)` or a
    /// `#` comment ends the extension and leaves the flags on for the rest of
    /// whatever encloses it. That last part is worth reading twice: TRE parses
    /// the remainder as a whole expression under the new flags, so `a(?i)b|c`
    /// is `a` followed by `b|c` rather than `ab` or `c`. Anything else is
    /// "Invalid regexp", which is why `(?x)` is refused.
    fn extension(&mut self) -> Result<Ast, Error> {
        let (nocase, newline) = (self.nocase, self.newline);
        let mut on = true;
        let opens = loop {
            match self.peek().ok_or(Error::BadPattern)? {
                b'i' => self.nocase = on,
                b'n' => self.newline = on,
                // Right associativity and ungreedy decide which match is
                // reported rather than whether there is one, and Redis asks
                // only whether there is one, so these are read and dropped.
                b'r' | b'U' => {}
                b'-' => on = false,
                b':' => {
                    self.at += 1;
                    break true;
                }
                b'#' => {
                    // A comment is every byte up to the first `)`.
                    while self.peek().is_some_and(|b| b != b')') {
                        self.at += 1;
                    }
                    if !self.eat(b')') {
                        return Err(Error::BadPattern);
                    }
                    break false;
                }
                b')' => {
                    self.at += 1;
                    break false;
                }
                _ => return Err(Error::BadPattern),
            }
            self.at += 1;
        };
        let inner = if opens {
            self.group()?
        } else {
            self.alternation()?
        };
        self.nocase = nocase;
        self.newline = newline;
        Ok(inner)
    }

    /// One byte as a class, case folded if `nocase`.
    fn literal(&self, b: u8) -> Class {
        let mut c = raw(b);
        if self.nocase {
            c.fold_case();
        }
        c
    }

    /// What follows a backslash outside a bracket expression.
    ///
    /// TRE looks at this in two rounds and the order is what decides several
    /// of the answers. First it checks a table of macros, which are letters
    /// that stand for a short pattern and get parsed as if the client had
    /// written that pattern out. Only if the letter is not a macro does it
    /// reach the switch that has the word boundaries, `\x` and backreferences
    /// in it. That is why `\d` is the digit class rather than backreference
    /// number thirteen, and it is why `\n` is a newline rather than the letter.
    fn escape(&mut self) -> Result<Ast, Error> {
        let b = self.peek().ok_or(Error::TrailingBackslash)?;
        if let Some(node) = self.macro_for(b) {
            self.at += 1;
            return Ok(node);
        }
        self.at += 1;
        match b {
            b'b' => Ok(Ast::Assert(Assert::Word(true))),
            b'B' => Ok(Ast::Assert(Assert::Word(false))),
            b'<' => Ok(Ast::Assert(Assert::StartOfWord)),
            b'>' => Ok(Ast::Assert(Assert::EndOfWord)),
            b'x' => self.hex(),
            // A digit is a backreference, which needs the engine to remember
            // what a group matched and a set of live states cannot. Note it and
            // carry on: whether it is refused, and with which of two sentences,
            // is decided once the pattern has been read to the end.
            b'0'..=b'9' => {
                let n = u32::from(b - b'0');
                self.backref = Some(self.backref.map_or(n, |m| m.max(n)));
                Ok(Ast::Empty)
            }
            // Anything else is the byte itself, so `\.` is a dot and `\\` is a
            // backslash. Not folded: see `raw`.
            _ => Ok(Ast::Class(raw(b))),
        }
    }

    /// TRE's macro table, from `tre_macros` in `deps/tre/lib/tre-parse.c`.
    ///
    /// Six of them are a control byte by another name and six are a character
    /// class. TRE writes them out as source and parses that, so `\w` is
    /// `[[:alnum:]_]` down to how a negation and NOCASE interact; building the
    /// same byte set directly gets to the same place without a second parse.
    fn macro_for(&self, b: u8) -> Option<Ast> {
        let byte = |v: u8| Some(Ast::Class(raw(v)));
        let class = |keep: fn(u8) -> bool, negate: bool| {
            let mut c = Class::empty();
            for x in 0..=255u8 {
                if keep(x) {
                    c.set(x);
                }
            }
            if negate {
                c.negate();
            }
            Some(Ast::Class(c))
        };
        // A space in TRE's `[[:space:]]` is the C locale's, which includes the
        // vertical tab, so this is the same set the named class builds.
        let space = |x: u8| x.is_ascii_whitespace() || x == 0x0b;
        let word = |x: u8| x.is_ascii_alphanumeric() || x == b'_';
        match b {
            b't' => byte(b'\t'),
            b'n' => byte(b'\n'),
            b'r' => byte(b'\r'),
            b'f' => byte(0x0c),
            b'a' => byte(0x07),
            b'e' => byte(0x1b),
            b'w' => class(word, false),
            b'W' => class(word, true),
            b's' => class(space, false),
            b'S' => class(space, true),
            b'd' => class(|x| x.is_ascii_digit(), false),
            b'D' => class(|x| x.is_ascii_digit(), true),
            _ => None,
        }
    }

    /// `\xNN` or `\x{NNNN}`, with the `x` already eaten.
    ///
    /// The braced form is a whole code point in TRE and this engine is bytes,
    /// so anything past 255 is a class with nothing in it, which is what a byte
    /// build of TRE ends up matching against: nothing. A bare `\x` at the end
    /// of the pattern is a NUL, which is TRE's `tre_ast_new_literal(mem, 0, 0)`
    /// rather than an error.
    fn hex(&mut self) -> Result<Ast, Error> {
        let one = |v: u32| {
            let mut c = Class::empty();
            if v <= 255 {
                c.set(v as u8);
            }
            c
        };
        if !self.eat(b'{') {
            let mut v: u32 = 0;
            for _ in 0..2 {
                match self.peek().and_then(|b| (b as char).to_digit(16)) {
                    Some(d) => {
                        v = v * 16 + d;
                        self.at += 1;
                    }
                    None => break,
                }
            }
            return Ok(Ast::Class(one(v)));
        }
        // TRE reads at most eight hex digits and anything that is not one, and
        // is not the closing brace, ends the pattern rather than the number.
        let mut v: u32 = 0;
        let mut digits = 0;
        loop {
            match self.peek() {
                Some(b'}') => {
                    self.at += 1;
                    return Ok(Ast::Class(one(v)));
                }
                Some(b) => match (b as char).to_digit(16) {
                    Some(d) if digits < 8 => {
                        v = v * 16 + d;
                        digits += 1;
                        self.at += 1;
                    }
                    // Past eight digits TRE stops storing but keeps reading,
                    // so the value is the first eight and the rest is skipped.
                    Some(_) => self.at += 1,
                    None => return Err(Error::MissingBrace),
                },
                None => return Err(Error::MissingBrace),
            }
        }
    }

    /// A bracket expression, with the `[` already eaten.
    ///
    /// The rules are all about position rather than about escaping: a `]` first
    /// is a literal, a `^` first negates, a `-` first or last is a literal, and
    /// a backslash is a backslash rather than an escape. That last one is what
    /// catches people, and it is what TRE does.
    ///
    /// The order the three item shapes are tried in matters, because a range is
    /// looked for before anything else. That is why `[]-a]` is the range from
    /// `]` to `a` and not an empty expression, and why `[a-[:x:]]` is a range
    /// whose ends are the wrong way round rather than a class.
    fn bracket(&mut self) -> Result<Ast, Error> {
        let mut class = Class::empty();
        let negate = self.eat(b'^');
        let first = self.at;
        loop {
            let b = self.peek().ok_or(Error::MissingBracket)?;
            if b == b']' && self.at > first {
                self.at += 1;
                break;
            }
            // A range needs a `-` and something after it that is not the `]`
            // closing the expression, so `[a-]` is an a and a dash.
            let dash = self.pat.get(self.at + 1) == Some(&b'-');
            let hi = self.pat.get(self.at + 2).copied();
            if dash && hi.is_some_and(|h| h != b']') {
                let hi = hi.expect("checked");
                if b > hi {
                    return Err(Error::BadRange);
                }
                class.set_range(b, hi);
                self.at += 3;
                continue;
            }
            if b == b'[' {
                match self.pat.get(self.at + 1) {
                    Some(b'.') | Some(b'=') => return Err(Error::Collate),
                    Some(b':') => {
                        self.named_class(&mut class)?;
                        continue;
                    }
                    _ => {}
                }
            }
            // A dash that is not the first item and could have opened a range
            // has already had its left end taken by the range before it, and
            // TRE's own comment for this is that two ranges are not allowed to
            // share an endpoint. So `[a-c-e]` is refused while `[a-c-]` is not.
            if b == b'-'
                && self.at != first
                && self.pat.get(self.at + 1).is_some_and(|&n| n != b']')
            {
                return Err(Error::BadRange);
            }
            class.set(b);
            self.at += 1;
        }
        if self.nocase {
            class.fold_case();
        }
        if negate {
            class.negate();
        }
        Ok(Ast::Class(class))
    }

    /// `[:name:]` inside a bracket expression, with the `[` at the cursor.
    fn named_class(&mut self, class: &mut Class) -> Result<(), Error> {
        let start = self.at + 2;
        let mut end = start;
        while end < self.pat.len() && self.pat[end] != b':' {
            end += 1;
        }
        if end + 1 >= self.pat.len() || self.pat[end + 1] != b']' {
            return Err(Error::CharClass);
        }
        let name = &self.pat[start..end];
        let keep: fn(u8) -> bool = match name {
            b"alnum" => |b| b.is_ascii_alphanumeric(),
            b"alpha" => |b| b.is_ascii_alphabetic(),
            // No `blank`. TRE has eleven classes and that is not one of them,
            // so `[[:blank:]]` is an unknown class name rather than a space and
            // a tab.
            b"cntrl" => |b| b.is_ascii_control(),
            b"digit" => |b| b.is_ascii_digit(),
            b"graph" => |b| b.is_ascii_graphic(),
            b"lower" => |b| b.is_ascii_lowercase(),
            b"print" => |b| b.is_ascii_graphic() || b == b' ',
            b"punct" => |b| b.is_ascii_punctuation(),
            b"space" => |b| b.is_ascii_whitespace() || b == 0x0b,
            b"upper" => |b| b.is_ascii_uppercase(),
            b"xdigit" => |b| b.is_ascii_hexdigit(),
            _ => return Err(Error::CharClass),
        };
        for b in 0..=255u8 {
            if keep(b) {
                class.set(b);
            }
        }
        self.at = end + 2;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Compiling
// ---------------------------------------------------------------------------

struct Compiler {
    prog: Vec<Inst>,
    classes: Vec<Class>,
}

/// Whether the node has a path through it that consumes no bytes.
fn matches_empty(ast: &Ast) -> bool {
    match ast {
        Ast::Empty | Ast::Assert(_) => true,
        Ast::Class(_) => false,
        Ast::Concat(parts) => parts.iter().all(matches_empty),
        Ast::Alt(arms) => arms.iter().any(matches_empty),
        Ast::Repeat(inner, min, _) => *min == 0 || matches_empty(inner),
    }
}

impl Compiler {
    fn push(&mut self, i: Inst) -> Result<u32, Error> {
        if self.prog.len() >= PROG_MAX {
            return Err(Error::Space);
        }
        self.prog.push(i);
        Ok(self.prog.len() as u32 - 1)
    }

    fn here(&self) -> u32 {
        self.prog.len() as u32
    }

    fn class(&mut self, c: Class) -> Result<(), Error> {
        // The same class twice is one entry, which matters for `{n,m}` because
        // the expansion emits the same subtree over and over.
        let idx = match self.classes.iter().position(|&e| e == c) {
            Some(i) => i as u32,
            None => {
                self.classes.push(c);
                self.classes.len() as u32 - 1
            }
        };
        self.push(Inst::Class(idx))?;
        Ok(())
    }

    fn node(&mut self, ast: &Ast) -> Result<(), Error> {
        match ast {
            Ast::Empty => Ok(()),
            Ast::Class(c) => self.class(*c),
            Ast::Assert(a) => {
                self.push(Inst::Assert(*a))?;
                Ok(())
            }
            Ast::Concat(parts) => {
                for p in parts {
                    self.node(p)?;
                }
                Ok(())
            }
            Ast::Alt(arms) => self.alt(arms),
            Ast::Repeat(inner, min, max) => self.repeat(inner, *min, *max),
        }
    }

    /// `a|b|c` as a chain of splits, each arm jumping to the same place after.
    fn alt(&mut self, arms: &[Ast]) -> Result<(), Error> {
        let mut ends = Vec::with_capacity(arms.len());
        for (i, arm) in arms.iter().enumerate() {
            if i + 1 == arms.len() {
                self.node(arm)?;
                break;
            }
            let split = self.push(Inst::Split(0, 0))?;
            let first = self.here();
            self.node(arm)?;
            ends.push(self.push(Inst::Jump(0))?);
            let second = self.here();
            self.prog[split as usize] = Inst::Split(first, second);
        }
        let after = self.here();
        for j in ends {
            self.prog[j as usize] = Inst::Jump(after);
        }
        Ok(())
    }

    /// A repetition, by emitting the subtree as many times as it can run.
    ///
    /// `a{2,4}` becomes `aa a? a?`, which is the same language: concatenation is
    /// contiguous, so a skipped optional cannot be made up by a later one.
    /// `a{2,}` becomes `aa a*`. This is why `PROG_MAX` exists.
    fn repeat(&mut self, inner: &Ast, min: u32, max: Option<u32>) -> Result<(), Error> {
        // A body that can match the empty string always runs at least once in
        // TRE, because its assertions sit on the transitions the skip would go
        // through. For a body with no assertions that changes nothing, since
        // the extra pass can match nothing. For one with them it is the whole
        // difference: `^*a` does not match "*a" and `x\b?y` does not match
        // "xy". The one exception is `{0}`, which TRE turns into an explicit
        // empty node that throws the operand away, and which lands in the
        // `Some(0)` arm below having emitted nothing at all.
        let min = if min == 0 && max != Some(0) && matches_empty(inner) {
            1
        } else {
            min
        };
        for _ in 0..min {
            self.node(inner)?;
        }
        match max {
            None => {
                // `x*`: split forwards or into the body, and loop back.
                let split = self.push(Inst::Split(0, 0))?;
                let body = self.here();
                self.node(inner)?;
                self.push(Inst::Jump(split))?;
                let after = self.here();
                self.prog[split as usize] = Inst::Split(body, after);
                Ok(())
            }
            Some(max) => {
                let mut splits = Vec::new();
                for _ in min..max {
                    let split = self.push(Inst::Split(0, 0))?;
                    let body = self.here();
                    self.prog[split as usize] = Inst::Split(body, 0);
                    splits.push(split);
                    self.node(inner)?;
                }
                // Every optional copy skips to the same place, which is the end
                // of the last one.
                let after = self.here();
                for s in splits {
                    let Inst::Split(body, _) = self.prog[s as usize] else {
                        unreachable!("only splits were recorded")
                    };
                    self.prog[s as usize] = Inst::Split(body, after);
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hits(pattern: &str, subject: &str) -> bool {
        let re = Regex::new(pattern.as_bytes(), false).expect("compiles");
        Matcher::new().is_match(&re, subject.as_bytes())
    }

    fn hits_nocase(pattern: &str, subject: &str) -> bool {
        let re = Regex::new(pattern.as_bytes(), true).expect("compiles");
        Matcher::new().is_match(&re, subject.as_bytes())
    }

    fn refuses(pattern: &str) -> Error {
        Regex::new(pattern.as_bytes(), false).expect_err("refused")
    }

    #[test]
    fn a_literal_matches_anywhere_in_the_subject() {
        assert!(hits("abc", "abc"));
        assert!(hits("abc", "xxabcxx"));
        assert!(!hits("abc", "ab"));
        assert!(hits("", ""));
        assert!(hits("", "anything"));
    }

    #[test]
    fn the_anchors_are_the_ends_of_the_subject_and_not_of_a_line() {
        assert!(hits("^abc", "abcdef"));
        assert!(!hits("^abc", "xabcdef"));
        assert!(hits("abc$", "xxabc"));
        assert!(!hits("abc$", "abcx"));
        assert!(hits("^abc$", "abc"));
        assert!(!hits("^abc$", "abc\n"));
        // No REG_NEWLINE, so a newline is an ordinary byte and `^` does not
        // start again after it.
        assert!(!hits("^b", "a\nb"));
        assert!(hits(".", "\n"));
    }

    #[test]
    fn the_three_unbounded_repetitions_do_what_they_say() {
        assert!(hits("^ab*c$", "ac"));
        assert!(hits("^ab*c$", "abbbbc"));
        assert!(!hits("^ab+c$", "ac"));
        assert!(hits("^ab+c$", "abc"));
        assert!(hits("^ab?c$", "ac"));
        assert!(hits("^ab?c$", "abc"));
        assert!(!hits("^ab?c$", "abbc"));
        // A repetition that can match nothing, which is the shape that loops
        // forever in an engine that does not mark states as already added.
        assert!(hits("^(a*)*$", ""));
        assert!(hits("^(a*)*$", "aaa"));
    }

    #[test]
    fn a_bound_counts_and_refuses_what_it_cannot_count() {
        assert!(hits("^a{3}$", "aaa"));
        assert!(!hits("^a{3}$", "aa"));
        assert!(!hits("^a{3}$", "aaaa"));
        assert!(hits("^a{2,}$", "aaaaa"));
        assert!(!hits("^a{2,}$", "a"));
        assert!(hits("^a{2,4}$", "aa"));
        assert!(hits("^a{2,4}$", "aaaa"));
        assert!(!hits("^a{2,4}$", "aaaaa"));
        assert!(hits("^a{0,2}$", ""));
        assert_eq!(refuses("a{3,2}"), Error::BadBrace);
        assert_eq!(refuses("a{256}"), Error::BadMax);
        assert_eq!(refuses("a{2"), Error::MissingBrace);
        assert_eq!(refuses("a{x}"), Error::BadBrace);
        // In atom position a brace is just a brace.
        assert!(hits("^[{]a$", "{a"));
    }

    #[test]
    fn alternation_tries_every_arm_and_an_empty_arm_is_an_arm() {
        assert!(hits("^(cat|dog|bird)$", "dog"));
        assert!(!hits("^(cat|dog|bird)$", "cow"));
        assert!(hits("^(ab|a)b$", "ab"));
        assert!(hits("^(a|)$", ""));
        assert!(hits("^()$", ""));
        assert!(hits("^a(b|c)*d$", "abcbcd"));
    }

    #[test]
    fn a_bracket_expression_follows_position_rather_than_escaping() {
        assert!(hits("^[abc]$", "b"));
        assert!(!hits("^[abc]$", "d"));
        assert!(hits("^[^abc]$", "d"));
        assert!(!hits("^[^abc]$", "a"));
        assert!(hits("^[a-z]+$", "hello"));
        assert!(!hits("^[a-z]+$", "Hello"));
        // A `]` first is a literal, a `-` first or last is a literal.
        assert!(hits("^[]a]$", "]"));
        assert!(hits("^[-a]$", "-"));
        assert!(hits("^[a-]$", "-"));
        assert!(hits("^[^]]$", "x"));
        // A backslash inside brackets is a backslash, which is the one that
        // catches people coming from a Perl style engine.
        assert!(hits("^[\\]$", "\\"));
        assert_eq!(refuses("[abc"), Error::MissingBracket);
        assert_eq!(refuses("[z-a]"), Error::BadRange);
        assert_eq!(refuses("[[.a.]]"), Error::Collate);
        assert_eq!(refuses("[[=a=]]"), Error::Collate);
    }

    #[test]
    fn the_named_classes_are_the_posix_ones() {
        assert!(hits("^[[:digit:]]+$", "12345"));
        assert!(!hits("^[[:digit:]]+$", "12a45"));
        assert!(hits("^[[:alpha:][:digit:]]+$", "ab12"));
        assert!(hits("^[[:space:]]$", "\t"));
        assert!(hits("^[^[:alpha:]]$", "1"));
        assert!(hits("^[[:xdigit:]]+$", "deadBEEF01"));
        assert_eq!(refuses("[[:nosuch:]]"), Error::CharClass);
    }

    #[test]
    fn the_escapes_are_tres_and_a_macro_beats_the_switch() {
        assert!(hits("^a\\.c$", "a.c"));
        assert!(!hits("^a\\.c$", "abc"));
        assert!(hits("^a\\*$", "a*"));
        assert!(hits("^\\x41$", "A"));
        assert!(hits("^\\x{41}$", "A"));
        // Past a byte there is no byte to match, so the class is empty.
        assert!(!hits("\\x{100}", "\u{100}"));
        assert!(!hits("\\x{100}", "\0"));
        // The six control macros.
        assert!(hits("^\\n$", "\n"));
        assert!(!hits("^\\n$", "n"));
        assert!(hits("^\\t\\r\\f\\a\\e$", "\t\r\x0c\x07\x1b"));
        // The six class macros, which the table reaches before the switch does,
        // so `\d` is a digit rather than backreference thirteen.
        assert!(hits("^\\d+$", "42"));
        assert!(!hits("^\\d+$", "4a"));
        assert!(hits("^\\D$", "a"));
        assert!(hits("^\\w+$", "a_1"));
        assert!(!hits("^\\w+$", "a-1"));
        assert!(hits("^\\W$", "-"));
        assert!(hits("^\\s+$", " \t\n"));
        assert!(hits("^\\S$", "x"));
        // And a macro is one atom, so a repetition applies to the whole class.
        assert!(hits("^\\d{3}$", "123"));
        assert_eq!(refuses("a\\"), Error::TrailingBackslash);
    }

    #[test]
    fn a_backreference_is_refused_with_one_of_two_sentences() {
        // Pointing at a group that is not there is TRE's own error. Pointing at
        // one that is there compiles in TRE and is refused a step later, and
        // that step is the one whose sentence this carries.
        assert_eq!(refuses("(a)\\1"), Error::Unsupported);
        assert_eq!(refuses("(a)(b)\\2"), Error::Unsupported);
        assert_eq!(refuses("\\0"), Error::Unsupported);
        assert_eq!(refuses("\\1"), Error::BackRef);
        assert_eq!(refuses("(a)\\2"), Error::BackRef);
        // A reference may point forwards, so this one is valid and therefore
        // gets the second sentence rather than the first.
        assert_eq!(refuses("\\1(a)"), Error::Unsupported);
        // Both answers wait for the whole parse, so a mistake anywhere else in
        // the pattern is reported instead.
        assert_eq!(refuses("((a)\\9"), Error::MissingParen);
        assert_eq!(refuses("\\1a{256}"), Error::BadMax);
        assert_eq!(
            Error::Unsupported.as_str(),
            "regular expression backreferences are not supported"
        );
    }

    #[test]
    fn a_backslash_in_a_bracket_expression_is_not_an_escape() {
        // The macros are expanded in atom position only, so inside brackets a
        // backslash is a backslash and `\d` is the two bytes it looks like.
        assert!(hits("^[\\d]+$", "\\d"));
        assert!(!hits("^[\\d]+$", "42"));
        assert!(hits("^[\\n]+$", "\\n"));
        assert!(!hits("^[\\n]$", "\n"));
    }

    #[test]
    fn the_word_boundaries_look_at_both_sides() {
        assert!(hits("\\bcat\\b", "the cat sat"));
        assert!(!hits("\\bcat\\b", "concatenate"));
        assert!(hits("\\Bcat\\B", "concatenate"));
        assert!(!hits("\\Bcat\\B", "the cat sat"));
        assert!(hits("\\<cat", "a cat"));
        assert!(!hits("\\<cat", "concat"));
        assert!(hits("cat\\>", "concat"));
        assert!(!hits("cat\\>", "cats"));
        // The two ends of the subject are a boundary to TRE whatever sits next
        // to them, so `\b` holds there and `\B` fails there even between two
        // bytes that are not word bytes at all.
        assert!(hits("\\b-", "-a"));
        assert!(!hits("\\B-", "-a"));
        assert!(hits("-\\b", "a-"));
        assert!(!hits("-\\B", "a-"));
        assert!(!hits("-\\b-", "---"));
        assert!(hits("-\\B-", "---"));
        // A nul byte reads as the end for the same reason, because TRE decides
        // by comparing the byte ahead against nul rather than by counting.
        assert!(hits("\0\\b\0", "\0\0\0"));
        assert!(!hits("\0\\B\0", "\0\0\0"));
        // `\<` and `\>` are the ordinary ones and do look at both sides.
        assert!(hits("\\>", "a"));
        assert!(!hits("\\>", "-"));
        assert!(hits("\\<", "a"));
    }

    #[test]
    fn nocase_folds_the_class_and_not_the_subject() {
        assert!(hits_nocase("^abc$", "ABC"));
        assert!(hits_nocase("^[a-c]+$", "ABC"));
        assert!(hits_nocase("^[A-C]+$", "abc"));
        // The fold happens before the negation, so a negated class refuses
        // both cases rather than refusing one and taking the other.
        assert!(!hits_nocase("^[^a]$", "A"));
        assert!(hits_nocase("^[^a]$", "b"));
        assert!(!hits_nocase("^abc$", "abd"));
        // A byte an escape produced is not folded, because the fold sits in
        // the branch of TRE's parser that reads a plain byte and an escape
        // never goes through it. This is the one place the register calls out,
        // because Redis's literal fast path does fold it.
        assert!(!hits_nocase("\\x41.", "ab"));
        assert!(hits_nocase("\\x41.", "Ab"));
        assert!(!hits_nocase("\\x{41}.", "ab"));
    }

    #[test]
    fn the_inline_flags_are_read_and_scoped_the_way_tre_scopes_them() {
        assert!(hits("(?i)abc", "ABC"));
        assert!(hits("(?i)ABC", "abc"));
        assert!(hits("(?i:a)b", "Ab"));
        assert!(!hits("(?i:a)b", "AB"));
        assert!(hits_nocase("(?-i)A", "A"));
        assert!(!hits_nocase("(?-i)A", "a"));
        // The rest of the enclosing expression is parsed under the new flags
        // as a whole, so this is `a` followed by `b|c` rather than `ab` or `c`.
        assert!(hits("a(?i)b|c", "ac"));
        assert!(!hits("a(?i)b|c", "c"));
        // `(?n)` is the only flag with a second effect, and only on the dot,
        // because the matcher reads the outer flags when it decides an anchor.
        assert!(!hits("(?n).", "\n"));
        assert!(hits("(?n).", "a"));
        assert!(!hits("(?n)^b", "a\nb"));
        // A comment is dropped and the letters that only pick between matches
        // are read and ignored.
        assert!(hits("(?#a comment)abc", "abc"));
        assert!(hits("(?U)a", "a"));
        assert!(hits("(?r)a", "a"));
        // Anything else is refused, including a truncated one.
        assert_eq!(refuses("(?x)a"), Error::BadPattern);
        assert_eq!(refuses("(?"), Error::BadPattern);
        assert_eq!(refuses("(?ia)"), Error::BadPattern);
        assert_eq!(refuses("(?#unterminated"), Error::BadPattern);
    }

    #[test]
    fn an_operator_with_nothing_in_front_of_it_repeats_nothing() {
        assert_eq!(refuses("(abc"), Error::MissingParen);
        // A `)` with no `(` is a byte, so this is a pattern about three bytes
        // rather than a mistake.
        assert!(hits("^a)b$", "a)b"));
        assert!(!hits("^a)b$", "ab"));
        assert!(hits("^(a)b$", "ab"));
        // A leading operator repeats an atom that matches the empty string.
        assert!(hits("^*a$", "a"));
        assert!(!hits("^*a$", "*a"));
        assert!(hits("^+a$", "a"));
        assert!(hits("^?a$", "a"));
        assert!(hits("^{2}a$", "a"));
        assert!(!hits("^{2}a$", "aa"));
        assert!(hits("^{,3}a$", "a"));
        // A `{` in that position is still a bound and still has to parse.
        assert_eq!(refuses("^{$"), Error::BadBrace);
        assert_eq!(refuses("{256}"), Error::BadMax);
        assert_eq!(refuses("{3,2}"), Error::BadBrace);
    }

    #[test]
    fn a_repetition_of_something_that_matches_nothing_still_runs_once() {
        // The body of these repetitions can only match the empty string, so a
        // pass through it costs nothing, and TRE takes that pass rather than
        // the skip. Every assertion in the body therefore stays mandatory.
        assert!(!hits("x\\b?y", "xy"));
        assert!(!hits("x\\b{0,3}y", "xy"));
        assert!(!hits("x(\\b)*y", "xy"));
        assert!(!hits("x\\b?y", "x-y"));
        assert!(!hits("(^|a)*b", "cb"));
        assert!(hits("(^|a)*b", "ab"));
        assert!(!hits("(\\b*)*x", "yx"));
        assert!(hits("(\\b*)*x", "y x"));
        // A body that can match the empty string without asserting anything is
        // unaffected, because the extra pass matches nothing.
        assert!(hits("(a|)*b", "b"));
        assert!(hits("(a*|^)*b", "cb"));
        assert!(hits("(|^)*b", "cb"));
        assert!(hits("(a?)*b", "cb"));
        // A body that cannot match the empty string keeps its skip.
        assert!(hits("(a\\b)*c", "c"));
        // `{0}` is the exception: TRE throws the operand away entirely, so the
        // anchor is gone rather than mandatory.
        assert!(hits("^{0}a$", "*a"));
        assert!(hits("^{0}a", "*a"));
    }

    #[test]
    fn a_second_repetition_operator_is_reserved_and_refused() {
        assert_eq!(refuses("a**"), Error::BadRepeat);
        assert_eq!(refuses("a*+"), Error::BadRepeat);
        assert_eq!(refuses("a+*"), Error::BadRepeat);
        assert_eq!(refuses("a?*"), Error::BadRepeat);
        assert_eq!(refuses("a?+"), Error::BadRepeat);
        assert_eq!(refuses("a{2}*"), Error::BadRepeat);
        assert_eq!(refuses("a{2}+"), Error::BadRepeat);
        // A `?` is the exception. It asks for the shortest match rather than
        // the longest, which changes nothing about whether there is one.
        assert!(hits("^a*?$", ""));
        assert!(hits("^a??$", ""));
        assert!(hits("^a{2}?$", "aa"));
        // A bound after anything is a repetition of a repetition, not a second
        // operator, so it counts rather than being refused.
        assert!(hits("^a{2}{3}$", "aaaaaa"));
        assert!(!hits("^a{2}{3}$", "aa"));
        assert!(hits("^a*{2}$", "aa"));
        assert!(hits("^a{1,2}{1,2}$", "aaaa"));
    }

    #[test]
    fn nothing_a_pattern_can_do_makes_the_walk_more_than_linear() {
        // The shape that is exponential in a backtracking engine. If this ever
        // stops returning promptly, the simulation has grown a backtrack.
        let re = Regex::new(b"^(a+)+b$", false).expect("compiles");
        let mut m = Matcher::new();
        let subject = vec![b'a'; 4096];
        assert!(!m.is_match(&re, &subject));
        assert!(m.is_match(&re, b"aaaab"));
    }

    #[test]
    fn a_pattern_that_would_compile_to_too_much_is_refused_rather_than_built() {
        // Each level multiplies, so three of them is sixteen million steps.
        assert_eq!(refuses("((a{255}){255}){255}"), Error::Space);
        // Deep nesting is refused before it reaches the parser's own stack.
        let deep = "(".repeat(200) + "a" + &")".repeat(200);
        assert_eq!(refuses(&deep), Error::Space);
        // And the thing just under the cap still compiles.
        assert!(Regex::new(b"(a{200}){200}", false).is_ok());
    }

    #[test]
    fn one_matcher_serves_every_pattern_it_is_given() {
        // This is the shape ARGREP uses: several compiled patterns, one lot of
        // scratch, many subjects. The scratch grows to the largest program and
        // a smaller one afterwards must not read the leftovers as live states.
        let big = Regex::new(b"^(abc|def){2,8}$", false).expect("compiles");
        let small = Regex::new(b"^x$", false).expect("compiles");
        let mut m = Matcher::new();
        for _ in 0..4 {
            assert!(m.is_match(&big, b"abcdefabc"));
            assert!(m.is_match(&small, b"x"));
            assert!(!m.is_match(&small, b"y"));
            assert!(!m.is_match(&big, b"abcdefa"));
        }
    }

    #[test]
    fn the_error_messages_are_the_ones_tre_would_have_printed() {
        assert_eq!(Error::MissingBracket.as_str(), "Missing ']'");
        assert_eq!(Error::MissingParen.as_str(), "Missing ')'");
        assert_eq!(Error::MissingBrace.as_str(), "Missing '}'");
        assert_eq!(Error::BadBrace.as_str(), "Invalid contents of {}");
        assert_eq!(Error::BadRange.as_str(), "Invalid character range");
        assert_eq!(Error::CharClass.as_str(), "Unknown character class name");
        assert_eq!(Error::Collate.as_str(), "Unknown collating element");
        assert_eq!(Error::TrailingBackslash.as_str(), "Trailing backslash");
        assert_eq!(
            Error::BadRepeat.as_str(),
            "Invalid use of repetition operators"
        );
        assert_eq!(
            Error::BadMax.as_str(),
            "Maximum repetition in {} larger than 255"
        );
        assert_eq!(Error::Space.as_str(), "Out of memory");
        assert_eq!(Error::BadPattern.as_str(), "Invalid regexp");
        assert_eq!(Error::BackRef.as_str(), "Invalid back reference");
    }

    #[test]
    fn a_subject_is_bytes_and_not_text() {
        let re = Regex::new(b"^.{3}$", false).expect("compiles");
        let mut m = Matcher::new();
        // Three bytes of anything, including a NUL and the top of the range,
        // and a three byte character is three bytes rather than one.
        assert!(m.is_match(&re, &[0x00, 0xff, 0x80]));
        assert!(m.is_match(&re, "☃".as_bytes()));
        assert!(!m.is_match(&re, "ab".as_bytes()));
        let hi = Regex::new(&[b'^', 0xff, b'$'], false).expect("compiles");
        assert!(m.is_match(&hi, &[0xff]));
        assert!(!m.is_match(&hi, &[0xfe]));
    }
}
