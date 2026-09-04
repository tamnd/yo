//! Turning the bytes of a query into the tree that answers it.
//!
//! Recursive descent straight off the bytes, with no token stream in between,
//! for the two reasons in the module above: the errors carry byte offsets, and
//! the grammar changes meaning depending on which bracket it is inside.
//!
//! # What the two dialects actually differ on
//!
//! The first dialect has three loose operators. `@a:`, `-` and `~` all reach as
//! far to the right as the enclosing group lets them, so `-hello world` is one
//! negation of two words rather than a negated word and a word. The second
//! dialect gives each of them exactly one thing. That is nearly the whole
//! difference, and it is why the first dialect needs the idea of a run of plain
//! words: `@a:one two three` narrows all three, `@a:one (two)` narrows only the
//! first, and the thing that tells those apart is whether the next thing along
//! is a bare word or a bracket.
//!
//! # Words are expanded here
//!
//! `hello` does not parse to a word, it parses to a union of `hello` and the
//! stem of `hello`, because that is the query that runs and that is what
//! `FT.EXPLAIN` prints. Doing it during the parse rather than in a pass
//! afterwards keeps the one case that has to skip it honest: a word inside a
//! quoted phrase is matched as typed, and it never reaches the expander because
//! the phrase never calls it.

use crate::english::English;
use crate::field::Kind;
use crate::index::Index;
use crate::query::explain::bit;
use crate::query::{Circle, EVERY, Mask, Node, Pair, Range, Vector, What, Word, expansion};
use crate::text;
use crate::token::{bare, control, escapes, fold, wordy};

/// Why a query was refused.
///
/// Each of these is a different reply prefix on the wire, so they are kept apart
/// here rather than flattened into one string with the wording baked in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Bad {
    /// The bytes do not parse, pointing at where it went wrong.
    Syntax {
        /// Byte offset into the query.
        at: usize,
        /// The token that offset lands on.
        near: Box<[u8]>,
    },
    /// A field the schema has never heard of.
    Unknown {
        /// Byte offset into the query.
        at: usize,
        /// The field that was named, where the query left anything to quote.
        near: Option<Box<[u8]>>,
    },
    /// A field that is there but cannot do what was asked of it.
    Wrong {
        /// What the query needed it to be, in the wording of the reply.
        kind: &'static str,
        /// Byte offset into the query.
        at: usize,
        /// The field that was named, where the query left anything to quote.
        near: Option<Box<[u8]>>,
    },
    /// An attribute clause named something that is not an attribute.
    Attribute(Box<[u8]>),
    /// An attribute was given something it cannot hold.
    Value {
        /// The attribute, spelled the way the client spelled it.
        name: Box<[u8]>,
        /// What was written after the colon.
        value: Box<[u8]>,
    },
    /// A parameter the query refers to was not passed with `PARAMS`.
    Missing(Box<[u8]>),
    /// A distance was named after something the schema already has.
    Taken(Box<[u8]>),
    /// A syntax error a real server words for itself, with no place in it.
    Plain(&'static str),
    /// Something a real server words for itself, such as a field that cannot be
    /// matched the way the query asked for.
    Refused(&'static str),
}

/// The wording a real server uses for a geo filter in a unit it does not know.
pub const BAD_UNIT: &str = "Invalid GeoFilter unit";

/// The units a geo filter may be written in.
const UNITS: [&[u8]; 4] = [b"m", b"km", b"mi", b"ft"];

/// What a piece of the grammar needs the field it names to be.
///
/// The kind is decided by the shape of the query rather than by the schema:
/// `[1 10]` is a numeric filter whatever the field turns out to hold, and
/// naming a text field there is an error about the field rather than about the
/// brackets.
#[derive(Clone, Copy)]
enum Want {
    Text,
    Numeric,
    Geo,
    Tag,
    Vector,
}

impl Want {
    /// The word the reply uses for it.
    const fn named(self) -> &'static str {
        match self {
            Want::Text => "TEXT",
            Want::Numeric => "NUMERIC",
            Want::Geo => "GEO",
            Want::Tag => "TAG",
            Want::Vector => "VECTOR",
        }
    }

    /// Whether a field of this kind will do.
    const fn fits(self, kind: &Kind) -> bool {
        matches!(
            (self, kind),
            (Want::Text, Kind::Text(_))
                | (Want::Numeric, Kind::Numeric)
                | (Want::Geo, Kind::Geo | Kind::GeoShape(_))
                | (Want::Tag, Kind::Tag(_))
                | (Want::Vector, Kind::Vector(_))
        )
    }
}

/// The wording a real server uses when an empty string is asked of a field that
/// was not created to hold one.
pub const NO_EMPTY: &str =
    "Use `INDEXEMPTY` in field creation in order to index and query for empty strings";

/// The wording a real server uses when a field cannot be matched phonetically.
pub const NO_PHONETICS: &str = "field does not support phonetics";

/// One piece written inside a `[`, as the bytes of it, whether a `(` stood in
/// front of it and where it began.
type Item = (Box<[u8]>, bool, usize);

/// The wording a real server uses when a query nests deeper than its parser
/// can hold.
pub const TOO_DEEP: &str = "Parser stack overflow. Try moving nested parentheses more to the left";

/// How deeply brackets may nest under the later dialects, which is what a real
/// server holds before it gives up.
const DEEPEST: usize = 253;

/// How deeply brackets may nest under the first dialect, which gives up sooner
/// and words it as an ordinary syntax error.
const DEEPEST_ONE: usize = 97;

/// How many operators may stand one in front of another under the later
/// dialects, which is one more than the brackets allow because a bracket costs
/// a real server an extra place of its own.
const DEEPEST_OPS: usize = 254;

/// The deepest a query may nest, counted in groups rather than in nodes.
///
/// A query is bytes off the wire and a client can send a hundred thousand
/// operators as easily as one, and the first dialect reaches over the rest of
/// the group for every one of them, so the descent is bounded rather than
/// trusted. Nothing anybody writes comes near it and the stack is nowhere near
/// it either.
const DEEPEST_ANY: usize = 512;

/// What a client asked for on top of the query itself.
///
/// These come off the command line rather than out of the query, and every one
/// of them changes what the same bytes parse into, so they travel together
/// rather than as four arguments that are easy to pass in the wrong order.
#[derive(Debug, Clone, Copy)]
pub struct Ask<'a> {
    /// Which version of the grammar to read the query under.
    pub dialect: u8,
    /// What `PARAMS` passed, for the `$name` references in the query.
    pub params: &'a [Pair],
    /// Whether words are matched exactly as typed, which is `VERBATIM`.
    pub verbatim: bool,
    /// Whether stopwords are dropped, which `NOSTOPWORDS` turns off.
    pub stopwords: bool,
}

impl Default for Ask<'_> {
    fn default() -> Ask<'static> {
        Ask {
            dialect: 1,
            params: &[],
            verbatim: false,
            stopwords: true,
        }
    }
}

/// Parses a query against the schema it will run on.
///
/// The schema is needed during the parse and not only after it, because
/// `@n:[1 10]` and `@loc:[1 2 3 km]` are the same brackets holding different
/// grammars and only the field kind says which.
///
/// # Errors
///
/// A `Bad` naming what went wrong and where.
pub fn parse(query: &[u8], index: &Index, ask: &Ask) -> Result<Node, Bad> {
    // A real server reads the query as a C string, so a zero byte in it ends the
    // query rather than sitting inside it: `aa\0bb` is the one word `aa`.
    let query = match query.iter().position(|b| *b == 0) {
        Some(end) => &query[..end],
        None => query,
    };
    let mut p = Parse {
        src: query,
        at: 0,
        index,
        ask: *ask,
        deep: 0,
        stemmer: English::default(),
        mark: 0,
        word: Box::default(),
        stop: 0,
        holding: false,
        hung: false,
        arrowed: false,
        dropped: 0,
        crossed: false,
        nested: 0,
        ops: 0,
        inside: false,
        shut: 0,
        solo: true,
        gone: Vec::new(),
        taken: None,
        unit: false,
        hush: false,
        fielded: false,
        barred: None,
    };
    let node = p.query()?;
    // These three wait until the whole query has parsed, because a real server
    // resolves parameters and checks the rest of a vector clause after it has a
    // tree rather than while it is building one, and a syntax error further
    // along the query is the one that gets reported.
    if let Some(name) = p.gone.into_iter().next() {
        return Err(Bad::Missing(name));
    }
    if p.unit {
        return Err(Bad::Plain(BAD_UNIT));
    }
    if let Some(name) = p.taken {
        return Err(Bad::Taken(name));
    }
    Ok(node)
}

/// The cursor and everything the grammar needs to read the bytes under it.
struct Parse<'a> {
    src: &'a [u8],
    at: usize,
    index: &'a Index,
    ask: Ask<'a>,
    deep: usize,
    stemmer: English,
    /// Where the token the parser is looking at starts.
    mark: usize,
    /// The last word it read, which is what an error quotes back.
    ///
    /// A syntax error names a position and a word, and they are not always the
    /// same token: `hello)` is refused at offset 5, which is the bracket, and
    /// quotes `hello`, which is the word before it. Keeping the two apart here
    /// is what makes that come out right rather than by accident.
    word: Box<[u8]>,
    /// Where the last `]` was, which is where a bracket holding the wrong
    /// number of things is refused.
    shut: usize,
    /// Whether the query so far is one thing, which is what a vector clause
    /// may hang off.
    solo: bool,
    /// The parameters the query named that nobody passed, in the order the
    /// tree that holds them is walked.
    ///
    /// A query with two of them names one, and which one is not the one written
    /// first: `$B $C $D` is a pair on the right with `$B` folded into it, and
    /// the pair is walked first, so the answer is `C`. The order is kept right
    /// here by swapping the two halves of the list wherever a fold swaps the
    /// two halves of the tree, which costs nothing on the ordinary query that
    /// passed everything it named.
    gone: Vec<Box<[u8]>>,
    /// The name a vector clause gave its distance that the schema already uses.
    taken: Option<Box<[u8]>>,
    /// Whether a geo filter named a unit that does not exist.
    unit: bool,
    /// Whether a pattern has been read, which empties the word an unknown or
    /// wrong field quotes back for the rest of the query.
    hush: bool,
    /// Whether a modifier has been read anywhere inside the union being read,
    /// which is what lets a later part of it carry one of its own.
    fielded: bool,
    /// How deep the union is whose part may not bring a field of its own, when
    /// one is being read.
    ///
    /// A field at that depth ends the union rather than joining the part, so
    /// `a1|b1|c1 @n:[1 10]` is the numeric over all three words. One written
    /// any deeper, inside a bracket or under an operator, has nowhere to go and
    /// is refused where it stands.
    barred: Option<usize>,
    /// Whether a field modifier is already open over whatever comes next.
    ///
    /// The second dialect allows one at a time, so this is what refuses the
    /// second rather than a count of how many are open.
    inside: bool,
    /// The operators that end a run of things, which is how far a `-` or a `~`
    /// reaches under the first dialect.
    stop: u8,
    /// Whether a first dialect run of words is being collected, in which case
    /// an attribute clause belongs to the whole run rather than to the word it
    /// happens to follow.
    holding: bool,
    /// Whether the last thing read carried an attribute clause of its own.
    ///
    /// A clause ends what a field modifier reaches over, so `@a:hello hel*`
    /// narrows both words where `@a:hello => {$weight: 2;}hel*` narrows only
    /// the first.
    hung: bool,
    /// Whether the error on its way out is a vector clause written inside a
    /// bracket, which a field modifier is asked about before that error is
    /// heard.
    arrowed: bool,
    /// Where a run a stopword put an end to left the cursor, or zero when the
    /// last thing read was not one of those.
    ///
    /// A `|` written straight after one takes the stopword on its left, which
    /// is nothing, so the run in front of it stands beside the union rather
    /// than inside it. It is kept as a place rather than as a yes or no so
    /// that anything else read in between says by itself that the stopword is
    /// no longer what the bar is written after.
    dropped: usize,
    /// Whether the last thing read was a bar the stopword in front of it took,
    /// which the next thing in the sequence is read across.
    ///
    /// An operator gives up its run where another one is written, and a bar is
    /// not part of that run at all, so `~@A:x the|~(@a:hello)` is one maybe
    /// over the pair where `~@A:x ~(@a:hello)` is two side by side.
    crossed: bool,
    /// How many brackets stand open around where the parser is.
    nested: usize,
    /// How many operators stand one in front of another where the parser is.
    ops: usize,
}

/// A `-` ends the run a `-` reaches over.
const STOP_NOT: u8 = 1;
/// A `~` ends the run either operator reaches over, which is the one place the
/// two differ: a negation gives way to another negation and to a maybe, and a
/// maybe gives way only to another maybe.
const STOP_MAYBE: u8 = 2;
/// A phrase ends the run a `-` reaches over, where a `~` reaches straight over
/// it: `-a1 "b1 c1" d1` leaves out only `a1` and `~a1 "b1 c1" d1` takes all
/// three. A phrase written after a `|` or inside a bracket is not one of these,
/// because the run is not what is being read there.
const STOP_PHRASE: u8 = 4;

/// Where the numeric literal starting at a position ends, if there is one.
///
/// A sign, digits, one point, and an exponent that only counts when it has
/// digits of its own, which is why `1e3` is a number and `1e` is a word. A
/// literal that runs straight into letters, such as `0x1f`, is a word after
/// all, and that is the one part of it that has to be checked at the end
/// rather than the start.
fn number_len(src: &[u8], from: usize) -> Option<usize> {
    let mut at = from;
    if matches!(src.get(at), Some(b'+' | b'-')) {
        at += 1;
    }
    let mut digits = 0;
    while matches!(src.get(at), Some(b'0'..=b'9')) {
        at += 1;
        digits += 1;
    }
    let mut dotted = false;
    if matches!(src.get(at), Some(b'.')) {
        at += 1;
        dotted = true;
        while matches!(src.get(at), Some(b'0'..=b'9')) {
            at += 1;
            digits += 1;
        }
    }
    if digits == 0 {
        return None;
    }
    if matches!(src.get(at), Some(b'e' | b'E')) {
        let mut ex = at + 1;
        if matches!(src.get(ex), Some(b'+' | b'-')) {
            ex += 1;
        }
        let mut more = 0;
        while matches!(src.get(ex), Some(b'0'..=b'9')) {
            ex += 1;
            more += 1;
        }
        if more > 0 {
            at = ex;
        }
    }
    // A dot somewhere in it settles the question of what this is, so `5.5a` is
    // the number and then the word while `0x1f` is one word after all.
    if !dotted && matches!(src.get(at), Some(b) if wordy(*b)) {
        return None;
    }
    Some(at)
}

/// Whether a byte ends a word without meaning anything itself.
///
/// These are the marks that split a word where they fall between two of them:
/// `ab-cd` is two words and so is `ab.cd`. The same marks mean something at the
/// start of a word, which is why they are only consumed straight after one.
const fn splitter(b: u8) -> bool {
    matches!(
        b,
        b'.' | b',' | b'/' | b'+' | b'=' | b'!' | b'?' | b'#' | b'&' | b'^' | b'<' | b'>' | b'`'
    )
}

impl Parse<'_> {
    // The cursor.

    fn peek(&self) -> Option<u8> {
        self.src.get(self.at).copied()
    }

    fn done(&self) -> bool {
        self.at >= self.src.len()
    }

    /// Whitespace, and under the first dialect the quote that has no meaning
    /// there either.
    ///
    /// The second dialect gives `'` two jobs, a phrase of its own and the
    /// pattern in `w'...'`, and the first gives it none, so under the first it
    /// is skipped wherever it turns up. That is what leaves `w'FOO*'` as the
    /// word `w` next to the prefix `FOO*` rather than as anything quoted.
    fn spaces(&mut self) {
        loop {
            match self.peek() {
                Some(b' ') => self.at += 1,
                // Every control byte ends a word here the way a space does, so
                // `a\nb` is two words. A document is tokenised by other rules
                // that drop them instead, which is a difference between the two
                // halves of a real server rather than one this invented.
                Some(b) if control(b) => self.at += 1,
                // A backslash that escapes nothing is not part of the word it
                // was written against and is not a token of its own either, so
                // `aa\yy` is the two words `aa` and `yy`.
                Some(b'\\') if !self.escaped(self.at) => self.at += 1,
                Some(b'\'') if self.ask.dialect == 1 => self.at += 1,
                _ => return,
            }
        }
    }

    fn eat(&mut self, b: u8) -> bool {
        if self.peek() == Some(b) {
            self.mark = self.at;
            self.at += 1;
            return true;
        }
        false
    }

    fn starts(&self, s: &[u8]) -> bool {
        self.src[self.at.min(self.src.len())..].starts_with(s)
    }

    fn at_word(&self) -> bool {
        matches!(self.peek(), Some(b) if wordy(b)) || self.escaped(self.at)
    }

    /// Whether a backslash at this offset takes the byte after it as a letter,
    /// which is the only way a backslash is part of a word at all.
    fn escaped(&self, at: usize) -> bool {
        self.src.get(at) == Some(&b'\\') && matches!(self.src.get(at + 1), Some(b) if escapes(*b))
    }

    /// Where the word starting at a position ends.
    fn word_end(&self, from: usize) -> usize {
        let mut at = from;
        while at < self.src.len() {
            if self.escaped(at) {
                at += 2;
            } else if wordy(self.src[at]) {
                at += 1;
            } else {
                break;
            }
        }
        at.min(self.src.len())
    }

    /// Skips an `@` that names no field, which is not a token at all.
    ///
    /// A real server's scanner has nothing to emit for a lone `@`, so it is not
    /// an empty node in the tree, it simply is not there: `@ hello` is `hello`,
    /// `@@` is an empty query, and `hello|@` is a union with nothing on its
    /// right hand side and is refused at the bar rather than at the `@`. A colon
    /// after it is a different thing, because then the query did mean to name a
    /// field.
    /// Whitespace, and the marks that mean nothing where a thing is about to
    /// start.
    ///
    /// `hello, world` is two words and so is `hello ,world`, because a comma
    /// is a mark that splits words wherever it falls rather than only after
    /// one. The same goes for the rest of them, so `,*` is the star and `,-x`
    /// is a negation, and a query with nothing in it but marks is a query with
    /// nothing in it. A sign in front of digits is part of the number instead,
    /// and an arrow is an arrow.
    fn lead(&mut self) {
        loop {
            self.spaces();
            if self.starts(b"=>") || self.at_number() {
                return;
            }
            match self.peek() {
                Some(b) if splitter(b) => self.at += 1,
                _ => return,
            }
        }
    }

    /// Whether the mark at this offset names nothing, which is what makes it
    /// noise rather than something the query meant.
    ///
    /// A sign with no name after it, a sign with no field after it and a quote
    /// that never closes are all nothing at all, so `@@*` and `$*` are the star
    /// as much as `*` is.
    fn lone_at(&self, at: usize) -> bool {
        match self.src.get(at) {
            Some(b'@') => !matches!(self.src.get(at + 1), Some(c) if wordy(*c) || *c == b':'),
            Some(b'$') => !matches!(self.src.get(at + 1), Some(c) if wordy(*c)),
            Some(b'\'') => !self.src[at + 1..].contains(&b'\''),
            _ => false,
        }
    }

    fn skip_lone(&mut self) {
        loop {
            match self.peek() {
                Some(b'@') => {
                    let end = self.word_end(self.at + 1);
                    if end != self.at + 1 || self.src.get(end) == Some(&b':') {
                        return;
                    }
                    self.at = end;
                }
                // A sign with no name after it names no parameter and is not
                // anything else either, so `x1 $ x1` is the two words.
                Some(b'$') if !matches!(self.src.get(self.at + 1), Some(b) if wordy(*b)) => {
                    self.at += 1;
                }
                // A quote that never closes opens nothing, so `x1 ' x1` is the
                // two words and `'a b` is the two words inside it.
                Some(b'\'') if !self.closed() => {
                    self.at += 1;
                }
                _ => return,
            }
            self.spaces();
        }
    }

    /// Where the field modifier at the head of the next part of a union is, if
    /// that is what it starts with.
    ///
    /// Operators and brackets are looked past, because `-@a:x` and `(@a:x)` are
    /// as much a modifier at the head as `@a:x` is, and a modifier further along
    /// is not one at the head at all.
    fn at_modifier(&self) -> Option<(usize, Box<[u8]>)> {
        if self.ask.dialect == 1 {
            return None;
        }
        let mut at = self.at;
        while matches!(
            self.src.get(at),
            Some(b' ' | b'\t' | b'\n' | b'\r' | b'-' | b'~' | b'(')
        ) {
            at += 1;
        }
        if self.src.get(at) != Some(&b'@') {
            return None;
        }
        let end = self.word_end(at + 1);
        if end == at + 1 {
            return None;
        }
        Some((at, self.src[at + 1..end].into()))
    }

    /// Whether a field modifier starts right here, which is what a union part
    /// that may not carry one stops at.
    fn at_field(&self) -> bool {
        if self.peek() != Some(b'@') {
            return false;
        }
        let mut end = self.word_end(self.at + 1);
        if end == self.at + 1 {
            return false;
        }
        // A modifier may name several fields, and the colon comes after the
        // last of them.
        while self.src.get(end) == Some(&b'|') {
            let next = self.word_end(end + 1);
            if next == end + 1 {
                return false;
            }
            end = next;
        }
        self.src.get(end) == Some(&b':')
    }

    /// The text an error quotes when it is refused at the start of something.
    ///
    /// The offset and the word in a syntax error come from the token the parser
    /// gave up on, and for most of them that is the token's own text: the name
    /// in `@field:` or in `$param`, the first word of a phrase, the stem of a
    /// prefix. A bracket, a `-` and a `%` carry no text of their own, so an
    /// error on one quotes whatever word was last read instead.
    fn near_at(&self, at: usize) -> Box<[u8]> {
        let from = match self.src.get(at) {
            // A phrase quotes the whole of itself, and one that never closes
            // has nothing to quote at all. The first dialect never reads a
            // phrase this far ahead and quotes the word it read last instead.
            Some(b'"') => {
                if self.ask.dialect == 1 {
                    return self.word.clone();
                }
                let start = at + 1;
                let Some(end) = self.src[start..].iter().position(|b| *b == b'"') else {
                    return Box::default();
                };
                return self.src[start..start + end].into();
            }
            Some(b'@' | b'$') => at + 1,
            Some(b'*') if matches!(self.src.get(at + 1), Some(b) if wordy(*b)) => at + 1,
            Some(b) if wordy(*b) => at,
            Some(b'\\') if self.escaped(at) => at,
            _ => return Box::default(),
        };
        self.src[from..self.word_end(from)].into()
    }

    /// Where a pattern written at `at` begins and what is in it, for an error
    /// that has to name one.
    ///
    /// A pattern is quoted by what stands between its quotes and points there
    /// rather than at the `w` in front of them, so `* w'a*b'` is refused at the
    /// body and names it.
    fn patternish(&self, at: usize) -> Option<(usize, Box<[u8]>)> {
        if self.ask.dialect == 1 || !self.src[at.min(self.src.len())..].starts_with(b"w'") {
            return None;
        }
        let body = at + 2;
        let end = self.src[body..].iter().position(|b| *b == b'\'')?;
        (end > 0).then(|| (body, self.src[body..body + end].into()))
    }

    // Errors.

    /// A syntax error at a place of the caller's choosing, quoting the last
    /// word read.
    fn syntax_at(&self, at: usize) -> Bad {
        Bad::Syntax {
            at,
            near: self.word.clone(),
        }
    }

    /// A syntax error at whatever the parser is looking at, or at `start` when
    /// there is nothing there to look at.
    ///
    /// A star glued to a word belongs to the word rather than standing as
    /// something of its own, so `%hello*` is refused back at the word where
    /// `%hello world` is refused at the word after it. Something with no text
    /// of its own, a bracket or a bar, quotes the last word read.
    fn stuck(&mut self, start: usize) -> Bad {
        if self.peek() == Some(b'*') {
            // The star is part of the word it is glued to, so there is nothing
            // else in the query to point at even when the query goes on:
            // `%hello* world` is refused back at `hello` the same way `%hello*`
            // on its own is.
            return Bad::Syntax {
                at: start,
                near: self.word.clone(),
            };
        }
        self.spaces();
        if self.done() {
            return Bad::Syntax {
                at: start,
                near: self.word.clone(),
            };
        }
        let near = self.near_at(self.at);
        let near = if near.is_empty() {
            self.word.clone()
        } else {
            near
        };
        Bad::Syntax { at: self.at, near }
    }

    /// The error for wherever the parse gave up.
    ///
    /// The offset is the token being looked at and the word is the last one
    /// read, which are the same thing when a word is what went wrong and are
    /// not when a bracket is. At the end of the query there is no token to
    /// point at, so it points at the last one there was.
    fn syntax(&self) -> Bad {
        let at = if self.done() { self.mark } else { self.at };
        Bad::Syntax {
            at,
            near: self.word.clone(),
        }
    }

    /// The same, quoting what is in front of the parser rather than the word it
    /// read last.
    ///
    /// A star that is standing where a word belongs is refused at the star and
    /// names the word after it, so `aa%*BB` is refused at the star near `BB`
    /// and not near the `aa` that came before. Where what follows has no text
    /// of its own there is nothing to name and the last word read is quoted
    /// after all.
    fn syntax_near(&self) -> Bad {
        let near = self.near_at(self.at);
        if near.is_empty() {
            return self.syntax();
        }
        Bad::Syntax { at: self.at, near }
    }

    /// Whether a field is there and can do what the query asked of it.
    ///
    /// Only the second dialect asks. The first takes any field name for a
    /// filter and works out afterwards that it matches nothing, which is
    /// generous and is what clients written against it expect.
    /// Whether a field is there at all, which is asked earlier than what it
    /// can do is.
    fn known(&self, name: &[u8], at: usize) -> Result<(), Bad> {
        if self.ask.dialect == 1 || self.index.field(name).is_some() {
            return Ok(());
        }
        Err(Bad::Unknown {
            at,
            near: (!self.hush).then(|| name.into()),
        })
    }

    fn expect(&self, name: &[u8], at: usize, want: Want) -> Result<(), Bad> {
        if self.ask.dialect == 1 {
            return Ok(());
        }
        // A pattern earlier in the query leaves nothing for these two to quote,
        // so they name the offset and stop there.
        let near: Option<Box<[u8]>> = (!self.hush).then(|| name.into());
        let Some(field) = self.index.field(name) else {
            return Err(Bad::Unknown { at, near });
        };
        if !want.fits(&field.kind) {
            return Err(Bad::Wrong {
                kind: want.named(),
                at,
                near,
            });
        }
        Ok(())
    }

    // The grammar.

    /// A whole query, which is a union and possibly a vector clause after it.
    fn query(&mut self) -> Result<Node, Bad> {
        self.lead();
        self.skip_lone();
        if self.done() {
            return Ok(Node::empty());
        }
        let mut node = self.union()?;
        self.spaces();
        if self.starts(b"=>") {
            let eq = self.at;
            self.at += 2;
            let bare = matches!(node.what, What::Wildcard);
            self.spaces();
            let near = self.arrow(eq);
            if self.peek() != Some(b'[') {
                self.mark = eq;
                self.word = near;
                return Err(self.after_arrow());
            }
            // The clause hangs off one thing: a word, a group, a filter. A
            // sequence or a union or anything with an operator on it is not
            // one thing, and neither is anything at all under the first
            // dialect, which has no vector syntax.
            if self.ask.dialect == 1 || !self.solo {
                let at = if bare && self.ask.dialect == 1 {
                    eq
                } else {
                    self.at
                };
                return Err(Bad::Syntax { at, near });
            }
            // A clause that runs out halfway is refused at the last thing that
            // was read, which is the bracket until something inside it has been.
            self.mark = self.at;
            self.word = near;
            node = self.knn(node)?;
        }
        self.spaces();
        if !self.done() {
            return Err(self.syntax());
        }
        Ok(node)
    }

    /// Things separated by `|`.
    ///
    /// The first dialect binds a `|` tighter than the space between two things,
    /// so `x hel*|z` asks for `x` and for either of the other two, while the
    /// later dialects read the same bytes as either `x hel*` or `z`. That is
    /// why the first dialect hands the whole job to the sequence below and only
    /// the later ones join sequences here.
    fn union(&mut self) -> Result<Node, Bad> {
        self.deeper()?;
        if self.ask.dialect == 1 {
            let node = self.seq()?;
            self.deep -= 1;
            return Ok(node);
        }
        let mut fields = self.at_modifier();
        let mut wide = 1;
        // A modifier anywhere inside a part counts, not only one at the head of
        // it, and only the parts of this union count: `a1|b1 @a:x|@g:{x}` is
        // taken where `a1|b1|@g:{x}` is refused, and a modifier outside the
        // brackets does nothing for a union written inside them.
        let held = std::mem::take(&mut self.fielded);
        let mut head = self.gone.len();
        let mut acc = self.seq()?;
        // The unions waiting on the one being read now, which is the last part
        // of the one under it. What each of them holds is its parts so far,
        // where in the missing parameters they start, and what it knew about
        // fields before this one took over.
        let mut stack: Vec<(Node, usize, bool)> = Vec::new();
        loop {
            if self.fielded {
                fields = fields.or(Some((0, Box::default())));
            }
            self.spaces();
            self.skip_lone();
            let bar = self.at;
            if !self.eat(b'|') {
                break;
            }
            if matches!(acc.what, What::Wildcard) {
                return Err(Bad::Syntax {
                    at: bar,
                    near: self.word.clone(),
                });
            }
            if self.deep == 1 {
                self.solo = false;
            }
            self.spaces();
            // A union takes a field of its own in its first two parts, and
            // after that only where it already had one. The rule reads like an
            // accident of somebody's grammar and it is one, but a client that
            // writes `a|b|@title:c` gets an error back and has to be told the
            // same thing here.
            let named = self.at_modifier();
            match (named.clone(), wide > 1 && fields.is_none()) {
                (Some((at, name)), true) => return Err(Bad::Syntax { at, near: name }),
                (Some(found), false) => fields = fields.or(Some(found)),
                (None, _) => {}
            }
            wide += 1;
            let mid = self.gone.len();
            // A union with a field of its own keeps its first two parts and
            // puts everything after them in a union of their own, which has the
            // same rules again: `@a:z|b1|c1|d1` is three parts with a pair as
            // the last of them, where `a1|b1|c1|d1` is four side by side.
            if wide > 2 && fields.is_some() {
                stack.push((acc, head, std::mem::take(&mut self.fielded)));
                head = mid;
                fields = named;
                wide = 1;
                acc = self.seq()?;
                continue;
            }
            // A part after the second one of a union with no field yet may not
            // bring the first, and what happens instead depends on where the
            // field is written. One standing on its own in the part ends the
            // union and takes it whole, and one buried inside a bracket or
            // under an operator has nothing it could take and is refused.
            let bars = wide > 2 && fields.is_none();
            let stood = self.barred;
            if bars {
                self.barred = Some(self.deep);
            }
            let next = self.seq()?;
            self.barred = stood;
            let either = self.ask.dialect >= 2;
            if Node::swaps(&acc, &next, either, false) {
                self.swap_gone(head, mid);
            }
            acc = Node::fold(acc, next, either, false);
            if bars && self.at_field() {
                // The union ends here and stands as the first thing in the
                // sequence the field belongs to, and a `|` after that starts a
                // union again with the whole sequence as its first part.
                acc = self.seq_from(acc, head)?;
                wide = 1;
            }
        }
        // The unions waiting under this one take it as their last part.
        while let Some((outer, spot, older)) = stack.pop() {
            let either = self.ask.dialect >= 2;
            if Node::swaps(&outer, &acc, either, false) {
                self.swap_gone(spot, head);
            }
            acc = Node::fold(outer, acc, either, false);
            self.fielded |= older;
            head = spot;
        }
        self.fielded |= held;
        self.deep -= 1;
        Ok(acc)
    }

    /// Things next to each other, which means all of them have to match.
    fn seq(&mut self) -> Result<Node, Bad> {
        let head = self.gone.len();
        let acc = self.bars()?;
        self.seq_from(acc, head)
    }

    /// The rest of a sequence, with something already read standing at the head
    /// of it.
    ///
    /// The union that ends where a field modifier stands carries on as the
    /// first thing in a sequence, which is what makes `a1|b1|c1 @n:[1 10]` the
    /// numeric over the whole union.
    fn seq_from(&mut self, acc: Node, head: usize) -> Result<Node, Bad> {
        let mut acc = acc;
        loop {
            self.lead();
            self.skip_lone();
            if self.done() || matches!(self.peek(), Some(b')' | b'|' | b']' | b'}')) {
                break;
            }
            if self.starts(b"=>") {
                break;
            }
            // A bar a stopword took is still a bar as far as an operator is
            // concerned, and the thing after it is a part of its own that the
            // operator reaches over rather than the end of what it reaches.
            let crossed = std::mem::take(&mut self.crossed);
            if !crossed
                && (matches!(self.peek(), Some(b'-') if self.stop & STOP_NOT != 0)
                    || matches!(self.peek(), Some(b'~') if self.stop & STOP_MAYBE != 0)
                    || matches!(self.peek(), Some(b'"') if self.stop & STOP_PHRASE != 0))
            {
                break;
            }
            // A union part that may not bring a field of its own ends where one
            // is written, and the caller takes the union and the field on from
            // there.
            if self.barred == Some(self.deep) && self.at_field() {
                break;
            }
            // A star with something after it is refused as that thing is
            // reached rather than after it has been read, so the error names
            // the first word of it and nothing further along.
            let mut spot = self.at;
            if matches!(acc.what, What::Wildcard) {
                // A sign that names nothing is passed over here even with a
                // colon after it, because the star has already been read and
                // there is no field left for the colon to belong to.
                if self.peek() == Some(b'@') && self.word_end(self.at + 1) == self.at + 1 {
                    spot += 1;
                }
                if let Some((body, near)) = self.patternish(spot) {
                    return Err(Bad::Syntax { at: body, near });
                }
                return Err(Bad::Syntax {
                    at: spot,
                    near: self.near_at(spot),
                });
            }
            let mid = self.gone.len();
            let next = self.bars()?;
            if self.deep == 1 {
                self.solo = false;
            }
            acc = self.chained(acc, next, head, mid)?;
        }
        Ok(acc)
    }

    /// Joins two things standing next to each other, and takes in whatever
    /// stands after them that binds tighter than the join does.
    ///
    /// Under the later dialects `hello world *llo` is `hello` and then
    /// `world *llo` rather than three things in a row, because a suffix reaches
    /// back to the word on its left before that word reaches back to its own.
    /// So the pair on the right is built first and the thing on the left is
    /// folded into it afterwards, which is what leaves `hello` last in the
    /// printout. The pairs waiting for that are held on a list rather than by
    /// calling back into here, so a long line of them costs no stack.
    fn chained(&mut self, left: Node, right: Node, head: usize, mid: usize) -> Result<Node, Bad> {
        let mut held: Vec<Node> = Vec::new();
        // Where in the list of missing parameters each of those things begins,
        // so that a fold which swaps two of them can swap their parameters too.
        let mut marks: Vec<usize> = Vec::new();
        let mut left = left;
        let mut right = right;
        let mut head = head;
        let mut mid = mid;
        while self.sticky() {
            held.push(left);
            marks.push(head);
            left = right;
            head = mid;
            mid = self.gone.len();
            right = self.factor()?;
        }
        let either = self.ask.dialect >= 2;
        if Node::swaps(&left, &right, either, true) {
            self.swap_gone(head, mid);
        }
        let mut node = Node::fold(left, right, either, true);
        while let Some(prev) = held.pop() {
            let start = marks.pop().unwrap_or(head);
            if Node::swaps(&prev, &node, either, true) {
                self.swap_gone(start, head);
            }
            node = Node::fold(prev, node, either, true);
            head = start;
        }
        Ok(node)
    }

    /// Adds to something already read whatever follows it that binds tighter
    /// than a space, which is what widens the reach of a `-`, a `~` and a field
    /// modifier past the one thing they otherwise take.
    fn tighter(&mut self, node: Node, head: usize) -> Result<Node, Bad> {
        if !self.sticky() {
            return Ok(node);
        }
        if self.deep == 1 {
            self.solo = false;
        }
        let mid = self.gone.len();
        let next = self.factor()?;
        self.chained(node, next, head, mid)
    }

    /// Whether what starts here binds to the thing on its left before that
    /// thing binds to whatever is on its own left.
    ///
    /// A prefix, a suffix, an infix, a fuzzy word, a pattern, a number and a
    /// parameter all do, and a plain word, a phrase, a bracket and a modifier
    /// do not. It is the whole of why `-hello hel*` leaves both words out while
    /// `-hello world` leaves out only the first. The first dialect does none of
    /// this and reads a sequence straight from left to right.
    fn sticky(&mut self) -> bool {
        if self.ask.dialect == 1 {
            return false;
        }
        self.spaces();
        // A sign that names nothing stands between two things without keeping
        // them apart, so `-(a1)@*ell*` leaves out the pair the way `-(a1)*ell*`
        // does.
        self.skip_lone();
        // A sign with no name after it is not a parameter and binds to nothing,
        // so `@a:hello $|world` reads the union rather than the parameter.
        let param =
            self.peek() == Some(b'$') && matches!(self.src.get(self.at + 1), Some(b) if wordy(*b));
        param || self.at_number() || self.at_wild_word()
    }

    /// One thing in a sequence, and everything a `|` joins to it.
    ///
    /// This is where the first dialect differs. A `|` there takes the thing on
    /// its left rather than the whole sequence, so `x y|z` is the run `x y` or
    /// `z` while `x hel*|z` is `x` and either of the other two, the run being
    /// the only thing that holds more than one word together. The later
    /// dialects join whole sequences instead and do that in `union` above, so
    /// there is nothing to do here for them.
    fn bars(&mut self) -> Result<Node, Bad> {
        let mut pending = self.factor()?;
        if self.ask.dialect != 1 {
            return Ok(pending);
        }
        let mut acc = Node::empty();
        loop {
            self.spaces();
            self.skip_lone();
            let bar = self.at;
            // Whether the run on the left is one a stopword ended has to be
            // asked before the bar is read, because reading it moves the cursor
            // off the spot the run left behind.
            let stopped = self.after_stop();
            if !self.eat(b'|') {
                break;
            }
            if matches!(pending.what, What::Wildcard) {
                return Err(Bad::Syntax {
                    at: bar,
                    near: self.word.clone(),
                });
            }
            if self.deep == 1 {
                self.solo = false;
            }
            // A run a stopword ended leaves the bar with nothing on its left,
            // so the bar is read and thrown away and what follows it is the
            // next thing in the sequence: `a1 b1 the|c1` is the flat three.
            if stopped {
                self.dropped = 0;
                self.crossed = true;
                break;
            }
            self.spaces();
            let next = self.factor()?;
            acc = Node::fold(acc, pending, false, false);
            pending = next;
        }
        Ok(Node::fold(acc, pending, false, false))
    }

    /// One thing, which may be an operator and everything it reaches over.
    fn factor(&mut self) -> Result<Node, Bad> {
        self.lead();
        // A sign in front of digits is part of the number, not an operator.
        if self.at_number() {
            return self.run();
        }
        let spot = self.at;
        match self.peek() {
            Some(b'-') => {
                self.mark = spot;
                self.at += 1;
                if self.deep == 1 {
                    self.solo = false;
                }
                self.operator()?;
                let child = self.reach(STOP_NOT | STOP_MAYBE | STOP_PHRASE);
                self.ops = self.ops.saturating_sub(1);
                Ok(wrap(child?, true))
            }
            Some(b'~') => {
                self.mark = spot;
                self.at += 1;
                if self.deep == 1 {
                    self.solo = false;
                }
                self.operator()?;
                let child = self.reach(STOP_MAYBE);
                self.ops = self.ops.saturating_sub(1);
                Ok(wrap(child?, false))
            }
            _ => self.run(),
        }
    }

    /// Whether a bare star is somewhere a real server takes one.
    ///
    /// The star stands for every document and is a whole query rather than a
    /// piece of one, so anything at all in front of it is refused, and it is
    /// refused at the star rather than at whatever came before. The later
    /// dialects allow brackets around it and the first allows nothing, which is
    /// why `(*)` is a query in one of them and an error in the other.
    fn fresh(&self) -> Result<(), Bad> {
        let bracketed = self.ask.dialect != 1;
        let mut at = 0;
        while at < self.mark {
            let b = self.src[at];
            if !b.is_ascii_whitespace()
                && !splitter(b)
                && !(bracketed && b == b'(')
                && !self.lone_at(at)
            {
                return Err(Bad::Syntax {
                    at: self.mark,
                    near: self.word.clone(),
                });
            }
            at += 1;
        }
        Ok(())
    }

    /// What a `-` or a `~` takes.
    ///
    /// The whole rest of the group under the first dialect, one thing under the
    /// second. This is the single difference that makes `-hello world` mean two
    /// opposite things depending on a number nobody sets.
    fn reach(&mut self, stop: u8) -> Result<Node, Bad> {
        if self.ask.dialect != 1 {
            let head = self.gone.len();
            let node = self.factor()?;
            return self.tighter(node, head);
        }
        let held = self.stop;
        // What an operator stops at is its own business and not the business of
        // whatever it is written inside, so a `~` in `-~a -b` reaches over the
        // `-b` even though the `-` in front of it would have stopped there.
        self.stop = stop;
        let node = self.union();
        self.stop = held;
        node
    }

    /// A run of plain words, or one thing that is not a plain word.
    ///
    /// Under the first dialect a run of bare words is one node, and that is what
    /// makes `@a:one two three` narrow all three while `@a:one (two)` narrows
    /// only the first. The run stops before a wildcard word rather than taking
    /// it in, which is what leaves `hel*` on its own for a `|` after it to bind
    /// to. The later dialects have no run at all, and what looks like one there
    /// comes from `sticky` instead.
    fn run(&mut self) -> Result<Node, Bad> {
        if self.ask.dialect != 1 || !self.at_word() || !self.at_plain_word() {
            return self.atom();
        }
        let mut acc = Node::empty();
        loop {
            // An attribute clause written after the last word of a run belongs
            // to the whole run under this dialect, so the words are read
            // without it and it is read once at the end.
            self.holding = true;
            let node = self.atom()?;
            // A modifier whose first word is a stopword reaches nothing, and
            // the words after it are read as if the modifier had not been
            // written at all, so `@a:the hello` asks every field for `hello`.
            if self.inside && matches!(acc.what, What::Empty) && matches!(node.what, What::Empty) {
                return self.attributes(node);
            }
            // A stopword in the middle ends the run the way a wildcard word
            // does, and the words after it are a run of their own, so
            // `x1 a b1 c1` is `x1` and then the pair rather than three in a row.
            if matches!(node.what, What::Empty) && !matches!(acc.what, What::Empty) {
                // A clause written after the stopword belongs to the stopword
                // rather than to the run in front of it, and a stopword is
                // nothing to hang one on, so `-b1 the => {$weight: 0.5;}` is
                // the negation with no weight on it at all.
                self.attributes(node)?;
                self.dropped = self.at;
                return Ok(acc);
            }
            acc = Node::fold(acc, node, false, true);
            self.spaces();
            // A quote is nothing at all under this dialect and does not end a
            // run of words, so `w'a*b' x1 y1` reads `b x1 y1` as one run.
            while self.peek() == Some(b'\'') {
                self.at += 1;
                self.spaces();
            }
            // A sign that names nothing does not end a run either, so
            // `@a:hello @ world` asks the field for both words.
            self.skip_lone();
            if !self.at_word() || !self.at_plain_word() {
                return self.attributes(acc);
            }
        }
    }

    /// Whether a word starting here is one of the kinds that binds tightly.
    ///
    /// A prefix, a suffix, an infix, a pattern and a fuzzy word all do, and a
    /// plain word does not, which is the whole of the difference between
    /// `-hello hel*` and `-hello world`.
    fn at_wild_word(&self) -> bool {
        match self.peek() {
            Some(b'%') => true,
            Some(b'*') => matches!(self.src.get(self.at + 1), Some(b) if wordy(*b)),
            // A phrase is a thing of its own and binds to nothing on its left,
            // unless a star is glued to its closing quote and makes it the
            // prefix that `-x "a b"*` leaves out along with the `x`.
            Some(b'"') => self.at_shut_star(),
            // A word with a star on the end is a prefix and binds tightly, and
            // a pattern does the same. A word with a fuzzy or a phrase written
            // after it is only a word, and it is what the fuzzy binds to rather
            // than something that binds to whatever is on its own left.
            Some(b) if wordy(b) || self.escaped(self.at) => {
                self.src.get(self.word_end(self.at)) == Some(&b'*')
                    || self.patternish(self.at).is_some()
            }
            _ => false,
        }
    }

    /// Whether the cursor stands where a run a stopword ended left it, with
    /// nothing but spaces read since.
    fn after_stop(&self) -> bool {
        self.dropped > 0
            && self.dropped <= self.at
            && self.src[self.dropped..self.at]
                .iter()
                .all(u8::is_ascii_whitespace)
    }

    /// Whether a phrase starting here has a star glued to its closing quote.
    fn at_shut_star(&self) -> bool {
        if self.ask.dialect == 1 {
            return false;
        }
        let start = self.at + 1;
        let Some(end) = self.src[start..].iter().position(|b| *b == b'"') else {
            return false;
        };
        end > 0 && self.src.get(start + end + 1) == Some(&b'*')
    }

    /// Whether the word starting here is a plain one rather than a wildcard or a
    /// pattern, which decides whether a run of words carries on past it.
    fn at_plain_word(&self) -> bool {
        match self.src.get(self.word_end(self.at)) {
            // The first dialect has no pattern, so a quote after a word is
            // nothing and the word in front of it is as plain as any other.
            Some(b'\'') => self.ask.dialect == 1,
            Some(b'*' | b'%') => false,
            _ => true,
        }
    }

    /// One indivisible thing, and whatever attributes were hung off it.
    fn atom(&mut self) -> Result<Node, Bad> {
        // Whoever is collecting a run wants the clause for itself, and only for
        // the one thing it asked for rather than for anything nested inside it.
        let holding = std::mem::take(&mut self.holding);
        self.spaces();
        self.skip_lone();
        if self.starts(b"=>") {
            let eq = self.at;
            self.at += 2;
            self.spaces();
            let near = self.arrow(eq);
            self.mark = eq;
            self.word = near.clone();
            return Err(Bad::Syntax { at: eq, near });
        }
        self.lead();
        let node = match self.peek() {
            None => return Err(self.syntax()),
            Some(b'(') => {
                // A query that runs out is answered for at the last thing that
                // was read, and a bracket is one of those things, so
                // `hello ((` points at the second of them.
                self.mark = self.at;
                self.at += 1;
                self.nest()?;
                let held = std::mem::take(&mut self.stop);
                let inner = self.union();
                self.stop = held;
                self.nested -= 1;
                let inner = inner?;
                self.spaces();
                if !self.eat(b')') {
                    return Err(self.syntax());
                }
                inner
            }
            Some(b'"') => self.phrase(b'"')?,
            Some(b'\'') if self.ask.dialect >= 2 && self.closed() => self.phrase(b'\'')?,
            Some(b'$') => self.substituted()?,
            Some(b'@') => self.modified()?,
            Some(b'%') => self.fuzzy()?,
            Some(b'*') => self.star()?,
            Some(b'-' | b'+' | b'.') if self.at_number() => self.plain()?,
            Some(b'-' | b'~') => self.factor()?,
            Some(b) if wordy(b) => self.plain()?,
            Some(b'\\') if self.escaped(self.at) => self.plain()?,
            _ => return Err(self.syntax()),
        };
        if holding {
            return Ok(node);
        }
        self.attributes(node)
    }

    /// `$name` where a word would go, which the second dialect fills in.
    ///
    /// The value takes the place of the text and is then read as text, so a
    /// parameter holding a stopword drops out and one holding an ordinary word
    /// expands the same way a typed word would. The first dialect has no
    /// parameters at all and refuses the dollar sign itself.
    fn substituted(&mut self) -> Result<Node, Bad> {
        let at = self.at;
        self.mark = at;
        self.at += 1;
        let name = self.word();
        if self.ask.dialect == 1 || name.is_empty() {
            self.word = name.clone();
            return Err(Bad::Syntax { at, near: name });
        }
        self.word = name.clone();
        let value: Box<[u8]> = fold(&self.param(&name)).into();
        // A star after the reference makes a prefix of whatever it held, and
        // the value goes in whole, spaces and digits and all.
        if self.eat(b'*') {
            // A parameter that turns into a prefix leaves nothing for a field
            // error later in the query to quote, the same way a pattern does.
            self.hush = true;
            return Ok(Node::new(What::Prefix(value)));
        }
        // A value with nothing in it is a word with nothing in it rather than
        // a stopword or something to stem.
        if value.is_empty() {
            return Ok(Node::term(&value));
        }
        // The value is read the way the same text would have been read where
        // it was written, and a number written there is a token of its own
        // that no stemmer and no stopword list ever sees.
        if number_len(&value, 0) == Some(value.len()) {
            return Ok(Node::term(&value));
        }
        // No stopword list is consulted, so a parameter holding `the` asks for
        // `the` where the same three letters written in the query would have
        // been dropped before they reached the tree.
        if self.ask.verbatim {
            return Ok(Node::term(&value));
        }
        Ok(self.expand(&value))
    }

    /// `*` on its own, or a word with one in front of it.
    fn star(&mut self) -> Result<Node, Bad> {
        self.mark = self.at;
        self.at += 1;
        // A phrase written where the word belongs is that word, so
        // `*"hello world"` is a suffix on both words and the space between them.
        let quote = self.at;
        if let Some(body) = self.glued() {
            if body.is_empty() {
                // A phrase with nothing in it leaves the star standing on its
                // own, so the star has to be allowed to stand there at all:
                // `hello *""` is refused at the star the way `hello *` is, and
                // only a star nothing came before is refused at the quotes.
                self.fresh()?;
                return Err(Bad::Syntax {
                    at: quote,
                    near: Box::default(),
                });
            }
            self.mark = quote;
            self.word = body.clone();
            self.hush = false;
            if self.eat(b'*') {
                return Ok(Node::new(What::Infix(body)));
            }
            return Ok(Node::new(What::Suffix(body)));
        }
        // A parameter stands in for the word the same way it does in a fuzzy
        // match, so `*$B` is a suffix on whatever was passed for `B`. A sign
        // with no name after it is not one and leaves the star on its own.
        let named = matches!(self.src.get(self.at + 1), Some(b) if wordy(*b));
        let word = if self.ask.dialect >= 2 && self.peek() == Some(b'$') && named {
            self.mark = self.at;
            self.at += 1;
            let name = self.word();
            self.word = name.clone();
            self.hush = true;
            self.param(&name)
        } else {
            if !self.at_word() {
                self.fresh()?;
                return Ok(Node::new(What::Wildcard));
            }
            // A star word written out in full leaves its own text for a field
            // error later in the query to quote, where a pattern and a
            // parameter leave nothing, so it undoes what those two did.
            self.hush = false;
            let word = self.word();
            let end = self.at;
            self.bodied(&word, end)
        };
        if self.eat(b'*') {
            return Ok(Node::new(What::Infix(word)));
        }
        Ok(Node::new(What::Suffix(word)))
    }

    /// `%word%`, where the number of signs is how far off a match may be.
    fn fuzzy(&mut self) -> Result<Node, Bad> {
        let mut wide = 0u8;
        while self.eat(b'%') {
            wide = wide.saturating_add(1);
        }
        // Where the last thing read starts, which is what an error names when
        // the closing signs run out: `%%hello` is refused at the word and
        // `%%hello%` at the sign that did arrive.
        let mut start = self.at;
        // The first dialect has no phrase in single quotes and passes the quote
        // over, so `%'A'%` is the fuzzy match `%A%` written the long way.
        while self.ask.dialect == 1 && self.peek() == Some(b'\'') {
            self.at += 1;
            start = self.at;
        }
        // A phrase stands in for the word here too, and an empty one is taken
        // where the same phrase after a star is refused: `%""%` is `FUZZY{}`.
        if let Some(body) = self.glued() {
            // The match is on the words in lower case and the error quotes the
            // phrase as it was written, so `%"A B"` is refused near `A B`.
            let word: Box<[u8]> = fold(&body).into();
            self.word = body;
            for _ in 0..wide {
                let spot = self.at;
                if !self.eat(b'%') {
                    return Err(self.stuck(start));
                }
                start = spot;
            }
            return Ok(Node::new(What::Fuzzy(word, wide)));
        }
        // A parameter stands in for the word, which is why the value it holds
        // is read here rather than anywhere a word could have been written.
        let word: Box<[u8]> = if self.peek() == Some(b'$') {
            let spot = self.at;
            self.mark = spot;
            self.at += 1;
            let name = self.word();
            if name.is_empty() {
                return Err(self.syntax());
            }
            if self.ask.dialect == 1 {
                return Err(self.syntax_at(spot));
            }
            fold(&self.param(&name)).into()
        } else {
            if !self.at_word() {
                return Err(self.syntax_near());
            }
            fold(&self.token()).into()
        };
        for _ in 0..wide {
            let spot = self.at;
            if !self.eat(b'%') {
                return Err(self.stuck(start));
            }
            start = spot;
        }
        Ok(Node::new(What::Fuzzy(word, wide)))
    }

    /// A bare word, which may turn out to be a prefix or a pattern instead.
    fn plain(&mut self) -> Result<Node, Bad> {
        let number = self.number_end();
        let word = match number {
            Some(end) => {
                self.mark = self.at;
                let out: Box<[u8]> = self.src[self.at..end].into();
                self.at = end;
                self.word = out.clone();
                out
            }
            None => self.token(),
        };
        // `w'foo*bar'` is a pattern rather than the word `w` and a phrase, and
        // the only thing that says so is the quote being flush against it.
        // A pattern needs something between its quotes and needs the second
        // quote: `w''` is the word and an empty phrase, and `w'x1` is the two
        // words with the quote passed over.
        let pattern = self.peek() == Some(b'\'')
            && self.src.get(self.at + 1) != Some(&b'\'')
            && self.closed();
        if self.ask.dialect >= 2 && &*word == b"w" && pattern {
            self.at += 1;
            let body = self.at;
            let pattern = self.until(b'\'')?;
            // The pattern is what an error further along the query quotes, and
            // it starts inside the quotes rather than at the `w`.
            self.mark = body;
            self.word = pattern.clone();
            self.hush = true;
            let pattern = self.patterned(pattern);
            return Ok(Node::new(What::Pattern(pattern)));
        }
        let spot = self.mark;
        let end = self.at;
        if self.eat(b'*') {
            // Two stars in a row are a prefix and then a suffix, and are that
            // only when there is a word after the second one: `hel**llo` is two
            // things and `hel**` and `hel***llo` are neither.
            if self.peek() == Some(b'*')
                && !matches!(self.src.get(self.at + 1), Some(b) if wordy(*b))
            {
                return Err(Bad::Syntax {
                    at: self.at,
                    near: word,
                });
            }
            // A prefix is the one thing the word and the star make together,
            // and a query that runs out after it points at where the word
            // began rather than at the star.
            self.mark = spot;
            self.hush = false;
            return Ok(Node::new(What::Prefix(self.bodied(&word, end))));
        }
        // Under the later dialects a number is matched as it stands. Nothing
        // stems it and no stopword list holds it, so it is done here. The first
        // dialect has no number where text goes and reads `5` as the word it
        // looks like, which is expanded the same way any other word is and has
        // the marks around it dropped the same way too, so `.6` is a six.
        if number.is_some() {
            self.skip_splitters();
            if self.ask.dialect >= 2 || self.ask.verbatim {
                return Ok(Node::term(&word));
            }
            let bare = word.strip_suffix(b".").unwrap_or(&word);
            let bare = bare
                .strip_prefix(b".")
                .or_else(|| bare.strip_prefix(b"+"))
                .unwrap_or(bare);
            return Ok(self.expand(bare));
        }
        self.skip_splitters();
        let word: Box<[u8]> = fold(&word).into();
        if self.drops(&word) {
            return Ok(Node::empty());
        }
        // `VERBATIM` is a promise that the words in the query are the words in
        // the documents, so the stemmer never runs and the node stays one term.
        if self.ask.verbatim {
            return Ok(Node::term(&word));
        }
        Ok(self.expand(&word))
    }

    /// The marks that end a word without meaning anything, eaten so that
    /// `ab.cd` is two words rather than one.
    ///
    /// A hyphen is not among them. `ab-cd` is `ab` and not `cd`, which reads
    /// like a mistake and is what a real server does, so a `-` is an operator
    /// wherever it appears rather than punctuation inside a word.
    fn skip_splitters(&mut self) {
        // `=>` is an arrow rather than two splitters, so a word flush against
        // one keeps it for whatever comes after the word.
        while matches!(self.peek(), Some(b) if splitter(b)) && !self.starts(b"=>") {
            // A sign belongs to the number after it, so `ab+3` is a word and
            // the number `+3` rather than a word, a splitter and a three.
            if self.at_number() {
                break;
            }
            self.at += 1;
        }
        // A quote flush against a word is a separator rather than the start of
        // a phrase, which is what leaves `w'FOO*'` as a word and a prefix under
        // the first dialect and one pattern under the second.
        if self.ask.dialect == 1 && self.peek() == Some(b'\'') {
            self.at += 1;
        }
    }

    /// Whether a numeric literal starts at the cursor.
    ///
    /// The first dialect has no number of its own where text goes, so a `+` or
    /// a `.` in front of the digits is a mark that splits words rather than
    /// part of them, and a `-` is the operator unless it is flush against a
    /// digit. That is why `-.5` is a negated five there and `-5` is not.
    fn at_number(&self) -> bool {
        if self.number_end().is_none() {
            return false;
        }
        if self.ask.dialect >= 2 {
            return true;
        }
        let digit = |b: Option<&u8>| matches!(b, Some(b) if b.is_ascii_digit());
        digit(self.src.get(self.at))
            || (self.peek() == Some(b'-') && digit(self.src.get(self.at + 1)))
    }

    /// Where the numeric literal at the cursor ends, if there is one.
    ///
    /// Numbers are their own token because a real server treats them as one:
    /// `3` is a term that never goes near the stemmer, `-3` is one token and
    /// not a negated three, and `3.5.6` is `3.5` next to `.6`. A number that
    /// runs straight into letters, such as `0x1f`, is a word after all, which
    /// is the one case that has to be checked at the end rather than the
    /// start.
    fn number_end(&self) -> Option<usize> {
        number_len(self.src, self.at)
    }

    /// One word, unescaped and as it was typed.
    ///
    /// The case is kept, because half the places a word can appear keep it: a
    /// prefix prints as it was written, a tag value is matched as it was
    /// written, and a parameter name is looked up as it was written. The places
    /// that fold it do so themselves, which is every place a word becomes a
    /// term.
    fn word(&mut self) -> Box<[u8]> {
        self.worded(false)
    }

    /// The same, with a `$` taken as an ordinary letter.
    ///
    /// It is one inside a phrase under the later dialects, where `"a$b"` is one
    /// word rather than a word and a parameter, and it is one nowhere else.
    fn worded(&mut self, dollar: bool) -> Box<[u8]> {
        let start = self.at;
        while let Some(b) = self.peek() {
            if self.escaped(self.at) {
                self.at += 2;
                continue;
            }
            if !wordy(b) && !(dollar && b == b'$') {
                break;
            }
            self.at += 1;
        }
        if self.at > start {
            // What a syntax error further along the query quotes is the word as
            // it was written, backslashes and capitals and all, rather than the
            // word the parser made of it. So `aa\-bb )` is refused near
            // `aa\-bb` and `"BB\A Cc"` near `A`.
            self.word = self.src[start..self.at].into();
        }
        bare(&self.src[start..self.at]).into()
    }

    /// The body of a prefix, a suffix or an infix, which is not quite the word
    /// it was made from.
    ///
    /// A backslash left on the end of it comes off, so `{*x\\}` looks for
    /// anything ending in `x` where the tag value `{x\\}` is a backslash on the
    /// end of an `x`. The one place it stays is a query that ends inside the
    /// word itself, where `*x\\` is the suffix `x\`, and there is no reason for
    /// the difference other than that a real server reads the end of its input
    /// through a different door than it reads the rest.
    fn bodied(&self, body: &[u8], end: usize) -> Box<[u8]> {
        if end >= self.src.len() {
            return body.into();
        }
        let mut cut = body.len();
        while cut > 0 && body[cut - 1] == b'\\' {
            cut -= 1;
        }
        body[..cut].into()
    }

    /// One word that is a token in its own right, which an error can point at.
    fn token(&mut self) -> Box<[u8]> {
        self.mark = self.at;
        self.word()
    }

    /// Everything up to a closing byte, taken as it stands.
    fn until(&mut self, shut: u8) -> Result<Box<[u8]>, Bad> {
        let start = self.at;
        while let Some(b) = self.peek() {
            if b == shut {
                let out: Box<[u8]> = self.src[start..self.at].into();
                self.at += 1;
                return Ok(out);
            }
            self.at += 1;
        }
        Err(self.syntax_at(start))
    }

    /// The stopwords this index drops, or nothing for the built in list.
    fn stopwords(&self) -> Option<&[Box<[u8]>]> {
        self.index.definition.stopwords.as_deref()
    }

    /// Whether a word is thrown away before it reaches the tree.
    ///
    /// `NOSTOPWORDS` turns the whole thing off for one query, which is how a
    /// client searches for the words the index itself would rather not have.
    fn drops(&self, word: &[u8]) -> bool {
        self.ask.stopwords && text::dropped(self.stopwords(), word)
    }

    /// A word and the forms of it the stemmer knows.
    ///
    /// Three nodes when the stem differs from the word and two when it does not,
    /// because the plain expansion of a word that stems to itself is the word
    /// and a real server does not print it twice.
    fn expand(&mut self, word: &[u8]) -> Node {
        let stem: Box<[u8]> = self.stemmer.stem(word).into();
        let mut list = vec![Node::term(word)];
        list.push(Node::new(What::Term(Word {
            word: stem.clone(),
            expanded: true,
            stem: true,
        })));
        if *stem != *word {
            list.push(Node::new(What::Term(Word {
                word: stem,
                expanded: true,
                stem: false,
            })));
        }
        Node::new(What::Union(list))
    }

    /// A quoted phrase, whose words are matched as typed and in order.
    fn phrase(&mut self, quote: u8) -> Result<Node, Bad> {
        let start = self.at;
        self.at += 1;
        let mut list = Vec::new();
        loop {
            self.spaces();
            match self.peek() {
                None => return Err(self.syntax()),
                Some(b) if b == quote => {
                    self.at += 1;
                    break;
                }
                // A `$` inside a phrase is not a parameter. The later
                // dialects take it as a letter and the first refuses the
                // query, quoting the name that followed it, unless there was
                // no name at all and there is nothing to refuse.
                Some(b'$') if self.ask.dialect == 1 => {
                    let spot = self.at;
                    self.at += 1;
                    let name = self.word();
                    if name.is_empty() {
                        continue;
                    }
                    return Err(Bad::Syntax {
                        at: spot,
                        near: name,
                    });
                }
                Some(b) if wordy(b) || b == b'$' || self.escaped(self.at) => {
                    let at = self.at;
                    self.mark = self.at;
                    let word: Box<[u8]> = fold(&self.worded(self.ask.dialect >= 2)).into();
                    // A star glued to the end of a word makes a prefix, and a
                    // prefix in a phrase is refused under the first dialect
                    // where the word started rather than where the star is.
                    // What it quotes is the word as it was typed, capitals and
                    // all, and not the word the phrase was going to look for.
                    if self.ask.dialect == 1 && self.peek() == Some(b'*') {
                        return Err(Bad::Syntax {
                            at,
                            near: self.word.clone(),
                        });
                    }
                    self.skip_splitters();
                    // The one place the dialects disagree about stopwords. The
                    // first refuses the whole query rather than quietly matching
                    // a phrase the client did not ask for, which is defensible
                    // and is also what it does.
                    if self.ask.dialect == 1 && self.drops(&word) {
                        // A phrase is a list of terms, and the grammar the
                        // first dialect is written in only knows how to drop a
                        // stopword out of a list that has two terms in it
                        // already. So `"x y be"` is the two words with the
                        // stopword gone and `"x be y"` is refused at the
                        // stopword, which is the same rule a tag list follows.
                        if list.len() < 2 {
                            return Err(Bad::Syntax {
                                at,
                                near: self.word.clone(),
                            });
                        }
                        continue;
                    }
                    list.push(Node::term(&word));
                }
                // The later dialects read a phrase as the words in it and
                // nothing else, so the punctuation between them is passed
                // over rather than refused.
                _ if self.ask.dialect >= 2 => {
                    self.at += 1;
                    continue;
                }
                _ => return Err(self.syntax_near()),
            }
        }
        // What the client wrote between the quotes, which is the word an error
        // later in the query quotes back and is also what a star glued to the
        // closing quote turns into a prefix, spaces and punctuation and all.
        let body: Box<[u8]> = self.src[start + 1..self.at - 1].into();
        if self.ask.dialect >= 2 {
            // The first dialect quotes the last word of the phrase instead,
            // which is whatever reading the words left behind, and points
            // inside the quotes rather than at them.
            self.word = body.clone();
            self.mark = start;
        }
        // A phrase with nothing in it takes no star, and the star belongs to
        // whatever was written after it instead: `""*ell*` is the empty word
        // beside the infix, where `""*` is the empty word beside a star that
        // has to stand on its own and cannot.
        if self.ask.dialect >= 2 && self.peek() == Some(b'*') && !body.is_empty() {
            self.at += 1;
            // A second star is a suffix on the word after it, the same as it is
            // when the prefix was a plain word rather than a phrase.
            if self.peek() == Some(b'*')
                && !matches!(self.src.get(self.at + 1), Some(b) if wordy(*b))
            {
                return Err(Bad::Syntax {
                    at: self.at,
                    near: body,
                });
            }
            // The body is kept the way it was written, so a phrase in capitals
            // stays in capitals where the words of an ordinary phrase would not.
            self.hush = false;
            return Ok(Node::new(What::Prefix(body)));
        }
        if list.len() == 1 && self.ask.dialect == 1 {
            // A phrase of one word is that word, matched as typed. It is not
            // expanded, which is the whole point of having quoted it.
            return Ok(list.pop().unwrap_or_else(Node::empty));
        }
        if list.is_empty() && self.ask.dialect == 1 {
            return Err(self.syntax_at(self.at - 1));
        }
        // A phrase with nothing at all between the quotes is the empty word,
        // which prints as the quotes themselves. One that held something and
        // came to no words is an exact match on no words, which does not.
        if list.is_empty() && self.at == start + 2 {
            return Ok(Node::term(b""));
        }
        Ok(Node::new(What::Exact(list)))
    }

    /// `@field:` and whatever it applies to.
    ///
    /// The field part is one name or several separated by `|`, and what comes
    /// after the colon decides which grammar the rest is read under: brackets
    /// are a range, braces are a tag, anything else is text.
    fn modified(&mut self) -> Result<Node, Bad> {
        let at = self.at;
        self.at += 1;
        let first = self.word();
        if first.is_empty() {
            // An `@` with no name after it is nothing at all rather than an
            // error, which is a real server being forgiving about a truncated
            // query. A colon after it is different: the query meant to name a
            // field and did not.
            if self.peek() != Some(b':') {
                return Ok(Node::empty());
            }
            return Err(self.syntax());
        }
        self.mark = at;
        self.word = first.clone();
        // The second dialect allows one field modifier over a piece of query,
        // not two, so a modifier inside the reach of another is refused rather
        // than ANDed into it the way the first dialect ANDs them.
        if self.ask.dialect >= 2 && self.inside {
            return Err(Bad::Syntax { at, near: first });
        }
        let mut names = vec![(first, at)];
        // Whitespace is allowed around the bars of a name list and in front of
        // the colon, so `@a | b :x` names the two fields that `@a|b:x` does.
        // A bar with no name after it breaks the list, and a colon written
        // after that is no longer the list's own, which is why `@a|@:x` is
        // refused at the colon rather than read as the one field.
        let mut broke = None;
        loop {
            let save = self.at;
            self.spaces();
            if !self.eat(b'|') {
                self.at = save;
                break;
            }
            let bar = self.at - 1;
            self.spaces();
            // A sign that names nothing is noise in front of a name the same
            // way it is anywhere else, so `@a|@ b:x` names both fields. One
            // that does name something is a modifier of its own, and a
            // modifier is not a name a list will take.
            while self.peek() == Some(b'@')
                && !matches!(self.src.get(self.at + 1), Some(b) if wordy(*b))
            {
                self.at += 1;
                self.spaces();
            }
            let spot = self.at;
            // A name after a bar may be written in quotes, and what is between
            // them is the name, spaces and all, so `@a|"b c":x` asks for a
            // field called `b c` and finds nobody has one. The first dialect
            // has no reading for a quoted name and stops at the quote.
            let name = if matches!(self.peek(), Some(b'"' | b'\'')) {
                match self.glued() {
                    Some(body) => body,
                    None => {
                        broke = Some(bar);
                        break;
                    }
                }
            } else {
                let word = self.word();
                if word.is_empty() {
                    broke = Some(bar);
                    break;
                }
                word
            };
            self.mark = spot;
            self.word = name.clone();
            // The first dialect reads the names after the first as ordinary
            // words, and a word that is a stopword is not a word it will take.
            if self.ask.dialect == 1 && self.drops(&name.to_ascii_lowercase()) {
                return Err(Bad::Syntax {
                    at: spot,
                    near: name,
                });
            }
            names.push((name, spot));
        }
        // The colon may stand away from the name it belongs to, and the space
        // in front of it is only the list's when the colon is really there.
        if broke.is_none() {
            let save = self.at;
            self.spaces();
            if self.peek() != Some(b':') {
                self.at = save;
            }
        }
        if broke.is_some() || !self.eat(b':') {
            // A modifier with no colon after it is a modifier that named a
            // field and then did nothing with it. A field nobody has heard of
            // is worth saying so about, and a real one is a plain syntax error
            // at whatever came instead of the colon.
            let missing = names
                .iter()
                .find(|(name, _)| self.index.field(name).is_none());
            if let Some((name, spot)) = missing.filter(|_| self.ask.dialect >= 2) {
                // A pattern earlier in the query leaves the error nothing to
                // quote here either, so `w'a*b' @flies` names the offset alone.
                return Err(Bad::Unknown {
                    at: *spot,
                    near: (!self.hush).then(|| name.clone()),
                });
            }
            let (name, spot) = &names[names.len() - 1];
            // A list that ran out at a bar is the bar's to answer for when
            // there is nothing after it to quote.
            self.mark = broke.unwrap_or(*spot);
            self.word = name.clone();
            self.spaces();
            // A sign that names nothing is noise here too, so `@a @` runs out
            // of query at the modifier and is refused back at it.
            self.skip_lone();
            return Err(self.stray());
        }
        // A part of a union that may not name a field is being read, and this
        // one is written too deep for the union to end at it, so there is
        // nowhere for it to go. The field itself is never looked up, which is
        // why a name the schema has never heard of is refused as a plain
        // syntax error here rather than as an unknown field.
        if self.barred.is_some() {
            let (name, spot) = names.swap_remove(0);
            return Err(Bad::Syntax {
                at: spot,
                near: name,
            });
        }
        self.spaces();
        // The union being read now has a field of its own, whatever it turns
        // out to narrow, and a later part of that union may name one too.
        self.fielded = true;
        let one = names.len() == 1;
        match self.peek() {
            // A range or a tag is about one field, so a list in front of one is
            // refused at the bracket rather than halfway through it. The names
            // are still read as the text fields a list is made of, so a list
            // holding a tag or a numeric field is refused at that field first.
            Some(b'[' | b'{') if !one => {
                for (name, at) in &names {
                    self.expect(name, *at, Want::Text)?;
                }
                Err(self.syntax())
            }
            Some(b'[') => self.bracket(&names[0].0, names[0].1),
            Some(b'{') => self.tag(&names[0].0, names[0].1),
            _ => self.scoped(&names),
        }
    }

    /// A modifier over text, which narrows everything it reaches to a field set.
    fn scoped(&mut self, names: &[(Box<[u8]>, usize)]) -> Result<Node, Bad> {
        let mut mask: Mask = 0;
        for (name, at) in names {
            // A field the schema has never heard of is answered for before
            // anything the modifier was written in front of is read at all,
            // where a field that is there and holds the wrong thing waits.
            self.known(name, *at)?;
            // The first dialect takes the fields that can hold text and drops
            // the rest, which leaves a query asking for no field at all when
            // none of them can.
            if let Some(b) = bit(self.index, name) {
                mask |= b;
            }
        }
        // What the names are is settled once the modifier has read everything
        // it reaches over, so anything written wrong in there is answered for
        // first. A pattern inside the reach is not one of the patterns that
        // leaves an error nothing to quote, so what the query looked like at
        // the name is what the answer is written from.
        let hushed = self.hush;
        // `@a:*` is the one thing a modifier will not take, and it is refused
        // at the star while still quoting the field name.
        let after = self.src.get(self.at + 1).copied();
        let suffix = matches!(after, Some(b) if wordy(b))
            || (self.ask.dialect >= 2
                && after == Some(b'$')
                && matches!(self.src.get(self.at + 2), Some(b) if wordy(*b)))
            // A phrase stands in for the word, so `@a:*"hello world"` is a
            // suffix on the field the same way `@a:*hello` is.
            || (self.ask.dialect >= 2 && matches!(after, Some(b'"' | b'\'')));
        if self.peek() == Some(b'*') && !suffix {
            return Err(Bad::Syntax {
                at: self.at,
                near: self.word.clone(),
            });
        }
        let held = self.inside;
        self.inside = true;
        // A modifier takes the whole of a first dialect union and one thing
        // under the later ones, where one thing still takes in whatever binds
        // tighter than a space, so `@a:hello hel*` narrows both words.
        let head = self.gone.len();
        let mut chain = false;
        let node = if self.ask.dialect == 1 {
            self.scoped_union()
        } else {
            // An attribute clause ends what the modifier reaches over, and
            // nothing else does, so `@a:hello => {$weight: 2;}hel*` asks the
            // field for the one word and every field for the prefix.
            self.hung = false;
            self.run().and_then(|node| {
                if self.hung {
                    return Ok(node);
                }
                chain = self.sticky();
                self.tighter(node, head)
            })
        };
        self.inside = held;
        let mut node = match node {
            Ok(node) => node,
            Err(bad) => {
                // A vector clause written inside a bracket is answered for
                // after the field is, unless something bound itself to the
                // modifier and had its own say first.
                if std::mem::take(&mut self.arrowed) && !chain {
                    self.hush = hushed;
                    for (name, at) in names {
                        self.expect(name, *at, Want::Text)?;
                    }
                }
                return Err(bad);
            }
        };
        // An arrow written straight after something that bound itself to the
        // modifier is read before the names are settled, so
        // `@n:hello hel*=>[KNN 2 @v $B]` is refused at the arrow where
        // `@n:hello=>[KNN 2 @v $B]` is refused at the field. An arrow the
        // modifier could take is gone by now, having been read as part of what
        // it was written after.
        self.spaces();
        if !(chain && self.starts(b"=>")) {
            let carried = std::mem::replace(&mut self.hush, hushed);
            for (name, at) in names {
                self.expect(name, *at, Want::Text)?;
            }
            self.hush = carried;
        }
        // A word with nothing in it cannot be asked of a text field unless the
        // field was created with `INDEXEMPTY`, and the whole query is refused
        // rather than the one word dropped.
        if self.ask.dialect >= 2 && hollow(&node) {
            return Err(Bad::Plain(NO_EMPTY));
        }
        node.narrow(mask);
        settle(&mut node);
        Ok(node)
    }

    /// What a first dialect modifier reaches over.
    ///
    /// A union whose parts are single things rather than sequences, which is
    /// what stops `@a:hello (world)` narrowing the bracket and lets
    /// `@a:hello|world` narrow both sides.
    fn scoped_union(&mut self) -> Result<Node, Bad> {
        let mut acc = self.factor()?;
        loop {
            self.spaces();
            // A sign that names nothing stands between the parts of a union
            // without ending it, so `@a:hello @|world` narrows both sides.
            self.skip_lone();
            // A stopword ends what the modifier reaches over, and the bar after
            // it belongs to whoever reads the rest of the sequence.
            if self.after_stop() {
                break;
            }
            if !self.eat(b'|') {
                break;
            }
            let next = self.factor()?;
            acc = Node::fold(acc, next, false, false);
        }
        Ok(acc)
    }

    /// `[` after a field, which is three different grammars sharing a bracket.
    ///
    /// Which one it is comes from what is inside rather than from what the
    /// field holds: two numbers are a range and four things ending in a unit
    /// are a circle, whatever the schema says. The field is checked against
    /// that afterwards, which is why naming a text field in front of `[1 10]`
    /// complains about a numeric field rather than about the brackets.
    fn bracket(&mut self, name: &[u8], at: usize) -> Result<Node, Bad> {
        self.mark = self.at;
        self.at += 1;
        self.spaces();
        if self.starts(b"VECTOR_RANGE") || self.starts(b"vector_range") {
            let spot = self.at;
            let word = self.token();
            if self.ask.dialect == 1 {
                return Err(Bad::Syntax {
                    at: spot,
                    near: word,
                });
            }
            self.expect(name, at, Want::Vector)?;
            return self.vector_range(name);
        }
        let items = self.items()?;
        match items.len() {
            // One number is a range with the same number at both ends, which is
            // the second dialect's shorthand and not something the first knows.
            1 if self.ask.dialect >= 2 => self.range(name, at, &items, true),
            2 => self.range(name, at, &items, false),
            4 => self.circle(name, at, &items),
            _ => Err(Bad::Syntax {
                at: self.shut,
                near: self.word.clone(),
            }),
        }
    }

    /// The pieces inside a `[`, read as words and turned into numbers after.
    ///
    /// They are read first and understood second because that is where a real
    /// server points when one of them is not a number: `[a b]` is refused at
    /// `b`, which is the last thing it read rather than the first thing that
    /// was wrong.
    fn items(&mut self) -> Result<Vec<Item>, Bad> {
        let mut items = Vec::new();
        loop {
            self.spaces();
            match self.peek() {
                None => return Err(self.syntax()),
                Some(b']') => {
                    self.shut = self.at;
                    self.at += 1;
                    return Ok(items);
                }
                Some(b) if wordy(b) || matches!(b, b'(' | b'$' | b'+' | b'-' | b'.') => {
                    // A circle is the longest thing a bracket holds, so a fifth
                    // item is refused where it stands rather than at the close.
                    if items.len() == 4 {
                        let spot = self.at;
                        let word = self.number_word();
                        return Err(Bad::Syntax {
                            at: spot,
                            near: word,
                        });
                    }
                    let open = self.eat(b'(');
                    self.spaces();
                    let spot = self.at;
                    let word = if self.peek() == Some(b'$') {
                        self.mark = self.at;
                        self.at += 1;
                        let name = self.word();
                        if name.is_empty() || self.ask.dialect == 1 {
                            return Err(Bad::Syntax {
                                at: spot,
                                near: name,
                            });
                        }
                        self.param(&name)
                    } else {
                        self.number_word()
                    };
                    if word.is_empty() {
                        return Err(self.syntax());
                    }
                    self.mark = spot;
                    self.word = word.clone();
                    items.push((word, open, spot));
                }
                _ => return Err(self.syntax()),
            }
        }
    }

    /// Where a bracket item that is not a number is refused.
    ///
    /// The first dialect turns each item into a number as it reads it, so it
    /// stops at the one that went wrong. The second reads them all first and
    /// then converts, so it is still pointing at the last one it read.
    fn not_a_number(&self, items: &[(Box<[u8]>, bool, usize)], spot: usize) -> Bad {
        if self.ask.dialect == 1 {
            let near = items.iter().find(|i| i.2 == spot).map(|i| i.0.clone());
            return Bad::Syntax {
                at: spot,
                near: near.unwrap_or_default(),
            };
        }
        match items.last() {
            Some((word, _, at)) => Bad::Syntax {
                at: *at,
                near: word.clone(),
            },
            None => Bad::Syntax {
                at: spot,
                near: Box::default(),
            },
        }
    }

    /// A number as it is written inside brackets, which is a word with a sign
    /// and a point allowed in it.
    fn number_word(&mut self) -> Box<[u8]> {
        let start = self.at;
        while matches!(self.peek(), Some(b) if wordy(b) || matches!(b, b'+' | b'-' | b'.')) {
            self.at += 1;
        }
        let out: Box<[u8]> = self.src[start..self.at].into();
        if !out.is_empty() {
            self.word = out.clone();
        }
        out
    }

    /// `[min max]`, with either end open and either end infinite.
    fn range(
        &mut self,
        name: &[u8],
        at: usize,
        items: &[(Box<[u8]>, bool, usize)],
        alone: bool,
    ) -> Result<Node, Bad> {
        let mut ends = Vec::new();
        for (word, open, spot) in items {
            let text = String::from_utf8_lossy(word).into_owned();
            let Some(value) = number(&text) else {
                return Err(self.not_a_number(items, *spot));
            };
            ends.push((value, *open));
        }
        let (min, min_open) = ends[0];
        let (max, max_open) = if alone { ends[0] } else { ends[1] };
        self.expect(name, at, Want::Numeric)?;
        if self.index.field(name).is_none() {
            return Ok(Node::empty());
        }
        let range = Range {
            field: name.into(),
            min,
            max,
            min_open,
            max_open,
        };
        Ok(Node::new(What::Numeric(range)))
    }

    /// `[lon lat radius unit]`.
    fn circle(
        &mut self,
        name: &[u8],
        at: usize,
        items: &[(Box<[u8]>, bool, usize)],
    ) -> Result<Node, Bad> {
        let mut ends = Vec::new();
        for (word, _, spot) in &items[..3] {
            let text = String::from_utf8_lossy(word).into_owned();
            let Some(value) = number(&text) else {
                return Err(self.not_a_number(items, *spot));
            };
            ends.push(value);
        }
        let unit = items[3].0.clone();
        self.expect(name, at, Want::Geo)?;
        // The unit is checked after the field under the second dialect and
        // before it under the first, which is the order each of them resolves
        // the clause in rather than a rule about units.
        if !UNITS.iter().any(|u| unit.eq_ignore_ascii_case(u)) {
            if self.ask.dialect == 1 {
                return Err(Bad::Plain(BAD_UNIT));
            }
            self.unit = true;
        }
        if self.index.field(name).is_none() {
            return Ok(Node::empty());
        }
        let circle = Circle {
            field: name.into(),
            lon: ends[0],
            lat: ends[1],
            radius: ends[2],
            unit,
        };
        Ok(Node::new(What::Geo(circle)))
    }

    /// `[VECTOR_RANGE radius $param]`.
    fn vector_range(&mut self, name: &[u8]) -> Result<Node, Bad> {
        self.spaces();
        let spot = self.at;
        let word = self.number_word();
        let text = String::from_utf8_lossy(&word).into_owned();
        let Some(radius) = number(&text) else {
            return Err(self.syntax_at(spot));
        };
        self.spaces();
        let param = self.dollar()?;
        self.param(&param);
        self.spaces();
        if !self.eat(b']') {
            return Err(self.syntax());
        }
        let vector = Vector {
            field: name.into(),
            param,
            k: None,
            radius: Some(radius),
            alias: None,
            options: Vec::new(),
            over: None,
        };
        Ok(Node::new(What::Vector(Box::new(vector))))
    }

    /// `{a|b}` after a tag field.
    ///
    /// Values separated by spaces are one intersection, values separated by
    /// `|` are alternatives, and the two nest that way round. Two rules sit on
    /// top of that and neither is obvious. A value with a `*` in it cannot be
    /// in a run with another value, and under the first dialect a stopword
    /// cannot either. The first dialect reports both at the closing brace and
    /// the second reports them at the value that broke the rule, which is the
    /// difference between finishing the list and giving up on it.
    fn tag(&mut self, name: &[u8], at: usize) -> Result<Node, Bad> {
        self.expect(name, at, Want::Tag)?;
        self.mark = self.at;
        self.at += 1;
        let mut list: Vec<Node> = Vec::new();
        let mut run: Vec<Node> = Vec::new();
        let mut odd = false;
        // Whether the one value the list has so far is a stopword, which is a
        // value nothing else may sit beside in the first dialect.
        let mut stopped = false;
        let mut refused = false;
        let mut more = false;
        // Whether the list ended at a modifier rather than at its brace, which
        // is the one ending that lets a stopword sit beside another value.
        let mut cut = false;
        let mut shut;
        'values: loop {
            self.spaces();
            // A mark that splits words splits tag values too, so `{a.b}` is two
            // values and `{,x}` is one with the comma gone. An arrow is not one
            // of them and is refused where it starts.
            while matches!(self.peek(), Some(b) if splitter(b)) && !self.at_number() {
                if self.starts(b"=>") {
                    // The first dialect ends the list where the arrow starts
                    // and leaves the arrow to whoever reads attributes, the
                    // same way it ends a list that simply ran out.
                    if self.ask.dialect == 1 {
                        // A list carried on past its brace and then handed an
                        // arrow instead of a value is refused at the arrow,
                        // where a list that got its value ends quietly and
                        // leaves the clause to whoever reads attributes.
                        if run.is_empty() {
                            let eq = self.at;
                            self.at += 2;
                            self.spaces();
                            return Err(Bad::Syntax {
                                at: eq,
                                near: self.arrow(eq),
                            });
                        }
                        shut = self.mark;
                        break 'values;
                    }
                    let eq = self.at;
                    self.at += 2;
                    return Err(Bad::Syntax {
                        at: eq,
                        near: self.arrow(eq),
                    });
                }
                self.at += 1;
                self.spaces();
            }
            // A list carried on past its brace takes a plain word beside a
            // value it already has and nothing else. A pattern, a stopword or
            // anything that is not a word at all ends the tag where it stands
            // and is read as ordinary text after it, so `@g:{a}|b c*` is the
            // two values with a prefix beside them rather than a refusal. A
            // bar carries the list on whatever went before it.
            if more
                && !run.is_empty()
                && !matches!(self.peek(), Some(b'|' | b'}') | None)
                && (odd || !self.at_plain())
            {
                shut = self.mark;
                break;
            }
            // A list whose only value is a stopword takes nothing beside it,
            // so `@g:{be x}` is the tag `be` with text after it and the brace
            // on the end belongs to nobody.
            if stopped && !matches!(self.peek(), Some(b'|' | b'}') | None) {
                shut = self.at;
                cut = true;
                break;
            }
            let mut spot = self.at;
            let node = match self.peek() {
                // The first dialect takes a tag that was never closed and
                // reads it as if the brace were there, which is forgiving in
                // a way nothing else in the grammar is.
                None if self.ask.dialect == 1 => {
                    // A list that ran out is refused at the last thing that
                    // was read, which is the brace it opened with or the bar
                    // it was in the middle of.
                    shut = self.mark;
                    break;
                }
                None => return Err(self.syntax()),
                // A list carried on past the brace ends where it runs out of
                // values, and the bracket it stops at belongs to whoever
                // opened it rather than to the tag.
                Some(b')') if more => {
                    shut = self.at;
                    break;
                }
                Some(b'}') => {
                    shut = self.at;
                    self.mark = self.at;
                    self.at += 1;
                    // Under the first dialect a `|` written after the brace
                    // carries the list on, so `@g:{a}|b` is one tag of two
                    // values rather than a tag or a word.
                    let Some(bar) = self.carried() else { break };
                    // A list that has already gone wrong is refused at the
                    // brace it closed with rather than wherever the values
                    // carried on past it stop being values.
                    if refused || (odd && run.len() > 1) {
                        return Err(Bad::Syntax {
                            at: shut,
                            near: self.word.clone(),
                        });
                    }
                    self.at = bar + 1;
                    self.mark = bar;
                    more = true;
                    odd = false;
                    stopped = false;
                    close_run(&mut list, &mut run);
                    continue;
                }
                // A modifier is not a tag value in any dialect, and it is
                // refused where it starts, quoting the name it gave rather
                // than the value before it.
                Some(b'@') => {
                    let at = self.at;
                    self.at += 1;
                    // A sign with no name after it names no field and is passed
                    // over, the same way one written outside a tag is, so
                    // `@g:{a @}` is the one value `a`.
                    let name = self.word();
                    if name.is_empty() {
                        continue;
                    }
                    // The first dialect ends the list where the modifier starts
                    // and reads the modifier as ordinary text after the tag, so
                    // `@g:{a @b c}` is refused at `c` the way `@b c` is. The
                    // later ones refuse the modifier where it starts, quoting
                    // the value before it.
                    if self.ask.dialect != 1 {
                        return Err(self.syntax_at(spot));
                    }
                    self.at = at;
                    self.word = name;
                    shut = at;
                    cut = true;
                    break;
                }
                // The first dialect has no parameters, so one here goes the
                // same way. The later ones take the value whole, spaces and
                // all, and let a `*` after it make a prefix of it.
                Some(b'$') => {
                    self.at += 1;
                    let name = self.word();
                    // A sign with no name after it is passed over here the same
                    // way it is outside a tag, so `@g:{a $ b}` is the two
                    // values.
                    if name.is_empty() {
                        continue;
                    }
                    if self.ask.dialect == 1 {
                        return Err(self.syntax_at(spot));
                    }
                    let value = self.param(&name);
                    self.word = value.clone();
                    self.mark = spot;
                    if self.eat(b'*') {
                        odd = true;
                        self.hush = true;
                        Node::new(What::Prefix(value))
                    } else {
                        Node::term(&value)
                    }
                }
                // The first dialect ends the list where something begins that
                // is a term but cannot be a value, and reads it as ordinary
                // text after the tag the same way it does a modifier. So
                // `@g:{aa-bb}` is a tag beside a NOT, and the brace on the end
                // belongs to nobody by then, which is where it is refused. The
                // five are every mark the grammar lets a term start with.
                Some(b'"' | b'%' | b'(' | b'-' | b'~')
                    if self.ask.dialect == 1 && !run.is_empty() =>
                {
                    shut = self.at;
                    cut = true;
                    break;
                }
                Some(b'|') => {
                    // An alternative with nothing in it is refused at the bar,
                    // which is the one place in a tag that points forwards
                    // rather than back at the value that went wrong.
                    if run.is_empty() {
                        return Err(Bad::Syntax {
                            at: spot,
                            near: self.word.clone(),
                        });
                    }
                    self.mark = self.at;
                    self.at += 1;
                    refused |= odd && run.len() > 1;
                    odd = false;
                    stopped = false;
                    close_run(&mut list, &mut run);
                    continue;
                }
                Some(b'"') if self.ask.dialect >= 2 => {
                    self.at += 1;
                    // A phrase that never closes is refused at the quote it
                    // opened with, where one written outside a tag is refused
                    // at the word after the quote.
                    let Ok(word) = self.until(b'"') else {
                        return Err(self.syntax_at(spot));
                    };
                    self.word = word.clone();
                    // A phrase with a star after it is a prefix on the whole
                    // phrase, spaces and all, so `{"a b"*}` looks for a tag
                    // that starts `a b`.
                    if self.eat(b'*') {
                        odd = true;
                        self.hush = false;
                        Node::new(What::Prefix(self.bodied(&bare(&word), self.at)))
                    } else {
                        Node::term(&word)
                    }
                }
                // A single quote holds a value the same way a double one does,
                // and one that never closes is passed over rather than read,
                // so `@g:{w'}` is the one value `w`.
                Some(b'\'') if self.ask.dialect >= 2 => {
                    if !self.closed() {
                        self.at += 1;
                        continue;
                    }
                    self.at += 1;
                    let word = self.until(b'\'')?;
                    self.word = word.clone();
                    if self.eat(b'*') {
                        odd = true;
                        self.hush = false;
                        Node::new(What::Prefix(self.bodied(&bare(&word), self.at)))
                    } else {
                        Node::term(&word)
                    }
                }
                Some(b'*') => {
                    self.mark = self.at;
                    self.at += 1;
                    odd = true;
                    // A phrase stands in for the value the way it does outside a
                    // tag, so `@g:{*"a b"}` is a suffix on both words.
                    let named = matches!(self.src.get(self.at + 1), Some(b) if wordy(*b));
                    let word = if let Some(body) = self.glued() {
                        // A phrase with nothing in it leaves the star with no
                        // value, and a star with no value inside a tag is
                        // refused at the star itself.
                        if body.is_empty() {
                            return Err(self.syntax_at(spot));
                        }
                        self.hush = false;
                        body
                    } else if self.ask.dialect >= 2 && self.peek() == Some(b'$') && named {
                        self.at += 1;
                        let name = self.word();
                        self.hush = true;
                        self.param(&name)
                    } else {
                        if !self.at_word() {
                            return Err(self.syntax_at(spot));
                        }
                        self.hush = false;
                        self.tag_word()
                    };
                    self.word = word.clone();
                    // A value that is matched against the front or the back of
                    // a tag loses its escapes where a whole value keeps them,
                    // so `{a\-b}` looks for `a\-b` and `{*a\-b}` looks for
                    // anything ending `a-b`.
                    let body = self.bodied(&bare(&word), self.at);
                    // A star on both ends of the value matches anywhere inside
                    // it, the same way it does outside a tag.
                    if self.eat(b'*') {
                        Node::new(What::Infix(body))
                    } else {
                        Node::new(What::Suffix(body))
                    }
                }
                Some(b) if wordy(b) || self.at_number() || self.escaped(self.at) => {
                    let word = self.tag_value();
                    if word.is_empty() {
                        return Err(self.syntax_at(spot));
                    }
                    self.mark = spot;
                    self.word = word.clone();
                    if self.eat(b'*') {
                        odd = true;
                        self.hush = false;
                        Node::new(What::Prefix(self.bodied(&bare(&word), self.at)))
                    } else if self.ask.dialect >= 2
                        && &*word == b"w"
                        && self.src.get(self.at + 1) != Some(&b'\'')
                        && self.closed()
                        && self.eat(b'\'')
                    {
                        // A pattern is a value no other value may sit beside,
                        // and it is refused at its body rather than at the `w`
                        // that introduced it.
                        odd = true;
                        spot = self.at;
                        let pattern = self.until(b'\'')?;
                        self.mark = spot;
                        self.word = pattern.clone();
                        Node::new(What::Pattern(self.patterned(pattern)))
                    } else if self.ask.dialect == 1 && self.drops(&fold(&word)) {
                        // A run of values is a phrase, and the first dialect
                        // drops a stopword out of a phrase that has two words
                        // in it already. One that arrives before that ends the
                        // list instead, so `@g:{x be y}` is the tag `x` with
                        // text after it and `@g:{x y be z}` is the three words
                        // that are not stopwords.
                        if run.len() >= 2 {
                            continue;
                        }
                        if run.is_empty() {
                            stopped = true;
                            Node::term(&word)
                        } else {
                            self.at = spot;
                            shut = spot;
                            cut = true;
                            break;
                        }
                    } else {
                        Node::term(&word)
                    }
                }
                _ => return Err(self.syntax()),
            };
            // A pattern next to another value is refused where a plain value
            // next to another value is not, and the second dialect says so as
            // soon as the second one arrives.
            if self.ask.dialect >= 2 && odd && !run.is_empty() {
                return Err(Bad::Syntax {
                    at: spot,
                    near: self.word.clone(),
                });
            }
            run.push(node);
        }
        // A stopword beside another value only sinks a list that reached its
        // brace, so `@g:{a b}` is refused where `@g:{a b @c}` hands the tag over
        // whole and is refused at the modifier after it.
        refused |= odd && run.len() > 1 && !cut;
        if refused || run.is_empty() {
            return Err(Bad::Syntax {
                at: shut,
                near: self.word.clone(),
            });
        }
        close_run(&mut list, &mut run);
        if self.index.field(name).is_none() {
            return Ok(Node::empty());
        }
        Ok(Node::new(What::Tag(name.into(), list)))
    }

    /// Where the `|` is that carries a tag list on past its closing brace, if
    /// there is one.
    ///
    /// Only the first dialect does this, and it does it across spaces on both
    /// sides, so `@g:{a} | b` is the same one tag as `@g:{a}|b`.
    fn carried(&self) -> Option<usize> {
        if self.ask.dialect != 1 {
            return None;
        }
        let mut at = self.at;
        loop {
            while matches!(self.src.get(at), Some(b' ' | b'\t' | b'\n' | b'\r')) {
                at += 1;
            }
            // A sign that names nothing is noise between the brace and the bar
            // the same way it is anywhere else, so `@g:{a} @|b` carries the
            // list on and `@g:{a} @b` hands it a modifier and ends it.
            if self.src.get(at) == Some(&b'@')
                && !matches!(self.src.get(at + 1), Some(b) if wordy(*b))
            {
                at += 1;
                continue;
            }
            break;
        }
        (self.src.get(at) == Some(&b'|')).then_some(at)
    }

    /// Whether a plain value stands at the cursor, which is a word that is
    /// neither a prefix nor a stopword.
    ///
    /// A value that is one of those two is the value no other value may sit
    /// beside, and a list carried on past its brace ends in front of it rather
    /// than being refused for it.
    fn at_plain(&mut self) -> bool {
        if !self.at_word() && !self.at_number() {
            return false;
        }
        let save = self.at;
        let word = self.tag_value();
        let plain = !word.is_empty()
            && self.peek() != Some(b'*')
            && !self.drops(&word.to_ascii_lowercase());
        self.at = save;
        plain
    }

    /// One value inside braces, which is a number where there is one and a word
    /// otherwise.
    ///
    /// `{5.5}` is one value and `{a.b}` is two, because the dot is part of a
    /// number and is a separator anywhere else. The first dialect drops a dot
    /// left on the end of a number the same way it drops one in front.
    fn tag_value(&mut self) -> Box<[u8]> {
        if !self.at_number() {
            return self.tag_word();
        }
        let end = self.number_end().unwrap_or(self.at);
        // A number and a word both start here and the longer one wins, which is
        // why `{1\-a}` is the one word it looks like while `{5.5\-a}` is the
        // number `5.5` beside the word `-a`: the escape is part of a word and
        // is not part of a number, so it only carries the value on when the
        // word it makes reaches further than the number does.
        if self.word_end(self.at) > end {
            return self.tag_word();
        }
        let word = &self.src[self.at..end];
        self.at = end;
        if self.ask.dialect == 1 {
            return word.strip_suffix(b".").unwrap_or(word).into();
        }
        word.into()
    }

    /// A tag value, which keeps its case and the escapes that were written in
    /// it, because both of those are matched rather than folded away.
    ///
    /// The escape reaches the same bytes it reaches anywhere else, so `{aa\-bb}`
    /// is one value and `{aa\xbb}` is two: the backslash in front of a letter
    /// escapes nothing, ends the value where it stands and is thrown away, and
    /// the letters after it are a value of their own.
    fn tag_word(&mut self) -> Box<[u8]> {
        let start = self.at;
        self.at = self.word_end(self.at);
        self.src[start..self.at].into()
    }

    /// `=> {$weight: 2;}` and the rest of the attributes a term can carry.
    ///
    /// More than one clause may follow one thing, so this reads them until
    /// there are no more, which is also what puts a vector clause written after
    /// an attribute clause in front of the check that refuses it.
    fn attributes(&mut self, node: Node) -> Result<Node, Bad> {
        let mut node = node;
        let mut any = false;
        loop {
            match self.attribute(node)? {
                (out, false) => {
                    // Whatever was read last is what a field modifier asks
                    // about, and the answer is this one rather than the answer
                    // for anything nested inside it, because the thing on the
                    // outside is always read last.
                    self.hung = any;
                    return Ok(out);
                }
                (out, true) => {
                    any = true;
                    node = out;
                }
            }
        }
    }

    /// The error for an arrow with something after it that is not a bracket.
    ///
    /// What gets quoted is the word of whatever token comes next, so `=> b` and
    /// `=> "b"` and `=> @b:c` and `=> hel*` all quote a word and point at where
    /// that token starts. A token with no word in it, a bracket or a `%` or a
    /// `-`, leaves the arrow quoted instead, and when there is nothing left at
    /// all the arrow is also what the error points at.
    fn after_arrow(&mut self) -> Bad {
        self.lead();
        let at = self.at;
        let sigil = matches!(self.peek(), Some(b'@' | b'$' | b'\''))
            || (self.peek() == Some(b'"') && self.ask.dialect >= 2)
            || (self.peek() == Some(b'*') && matches!(self.src.get(at + 1), Some(b) if wordy(*b)));
        if sigil {
            self.at += 1;
        }
        if self.at_word() {
            let word = self.word();
            self.word = word.clone();
            return Bad::Syntax { at, near: word };
        }
        self.at = at;
        self.syntax()
    }

    /// What an error quotes back when the arrow at `eq` is what went wrong.
    ///
    /// It is the tail of the arrow and whatever follows it, which is the spaces
    /// when there are any and the one byte flush against it when there are
    /// none, so `=> {` quotes `> ` and `=>{` quotes `>{`.
    fn arrow(&self, eq: usize) -> Box<[u8]> {
        let end = if self.at == eq + 2 {
            (eq + 3).min(self.src.len())
        } else {
            self.at
        };
        self.src[eq + 1..end].into()
    }

    /// One attribute clause, and whether there may be another after it.
    fn attribute(&mut self, node: Node) -> Result<(Node, bool), Bad> {
        let save = self.at;
        self.spaces();
        // A sign that names nothing is noise here as it is anywhere else, so
        // `hello @ => {$weight: 0.5;}` weighs `hello` rather than being refused
        // at the arrow. Nothing is given up by looking past it, because the
        // cursor goes back where it was when no arrow turns up.
        self.skip_lone();
        if !self.starts(b"=>") {
            self.at = save;
            return Ok((node, false));
        }
        let eq = self.at;
        self.at += 2;
        self.spaces();
        // A star carries no attributes and no vector clause either. Inside a
        // bracket, and under the first dialect anywhere, the arrow itself is
        // what the error names, whatever was written after it.
        let star = matches!(node.what, What::Wildcard);
        if star && (self.deep > 1 || self.ask.dialect == 1) {
            return Err(Bad::Syntax {
                at: eq,
                near: self.arrow(eq),
            });
        }
        // A vector clause hangs off a whole query, so one written inside a
        // bracket is refused where a real server refuses it, at the bracket
        // itself and quoting the arrow along with it. The one at the top of a
        // query is the caller's business rather than this one's.
        if self.peek() == Some(b'[') && self.deep > 1 {
            self.arrowed = true;
            return Err(Bad::Syntax {
                at: self.at,
                near: self.arrow(eq),
            });
        }
        if self.peek() != Some(b'{') {
            // Inside a bracket there is nobody further out to read whatever the
            // arrow was written in front of, so it is refused here rather than
            // left for a closing bracket that is not going to come. With
            // nothing after it to quote, the arrow itself is quoted.
            if self.deep > 1 {
                self.mark = self.at;
                self.word = self.arrow(eq);
                return Err(self.after_arrow());
            }
            self.at = save;
            return Ok((node, false));
        }
        // The later dialects say a star carries no attributes at the brace.
        if star {
            return Err(Bad::Syntax {
                at: self.at,
                near: self.arrow(eq),
            });
        }
        self.mark = self.at;
        self.word = self.arrow(eq);
        self.at += 1;
        let mut node = node;
        loop {
            self.spaces();
            match self.peek() {
                None => return Err(self.syntax()),
                Some(b'}') => {
                    self.at += 1;
                    break;
                }
                Some(b';') => {
                    self.at += 1;
                }
                Some(b'$') => {
                    // Read by hand rather than through `dollar`, so that a sign
                    // with no name after it leaves the last token where it was.
                    // A name is what makes this an attribute at all: `{$}` and
                    // `{$;}` hold nothing and are not an error, and only a colon
                    // after the empty name is.
                    let at = self.at;
                    self.at += 1;
                    let name = self.word();
                    if name.is_empty() {
                        self.spaces();
                        if self.peek() == Some(b':') {
                            return Err(Bad::Syntax {
                                at: self.at,
                                near: self.word.clone(),
                            });
                        }
                        continue;
                    }
                    self.mark = at;
                    self.word = name.clone();
                    self.spaces();
                    if !self.eat(b':') {
                        return Err(self.stray());
                    }
                    self.spaces();
                    let value = self.attribute_value();
                    if value.is_empty() {
                        return Err(self.syntax());
                    }
                    self.apply(&mut node, &name, &value)?;
                }
                _ => return Err(self.stray()),
            }
        }
        Ok((node, true))
    }

    /// The text of one attribute, which is one word and not the rest of the
    /// clause.
    ///
    /// `{$weight: 2 3}` is refused at the `3` rather than read as the value
    /// `2 3`, so what a value runs to is a space as much as a semicolon.
    fn attribute_value(&mut self) -> Box<[u8]> {
        let start = self.at;
        while !matches!(
            self.peek(),
            None | Some(b';' | b'}' | b' ' | b'\t' | b'\n' | b'\r')
        ) {
            self.at += 1;
        }
        let out: Box<[u8]> = self.src[start..self.at].into();
        if !out.is_empty() {
            self.mark = start;
            self.word = out.clone();
        }
        out
    }

    /// The error for something in an attribute clause that has no business
    /// there, which quotes the word it starts with when it starts with one.
    ///
    /// `{$weight 2}` is refused at the `2` and quotes it, because the value was
    /// read before anything noticed that the colon in front of it was missing.
    fn stray(&mut self) -> Bad {
        let at = self.at;
        // A pattern is a thing of its own and is quoted by what is between its
        // quotes, so `@a w'a*b'` is refused at the body rather than at the `w`.
        if let Some((body, near)) = self.patternish(at) {
            self.mark = body;
            self.word = near.clone();
            return Bad::Syntax { at: body, near };
        }
        if self.at_word() {
            let word = self.word();
            self.at = at;
            self.mark = at;
            self.word = word.clone();
            return Bad::Syntax { at, near: word };
        }
        // An arrow quotes itself and the little that follows it, the same way
        // it does when it turns up where a vector clause cannot go, so
        // `@a => {$weight: 2;}` is refused at the arrow rather than at `a`.
        if self.starts(b"=>") {
            let eq = at;
            self.at += 2;
            let near = self.arrow(eq);
            self.at = at;
            self.mark = eq;
            self.word = near.clone();
            return Bad::Syntax { at: eq, near };
        }
        // A modifier and a parameter are each a thing of their own and quote
        // the name they gave rather than the word before them, so `@a @b:x` is
        // refused at the second sign near `b`.
        if matches!(self.peek(), Some(b'@' | b'$'))
            && matches!(self.src.get(at + 1), Some(b) if wordy(*b))
        {
            let name = self.near_at(at);
            self.mark = at;
            self.word = name.clone();
            return Bad::Syntax { at, near: name };
        }
        // A star with a word after it is the word's, so `@a *x` is refused at
        // the star and names the `x` rather than the field.
        self.syntax_near()
    }

    /// Hangs one attribute off the node it was written after.
    ///
    /// The weight lands on the word rather than on the union the word expanded
    /// into, which is where a real server puts it and where it has to be for the
    /// stem not to inherit a weight the client meant for the word.
    fn apply(&mut self, node: &mut Node, name: &[u8], value: &[u8]) -> Result<(), Bad> {
        let text = String::from_utf8_lossy(value).into_owned();
        let wrong = || Bad::Value {
            name: name.into(),
            value: value.into(),
        };
        match name.to_ascii_lowercase().as_slice() {
            b"weight" => {
                // A weight is read the way C reads one, so `inf` and `nan` are
                // both weights, and a number too big to hold is not one at all
                // even though it comes out as an infinity.
                let Ok(weight) = text.parse::<f64>() else {
                    return Err(wrong());
                };
                if weight < 0.0 || (weight.is_infinite() && !text.to_lowercase().contains("inf")) {
                    return Err(wrong());
                }
                carrier(node).weight = Some(weight);
            }
            b"slop" => {
                let Ok(slop) = text.parse::<i64>() else {
                    return Err(wrong());
                };
                if slop < -1 {
                    return Err(wrong());
                }
                carrier(node).slop = Some(slop);
            }
            b"inorder" => {
                let Some(order) = flag(&text) else {
                    return Err(wrong());
                };
                carrier(node).inorder = order;
            }
            b"phonetic" => {
                let Some(_) = flag(&text) else {
                    return Err(wrong());
                };
                // `VERBATIM` means no word is expanded at all, so nothing ever
                // asks the field for a phonetic form and nothing complains that
                // it has none. Asking for one either way is refused otherwise,
                // including asking for it to be off.
                if !self.ask.verbatim {
                    return Err(Bad::Refused(NO_PHONETICS));
                }
            }
            b"yield_distance_as" if matches!(node.what, What::Vector(_)) => {
                if let What::Vector(vector) = &mut node.what {
                    vector.alias = Some(value.into());
                }
            }
            _ => return Err(Bad::Attribute(name.into())),
        }
        Ok(())
    }

    /// `=>[KNN k @field $param ...]` after a whole query.
    fn knn(&mut self, over: Node) -> Result<Node, Bad> {
        // A parameter the clause itself is missing beats one the query text was
        // missing, because a real server settles the vector clause before it
        // walks the text under it. `$k=>[KNN 3 @v $B]` names `B` and not `k`.
        let outer = std::mem::take(&mut self.gone);
        self.at += 1;
        self.spaces();
        let start = self.at;
        let word = self.word();
        if !word.eq_ignore_ascii_case(b"knn") {
            return Err(self.syntax());
        }
        // The keyword is now the last thing read, so a clause that runs out
        // after it is refused there rather than back at the bracket.
        self.mark = start;
        self.word = word;
        self.spaces();
        let at = self.at;
        // The count may be a parameter, but it is not looked up yet: a real
        // server settles the field and the vector itself first, so a query
        // with two unknown parameters in it names the vector's one.
        let held = if self.peek() == Some(b'$') {
            self.at += 1;
            let name = self.word();
            if name.is_empty() {
                return Err(self.syntax_at(at));
            }
            self.word = name.clone();
            Some(name)
        } else {
            let held = self.mark;
            if self.token().is_empty() {
                // Nothing was read, so the keyword before it is still the last
                // token and is what an error points at.
                self.mark = held;
            }
            None
        };
        let count = self.src[at..self.at].to_vec();
        self.spaces();
        let spot = self.at;
        if !self.eat(b'@') {
            return Err(self.syntax());
        }
        let field = self.token();
        // The sign is part of the field token, so an error after it points at
        // the sign rather than at the first letter of the name.
        self.mark = spot;
        self.expect(&field, spot, Want::Vector)?;
        self.spaces();
        let param = self.dollar()?;
        self.param(&param);
        let count = match held {
            Some(name) => self.param(&name),
            None => count.into(),
        };
        let Ok(k) = String::from_utf8_lossy(&count).parse::<u64>() else {
            return Err(self.syntax_at(at));
        };
        let mut vector = Vector {
            field,
            param,
            k: Some(k),
            radius: None,
            alias: None,
            options: Vec::new(),
            // A star is every document and so is a query that came to nothing,
            // and either way there is no filter left to run the search over, so
            // `the=>[KNN 2 @v $B]` is the plain shape a bare `*` gives.
            over: (!matches!(over.what, What::Wildcard | What::Empty)).then(|| Box::new(over)),
        };
        loop {
            self.spaces();
            match self.peek() {
                None => return Err(self.syntax()),
                Some(b']') => {
                    self.at += 1;
                    break;
                }
                Some(b) if wordy(b) => {
                    let name = self.token();
                    self.word = name.clone();
                    self.spaces();
                    // A runtime option may be given as a parameter the same way
                    // the count and the vector are.
                    let value = if self.peek() == Some(b'$') {
                        let held = self.dollar()?;
                        self.param(&held)
                    } else {
                        let word = self.word();
                        // An option with nothing after it is refused where it
                        // ran out, ahead of any parameter the client forgot.
                        if word.is_empty() {
                            return Err(self.syntax());
                        }
                        word
                    };
                    if name.eq_ignore_ascii_case(b"as") {
                        // The distance is another property in the reply, so a
                        // name the schema already uses would be two of them.
                        // It is held rather than raised, because a parameter
                        // the client forgot is reported ahead of it.
                        if self.index.field(&value).is_some() && self.taken.is_none() {
                            self.taken = Some(value.clone());
                        }
                        vector.alias = Some(value);
                    } else {
                        vector
                            .options
                            .push((name.to_ascii_uppercase().into(), value));
                    }
                }
                _ => return Err(self.syntax()),
            }
        }
        if self.gone.is_empty() {
            self.gone = outer;
        }
        let node = Node::new(What::Vector(Box::new(vector)));
        self.attributes(node)
    }

    /// `$name`, which is how a query refers to something passed with `PARAMS`.
    fn dollar(&mut self) -> Result<Box<[u8]>, Bad> {
        if !self.eat(b'$') {
            return Err(self.syntax());
        }
        let name = self.word();
        if name.is_empty() {
            return Err(self.syntax());
        }
        Ok(name)
    }

    /// Whether the quote under the cursor has another one after it.
    ///
    /// A single quote that never closes is not a quote at all, and neither the
    /// phrase nor the pattern it looks like is read as one.
    fn closed(&self) -> bool {
        self.src[self.at + 1..].contains(&b'\'')
    }

    /// The body of a phrase written where a word belongs, such as the one in
    /// `*"hello world"`, with the cursor left after the closing quote.
    ///
    /// The body is what was written between the quotes, spaces and capitals and
    /// stars and all, and a phrase that never closes is not one of these. The
    /// first dialect has no use for any of it and reads none of it.
    fn glued(&mut self) -> Option<Box<[u8]>> {
        if self.ask.dialect == 1 {
            return None;
        }
        let quote = match self.peek() {
            Some(b'"') => b'"',
            Some(b'\'') if self.closed() => b'\'',
            _ => return None,
        };
        let start = self.at + 1;
        let end = start + self.src[start..].iter().position(|b| *b == quote)?;
        self.at = end + 1;
        Some(self.src[start..end].into())
    }

    /// The body of a pattern, which is a parameter when it starts with a sign
    /// and is the text as it was written otherwise.
    ///
    /// The whole of the rest of the body is the name, so `w'$B*'` asks for a
    /// parameter called `B*` rather than for a pattern that ends in a star.
    fn patterned(&mut self, body: Box<[u8]>) -> Box<[u8]> {
        if self.ask.dialect == 1 {
            return body;
        }
        match body.strip_prefix(b"$".as_slice()) {
            // A body of nothing but the sign asks for a parameter with no name
            // at all, which nobody ever passed and which is said so about.
            Some(name) => self.param(name),
            None => body,
        }
    }

    /// Puts the parameters read since `mid` in front of the ones read between
    /// `head` and `mid`, which is what a fold that swapped its two sides did to
    /// the tree those parameters sit in.
    fn swap_gone(&mut self, head: usize, mid: usize) {
        if head < mid && mid < self.gone.len() {
            self.gone[head..].rotate_left(mid - head);
        }
    }

    /// What was passed for a parameter, or the error naming it.
    fn param(&mut self, name: &[u8]) -> Box<[u8]> {
        let found = self
            .ask
            .params
            .iter()
            .find(|(k, _)| **k == *name)
            .map(|(_, v)| v.clone());
        match found {
            Some(value) => value,
            None => {
                // The rest of the query still has to parse, so the reference
                // stands in as a number, which is the one shape that reads
                // sensibly everywhere a parameter can appear.
                //
                // The name the client is told about is the one they wrote, so
                // `$p\{a` is missing under that name and not under the `p{a`
                // the escape made of it. The word last read is that name when
                // nothing has been read since, which is the case everywhere a
                // parameter is looked up.
                let told = if bare(&self.word) == name {
                    self.word.clone()
                } else {
                    name.into()
                };
                self.gone.push(told);
                b"0".to_vec().into()
            }
        }
    }

    /// Counts one more level of nesting, and refuses to go deeper than the
    /// stack can take.
    ///
    /// This counts every group, which under the first dialect means every
    /// operator as well, because each of them reaches over the rest of what it
    /// is written in. Brackets have a count of their own in `bracket` below,
    /// and it is a much smaller one.
    fn deeper(&mut self) -> Result<(), Bad> {
        self.deep += 1;
        if self.deep > DEEPEST_ANY {
            return Err(Bad::Plain(TOO_DEEP));
        }
        Ok(())
    }

    /// Counts one more open bracket, and refuses a query that nests them
    /// deeper than a real server does.
    ///
    /// The first dialect gives up at a much shallower place than the later ones
    /// and words it as an ordinary syntax error, pointing past the brackets at
    /// whatever was written inside them.
    fn nest(&mut self) -> Result<(), Bad> {
        self.nested += 1;
        if self.ask.dialect == 1 {
            if self.nested > DEEPEST_ONE {
                // What the error quotes is what was written inside all those
                // brackets rather than the brackets themselves, so a wall of
                // them followed by a word names the word.
                let mut spot = self.at;
                while matches!(self.src.get(spot), Some(b'(' | b' ')) {
                    spot += 1;
                }
                return Err(Bad::Syntax {
                    at: self.at,
                    near: self.near_at(spot),
                });
            }
        } else if self.nested + self.ops > DEEPEST {
            return Err(Bad::Plain(TOO_DEEP));
        }
        Ok(())
    }

    /// Counts one more operator standing in front of another, which a real
    /// server holds on the same stack as its brackets under the later dialects
    /// and holds without counting at all under the first.
    fn operator(&mut self) -> Result<(), Bad> {
        if self.ask.dialect == 1 {
            return Ok(());
        }
        self.ops += 1;
        if self.nested + self.ops > DEEPEST_OPS {
            return Err(Bad::Plain(TOO_DEEP));
        }
        Ok(())
    }
}

/// Wraps a node in a negation or an optional, unless there is nothing to wrap.
///
/// `hello -the` is `hello`, because the stopword left nothing behind and a
/// negation of nothing is not a filter that excludes everything.
/// Whether a word with nothing in it is anywhere under a node.
fn hollow(node: &Node) -> bool {
    match &node.what {
        What::Term(word) => word.word.is_empty(),
        What::Union(list) | What::Intersect(list) | What::Exact(list) | What::Tag(_, list) => {
            list.iter().any(hollow)
        }
        What::Not(child) | What::Optional(child) => hollow(child),
        _ => false,
    }
}

fn wrap(child: Node, negate: bool) -> Node {
    if child.is_empty() {
        return Node::empty();
    }
    // Two negations cancel, which they have to do here rather than at print
    // time because the tree is what a reader walks. Two `~` do not cancel and
    // are printed nested, which is the module's own asymmetry: an optional
    // clause changes a score rather than a set, so it is not its own inverse.
    // Whatever the inner negation was carrying goes with it, so the weight in
    // `-(-hello) => {$weight: 2;}` is written and then thrown away.
    if negate
        && child.mask == EVERY
        && let What::Not(inner) = child.what
    {
        return *inner;
    }
    let child = Box::new(child);
    Node::new(if negate {
        What::Not(child)
    } else {
        What::Optional(child)
    })
}

/// Closes off a run of tag values that had only spaces between them.
fn close_run(list: &mut Vec<Node>, run: &mut Vec<Node>) {
    match run.len() {
        0 => {}
        1 => list.push(run.pop().unwrap_or_else(Node::empty)),
        _ => {
            // A value of one word is matched exactly as it was written, and a
            // value of several is not a value at all: it is a phrase, and a
            // phrase is folded and unescaped the way every other phrase in the
            // language is. So `{AA}` looks for `AA` and `{AA BB}` looks for
            // `aa` next to `bb`, which reads like a mistake until you see that
            // the second one was never going to match a tag with a space in it
            // either way.
            for node in run.iter_mut() {
                if let What::Term(w) = &mut node.what {
                    w.word = fold(&bare(&w.word)).into();
                }
            }
            list.push(Node::new(What::Intersect(std::mem::take(run))));
        }
    }
    run.clear();
}

/// The true or false an attribute is written with, or nothing when it is
/// neither.
///
/// A real server takes the two words and the two digits and nothing else, so
/// `yes` and `t` are both refused where `TRUE` and `1` are taken.
fn flag(text: &str) -> Option<bool> {
    match text.to_ascii_lowercase().as_str() {
        "true" | "1" => Some(true),
        "false" | "0" => Some(false),
        _ => None,
    }
}

/// The node an attribute written after something actually lands on.
///
/// The union a word expanded into passes it to the word, so `hello => {$weight:
/// 2}` weighs `hello` rather than the stem beside it. Nothing else passes it on:
/// a negation written in front of a word never sees the clause, because the word
/// took it before the negation took the word.
fn carrier(node: &mut Node) -> &mut Node {
    if !matches!(&node.what, What::Union(list) if expansion(list)) {
        return node;
    }
    match &mut node.what {
        What::Union(list) => &mut list[0],
        _ => unreachable!("the shape was just checked"),
    }
}

/// A number the way a query writes one, including the infinities.
fn number(text: &str) -> Option<f64> {
    let text = text.trim();
    match text {
        "inf" | "+inf" | "INF" | "+INF" => return Some(f64::INFINITY),
        "-inf" | "-INF" => return Some(f64::NEG_INFINITY),
        _ => {}
    }
    text.parse::<f64>().ok().filter(|v| !v.is_nan())
}

impl Node {
    /// Narrows this node to no field at all, undoing an expansion on the way.
    ///
    /// Kept beside the parser rather than beside `narrow` because the reason is
    /// a parser reason: the expansion is something the parse added, and a node
    /// that can never match is one a real server never bothered to expand.
    fn collapse(&mut self) {
        if let What::Union(list) = &mut self.what
            && expansion(list)
        {
            let first = list.swap_remove(0);
            self.what = first.what;
            // A weight written after the word landed on the word rather than on
            // the union it expanded into, so it comes up with it.
            self.weight = self.weight.or(first.weight);
        }
    }
}

/// Narrows a whole tree and folds away the expansions that can no longer match.
pub(crate) fn settle(node: &mut Node) {
    if node.mask == 0 {
        node.collapse();
    }
    match &mut node.what {
        What::Union(list) | What::Intersect(list) | What::Exact(list) | What::Tag(_, list) => {
            for child in list {
                settle(child);
            }
        }
        What::Not(child) | What::Optional(child) => settle(child),
        What::Vector(vector) => {
            if let Some(over) = &mut vector.over {
                settle(over);
            }
        }
        _ => {}
    }
}

const _: () = assert!(
    EVERY != 0,
    "every field has to be a different mask from no field"
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{Field, Kind, Text};
    use crate::index::{Definition, Index};
    use crate::query::explain::explain;

    fn index() -> Index {
        let schema = vec![
            Field::new(b"a", Kind::Text(Text::default())),
            Field::new(b"b", Kind::Text(Text::default())),
            Field::new(b"n", Kind::Numeric),
            Field::new(b"g", Kind::Tag(crate::field::Tag::default())),
            Field::new(
                b"v",
                Kind::Vector(crate::field::Vector::new(
                    crate::field::Algo::Flat,
                    crate::field::Width::Float32,
                    4,
                    yo_shape::Metric::L2,
                )),
            ),
            Field::new(b"loc2", Kind::Geo),
        ];
        Index::new(b"i", Definition::default(), schema)
    }

    fn shown(query: &str, dialect: u8) -> String {
        let index = index();
        let ask = Ask {
            dialect,
            ..Ask::default()
        };
        let node = parse(query.as_bytes(), &index, &ask).expect("the query parses");
        String::from_utf8(explain(&node, &index)).expect("the printout is text")
    }

    #[test]
    fn a_word_becomes_itself_and_its_stem() {
        assert_eq!(
            shown("running", 1),
            "UNION {\n  running\n  +run(expanded)\n  run(expanded)\n}\n"
        );
    }

    /// A word that stems to itself is not printed twice.
    #[test]
    fn a_word_that_is_its_own_stem_expands_to_two_and_not_three() {
        assert_eq!(
            shown("hello", 1),
            "UNION {\n  hello\n  +hello(expanded)\n}\n"
        );
    }

    #[test]
    fn a_query_of_nothing_but_stopwords_matches_nothing() {
        assert_eq!(shown("the a of", 1), "<empty>\n");
    }

    #[test]
    fn a_stopword_in_the_middle_is_dropped_and_the_rest_still_intersects() {
        assert!(shown("hello the world", 1).starts_with("INTERSECT {\n"));
        assert_eq!(shown("hello the world", 1).matches("UNION {").count(), 2);
    }

    /// The difference between the dialects, on the query that shows it.
    #[test]
    fn a_modifier_reaches_over_the_rest_in_one_dialect_and_one_word_in_the_other() {
        assert!(shown("@a:hello world", 1).starts_with("@a:INTERSECT {"));
        assert!(shown("@a:hello world", 2).starts_with("INTERSECT {"));
    }

    #[test]
    fn a_negation_reaches_over_the_rest_in_one_dialect_and_one_word_in_the_other() {
        assert!(shown("-hello world", 1).starts_with("NOT{"));
        assert!(shown("-hello world", 2).starts_with("INTERSECT {"));
    }

    /// A bracket on the left stops a run of words being one node, which is what
    /// makes these two queries different trees for the same set of documents.
    #[test]
    fn a_bracket_on_the_left_changes_how_a_run_of_words_nests() {
        assert_eq!(shown("hello foo bar", 1).matches("INTERSECT {").count(), 1);
        assert_eq!(
            shown("(hello) foo bar", 1).matches("INTERSECT {").count(),
            2
        );
    }

    #[test]
    fn a_union_folds_to_the_left_and_not_to_the_right() {
        assert_eq!(shown("(a1|b1)|c1", 1).matches("UNION {").count(), 4);
        assert_eq!(shown("a1|(b1|c1)", 1).matches("UNION {").count(), 5);
    }

    #[test]
    fn a_range_prints_its_ends_at_six_places_and_its_infinities_by_name() {
        assert_eq!(
            shown("@n:[1 10]", 1),
            "NUMERIC {1.000000 <= @n <= 10.000000}\n"
        );
        assert_eq!(shown("@n:[(1 +inf]", 1), "NUMERIC {1.000000 < @n <= inf}\n");
    }

    #[test]
    fn a_tag_lists_its_values_without_a_union_around_them() {
        assert_eq!(shown("@g:{x|y}", 1), "TAG:@g {\n  x\n  y\n}\n");
    }

    /// A tag value of one word is matched byte for byte and a value of several
    /// is not a value at all, it is a phrase, and every phrase in the language
    /// is folded and unescaped the same way.
    #[test]
    fn a_tag_value_of_one_word_keeps_what_a_value_of_several_loses() {
        assert_eq!(shown("@g:{AA}", 2), "TAG:@g {\n  AA\n}\n");
        assert_eq!(shown(r"@g:{aa\-bb}", 2), "TAG:@g {\n  aa\\-bb\n}\n");
        assert_eq!(
            shown("@g:{AA BB}", 2),
            "TAG:@g {\n  INTERSECT {\n    aa\n    bb\n  }\n}\n"
        );
        assert_eq!(
            shown(r"@g:{aa\-BB cc}", 2),
            "TAG:@g {\n  INTERSECT {\n    aa-bb\n    cc\n  }\n}\n"
        );
    }

    /// A backslash in front of a letter escapes nothing and ends the value, so
    /// these are two values where the one in front of a mark is one.
    #[test]
    fn an_escape_that_reaches_nothing_splits_a_tag_value_in_two() {
        assert_eq!(
            shown(r"@g:{aa\xbb}", 2),
            "TAG:@g {\n  INTERSECT {\n    aa\n    xbb\n  }\n}\n"
        );
    }

    /// A value matched against the front or the back of a tag loses its escapes
    /// where a whole value keeps them, and keeps its capitals either way.
    #[test]
    fn a_tag_value_with_a_star_on_it_is_unescaped_and_not_folded() {
        assert_eq!(shown(r"@g:{a\-b*}", 2), "TAG:@g {\n  PREFIX{a-b*}\n}\n");
        assert_eq!(shown(r"@g:{*a\-b}", 2), "TAG:@g {\n  SUFFIX{*a-b}\n}\n");
        assert_eq!(shown(r"@g:{*a\-b*}", 2), "TAG:@g {\n  INFIX{*a-b*}\n}\n");
        assert_eq!(shown("@g:{AA*}", 2), "TAG:@g {\n  PREFIX{AA*}\n}\n");
        assert_eq!(
            shown(r#"@g:{"aa bb"*}"#, 2),
            "TAG:@g {\n  PREFIX{aa bb*}\n}\n"
        );
    }

    /// A backslash left on the end of a pattern comes off, and the one place it
    /// stays is a query that ends inside the word.
    #[test]
    fn a_trailing_escape_comes_off_a_pattern_and_not_off_a_value() {
        assert_eq!(shown(r"@g:{*x\\}", 2), "TAG:@g {\n  SUFFIX{*x}\n}\n");
        assert_eq!(shown(r"@g:{x\\*}", 2), "TAG:@g {\n  PREFIX{x*}\n}\n");
        assert_eq!(shown(r"@g:{*x\\y\\}", 2), "TAG:@g {\n  SUFFIX{*x\\y}\n}\n");
        assert_eq!(shown(r"@g:{x\\}", 2), "TAG:@g {\n  x\\\\\n}\n");
        assert_eq!(shown(r"*x\\", 2), "SUFFIX{*x\\}\n");
    }

    /// A tag value is the longer of the number and the word that both start
    /// where it does, which is the only thing that tells these three apart.
    #[test]
    fn a_tag_value_is_the_longer_of_a_number_and_a_word() {
        assert_eq!(shown(r"@g:{1\-a}", 2), "TAG:@g {\n  1\\-a\n}\n");
        assert_eq!(
            shown(r"@g:{5.5\-a}", 2),
            "TAG:@g {\n  INTERSECT {\n    5.5\n    -a\n  }\n}\n"
        );
        assert_eq!(
            shown("@g:{1.5.5}", 2),
            "TAG:@g {\n  INTERSECT {\n    1.5\n    .5\n  }\n}\n"
        );
    }

    /// The first dialect has no reading for a tag value that starts with one of
    /// these five, so the list ends there and the rest is ordinary text, which
    /// leaves the closing brace belonging to nobody.
    #[test]
    fn the_first_dialect_ends_a_tag_list_at_a_mark_a_value_cannot_start_with() {
        assert_eq!(
            refused("@g:{aa-bb}", 1),
            Bad::Syntax {
                at: 9,
                near: b"bb".to_vec().into()
            }
        );
    }

    /// A tag list and a phrase are both term lists, and the grammar the first
    /// dialect is written in only drops a stopword out of one that has two
    /// terms in it already.
    #[test]
    fn a_stopword_needs_two_terms_in_front_of_it_before_it_is_dropped() {
        assert_eq!(shown("@g:{be}", 1), "TAG:@g {\n  be\n}\n");
        assert!(matches!(refused("@g:{x be}", 1), Bad::Syntax { at: 8, .. }));
        assert!(matches!(
            refused("@g:{x be y}", 1),
            Bad::Syntax { at: 10, .. }
        ));
        assert_eq!(
            shown("@g:{x y be z}", 1),
            "TAG:@g {\n  INTERSECT {\n    x\n    y\n    z\n  }\n}\n"
        );
        assert!(matches!(
            refused(r#""x be y""#, 1),
            Bad::Syntax { at: 3, .. }
        ));
        assert_eq!(shown(r#""x y be""#, 1), "EXACT {\n  x\n  y\n}\n");
    }

    /// What an error quotes is the word the client wrote and not the word the
    /// parser made of it.
    #[test]
    fn an_error_quotes_the_word_as_it_was_typed() {
        assert_eq!(
            refused(r"aa\-bb )", 1),
            Bad::Syntax {
                at: 7,
                near: br"aa\-bb".to_vec().into()
            }
        );
        assert_eq!(
            refused(r#"@g:{aa  "BB\A Cc"} @a:x"#, 1),
            Bad::Syntax {
                at: 12,
                near: b"A".to_vec().into()
            }
        );
    }

    /// A star is part of the word it is glued to, so there is nothing else to
    /// point at even when the query goes on.
    #[test]
    fn a_fuzzy_word_with_a_star_on_it_is_refused_back_at_the_word() {
        assert!(matches!(
            refused("%hello* world", 1),
            Bad::Syntax { at: 1, .. }
        ));
        assert_eq!(
            refused("aa%*BB", 1),
            Bad::Syntax {
                at: 3,
                near: b"BB".to_vec().into()
            }
        );
    }

    /// Two modifiers with no field in common leave a node that cannot match, and
    /// the word under it is not expanded because there is nothing to expand for.
    #[test]
    fn a_word_narrowed_to_no_field_is_left_unexpanded() {
        assert_eq!(shown("@a:hello|@b:foo", 1).matches("@NULL:foo").count(), 1);
    }

    #[test]
    fn a_phrase_of_one_word_is_that_word_and_is_not_expanded() {
        assert_eq!(shown("\"hello\"", 1), "hello\n");
    }

    #[test]
    fn a_star_on_its_own_is_every_document() {
        assert_eq!(shown("*", 1), "<WILDCARD>\n");
    }

    #[test]
    fn a_star_at_either_end_of_a_word_means_different_things() {
        assert_eq!(shown("hel*", 1), "PREFIX{hel*}\n");
        assert_eq!(shown("*llo", 1), "SUFFIX{*llo}\n");
        assert_eq!(shown("*ell*", 1), "INFIX{*ell*}\n");
    }

    #[test]
    fn a_weight_lands_on_the_word_and_not_on_the_union_it_expanded_into() {
        let out = shown("hello => {$weight: 2;}", 1);
        assert_eq!(
            out,
            "UNION {\n  hello => {$weight: 2;}\n  +hello(expanded)\n}\n"
        );
    }

    #[test]
    fn a_query_that_does_not_parse_says_where_it_gave_up() {
        let index = index();
        let bad =
            parse(b"\"hello the\"", &index, &Ask::default()).expect_err("a stopword in a phrase");
        assert_eq!(
            bad,
            Bad::Syntax {
                at: 7,
                near: b"the".to_vec().into()
            }
        );
    }

    /// What the parser gives back when it refuses a query.
    ///
    /// A parameter is passed along the way because the queries that ask for a
    /// vector name one, and what is being measured is where the refusal lands
    /// rather than whether the parameter was there.
    fn refused(query: &str, dialect: u8) -> Bad {
        let index = index();
        let params = vec![(b"B".to_vec().into(), b"aaaa".to_vec().into())];
        let ask = Ask {
            dialect,
            params: &params,
            ..Ask::default()
        };
        parse(query.as_bytes(), &index, &ask).expect_err("the query is refused")
    }

    #[test]
    fn an_attribute_clause_ends_what_a_field_modifier_reaches_over() {
        assert_eq!(
            shown("@a:hello hel*", 2),
            "@a:INTERSECT {\n  @a:UNION {\n    @a:hello\n    @a:+hello(expanded)\n  }\n  @a:PREFIX{hel*}\n}\n"
        );
        assert_eq!(
            shown("@a:hello => {$weight: 2;}hel*", 2),
            "INTERSECT {\n  @a:UNION {\n    @a:hello => {$weight: 2;}\n    @a:+hello(expanded)\n  }\n  PREFIX{hel*}\n}\n"
        );
    }

    #[test]
    fn a_sign_that_names_nothing_does_not_end_what_a_modifier_reaches_over() {
        assert_eq!(
            shown("@a:hello @ world", 1),
            "@a:INTERSECT {\n  @a:UNION {\n    @a:hello\n    @a:+hello(expanded)\n  }\n  @a:UNION {\n    @a:world\n    @a:+world(expanded)\n  }\n}\n"
        );
        assert_eq!(
            shown("@a:hello @|world", 1),
            "@a:UNION {\n  @a:UNION {\n    @a:hello\n    @a:+hello(expanded)\n  }\n  @a:UNION {\n    @a:world\n    @a:+world(expanded)\n  }\n}\n"
        );
    }

    #[test]
    fn a_sign_with_no_name_after_it_is_not_a_parameter_and_binds_to_nothing() {
        assert_eq!(
            shown("@a:hello $|world", 2),
            "UNION {\n  @a:UNION {\n    @a:hello\n    @a:+hello(expanded)\n  }\n  UNION {\n    world\n    +world(expanded)\n  }\n}\n"
        );
    }

    #[test]
    fn an_attribute_clause_after_a_stopword_belongs_to_the_stopword() {
        assert_eq!(
            shown("-b1 the => {$weight: 0.5;}", 1),
            "NOT{\n  UNION {\n    b1\n    +b1(expanded)\n  }\n}\n"
        );
    }

    #[test]
    fn a_phrase_with_a_star_on_its_quote_binds_to_the_word_on_its_left() {
        assert_eq!(
            shown("-x \"a b\"*", 2),
            "NOT{\n  INTERSECT {\n    UNION {\n      x\n      +x(expanded)\n    }\n    PREFIX{a b*}\n  }\n}\n"
        );
        assert_eq!(
            shown("-\"the\"hello%%hell%%", 2),
            "INTERSECT {\n  UNION {\n    hello\n    +hello(expanded)\n  }\n  FUZZY{hell}\n  NOT{\n    EXACT {\n      the\n    }\n  }\n}\n"
        );
        assert_eq!(
            shown("-x b1w'a*b'", 2),
            "INTERSECT {\n  NOT{\n    UNION {\n      x\n      +x(expanded)\n    }\n  }\n  UNION {\n    b1w\n    +b1w(expanded)\n  }\n  EXACT {\n    a\n    b\n  }\n}\n"
        );
    }

    #[test]
    fn a_bar_after_a_stopword_takes_the_stopword_and_not_the_run_before_it() {
        assert_eq!(
            shown("hello the|world", 1),
            "INTERSECT {\n  UNION {\n    hello\n    +hello(expanded)\n  }\n  UNION {\n    world\n    +world(expanded)\n  }\n}\n"
        );
        assert_eq!(
            shown("a1 b1 the|c1", 1),
            "INTERSECT {\n  UNION {\n    a1\n    +a1(expanded)\n  }\n  UNION {\n    b1\n    +b1(expanded)\n  }\n  UNION {\n    c1\n    +c1(expanded)\n  }\n}\n"
        );
        assert_eq!(
            shown("a1 the|b1 c1", 1),
            "INTERSECT {\n  UNION {\n    a1\n    +a1(expanded)\n  }\n  INTERSECT {\n    UNION {\n      b1\n      +b1(expanded)\n    }\n    UNION {\n      c1\n      +c1(expanded)\n    }\n  }\n}\n"
        );
        assert_eq!(
            shown("@a:hello the|world", 1),
            "INTERSECT {\n  @a:UNION {\n    @a:hello\n    @a:+hello(expanded)\n  }\n  UNION {\n    world\n    +world(expanded)\n  }\n}\n"
        );
        assert_eq!(
            shown("(a1 the)|b1", 1),
            "UNION {\n  UNION {\n    a1\n    +a1(expanded)\n  }\n  UNION {\n    b1\n    +b1(expanded)\n  }\n}\n"
        );
        assert_eq!(
            shown("~@A:x the|~(@a|b:hello)", 1),
            "OPTIONAL{\n  INTERSECT {\n    @NULL:x\n    OPTIONAL{\n      @a|b:UNION {\n        @a|b:hello\n        @a|b:+hello(expanded)\n      }\n    }\n  }\n}\n"
        );
    }

    #[test]
    fn a_sign_that_names_nothing_is_not_something_written_before_a_star() {
        assert_eq!(shown("@@*", 2), "<WILDCARD>\n");
        assert_eq!(shown("(@*)", 2), "<WILDCARD>\n");
        assert_eq!(shown("$*", 1), "<WILDCARD>\n");
    }

    #[test]
    fn a_sign_that_names_nothing_does_not_keep_a_prefix_from_binding() {
        assert_eq!(
            shown("-(a1)@*ell*", 2),
            "NOT{\n  INTERSECT {\n    UNION {\n      a1\n      +a1(expanded)\n    }\n    INFIX{*ell*}\n  }\n}\n"
        );
    }

    /// Whether a field is there at all and what it can hold are two questions
    /// asked at different times, so a name the schema has never heard of is
    /// answered for before the modifier has read anything and a name that holds
    /// the wrong thing is answered for after it has read everything.
    #[test]
    fn a_field_nobody_has_heard_of_is_answered_for_before_the_body_is_read() {
        assert_eq!(
            refused("@zzz:@n:[1 10]", 2),
            Bad::Unknown {
                at: 0,
                near: Some(b"zzz".to_vec().into())
            }
        );
        assert_eq!(
            refused("@n:%%%", 2),
            Bad::Syntax {
                at: 5,
                near: b"n".to_vec().into()
            }
        );
        assert_eq!(
            refused("@n:hello (", 2),
            Bad::Wrong {
                kind: "TEXT",
                at: 0,
                near: Some(b"n".to_vec().into())
            }
        );
    }

    /// An arrow written straight after something that bound itself to the
    /// modifier is read before the field is settled, and one written after
    /// anything else is not.
    #[test]
    fn a_broken_arrow_after_something_that_bound_itself_beats_the_field_check() {
        assert_eq!(
            refused("@n:hello hel*=>[KNN 2 @v $B]", 2),
            Bad::Syntax {
                at: 15,
                near: b">[".to_vec().into()
            }
        );
        assert_eq!(
            refused("@n:hello=>[KNN 2 @v $B]", 2),
            Bad::Wrong {
                kind: "TEXT",
                at: 0,
                near: Some(b"n".to_vec().into())
            }
        );
        assert_eq!(
            refused("@n:hello hel* => {$weight: 2;}", 2),
            Bad::Wrong {
                kind: "TEXT",
                at: 0,
                near: Some(b"n".to_vec().into())
            }
        );
        assert_eq!(
            refused("@zz:hello hel*=>[KNN 2 @v $B]", 2),
            Bad::Unknown {
                at: 0,
                near: Some(b"zz".to_vec().into())
            }
        );
    }

    #[test]
    fn a_vector_clause_in_a_bracket_is_answered_for_after_the_field_is() {
        assert_eq!(
            refused("(@n:hello=>[KNN 2 @v $B])", 2),
            Bad::Wrong {
                kind: "TEXT",
                at: 1,
                near: Some(b"n".to_vec().into())
            }
        );
        assert_eq!(
            refused("(@n:hello hel*=>[KNN 2 @v $B])", 2),
            Bad::Syntax {
                at: 16,
                near: b">[".to_vec().into()
            }
        );
    }

    /// A query that runs out is answered for where the last thing that was read
    /// began, and a prefix begins at its word rather than at its star.
    #[test]
    fn a_query_that_runs_out_is_answered_for_at_the_last_thing_that_was_read() {
        assert_eq!(
            refused("world (", 2),
            Bad::Syntax {
                at: 6,
                near: b"world".to_vec().into()
            }
        );
        assert_eq!(
            refused("hello ((", 2),
            Bad::Syntax {
                at: 7,
                near: b"hello".to_vec().into()
            }
        );
        assert_eq!(
            refused("((hello)", 2),
            Bad::Syntax {
                at: 7,
                near: b"hello".to_vec().into()
            }
        );
        assert_eq!(
            refused("hello (hel*", 2),
            Bad::Syntax {
                at: 7,
                near: b"hel".to_vec().into()
            }
        );
        assert_eq!(
            refused("(\"a b\"", 2),
            Bad::Syntax {
                at: 1,
                near: b"a b".to_vec().into()
            }
        );
    }

    #[test]
    fn a_pattern_is_quoted_by_what_is_between_its_quotes() {
        assert_eq!(
            refused("* w'a*b'", 2),
            Bad::Syntax {
                at: 4,
                near: b"a*b".to_vec().into()
            }
        );
        assert_eq!(
            refused("@a w'a*b'", 2),
            Bad::Syntax {
                at: 5,
                near: b"a*b".to_vec().into()
            }
        );
    }

    /// A pattern anywhere in front of an error leaves it nothing to quote, and
    /// a modifier with no colon after it is no different in that.
    #[test]
    fn a_pattern_leaves_a_later_error_with_nothing_to_quote() {
        assert_eq!(
            refused("w'a*b' @flies", 2),
            Bad::Unknown { at: 7, near: None }
        );
        assert_eq!(
            refused("x1 @flies", 2),
            Bad::Unknown {
                at: 3,
                near: Some(b"flies".to_vec().into())
            }
        );
    }

    /// Runs something where there is room for the whole descent.
    ///
    /// A build with no optimisation spends more than twenty times the stack on
    /// each level that a release build does, and the limits here are the ones a
    /// real server has rather than ones picked to fit a test runner, so the
    /// deep queries are read on a thread with room to spare.
    fn roomy(job: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(job)
            .expect("a thread with a bigger stack")
            .join()
            .expect("the descent to end in an error rather than a crash");
    }

    #[test]
    fn nesting_far_deeper_than_anyone_writes_is_refused_rather_than_crashing() {
        roomy(|| {
            let index = index();
            for dialect in [1, 2, 3] {
                let query = "(".repeat(4096);
                let ask = Ask {
                    dialect,
                    ..Ask::default()
                };
                assert!(parse(query.as_bytes(), &index, &ask).is_err(), "{dialect}");
            }
        });
    }

    #[test]
    fn brackets_are_taken_as_deep_as_a_real_server_takes_them() {
        roomy(|| {
            let index = index();
            for (dialect, most) in [(1, DEEPEST_ONE), (2, DEEPEST), (3, DEEPEST)] {
                let ask = Ask {
                    dialect,
                    ..Ask::default()
                };
                let fits = format!("{}hello{}", "(".repeat(most), ")".repeat(most));
                parse(fits.as_bytes(), &index, &ask).expect("the deepest a server takes");
                let over = format!("{}hello{}", "(".repeat(most + 1), ")".repeat(most + 1));
                assert!(parse(over.as_bytes(), &index, &ask).is_err(), "{dialect}");
            }
        });
    }

    #[test]
    fn operators_stacked_far_deeper_than_anyone_writes_are_refused_too() {
        roomy(|| {
            let index = index();
            for sign in ["-", "~"] {
                for dialect in [1, 2, 3] {
                    let query = format!("{}hello", sign.repeat(4096));
                    let ask = Ask {
                        dialect,
                        ..Ask::default()
                    };
                    assert!(
                        parse(query.as_bytes(), &index, &ask).is_err(),
                        "{sign} {dialect}"
                    );
                }
            }
        });
    }

    #[test]
    fn operators_are_taken_as_deep_as_a_real_server_takes_them() {
        roomy(|| {
            let index = index();
            for dialect in [2, 3] {
                let ask = Ask {
                    dialect,
                    ..Ask::default()
                };
                let fits = format!("{}hello", "-".repeat(DEEPEST_OPS));
                parse(fits.as_bytes(), &index, &ask).expect("the deepest a server takes");
                let over = format!("{}hello", "-".repeat(DEEPEST_OPS + 1));
                assert_eq!(
                    parse(over.as_bytes(), &index, &ask),
                    Err(Bad::Plain(TOO_DEEP))
                );
            }
        });
    }
}
