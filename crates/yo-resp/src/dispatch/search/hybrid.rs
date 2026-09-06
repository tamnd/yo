//! `FT.HYBRID`, which runs a text query and a vector query over one index and
//! folds the two answers into one.
//!
//! The command is three sections and a pipeline. `SEARCH` says what the text
//! branch looks for, `VSIM` says what the vector branch looks for, `COMBINE`
//! says how the two rankings become one number, and everything after that is
//! the same pipeline `FT.AGGREGATE` runs, read by the same code.
//!
//! Neither branch is a window on the other. Each one is cut to its own top few
//! rows first and the union of the two cuts is the answer, which is why a
//! document the text branch never found can still come first.

use std::time::Instant;

use yo_common::Result;
use yo_common::num::parse_f64;
use yo_common::parse_i64;
use yo_search::Index;
use yo_search::field::Kind;
use yo_search::query::{self, Ask, Node, What, Yield};
use yo_search::score::Scorer;

use super::aggregate::{self, Reads};
use super::{Args, Asked, Order, Row, Rows, Watch};
use crate::dispatch::Server;
use crate::dispatch::args;
use crate::reply::Out;

/// How many rows of each branch reach the merge when nothing said otherwise.
const WINDOW: usize = 20;

/// What an `RRF` divides by when nothing said otherwise.
const CONSTANT: f64 = 60.0;

/// What a `LINEAR` weighs the text branch by when nothing said otherwise.
const ALPHA: f64 = 0.3;

/// What a `LINEAR` weighs the vector branch by when nothing said otherwise.
const BETA: f64 = 0.7;

/// The name the vector branch's distance is carried under while the two
/// answers are being merged.
///
/// A client never sees it. It is only there so the merge can read the distance
/// back off a row whether or not a `YIELD_SCORE_AS` asked for one.
const AWAY: &[u8] = b"__yo_hybrid_distance";

/// The line a word nobody knows gets, wherever in the argument list it stood.
fn unknown(name: &[u8]) -> Vec<u8> {
    about(name, b"Unknown argument")
}

/// The line a word that was already written gets when it is written again.
fn twice(name: &[u8]) -> Vec<u8> {
    about(name, b"Argument specified multiple times")
}

/// The line a value that will not read as the type behind its keyword gets.
fn convert(name: &[u8]) -> Vec<u8> {
    about(name, b"Could not convert argument to expected type")
}

/// One of the argument errors, which are all a keyword, a colon and a reason.
fn about(name: &[u8], why: &[u8]) -> Vec<u8> {
    let mut out = b"SEARCH_PARSE_ARGS ".to_vec();
    out.extend_from_slice(name);
    out.extend_from_slice(b": ");
    out.extend_from_slice(why);
    out
}

/// The line a word gets when it stood inside a section that has its own small
/// list of words.
fn inside(name: &[u8], section: &[u8]) -> Vec<u8> {
    let mut out = b"SEARCH_ARG_UNRECOGNIZED Unknown argument `".to_vec();
    out.extend_from_slice(name);
    out.extend_from_slice(b"` in ");
    out.extend_from_slice(section);
    out
}

/// A line with nothing of the client's in it.
fn plain(text: &str) -> Vec<u8> {
    text.as_bytes().to_vec()
}

/// The line a value gets inside a `KNN` or a `RANGE`, which is its own family
/// rather than the one the pipeline's words are refused in.
fn value(name: &[u8]) -> Vec<u8> {
    let mut out = b"SEARCH_SYNTAX Invalid ".to_vec();
    out.extend_from_slice(name);
    out.extend_from_slice(b" value");
    out
}

/// The longest a foreground query may be given before a real server says it is
/// capping it, in milliseconds.
const LONGEST: i64 = 60_000;

/// What it says when it caps one.
const CAPPED: &str = "Query TIMEOUT exceeded the configured maximum (search-_max-foreground-timeout-limit) while search-workers is disabled; effective timeout was capped";

/// How the two rankings become one number.
enum Combine {
    /// Reciprocal rank fusion, which only reads where a document came in each
    /// branch and not what it scored there.
    Rrf { constant: f64, window: usize },
    /// A weighted sum of the two scores.
    Linear {
        alpha: f64,
        beta: f64,
        window: usize,
    },
}

impl Combine {
    /// How many rows of each branch reach the merge.
    const fn window(&self) -> usize {
        match self {
            Combine::Rrf { window, .. } | Combine::Linear { window, .. } => *window,
        }
    }
}

impl Default for Combine {
    fn default() -> Combine {
        Combine::Rrf {
            constant: CONSTANT,
            window: WINDOW,
        }
    }
}

/// What the vector branch was asked for.
enum Reach {
    /// A count of nearest neighbours, which cuts that branch on its own rather
    /// than through the window.
    Knn(u64),
    /// Everything within a distance, as the client wrote it.
    Range(Box<[u8]>),
}

/// What goes on every row before the pipeline has said anything.
///
/// The two properties a row carries without being asked are only there when
/// nothing was loaded, because a `LOAD` is the client saying what it wants and
/// these are not part of it.
#[derive(Clone, Copy)]
struct Held<'a> {
    front: &'a [Yield],
    back: &'a [Yield],
    defaults: bool,
}

/// The whole argument list, read.
struct Asks<'a> {
    /// The text branch's query.
    text: &'a [u8],
    /// The scorer the text branch ranks with.
    scorer: Scorer,
    /// What the text branch's score is answered as, when something asked.
    scored: Option<Box<[u8]>>,
    /// The vector field, without the `@`.
    field: &'a [u8],
    /// The parameter holding the vector, without the `$`.
    param: &'a [u8],
    /// What the vector branch was asked for, when it was asked for anything.
    ///
    /// Neither `KNN` nor `RANGE` has to be written. Without one the branch is
    /// a nearest neighbour search as wide as the window, which is the same
    /// number of rows a `KNN` written out in full would have to reach the
    /// merge with.
    reach: Option<Reach>,
    /// The runtime options written beside it, as they were written.
    options: Vec<(&'a [u8], &'a [u8])>,
    /// What the vector branch looks at before it measures anything, when the
    /// client narrowed it down first.
    ///
    /// Read by the query parser rather than by the expression parser, so this
    /// is query text and not a pipeline `FILTER`.
    filter: Option<&'a [u8]>,
    /// What the vector branch's score is answered as, when something asked.
    nears: Option<Box<[u8]>>,
    /// How the two rankings become one.
    combine: Combine,
    /// The pipeline, and everything it needs to run.
    asked: Asked<'a>,
    /// Whether a `LOAD` was written, which takes the two properties a row
    /// carries without being asked off it.
    loaded: bool,
    /// Which words that may only be written once have been.
    once: Vec<&'static [u8]>,
    /// What the reply has to say about how the command was written, which today
    /// is only ever the capped timeout.
    warn: Option<&'static str>,
}

/// `FT.HYBRID index [count] SEARCH ... VSIM ... [COMBINE ...] [pipeline]`.
pub(crate) fn hybrid(server: &Server, db: usize, args: Args<'_>, out: &mut Out) -> Result<()> {
    let name = args.get(1);
    let clock = Instant::now();
    let mut reg = server.search.lock();
    let Some(index) = reg.open(name) else {
        super::Fail::naming(super::MISSING, name).write(out);
        return Ok(());
    };
    // Two passes over the words, for the same reason an aggregation makes two:
    // a step names properties, the properties a step may name include the ones
    // the vector branch yields, and nothing is known about those until the
    // queries have been read, which cannot happen until every word has been.
    let read = match reads(args, index, None) {
        Ok(read) => read,
        Err(text) => {
            out.error(&text);
            return Ok(());
        }
    };
    let (text, vector) = match branches(&read, index) {
        Ok(pair) => pair,
        Err(text) => {
            out.error(&text);
            return Ok(());
        }
    };
    // The text branch's score goes on the row in front of everything and the
    // vector branch's goes on the end of it, which is measured and is why the
    // two are carried apart rather than as one list.
    let front: Vec<Yield> = read.scored.iter().map(|name| named(name)).collect();
    let back: Vec<Yield> = read.nears.iter().map(|name| named(name)).collect();
    let held = Held {
        front: &front,
        back: &back,
        defaults: !read.loaded,
    };
    let asks = match reads(args, index, Some(held)) {
        Ok(asks) => asks,
        Err(text) => {
            out.error(&text);
            return Ok(());
        }
    };
    let (total, rows) = merged(index, &asks, text, vector);
    drop(reg);
    let spent = clock.elapsed();
    writes(server, db, total, &rows, &asks, spent, out);
    Ok(())
}

/// A yield that carries a score the merge worked out rather than a distance a
/// walk measured.
///
/// The pipeline reads both the same way, through the name, so what matters here
/// is only that the name is on the row before a step can look for it.
fn named(name: &[u8]) -> Yield {
    Yield {
        name: name.into(),
        field: Box::default(),
        asked: Box::default(),
        ordered: false,
    }
}

/// Reads every word of the argument list.
///
/// The second pass is the one with the yields, and it is the one whose answer
/// is used. The first exists so that a word can be refused before a property is
/// looked up, which is the order a real server does it in.
fn reads<'a>(
    args: Args<'a>,
    index: &Index,
    held: Option<Held<'_>>,
) -> core::result::Result<Asks<'a>, Vec<u8>> {
    let mut asked = Asked::default();
    asked.rows.count = usize::MAX;
    if let Some(held) = held {
        asked.pipe.binding = true;
        for want in held.front {
            asked.pipe.base.push((want.name.clone(), Reads::Distance));
        }
        if held.defaults {
            asked.pipe.base.push((b"__key".to_vec().into(), Reads::Key));
            asked
                .pipe
                .base
                .push((b"__score".to_vec().into(), Reads::Score));
        }
        for want in held.back {
            asked.pipe.base.push((want.name.clone(), Reads::Distance));
        }
    }
    let mut at = 2;
    // The word after the index is either `SEARCH` or a count of subqueries, and
    // the count is the only thing two is ever allowed to be.
    if !args::is(args.get(at), b"SEARCH") {
        let Some(count) = counting(args.get(at)) else {
            return Err(plain(
                "SEARCH_SYNTAX Invalid subqueries count: expected an unsigned integer",
            ));
        };
        if count != 2 {
            return Err(plain(
                "SEARCH_PARSE_ARGS FT.HYBRID currently supports only two subqueries",
            ));
        }
        at += 1;
        if !args::is(args.get(at), b"SEARCH") {
            return Err(plain("SEARCH_PARSE_ARGS Missing required argument SEARCH"));
        }
    }
    let text = args.get(at + 1);
    at += 2;
    let mut named_scorer = None;
    let mut scored = None;
    // The text branch takes two words of its own and nothing else. Every other
    // word between here and the `VSIM` is refused by name, which is what makes
    // a `FILTER` written in the wrong place a clear line rather than a query
    // that quietly does something else.
    while at < args.len() && !args::is(args.get(at), b"VSIM") {
        let word = args.get(at);
        if args::is(word, b"SCORER") {
            let Some(name) = args.opt(at + 1) else {
                return Err(inside(word, b"SEARCH"));
            };
            named_scorer = Some(name);
            at += 2;
            continue;
        }
        if args::is(word, b"YIELD_SCORE_AS") {
            let Some(name) = args.opt(at + 1) else {
                return Err(inside(word, b"SEARCH"));
            };
            scored = Some(name.into());
            at += 2;
            continue;
        }
        return Err(inside(word, b"SEARCH"));
    }
    // The name is looked up only once the section has been read through, which
    // is why `SCORER VSIM @v $q` is refused for the `@v` it took the `VSIM` for
    // a scorer name and left standing rather than for the scorer.
    let mut scorer = Scorer::default_scorer();
    if let Some(name) = named_scorer {
        let Some(found) = Scorer::named(name) else {
            let mut out = b"SEARCH_QUERY_BAD No such scorer ".to_vec();
            out.extend_from_slice(name);
            return Err(out);
        };
        scorer = found;
    }
    if at >= args.len() {
        return Err(plain("SEARCH_PARSE_ARGS Missing required argument VSIM"));
    }
    at += 1;
    let field = args.get(at);
    let Some(field) = field.strip_prefix(b"@") else {
        return Err(plain(
            "SEARCH_SYNTAX Missing @ prefix for vector field name",
        ));
    };
    let Some(param) = args.opt(at + 1).and_then(|word| word.strip_prefix(b"$")) else {
        return Err(plain(
            "SEARCH_SYNTAX Invalid vector argument, expected a parameter name starting with $",
        ));
    };
    // Named here rather than left to the query parser, because the field
    // arrived as a word of its own and there is no query for an offset to
    // point into.
    match index.field(field).map(|held| &held.kind) {
        Some(Kind::Vector(_)) => {}
        Some(_) => {
            let mut out = b"SEARCH_SYNTAX Expected a VECTOR field `".to_vec();
            out.extend_from_slice(field);
            out.push(b'`');
            return Err(out);
        }
        None => {
            let mut out = b"SEARCH_SYNTAX Unknown field `".to_vec();
            out.extend_from_slice(field);
            out.push(b'`');
            return Err(out);
        }
    }
    at += 2;
    let mut reach = None;
    let mut options = Vec::new();
    let mut nears = None;
    let mut filter = None;
    // The section's three words are written in this order or not at all, and
    // each of them closes the ones in front of it, which is why a `KNN` after a
    // `FILTER` is not a second look at the vector branch but a word the
    // pipeline has never heard of.
    let mut stage = 0;
    while at < args.len() {
        let word = args.get(at);
        let knn = args::is(word, b"KNN");
        if stage < 1 && (knn || args::is(word, b"RANGE")) {
            let section: &[u8] = match knn {
                true => b"KNN",
                false => b"RANGE",
            };
            at = neighbours(args, at, section, &mut reach, &mut options)?;
            stage = 1;
            continue;
        }
        if stage < 2 && args::is(word, b"FILTER") {
            let Some(src) = args.opt(at + 1) else {
                return Err(inside(word, b"VSIM"));
            };
            filter = Some(src);
            at += 2;
            stage = 2;
            continue;
        }
        if stage < 3 && args::is(word, b"YIELD_SCORE_AS") {
            let Some(name) = args.opt(at + 1) else {
                return Err(inside(word, b"VSIM"));
            };
            nears = Some(name.into());
            at += 2;
            stage = 3;
            continue;
        }
        break;
    }
    let mut asks = Asks {
        text,
        scorer,
        scored,
        field,
        param,
        reach,
        options,
        filter,
        nears,
        combine: Combine::default(),
        asked,
        loaded: false,
        once: Vec::new(),
        warn: None,
    };
    pipeline(args, at, index, &mut asks)?;
    // A `LOAD` can name either of the two properties a row carries without
    // being asked, and neither of them is a field of the key, so what the load
    // wrote down is put back to what it meant.
    for (name, from) in &mut asks.asked.pipe.base {
        if !matches!(from, Reads::Field(..)) {
            continue;
        }
        if &**name == b"__key" {
            *from = Reads::Key;
        } else if &**name == b"__score" {
            *from = Reads::Score;
        }
    }
    Ok(asks)
}

/// The words after the `VSIM` section, which are `COMBINE`, the pipeline and
/// the handful that say how the whole command runs.
fn pipeline<'a>(
    args: Args<'a>,
    from: usize,
    index: &Index,
    asks: &mut Asks<'a>,
) -> core::result::Result<(), Vec<u8>> {
    let mut at = from;
    while at < args.len() {
        let word = args.get(at);
        if args::is(word, b"COMBINE") {
            // A second `COMBINE` is read as the word and no more, so what gets
            // refused is the algorithm behind it rather than the word itself,
            // which is measured.
            if asks.only(b"COMBINE").is_err() {
                at += 1;
                continue;
            }
            at = combined(args, at, asks)?;
            continue;
        }
        if args::is(word, b"DIALECT") {
            return Err(plain(
                "SEARCH_PARSE_ARGS DIALECT is not supported in FT.HYBRID or any of its subqueries. Please check the documentation on search-default-dialect configuration.",
            ));
        }
        // A grouping over nothing at all is a whole table in one row, which an
        // aggregation takes and this does not.
        if args::is(word, b"GROUPBY") && args.opt(at + 1).and_then(counting).is_none() {
            return Err(about(b"GROUPBY", b"Invalid argument count"));
        }
        if let Some(next) = super::step(args, at, &mut asks.asked, index)? {
            asks.loaded |= args::is(word, b"LOAD");
            at = next;
            continue;
        }
        if args::is(word, b"PARAMS") {
            asks.only(b"PARAMS")?;
            // The count is looked at here rather than left to the shared reader,
            // because a count of nought is an argument error in this command
            // where it is a pairing error in the others.
            if args.opt(at + 1).and_then(counting).is_none() {
                return Err(about(b"PARAMS", b"Invalid argument count"));
            }
            at = super::params(args, at, &mut asks.asked)?;
            continue;
        }
        if args::is(word, b"LIMIT") {
            asks.only(b"LIMIT")?;
        }
        if args::is(word, b"TIMEOUT") {
            asks.only(b"TIMEOUT")?;
            let Some(value) = args.opt(at + 1) else {
                return Err(convert(b"TIMEOUT"));
            };
            let Some(read) = parse_i64(value) else {
                return Err(convert(b"TIMEOUT"));
            };
            // Nought is no limit at all, which is longer than the longest a
            // foreground query is allowed to run for, so it is capped and said
            // so about the same way a number over the limit is.
            if read <= 0 || read > LONGEST {
                asks.warn = Some(CAPPED);
            }
            at += 2;
            continue;
        }
        if args::is(word, b"EXPLAINSCORE") {
            asks.only(b"EXPLAINSCORE")?;
            at += 1;
            continue;
        }
        if let Some(next) = super::plan(args, at, &mut asks.asked, super::Mode::Aggregate)? {
            at = next;
            continue;
        }
        return Err(unknown(word));
    }
    Ok(())
}

impl Asks<'_> {
    /// Refuses a word that has already been written.
    fn only(&mut self, name: &'static [u8]) -> core::result::Result<(), Vec<u8>> {
        if self.once.contains(&name) {
            return Err(twice(name));
        }
        self.once.push(name);
        Ok(())
    }
}

/// `KNN <count> K <k> [<option> <value>]...` and the range form beside it.
///
/// The count is a count of words rather than of pairs, so it has to be even and
/// it says exactly how far this section reaches: a count that runs past the end
/// of the section reads whatever came next as a pair of its own and refuses it
/// by name.
fn neighbours<'a>(
    args: Args<'a>,
    at: usize,
    section: &[u8],
    reach: &mut Option<Reach>,
    options: &mut Vec<(&'a [u8], &'a [u8])>,
) -> core::result::Result<usize, Vec<u8>> {
    let knn = section == b"KNN";
    let Some(count) = args.opt(at + 1).and_then(counting_zero) else {
        return Err(plain(
            "SEARCH_PARSE_ARGS Invalid argument count: expected an unsigned integer",
        ));
    };
    if count == 0 || count % 2 != 0 {
        let mut out = b"SEARCH_SYNTAX Invalid argument count: ".to_vec();
        out.extend_from_slice(count.to_string().as_bytes());
        out.extend_from_slice(b" (must be a positive even number for key/value pairs)");
        return Err(out);
    }
    let count = usize::try_from(count).unwrap_or(0);
    // Each of the two sections takes its own word and one runtime option, and
    // not the other's. Whether the field behind the clause actually takes that
    // option is a different question, and it is the query parser's to answer.
    let (want, extra): (&[u8], &[u8]) = match knn {
        true => (b"K", b"EF_RUNTIME"),
        false => (b"RADIUS", b"EPSILON"),
    };
    let mut found = None;
    let mut next = at + 2;
    let end = next + count;
    while next < end {
        let (Some(name), Some(held)) = (args.opt(next), args.opt(next + 1)) else {
            return Err(plain(
                "SEARCH_PARSE_ARGS Invalid argument count: expected an unsigned integer",
            ));
        };
        if args::is(name, want) {
            found = Some(held);
        } else if args::is(name, extra) {
            // An `EF_RUNTIME` counts and an `EPSILON` measures, and both of
            // them have to be above nought.
            let fine = match knn {
                true => counting(held).is_some(),
                false => parse_f64(held).is_some_and(|read| read > 0.0),
            };
            if !fine {
                return Err(value(extra));
            }
            options.push((name, held));
        } else {
            return Err(inside(name, section));
        }
        next += 2;
    }
    let Some(found) = found else {
        let mut out = b"SEARCH_PARSE_ARGS Missing required argument ".to_vec();
        out.extend_from_slice(want);
        return Err(out);
    };
    *reach = Some(match knn {
        true => match counting(found) {
            Some(k) => Reach::Knn(k),
            None => return Err(value(b"K")),
        },
        // A radius of nought is a real question with an empty answer, so only a
        // negative one and a word are refused.
        false => match parse_f64(found).filter(|read| *read >= 0.0) {
            Some(_) => Reach::Range(found.into()),
            None => return Err(value(b"RADIUS")),
        },
    });
    Ok(end)
}

/// `COMBINE RRF <count> [CONSTANT c] [WINDOW w]` and the linear form.
fn combined(
    args: Args<'_>,
    at: usize,
    asks: &mut Asks<'_>,
) -> core::result::Result<usize, Vec<u8>> {
    let algo = args.get(at + 1);
    let rrf = args::is(algo, b"RRF");
    if !rrf && !args::is(algo, b"LINEAR") {
        return Err(about(b"COMBINE", b"Invalid value for argument"));
    }
    let name: &[u8] = match rrf {
        true => b"RRF",
        false => b"LINEAR",
    };
    let Some(count) = args.opt(at + 2).and_then(counting_zero) else {
        let mut out = b"SEARCH_PARSE_ARGS Invalid ".to_vec();
        out.extend_from_slice(name);
        out.extend_from_slice(
            b" argument count, error: Could not convert argument to expected type",
        );
        return Err(out);
    };
    if count % 2 != 0 {
        let mut out = b"SEARCH_PARSE_ARGS ".to_vec();
        out.extend_from_slice(name);
        out.extend_from_slice(
            b" expects pairs of key value arguments, argument count must be an even number",
        );
        return Err(out);
    }
    let count = usize::try_from(count).unwrap_or(0);
    let mut constant = CONSTANT;
    let mut window = WINDOW;
    let mut alpha = None;
    let mut beta = None;
    let mut next = at + 3;
    let end = next + count;
    while next < end {
        let held = args.get(next);
        let Some(value) = args.opt(next + 1) else {
            let mut out = b"SEARCH_SYNTAX Missing value for ".to_vec();
            out.extend_from_slice(held);
            return Err(out);
        };
        let number = |name: &[u8]| parse_f64(value).ok_or_else(|| convert(name));
        if args::is(held, b"WINDOW") {
            let Some(read) = counting(value) else {
                return match parse_i64(value).is_some() {
                    true => Err(about(b"WINDOW", b"Value below minimum")),
                    false => Err(convert(b"WINDOW")),
                };
            };
            window = usize::try_from(read).unwrap_or(usize::MAX);
        } else if rrf && args::is(held, b"CONSTANT") {
            constant = number(b"CONSTANT")?;
        } else if !rrf && args::is(held, b"ALPHA") {
            alpha = Some(number(b"ALPHA")?);
        } else if !rrf && args::is(held, b"BETA") {
            beta = Some(number(b"BETA")?);
        } else {
            return Err(unknown(held));
        }
        next += 2;
    }
    // Both weights are required together, and a list that ran out before the
    // second one is refused for the value that never arrived rather than for
    // the keyword.
    if !rrf && (alpha.is_none() || beta.is_none()) {
        let missing: &[u8] = match alpha.is_none() {
            true => b"ALPHA",
            false => b"BETA",
        };
        let mut out = b"SEARCH_SYNTAX Missing value for ".to_vec();
        out.extend_from_slice(missing);
        return Err(out);
    }
    asks.combine = match rrf {
        true => Combine::Rrf { constant, window },
        false => Combine::Linear {
            alpha: alpha.unwrap_or(ALPHA),
            beta: beta.unwrap_or(BETA),
            window,
        },
    };
    Ok(end)
}

/// A count that has to be there and has to be above nought.
fn counting(src: &[u8]) -> Option<u64> {
    counting_zero(src).filter(|held| *held > 0)
}

/// The same, with nought allowed.
fn counting_zero(src: &[u8]) -> Option<u64> {
    let text = core::str::from_utf8(src).ok()?;
    let body = text.strip_prefix('+').unwrap_or(text);
    match body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(hex, 16).ok(),
        None => body.parse::<u64>().ok(),
    }
}

/// The two queries the two branches run, built out of what was read.
fn branches(asks: &Asks<'_>, index: &Index) -> core::result::Result<(Node, Node), Vec<u8>> {
    let ask = Ask {
        dialect: 2,
        params: &asks.asked.params,
        verbatim: asks.asked.verbatim,
        stopwords: asks.asked.stopwords,
    };
    let text = query::parse(asks.text, index, &ask).map_err(|bad| super::refused(&bad))?;
    // The vector branch is written out as a query and read by the query parser,
    // so a runtime option the field will not take is refused in exactly the
    // words a bracket written by hand is refused in.
    let mut src: Vec<u8> = Vec::new();
    let wide = Reach::Knn(asks.combine.window() as u64);
    match asks.reach.as_ref().unwrap_or(&wide) {
        Reach::Knn(k) => {
            src.extend_from_slice(b"*=>[KNN ");
            src.extend_from_slice(k.to_string().as_bytes());
            src.extend_from_slice(b" @");
            src.extend_from_slice(asks.field);
            src.extend_from_slice(b" $");
            src.extend_from_slice(asks.param);
            for (name, value) in &asks.options {
                src.push(b' ');
                src.extend_from_slice(name);
                src.push(b' ');
                src.extend_from_slice(value);
            }
            src.extend_from_slice(b" AS ");
            src.extend_from_slice(AWAY);
            src.push(b']');
        }
        Reach::Range(radius) => {
            src.push(b'@');
            src.extend_from_slice(asks.field);
            src.extend_from_slice(b":[VECTOR_RANGE ");
            src.extend_from_slice(radius);
            src.extend_from_slice(b" $");
            src.extend_from_slice(asks.param);
            src.extend_from_slice(b"]=>{$yield_distance_as: ");
            src.extend_from_slice(AWAY);
            for (name, value) in &asks.options {
                src.extend_from_slice(b"; $");
                src.extend_from_slice(&name.to_ascii_lowercase());
                src.extend_from_slice(b": ");
                src.extend_from_slice(value);
            }
            src.push(b'}');
        }
    }
    let mut vector = query::parse(&src, index, &ask).map_err(|bad| super::refused(&bad))?;
    // The `VSIM` filter is query text and is read on its own, so what it is
    // refused for names an offset into what the client wrote rather than into
    // the clause that was built around it.
    if let Some(src) = asks.filter {
        let over = query::parse(src, index, &ask).map_err(|bad| super::refused(&bad))?;
        match &mut vector.what {
            // A nearest neighbour clause has a place of its own for what it
            // takes its neighbours from.
            What::Vector(held) if held.k.is_some() => held.over = Some(Box::new(over)),
            // A range clause has none, and narrowing it afterwards is the same
            // question asked the other way round.
            _ => vector = Node::new(What::Intersect(vec![over, vector])),
        }
    }
    Ok((text, vector))
}

/// Runs both branches and folds them into one list of rows.
fn merged(index: &Index, asks: &Asks<'_>, text: Node, vector: Node) -> (usize, Vec<Row>) {
    let window = asks.combine.window();
    let mut want = Rows {
        scorer: asks.scorer,
        count: usize::MAX,
        ..Rows::default()
    };
    let shaped = super::shape(text, index, &want);
    let (_, texts, _) = super::gather(index, shaped, &want, Order::Ranked, true, false);
    let held = query::yields(&vector);
    want.nearest = held
        .iter()
        .find(|held| held.ordered)
        .map(|held| held.name.clone());
    want.distance = held;
    let shaped = super::shape(vector, index, &want);
    let (_, mut nears, _) = super::gather(index, shaped, &want, Order::Forwards, true, false);
    // A range clause does not put its answer in distance order the way a
    // nearest neighbour clause does, and the merge ranks by distance either
    // way, so the order is settled here rather than left to the walk.
    nears.sort_by(|left, right| {
        let held = |row: &Row| row.away(AWAY).unwrap_or(f64::INFINITY);
        held(left)
            .partial_cmp(&held(right))
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    // The window cuts both branches, and a `KNN` count only narrows its own
    // further, so a `KNN 2 K 4` beside a `WINDOW 1` merges one row of vectors
    // and not four.
    let reach = match &asks.reach {
        Some(Reach::Knn(k)) => usize::try_from(*k).unwrap_or(usize::MAX).min(window),
        Some(Reach::Range(_)) | None => window,
    };
    let mut rows: Vec<Row> = Vec::new();
    let mut where_at: Vec<(Box<[u8]>, usize)> = Vec::new();
    let place =
        |rows: &mut Vec<Row>, where_at: &mut Vec<(Box<[u8]>, usize)>, key: &[u8]| match where_at
            .iter()
            .find(|(held, _)| **held == *key)
        {
            Some((_, at)) => *at,
            None => {
                rows.push(Row {
                    key: key.into(),
                    score: 0.0,
                    payload: None,
                    note: None,
                    sort: None,
                    dists: Vec::new(),
                });
                where_at.push((key.into(), rows.len() - 1));
                rows.len() - 1
            }
        };
    for (rank, row) in texts.iter().take(window).enumerate() {
        let at = place(&mut rows, &mut where_at, &row.key);
        rows[at].score += match &asks.combine {
            Combine::Rrf { constant, .. } => 1.0 / (constant + (rank + 1) as f64),
            Combine::Linear { alpha, .. } => alpha * row.score,
        };
        if let Some(name) = &asks.scored {
            rows[at].dists.push((name.clone(), row.score));
        }
    }
    for (rank, row) in nears.iter().take(reach).enumerate() {
        let away = row.away(AWAY).unwrap_or(f64::INFINITY);
        let close = 1.0 / (1.0 + away);
        let at = place(&mut rows, &mut where_at, &row.key);
        rows[at].score += match &asks.combine {
            Combine::Rrf { constant, .. } => 1.0 / (constant + (rank + 1) as f64),
            Combine::Linear { beta, .. } => beta * close,
        };
        if let Some(name) = &asks.nears {
            rows[at].dists.push((name.clone(), close));
        }
    }
    // Best first, and a document that came out of both branches with the same
    // number as one that came out of one keeps the order the branches were read
    // in.
    rows.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then_with(|| left.key.cmp(&right.key))
    });
    let total = rows.len();
    (total, rows)
}

/// Runs the pipeline over the merged rows and writes the reply.
fn writes(
    server: &Server,
    db: usize,
    total: usize,
    rows: &[Row],
    asks: &Asks<'_>,
    spent: core::time::Duration,
    out: &mut Out,
) {
    let watch: Option<&mut Watch> = None;
    let made = aggregate::runs(server, db, total, rows, &asks.asked, watch);
    let shown: Vec<(Option<&Row>, &Vec<yo_search::expr::Value>)> = made
        .table
        .iter()
        .map(|held| (held.from.map(|at| &rows[at]), &held.values))
        .collect();
    let deep = out.proto().is_resp3();
    match deep {
        true => out.map(4),
        false => out.array(8),
    }
    out.bulk(b"total_results");
    out.int((made.start - made.gone.min(made.start)) as i64);
    out.bulk(b"results");
    out.array(shown.len());
    for (_, values) in &shown {
        aggregate::mapped(&made.names, values, None, out);
    }
    out.bulk(b"warnings");
    let said = made.warning.as_deref().or(asks.warn.map(str::as_bytes));
    match said {
        Some(warning) => {
            out.array(1);
            out.bulk(warning);
        }
        None => out.array(0),
    }
    out.bulk(b"execution_time");
    let ms = spent.as_secs_f64() * 1000.0;
    match deep {
        true => out.double(ms),
        false => out.bulk(format!("{ms:.6}").as_bytes()),
    }
}
