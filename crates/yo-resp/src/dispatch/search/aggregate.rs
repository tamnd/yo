//! The part of `FT.AGGREGATE` that reshapes an answer rather than reading it.
//!
//! A search hands back one row per document. A pipeline step folds those rows
//! into other rows, and from the first step on a row is a thing of its own
//! rather than a view of a key, which is why it lives here as a list of
//! [`Value`] rather than as a list of borrowed field pairs.
//!
//! The parsing side of a step and the running side of it are both in here,
//! because the two halves agree about where a property is read from through a
//! position in that list and nothing else.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use yo_common::num::parse_f64;
use yo_common::parse_i64;
use yo_search::Index;
use yo_search::expr::{Expr, Value, seventeen};
use yo_search::field::Kind;
use yo_search::reduce;
use yo_search::token;

use super::{
    AS_SHORT, Args, Asked, BAD_ARGS, COUNT_ONLY, DUPLICATE_END, DUPLICATE_PROP, GROUP_COUNT,
    GROUP_SHORT, MAX_COUNT, MISSING_ARGS, MOST_SAMPLE, NO_AT, NO_AT_END, NO_AT_MID, NO_PROPERTY,
    NO_REDUCER, NOT_A_NUMBER, NOT_LOADED, NOT_MAIN, NOT_THERE, OUT_OF_RANGE, PERCENTAGE, QUOTE_END,
    REDUCE_BARE, RESOLUTION, RESOLUTION_ARG, Row, SAMPLE_BIG, SAMPLE_SIZE, SORT_BOUNDS, SORT_COUNT,
    SORT_PROP, SORT_PROP_END, SORT_SHORT, SORT_TWICE, SORT_WAY, SORT_WAY_END, STEP_SHORT, UNKNOWN,
    line, twelve,
};
use crate::dispatch::Server;
use crate::dispatch::args;
use crate::dispatch::indexing;
use crate::reply::Out;

/// The half of the argument list only `FT.AGGREGATE` fills in.
///
/// The other two commands leave every field of this alone, because none of the
/// words that write it are words they take.
#[derive(Default)]
pub(super) struct Pipe<'a> {
    /// The fields to read off each key and the names to answer them under, in
    /// the order the client named them.
    pub(super) load: Vec<(&'a [u8], &'a [u8])>,
    /// Whether `LOAD *` asked for everything the key holds as well.
    pub(super) all: bool,
    /// Whether anything at all asked for a field, which is the one thing that
    /// decides how the count at the front of the reply is worked out.
    pub(super) loader: bool,
    /// Whether the score of each document is answered as a `__score` property.
    pub(super) addscores: bool,
    /// Whether a sort key element goes on each row. It holds the value the
    /// first `SORTBY` key found on that row and is null on a pipeline that
    /// never sorted, on a row a group step made after the sort, and on a row
    /// the sort could not find the property on.
    pub(super) sortkeys: bool,
    /// A `LOAD` whose count ran out on the `AS`, which is not an error on its
    /// own: it is only reported once the rest of the list has read cleanly.
    /// `LOAD 2 @t AS LIMIT 0 1` is refused for the missing name and
    /// `LOAD 2 @t AS VERBATIM` is refused for the `VERBATIM`, because the word
    /// after the `AS` is read as an argument of its own either way.
    pub(super) pending: bool,
    /// Whether a step that builds the pipeline has been read. `LOAD` is one and
    /// `LIMIT`, `TIMEOUT`, `DIALECT` and `PARAMS` are not, and after one of
    /// them the words about the search itself stop being taken: a real server
    /// answers `Unknown argument VERBATIM` for `LOAD 1 @t VERBATIM` and takes
    /// the same `VERBATIM` after a `LIMIT`.
    pub(super) stepped: bool,
    /// The row a document turns into before any step has run: where each
    /// property is read from and what it is called.
    ///
    /// A `LOAD` writes one of these, and so does every schema field a step
    /// names, because a schema field is readable without being loaded and a
    /// field that is not in the schema is not. There is one more rule behind
    /// that which only shows up beside a `LOAD *`: after one of those any name
    /// at all is readable, because the key might turn out to hold it.
    pub(super) base: Vec<(Box<[u8]>, Reads)>,
    /// The steps that reshape the rows, in the order they were written.
    pub(super) steps: Vec<Step>,
    /// Where the sorting step that `SORTBY`, `MAX` and `LIMIT` all write to
    /// sits in [`Pipe::steps`], when one of them has opened one.
    ///
    /// The three of them share a step rather than each having one of their own,
    /// which is measured and is the whole reason this is here.
    /// `SORTBY 1 @n MAX 3 APPLY '1' AS y LIMIT 0 5` answers five rows, so the
    /// `LIMIT` reached back past the `APPLY` and overwrote the `MAX`, and
    /// `SORTBY 1 @n FILTER '@n > 2' LIMIT 0 1` answers no rows at all, because
    /// the window it reached back to cut the rows down to one before the filter
    /// ever saw them. A `GROUPBY` closes the step, so a `LIMIT` after one is a
    /// window on the groups and not on the documents they were folded from.
    pub(super) arrange: Option<usize>,
    /// What the row looks like where the parser has got to, which is the base
    /// row until the first `GROUPBY` and that step's own output after it.
    ///
    /// This is what makes a second `GROUPBY` see the first one's answer and
    /// nothing else. A schema field is not readable after a step, because the
    /// document it would have been read off is not there any more.
    pub(super) stage: Option<Vec<Box<[u8]>>>,
}

/// Where one property of the base row is read from.
pub(super) enum Reads {
    /// A field of the key. The field and the name the property answers under
    /// differ whenever an `AS` renamed it or the schema declared it with one.
    Field(Box<[u8]>, Shape),
    /// The score the query gave the document, which is only a property when
    /// `ADDSCORES` asked for it.
    Score,
    /// Nothing off the key at all: a slot an `APPLY` fills in.
    Made,
}

/// What the bytes of a field become when they are read onto a row.
///
/// A field with a number behind it is a number and not the digits it was
/// written with, which is measured rather than assumed: `strlen(@n)` over a
/// `NUMERIC` field is refused for the type of its argument whether or not a
/// `LOAD` named it. The third one is the sortable copy, which a real server
/// keeps folded unless the schema said `UNF`, so a field holding `HeLLo` reads
/// back as `hello` through the sort and as `HeLLo` through a `LOAD`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Shape {
    Words,
    Number,
    Folded,
}

/// One step of the pipeline.
pub(super) enum Step {
    Group(Group),
    /// An expression written to a slot of the row, which is a new one unless
    /// the name it was given already meant something.
    Apply {
        at: usize,
        name: Box<[u8]>,
        expr: Expr,
    },
    /// An expression every row has to answer true to.
    Filter(Expr),
    /// An order to put the rows in and a window to keep of them, either of
    /// which may be there without the other.
    Sort(Sort),
}

/// One sorting step, which is what a `SORTBY` and a `LIMIT` both write.
#[derive(Default)]
pub(super) struct Sort {
    /// Where each property the rows are ordered by is read from, and whether
    /// that one runs the other way round.
    keys: Vec<(usize, bool)>,
    /// Where the window starts and how many rows it keeps, which is every row
    /// left when nothing said.
    offset: usize,
    count: Option<usize>,
}

/// Which properties a step may read.
///
/// A `GROUPBY` and a `REDUCE` read any field of the schema whether or not
/// anything loaded it. An `APPLY` and a `FILTER` read what is on the row and
/// nothing else, except that a sortable field is on the row without being
/// loaded, because a real server keeps a copy of one beside the document.
///
/// A `SORTBY` reads any field of the schema as well, and there it stops: it is
/// the one step a `LOAD *` does not open up, so `LOAD * SORTBY 1 @e` over a key
/// holding an `e` that the schema never heard of is refused.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Look {
    Any,
    Named,
    Sorted,
}

/// One `GROUPBY` and the reducers hanging off it.
pub(super) struct Group {
    /// The properties the rows are gathered by: where each is read from in the
    /// row in front of this step, and what it is called in the row after it.
    by: Vec<(usize, Box<[u8]>)>,
    /// What each group is folded to.
    folds: Vec<Reducer>,
}

/// One `REDUCE`.
pub(super) struct Reducer {
    kind: reduce::Kind,
    /// Where the value it folds is read from, which `COUNT` does not have
    /// because it counts documents rather than values.
    of: Option<usize>,
    /// Where the value it orders by is read from, which only `FIRST_VALUE` has
    /// and only when it was given a `BY`.
    by: Option<usize>,
    /// The name the answer comes back under, which is either the `AS` or the
    /// generated one [`generated`] builds.
    name: Box<[u8]>,
}

/// `GROUPBY nargs @property... [REDUCE ...]...`.
///
/// The properties come first and every one of them has to carry its `@`, which
/// is the one place in the whole command where leaving it off is named rather
/// than read as something else. Then any number of `REDUCE` clauses, each of
/// which folds the group to one more property.
pub(super) fn group<'a>(
    args: Args<'a>,
    at: usize,
    asked: &mut Asked<'a>,
    index: &Index,
) -> core::result::Result<usize, Vec<u8>> {
    asked.pipe.stepped = true;
    let mut at = at + 1;
    let Some(count) = args.opt(at) else {
        return Err(GROUP_SHORT.as_bytes().to_vec());
    };
    at += 1;
    let Some(count) = parse_i64(count) else {
        return Err(GROUP_COUNT.as_bytes().to_vec());
    };
    // A count below nought is read as a count that ran off the end rather than
    // as a number out of range, which is measured: `GROUPBY -1` and `GROUPBY 9`
    // over a list with nothing left in it answer the same line.
    let Ok(count) = usize::try_from(count) else {
        return Err(GROUP_SHORT.as_bytes().to_vec());
    };
    let mut by: Vec<(usize, Box<[u8]>)> = Vec::with_capacity(count);
    let mut names: Vec<Box<[u8]>> = Vec::with_capacity(count);
    for _ in 0..count {
        let Some(word) = args.opt(at) else {
            return Err(GROUP_SHORT.as_bytes().to_vec());
        };
        at += 1;
        let Some(name) = word.strip_prefix(b"@") else {
            let mut out = line(NO_AT, word, NO_AT_MID);
            out.extend_from_slice(word);
            out.extend_from_slice(NO_AT_END.as_bytes());
            return Err(out);
        };
        if names.iter().any(|held| **held == *name) {
            return Err(line(DUPLICATE_PROP, name, DUPLICATE_END));
        }
        let Some(from) = locate(&mut asked.pipe, index, name, Look::Any) else {
            return Err(line(NO_PROPERTY, name, QUOTE_END));
        };
        by.push((from, name.into()));
        names.push(name.into());
    }
    let mut folds: Vec<Reducer> = Vec::new();
    while args.opt(at).is_some_and(|word| args::is(word, b"REDUCE")) {
        let fold = reducer(args, &mut at, asked, index)?;
        if names.contains(&fold.name) {
            return Err(line(DUPLICATE_PROP, &fold.name, DUPLICATE_END));
        }
        names.push(fold.name.clone());
        folds.push(fold);
    }
    asked.pipe.steps.push(Step::Group(Group { by, folds }));
    // A sorting step in front of a group step is finished with: a `LIMIT` after
    // this one is a window on the groups.
    asked.pipe.arrange = None;
    // Everything the row held before this step is gone, and what is left is the
    // group properties and whatever the reducers answered.
    asked.pipe.stage = Some(names);
    Ok(at)
}

/// One `REDUCE FUNC nargs arg... [AS name]`, with `at` left on the word after
/// it.
///
/// The order the pieces are checked in is measured and is not the order they
/// are written in. The count is read before the function is looked up, so
/// `REDUCE NOPE` complains about the missing count rather than the unknown
/// reducer. The arguments are read before that lookup too, so `REDUCE NOPE 5
/// @n` complains about the four arguments that are not there. Only then does
/// the name have to mean something.
fn reducer<'a>(
    args: Args<'a>,
    at: &mut usize,
    asked: &mut Asked<'a>,
    index: &Index,
) -> core::result::Result<Reducer, Vec<u8>> {
    *at += 1;
    let Some(func) = args.opt(*at) else {
        return Err(REDUCE_BARE.as_bytes().to_vec());
    };
    *at += 1;
    let Some(count) = args.opt(*at) else {
        return Err(line(BAD_ARGS, func, NOT_THERE));
    };
    *at += 1;
    let Some(count) = parse_i64(count) else {
        return Err(line(BAD_ARGS, func, NOT_A_NUMBER));
    };
    let Ok(count) = usize::try_from(count) else {
        return Err(line(BAD_ARGS, func, OUT_OF_RANGE));
    };
    let mut words: Vec<&[u8]> = Vec::with_capacity(count);
    for _ in 0..count {
        let Some(word) = args.opt(*at) else {
            return Err(line(BAD_ARGS, func, NOT_THERE));
        };
        *at += 1;
        words.push(word);
    }
    let (kind, of, by) = fold(func, &words, asked, index)?;
    // The name, which is either the one the client gave or the one a real
    // server builds out of everything it just read.
    let name = match args.opt(*at).is_some_and(|word| args::is(word, b"AS")) {
        false => generated(func, &words),
        true => {
            *at += 1;
            let Some(name) = args.opt(*at) else {
                return Err(line(BAD_ARGS, b"AS", NOT_THERE));
            };
            *at += 1;
            name.into()
        }
    };
    Ok(Reducer { kind, of, by, name })
}

/// Which reducer a name means and what it was pointed at.
///
/// The arguments past the ones a reducer reads are not an error for eight of
/// the twelve: `REDUCE SUM 2 @n @m` sums `@n` and puts `@m` in the generated
/// name and nowhere else. The two that pick a document and the one that takes a
/// quantile are strict about them instead, and name the position inside the
/// reducer's own list rather than inside the command.
#[expect(clippy::too_many_lines, reason = "twelve reducers and their arguments")]
fn fold(
    func: &[u8],
    words: &[&[u8]],
    asked: &mut Asked<'_>,
    index: &Index,
) -> core::result::Result<(reduce::Kind, Option<usize>, Option<usize>), Vec<u8>> {
    // Every reducer but `COUNT` reads its property out of the first argument,
    // and the `@` on it is optional there: `REDUCE SUM 1 n` is the same
    // question as `REDUCE SUM 1 @n`.
    let first = |asked: &mut Asked<'_>| match words.first() {
        None => Err(line(MISSING_ARGS, func, "")),
        Some(word) => {
            let name = word.strip_prefix(b"@").unwrap_or(word);
            match locate(&mut asked.pipe, index, name, Look::Any) {
                Some(from) => Ok(from),
                None => Err(line(NOT_LOADED, name, QUOTE_END)),
            }
        }
    };
    if args::is(func, b"COUNT") {
        if !words.is_empty() {
            return Err(COUNT_ONLY.as_bytes().to_vec());
        }
        return Ok((reduce::Kind::Count, None, None));
    }
    for (name, kind) in [
        (b"SUM".as_slice(), reduce::Kind::Sum),
        (b"MIN", reduce::Kind::Min),
        (b"MAX", reduce::Kind::Max),
        (b"AVG", reduce::Kind::Avg),
        (b"STDDEV", reduce::Kind::Stddev),
        (b"TOLIST", reduce::Kind::ToList),
        (b"COUNT_DISTINCT", reduce::Kind::Distinct),
        (b"COUNT_DISTINCTISH", reduce::Kind::Distinctish),
    ] {
        if args::is(func, name) {
            return Ok((kind, Some(first(asked)?), None));
        }
    }
    if args::is(func, b"QUANTILE") {
        let of = first(asked)?;
        // A real server reads the fraction without checking that it is there
        // and dereferences a null, which takes the whole process down. This
        // side answers the line the reducer with no arguments at all answers,
        // and that is divergence D-71.
        let Some(want) = words.get(1) else {
            return Err(line(MISSING_ARGS, func, ""));
        };
        let Some(want) = parse_f64(want) else {
            return Err(line(BAD_ARGS, func, NOT_A_NUMBER));
        };
        if !(0.0..=1.0).contains(&want) {
            return Err(PERCENTAGE.as_bytes().to_vec());
        }
        // The resolution decides how fine the sketch a real server keeps is. It
        // is read and checked and then dropped, because this side sorts the
        // group and takes the value rather than estimating it.
        if let Some(fine) = words.get(2) {
            let Some(fine) = parse_i64(fine) else {
                return Err(line(RESOLUTION_ARG, b"", NOT_A_NUMBER));
            };
            if !(1..=MOST_SAMPLE).contains(&fine) {
                return Err(RESOLUTION.as_bytes().to_vec());
            }
        }
        if let Some(word) = words.get(3) {
            return Err(inside(word, 3, func));
        }
        return Ok((reduce::Kind::Quantile(want), Some(of), None));
    }
    if args::is(func, b"RANDOM_SAMPLE") {
        let of = first(asked)?;
        let Some(size) = words.get(1) else {
            return Err(line(SAMPLE_SIZE, b"", NOT_THERE));
        };
        let Some(size) = parse_i64(size) else {
            return Err(line(SAMPLE_SIZE, b"", NOT_A_NUMBER));
        };
        if size < 0 {
            return Err(line(SAMPLE_SIZE, b"", OUT_OF_RANGE));
        }
        if size > MOST_SAMPLE {
            return Err(SAMPLE_BIG.as_bytes().to_vec());
        }
        let size = usize::try_from(size).unwrap_or(0);
        return Ok((reduce::Kind::Sample(size), Some(of), None));
    }
    if args::is(func, b"FIRST_VALUE") {
        let of = first(asked)?;
        let mut order = reduce::Order {
            by: false,
            desc: false,
            numeric: false,
        };
        let mut key = None;
        if let Some(word) = words.get(1) {
            if !args::is(word, b"BY") {
                return Err(inside(word, 1, func));
            }
            let Some(word) = words.get(2) else {
                return Err(line(MISSING_ARGS, func, ""));
            };
            let name = word.strip_prefix(b"@").unwrap_or(word);
            let Some(from) = locate(&mut asked.pipe, index, name, Look::Any) else {
                return Err(line(NOT_LOADED, name, QUOTE_END));
            };
            order.by = true;
            // Whether two of these values are put in order as numbers is
            // decided by the field they came off and not by what they hold, so
            // a `NUMERIC` field orders 9 before 10 and a field with no type
            // behind it orders 10 before 9.
            order.numeric = index
                .field(name)
                .is_some_and(|f| matches!(f.kind, Kind::Numeric));
            key = Some(from);
            if let Some(word) = words.get(3) {
                match () {
                    () if args::is(word, b"ASC") => {}
                    () if args::is(word, b"DESC") => order.desc = true,
                    () => return Err(inside(word, 3, func)),
                }
            }
            if let Some(word) = words.get(4) {
                return Err(inside(word, 4, func));
            }
        }
        return Ok((reduce::Kind::First(order), Some(of), key));
    }
    Err(line(NO_REDUCER, func, ""))
}

/// The line a reducer answers for an argument of its own that nobody knows.
///
/// The position counts from nought inside the reducer's argument list rather
/// than from the query, which is why this is not [`unknown`].
fn inside(word: &[u8], position: usize, func: &[u8]) -> Vec<u8> {
    let mut out = UNKNOWN.as_bytes().to_vec();
    out.extend_from_slice(word);
    out.extend_from_slice(NOT_MAIN.as_bytes());
    out.extend_from_slice(position.to_string().as_bytes());
    out.extend_from_slice(b" for ");
    out.extend_from_slice(func);
    out
}

/// The name a reducer answers under when the client did not give it one.
///
/// A fixed head, then the function and every argument it was given with the
/// `@` taken off, joined by commas and folded to lower case. So
/// `REDUCE FIRST_VALUE 4 @n BY @n DESC` answers under
/// `__generated_aliasfirst_valuen,by,n,desc`.
fn generated(func: &[u8], words: &[&[u8]]) -> Box<[u8]> {
    let mut out = b"__generated_alias".to_vec();
    out.extend_from_slice(&func.to_ascii_lowercase());
    for (at, word) in words.iter().enumerate() {
        if at > 0 {
            out.push(b',');
        }
        let word = word.strip_prefix(b"@").unwrap_or(word);
        out.extend_from_slice(&word.to_ascii_lowercase());
    }
    out.into()
}

/// Where a step reads a property from, adding it to the base row when it is a
/// field of the key that nothing has read yet.
///
/// Before the first group step a property is one of four things: something a
/// `LOAD` or an `APPLY` already named, the score when `ADDSCORES` asked for it,
/// a field of the schema, or anything at all once a `LOAD *` has said the whole
/// key is coming. After a group step it is only one thing, a property that step
/// answered, because the key the rest would have been read off is not under the
/// row any more.
///
/// Which fields of the schema count is the one thing the two kinds of step
/// disagree about, and [`Look`] carries the answer.
fn locate(pipe: &mut Pipe<'_>, index: &Index, name: &[u8], look: Look) -> Option<usize> {
    if let Some(stage) = &pipe.stage {
        return stage.iter().position(|held| **held == *name);
    }
    if let Some(at) = pipe.base.iter().position(|(held, _)| **held == *name) {
        return Some(at);
    }
    if name == b"__score" {
        if !pipe.addscores {
            return None;
        }
        pipe.base.push((name.into(), Reads::Score));
        return Some(pipe.base.len() - 1);
    }
    let (from, shape) = match index.field(name) {
        Some(field) => {
            if look == Look::Sorted && !field.sortable && !pipe.all {
                return None;
            }
            let shape = match () {
                () if field.kind == Kind::Numeric => Shape::Number,
                // A field read through the sorting vector rather than off the
                // key comes back the way the index folded it, which is the one
                // place a value changes on its way onto a row.
                () if field.sortable && !field.is_unf() => Shape::Folded,
                () => Shape::Words,
            };
            (field.identifier.clone(), shape)
        }
        None if pipe.all && look != Look::Named => (name.into(), Shape::Words),
        None => return None,
    };
    pipe.base.push((name.into(), Reads::Field(from, shape)));
    Some(pipe.base.len() - 1)
}

/// Where an `APPLY` writes, which is a slot of its own unless the name it was
/// given already means something on the row.
///
/// An `APPLY` that names a property that is already there overwrites it where
/// it stands rather than answering it twice, so `LOAD 1 @n APPLY '@n' AS n`
/// answers one `n`.
fn makes(pipe: &mut Pipe<'_>, name: &[u8]) -> usize {
    if let Some(stage) = &mut pipe.stage {
        if let Some(at) = stage.iter().position(|held| **held == *name) {
            return at;
        }
        stage.push(name.into());
        return stage.len() - 1;
    }
    if let Some(at) = pipe.base.iter().position(|(held, _)| **held == *name) {
        return at;
    }
    pipe.base.push((name.into(), Reads::Made));
    pipe.base.len() - 1
}

/// Reads an expression and tells every property in it where it is read from.
fn built(src: &[u8], asked: &mut Asked<'_>, index: &Index) -> core::result::Result<Expr, Vec<u8>> {
    let mut expr = Expr::parse(src)?;
    expr.bind(&mut |name| locate(&mut asked.pipe, index, name, Look::Sorted))
        .map_err(|missing| line(NOT_LOADED, &missing.0, QUOTE_END))?;
    Ok(expr)
}

/// `APPLY expression [AS name]`.
///
/// Without the `AS` the property is answered under the expression as the client
/// wrote it, spaces and all, so `APPLY '1  +   1'` answers under `1  +   1`.
pub(super) fn apply<'a>(
    args: Args<'a>,
    at: usize,
    asked: &mut Asked<'a>,
    index: &Index,
) -> core::result::Result<usize, Vec<u8>> {
    asked.pipe.stepped = true;
    let Some(src) = args.opt(at + 1) else {
        return Err(STEP_SHORT.as_bytes().to_vec());
    };
    let mut at = at + 2;
    let name: Box<[u8]> = match args.opt(at).is_some_and(|word| args::is(word, b"AS")) {
        false => src.into(),
        true => {
            at += 1;
            let Some(name) = args.opt(at) else {
                return Err(AS_SHORT.as_bytes().to_vec());
            };
            at += 1;
            name.into()
        }
    };
    // The expression is read against the row in front of this step, which is
    // why the slot it writes to is worked out after it: `APPLY '@x + 1' AS x`
    // reads the old `x` and answers the new one.
    let expr = built(src, asked, index)?;
    let slot = makes(&mut asked.pipe, &name);
    asked.pipe.steps.push(Step::Apply {
        at: slot,
        name,
        expr,
    });
    Ok(at)
}

/// `FILTER expression`.
pub(super) fn keeps<'a>(
    args: Args<'a>,
    at: usize,
    asked: &mut Asked<'a>,
    index: &Index,
) -> core::result::Result<usize, Vec<u8>> {
    asked.pipe.stepped = true;
    let Some(src) = args.opt(at + 1) else {
        return Err(STEP_SHORT.as_bytes().to_vec());
    };
    let expr = built(src, asked, index)?;
    asked.pipe.steps.push(Step::Filter(expr));
    Ok(at + 2)
}

/// `SORTBY nargs [@property [ASC|DESC]]... [MAX num]`.
///
/// The count is a count of words rather than of properties, so a direction eats
/// one of them: `SORTBY 4 @g ASC @n DESC` orders by two properties and
/// `SORTBY 3 @n @g DESC` runs the second one backwards and the first one
/// forwards. A word in that window that is neither a property nor a direction
/// is named as a missing direction, and a direction with no property in front
/// of it is dropped, so `SORTBY 1 ASC` orders by nothing at all and is taken.
///
/// The whole list is read before any of it is looked up, which is measured:
/// `SORTBY 2 @zz NOPE` is refused for the `NOPE` and not for the `zz`.
pub(super) fn sorts<'a>(
    args: Args<'a>,
    at: usize,
    asked: &mut Asked<'a>,
    index: &Index,
) -> core::result::Result<usize, Vec<u8>> {
    let Some(count) = args.opt(at + 1) else {
        return Err(SORT_SHORT.as_bytes().to_vec());
    };
    let Some(count) = parse_i64(count) else {
        return Err(SORT_COUNT.as_bytes().to_vec());
    };
    // A count below nought is out of bounds where a `GROUPBY` reads the same
    // thing as a list that ran off the end. The two words do not answer alike.
    let Ok(count) = usize::try_from(count) else {
        return Err(SORT_BOUNDS.as_bytes().to_vec());
    };
    let mut at = at + 2;
    let mut keys: Vec<(&[u8], bool)> = Vec::new();
    for _ in 0..count {
        let Some(word) = args.opt(at) else {
            return Err(SORT_SHORT.as_bytes().to_vec());
        };
        at += 1;
        if let Some(name) = word.strip_prefix(b"@") {
            keys.push((name, false));
            continue;
        }
        let down = match () {
            () if args::is(word, b"ASC") => false,
            () if args::is(word, b"DESC") => true,
            () => return Err(line(SORT_WAY, word, SORT_WAY_END)),
        };
        if let Some((_, way)) = keys.last_mut() {
            *way = down;
        }
    }
    // `MAX` belongs to this word and to nowhere else: on its own or after the
    // `LIMIT` that followed a `SORTBY` it is an unknown argument.
    let mut most = None;
    if args.opt(at).is_some_and(|word| args::is(word, b"MAX")) {
        let Some(number) = args.opt(at + 1).and_then(parse_i64).filter(|n| *n >= 0) else {
            return Err(MAX_COUNT.as_bytes().to_vec());
        };
        // `MAX 0` is no cap at all, where `LIMIT 0 0` is a window holding
        // nothing, so the two write the same field with different words for
        // the same number.
        most = usize::try_from(number).ok().filter(|n| *n > 0);
        at += 2;
    }
    let spot = arranged(&mut asked.pipe);
    // Two of these in one pipeline is refused even when a step stands between
    // them, because both of them reach back to the same sorting step.
    if matches!(&asked.pipe.steps[spot], Step::Sort(sort) if !sort.keys.is_empty()) {
        return Err(SORT_TWICE.as_bytes().to_vec());
    }
    let mut found = Vec::with_capacity(keys.len());
    for (name, down) in keys {
        let Some(from) = locate(&mut asked.pipe, index, name, Look::Named) else {
            let mut out = SORT_PROP.as_bytes().to_vec();
            out.extend_from_slice(name);
            out.extend_from_slice(SORT_PROP_END.as_bytes());
            return Err(out);
        };
        found.push((from, down));
    }
    // A sort reads its properties onto the row without anything loading them,
    // and a row carrying a property is one of the things that decides how the
    // count at the front of the reply is worked out.
    asked.pipe.loader |= !found.is_empty();
    if let Some(most) = most {
        asked.rows.count = most;
    }
    if let Step::Sort(sort) = &mut asked.pipe.steps[spot] {
        sort.keys = found;
        if let Some(most) = most {
            sort.count = Some(most);
        }
    }
    Ok(at)
}

/// Where the sorting step the next window goes into sits, opening one on the
/// end of the pipeline when nothing has yet.
fn arranged(pipe: &mut Pipe<'_>) -> usize {
    if let Some(at) = pipe.arrange {
        return at;
    }
    pipe.steps.push(Step::Sort(Sort::default()));
    pipe.arrange = Some(pipe.steps.len() - 1);
    pipe.steps.len() - 1
}

/// What a `LIMIT` does to an aggregation, which is not what it does to a
/// search: it is a step of the pipeline standing where the client wrote it
/// rather than a window on the answer.
pub(super) fn windows(asked: &mut Asked<'_>, offset: usize, count: usize) {
    let spot = arranged(&mut asked.pipe);
    if let Step::Sort(sort) = &mut asked.pipe.steps[spot] {
        sort.offset = offset;
        sort.count = Some(count);
    }
}

/// One row of the pipeline and how many rows were thrown away before it.
///
/// The second half of that is what the count at the front of the reply is made
/// of, and it is carried per row because the count a client sees is the one
/// that stood when the first row of the reply was written.
struct Held {
    values: Vec<Value>,
    dropped: usize,
    /// Which document the row came off, which a row a group step made has
    /// nothing to answer.
    from: Option<usize>,
}

/// Reads the keys a pipeline needs, runs it, and writes what it made.
///
/// Every document that answered is read whatever the window is, because a step
/// that throws rows away changes which document the window lands on.
pub(super) fn piped(
    server: &Server,
    db: usize,
    total: usize,
    rows: &[Row],
    asked: &Asked<'_>,
    out: &mut Out,
) {
    let pipe = &asked.pipe;
    let mut names: Vec<Box<[u8]>> = pipe.base.iter().map(|(name, _)| name.clone()).collect();
    // The key only has to be read when something on the row comes off it, which
    // a pipeline grouping by `@__score` alone does not.
    let reads = pipe.all
        || pipe
            .base
            .iter()
            .any(|(_, from)| matches!(from, Reads::Field(..)));
    let mut table: Vec<Held> = Vec::with_capacity(rows.len());
    let mut lost = 0;
    for (at, row) in rows.iter().enumerate() {
        let doc = match reads {
            false => None,
            // A key that answered the query and is no longer there is left out
            // of the answer, the same way a search leaves it out of the reply.
            true => match indexing::read(&server.dbs[db], &row.key) {
                Some(doc) => Some(doc),
                None => {
                    lost += 1;
                    continue;
                }
            },
        };
        let pairs = doc.as_ref().map(indexing::Document::pairs);
        let mut made: Vec<Value> = Vec::with_capacity(pipe.base.len());
        for (_, from) in &pipe.base {
            made.push(match from {
                Reads::Made => Value::Missing,
                Reads::Score => Value::Text(twelve(row.score).into_bytes().into()),
                Reads::Field(id, shape) => {
                    match pairs.iter().flatten().find(|(held, _)| *held == &**id) {
                        Some((_, value)) => shaped(value, *shape),
                        None => Value::Missing,
                    }
                }
            });
        }
        // `LOAD *` puts whatever else the key holds on the end of the row.
        // Nothing can name one of those in an expression, because a name that
        // was written down has a slot of its own already.
        if pipe.all {
            for (field, value) in pairs.iter().flatten() {
                if pipe.base.iter().any(|(held, _)| **held == **field) {
                    continue;
                }
                let spot = match names.iter().position(|held| **held == **field) {
                    Some(spot) => spot,
                    None => {
                        names.push((*field).into());
                        names.len() - 1
                    }
                };
                made.resize_with(made.len().max(spot + 1), || Value::Missing);
                made[spot] = Value::Text((*value).into());
            }
        }
        made.resize_with(names.len(), || Value::Missing);
        table.push(Held {
            values: made,
            dropped: 0,
            from: Some(at),
        });
    }
    let mut start = total - lost;
    let mut warning = None;
    let mut grouped = false;
    let mut ranked = false;
    let mut gone = 0;
    // Which column the sort key beside each row is read from, which only a
    // `SORTBY` writes and a `GROUPBY` after one takes away again.
    let mut sorted = None;
    for step in &pipe.steps {
        match step {
            Step::Group(group) => {
                table = folded(group, &table);
                names = group
                    .by
                    .iter()
                    .map(|(_, name)| name.clone())
                    .chain(group.folds.iter().map(|fold| fold.name.clone()))
                    .collect();
                // A group is a row of its own with no key under it, so the
                // count starts again from the number of groups and the rows
                // that were folded into them are not in it.
                start = table.len();
                grouped = true;
                sorted = None;
            }
            Step::Apply { at, name, expr } => {
                match *at < names.len() {
                    true => names[*at] = name.clone(),
                    false => names.push(name.clone()),
                }
                let mut done = 0;
                for held in &mut table {
                    match expr.eval(&held.values) {
                        Ok(value) => {
                            held.values.resize_with(names.len(), || Value::Missing);
                            held.values[*at] = value;
                        }
                        Err(bad) => {
                            warning = Some(bad);
                            break;
                        }
                    }
                    done += 1;
                }
                // A row that cannot be worked out stops the pipeline where it
                // stands. What was written stays written and the rest is never
                // sent, which is why this is a warning and not an error.
                if warning.is_some() {
                    table.truncate(done);
                }
            }
            Step::Filter(expr) => {
                let mut over = 0;
                let mut stop = false;
                let seen = table.len();
                table.retain_mut(|held| {
                    if stop {
                        return false;
                    }
                    match expr.eval(&held.values) {
                        Ok(value) if value.truth() => {
                            held.dropped += over;
                            true
                        }
                        Ok(_) => {
                            over += 1;
                            false
                        }
                        Err(bad) => {
                            warning = Some(bad);
                            stop = true;
                            false
                        }
                    }
                });
                // Every row the step threw away counts against the number at the
                // front of the reply, whether or not a row lived long enough to
                // carry the tally, so the running total is kept beside it.
                gone += seen - table.len();
            }
            Step::Sort(sort) => {
                if let Some((first, _)) = sort.keys.first() {
                    // Rows that order the same keep the order they arrived in,
                    // which is the order the documents were walked in.
                    table.sort_by(|left, right| ordered(&sort.keys, left, right));
                    ranked = true;
                    // Only the first key of the list is answered as the sort
                    // key, however many the sort ran on.
                    sorted = Some(*first);
                }
                table.drain(..sort.offset.min(table.len()));
                if let Some(count) = sort.count {
                    table.truncate(count);
                }
            }
        }
        if warning.is_some() {
            break;
        }
    }
    let shown: Vec<(Option<&Row>, &Vec<Value>)> = table
        .iter()
        .map(|held| (held.from.map(|at| &rows[at]), &held.values))
        .collect();
    // A step that could not work a row out before a single row of the reply had
    // been written leaves nothing half sent, so the whole command answers the
    // error instead of a reply with the error hung off the end of it.
    if let (Some(bad), true) = (&warning, shown.is_empty()) {
        out.error(bad);
        return;
    }
    let counted = counting(
        &table,
        start,
        gone,
        asked,
        grouped || ranked,
        shown.len(),
        out,
    );
    writes(
        counted,
        &names,
        &shown,
        sorted,
        asked,
        warning.as_deref(),
        out,
    );
}

/// The number at the front of the reply.
///
/// It is not the number of rows that answered, except when it is. What it
/// really reports is how far the reply had got when the number went on the
/// wire, which is why it moves with the protocol and with whether anything read
/// a field. A pipeline that groups or sorts is the case where the whole answer
/// has to exist before any of it can be written, so there is nothing half done
/// to report and the number is the real one.
fn counting(
    table: &[Held],
    start: usize,
    gone: usize,
    asked: &Asked<'_>,
    settled: bool,
    shown: usize,
    out: &Out,
) -> usize {
    let want = &asked.rows;
    // Every row a `FILTER` threw away before the first row of the reply takes
    // one off the number, and one thrown away after it does not, because by
    // then the number has already gone. When no row of the reply was written at
    // all the number never went, so every row thrown away counts.
    let dropped = match table.first() {
        Some(held) => held.dropped,
        None => gone,
    };
    let count = start - dropped.min(start);
    let deep = out.proto().is_resp3();
    let whole = settled
        || want.count == 0
        || super::buffered(asked)
        || (asked.pipe.loader && want.offset == 0);
    if whole {
        return count;
    }
    let reached = match deep || asked.pipe.loader {
        true => shown,
        false => 1,
    };
    want.offset.saturating_add(reached).min(count)
}

/// Where one row goes beside another under a list of sort keys.
///
/// Two values order the way the four comparison operators order them, with one
/// thing on top of that: a row that does not hold the property at all goes to
/// the end whichever way the key runs, so `SORTBY 2 @m ASC` and
/// `SORTBY 2 @m DESC` both answer the document with no `m` last. A group key
/// that nothing filled in is not that. It is a null, it is a value in its own
/// right, and it sorts under everything else and turns round with the key.
fn ordered(keys: &[(usize, bool)], left: &Held, right: &Held) -> core::cmp::Ordering {
    for (from, down) in keys {
        let this = left.values.get(*from).unwrap_or(&Value::Missing);
        let that = right.values.get(*from).unwrap_or(&Value::Missing);
        let held = |value: &Value| !matches!(value, Value::Missing);
        match (held(this), held(that)) {
            (false, false) => continue,
            (false, true) => return core::cmp::Ordering::Greater,
            (true, false) => return core::cmp::Ordering::Less,
            (true, true) => {}
        }
        // Two values with nothing to compare are left where they are rather
        // than being an error, because there is no reply to hang one off here.
        let Some(way) = yo_search::expr::order(this, that) else {
            continue;
        };
        if way != core::cmp::Ordering::Equal {
            return match down {
                true => way.reverse(),
                false => way,
            };
        }
    }
    core::cmp::Ordering::Equal
}

/// The bytes of a field as the row holds them.
fn shaped(value: &[u8], shape: Shape) -> Value {
    match shape {
        Shape::Words => Value::Text(value.into()),
        Shape::Folded => Value::Text(token::fold(value).into()),
        // A field with a number behind it always holds one, because a document
        // whose value would not read as one was never indexed.
        Shape::Number => match yo_common::num::parse_f64(value) {
            Some(number) => Value::Number(number),
            None => Value::Text(value.into()),
        },
    }
}

/// One grouping step over the rows in front of it.
///
/// The groups come back in the order they were first seen, which is the order
/// the documents were walked in. A real server hands them back in the order its
/// hash table happens to hold them, and that order is not even the same between
/// two processes holding the same documents, so there is nothing to copy here.
/// Divergence D-68.
fn folded(group: &Group, table: &[Held]) -> Vec<Held> {
    let mut order: Vec<Vec<Value>> = Vec::new();
    let mut folds: Vec<Vec<reduce::Fold>> = Vec::new();
    let mut seen: HashMap<Vec<u8>, usize> = HashMap::new();
    // A reducer that hands a value back rather than working one out hands back
    // the type the property had, so a `FIRST_VALUE` over a number is a number
    // and not the digits it was spelled with.
    let mut counts = vec![false; group.folds.len()];
    for row in table {
        let key: Vec<Value> = group
            .by
            .iter()
            // A group key that the document had nothing under is a value in
            // its own right and comes back as a null, which is not the same as
            // a loaded field that is not there and is left out of the row.
            .map(|(from, _)| match &row.values[*from] {
                Value::Missing => Value::Nil,
                held => held.clone(),
            })
            .collect();
        let at = match seen.entry(tagged(&key)) {
            Entry::Occupied(held) => *held.get(),
            Entry::Vacant(spot) => {
                spot.insert(order.len());
                order.push(key);
                folds.push(
                    group
                        .folds
                        .iter()
                        .map(|fold| reduce::Fold::new(fold.kind.clone()))
                        .collect(),
                );
                order.len() - 1
            }
        };
        for ((fold, what), numeric) in folds[at]
            .iter_mut()
            .zip(&group.folds)
            .zip(counts.iter_mut())
        {
            let spelled;
            let value = match what.of.map(|at| &row.values[at]) {
                Some(Value::Text(text)) => Some(&**text),
                Some(Value::Number(number)) => {
                    *numeric = true;
                    spelled = twelve(*number).into_bytes();
                    Some(&spelled[..])
                }
                _ => None,
            };
            let ordered;
            let by = match what.by.map(|at| &row.values[at]) {
                Some(Value::Text(text)) => Some(&**text),
                Some(Value::Number(number)) => {
                    ordered = twelve(*number).into_bytes();
                    Some(&ordered[..])
                }
                _ => None,
            };
            fold.add(value, by);
        }
    }
    order
        .into_iter()
        .zip(folds)
        .map(|(key, folds)| Held {
            values: key
                .into_iter()
                .chain(
                    folds
                        .into_iter()
                        .zip(&counts)
                        .map(|(fold, numeric)| answered(fold.done(), *numeric)),
                )
                .collect(),
            dropped: 0,
            from: None,
        })
        .collect()
}

/// The bytes a group is keyed by, which are the values with their lengths in
/// front of them so two different groups cannot run together into one.
///
/// A number is keyed by the number and not by the digits it arrived as, so a
/// field holding `1` and a field holding `1.0` are one group.
fn tagged(values: &[Value]) -> Vec<u8> {
    let mut out = Vec::new();
    for value in values {
        match value {
            Value::Missing | Value::Nil => out.push(0),
            Value::Text(text) => {
                out.push(1);
                out.extend_from_slice(&text.len().to_le_bytes());
                out.extend_from_slice(text);
            }
            Value::Number(number) => {
                out.push(3);
                out.extend_from_slice(&number.to_le_bytes());
            }
            Value::List(list) => {
                out.push(2);
                out.extend_from_slice(&list.len().to_le_bytes());
                for item in list {
                    out.extend_from_slice(&tagged(core::slice::from_ref(item)));
                }
            }
        }
    }
    out
}

/// What a fold came to, as a property of the row after the step.
///
/// Nine of the twelve work a number out and hand back a number. The three that
/// pick values out of the group hand back what they picked, so they are numbers
/// when the property they read was one.
fn answered(answer: reduce::Answer, numeric: bool) -> Value {
    let held = |text: Box<[u8]>| match numeric.then(|| yo_common::num::parse_f64(&text)).flatten() {
        Some(number) => Value::Number(number),
        None => Value::Text(text),
    };
    match answer {
        reduce::Answer::Number(number) => Value::Number(number),
        reduce::Answer::Text(text) => held(text),
        reduce::Answer::List(list) => Value::List(list.into_iter().map(held).collect()),
        reduce::Answer::Nil => Value::Nil,
    }
}

/// The reply a pipeline answers with, on either protocol.
fn writes(
    count: usize,
    names: &[Box<[u8]>],
    shown: &[(Option<&Row>, &Vec<Value>)],
    sorted: Option<usize>,
    asked: &Asked<'_>,
    warning: Option<&[u8]>,
    out: &mut Out,
) {
    let pipe = &asked.pipe;
    let want = &asked.rows;
    let extras = usize::from(want.scores) + usize::from(want.payloads) + usize::from(pipe.sortkeys);
    // A row a group step made has no document under it and so has no score of
    // its own, which is why this is asked per row rather than once.
    let scored = pipe.addscores && !names.iter().any(|name| **name == *b"__score");
    let scoring = |row: Option<&Row>| scored.then(|| row.map(|row| row.score)).flatten();
    if out.proto().is_resp3() {
        out.map(5);
        out.simple(b"attributes");
        out.array(0);
        out.simple(b"format");
        out.simple(b"STRING");
        out.simple(b"results");
        out.array(shown.len());
        for (row, values) in shown {
            out.map(1 + extras + usize::from(want.content));
            if want.scores {
                out.simple(b"score");
                out.double(row.map_or(0.0, |row| row.score));
            }
            if want.payloads {
                out.simple(b"payload");
                match row.and_then(|row| row.payload.as_ref()) {
                    Some(payload) => out.bulk(payload),
                    None => out.nil(),
                }
            }
            if pipe.sortkeys {
                out.simple(b"sortkey");
                match keyed(values, sorted) {
                    Some(key) => out.bulk(&key),
                    None => out.nil(),
                }
            }
            if want.content {
                out.simple(b"extra_attributes");
                mapped(names, values, scoring(*row), out);
            }
            out.simple(b"values");
            out.array(0);
        }
        out.simple(b"total_results");
        out.int(count as i64);
        out.simple(b"warning");
        match warning {
            Some(warning) => {
                out.array(1);
                out.bulk(warning);
            }
            None => out.array(0),
        }
        return;
    }
    out.array(1 + shown.len() * (extras + usize::from(want.content)));
    out.int(count as i64);
    for (row, values) in shown {
        // A grouped row has no document behind it, so the score is nought and
        // the payload is nothing, which is what a real server sends for both.
        if want.scores {
            out.double(row.map_or(0.0, |row| row.score));
        }
        if want.payloads {
            match row.and_then(|row| row.payload.as_ref()) {
                Some(payload) => out.bulk(payload),
                None => out.nil(),
            }
        }
        if pipe.sortkeys {
            match keyed(values, sorted) {
                Some(key) => out.bulk(&key),
                None => out.nil(),
            }
        }
        if want.content {
            mapped(names, values, scoring(*row), out);
        }
    }
}

/// The sort key element beside a row, when `WITHSORTKEYS` asked for one.
///
/// A number is written after a hash and text after a dollar, and the number is
/// written wider than the same number on the row is: the sort key is the value
/// the sort compared and the row holds the value the client reads.
fn keyed(values: &[Value], sorted: Option<usize>) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    match values.get(sorted?)? {
        Value::Number(number) => {
            out.push(b'#');
            out.extend_from_slice(seventeen(*number).as_bytes());
        }
        Value::Text(text) => {
            out.push(b'$');
            out.extend_from_slice(text);
        }
        // A row the sort found nothing on has no key of its own, which is the
        // same null a pipeline that never sorted answers.
        _ => return None,
    }
    Some(out)
}

/// One row as a map of names to what the pipeline put under them.
///
/// A property the row never had is left out, which is what a `LOAD` of a field
/// the key does not hold does. A property that is there and holds nothing is
/// sent as a null, which is what a group key over a field nothing filled in
/// does. The two look the same from the outside and are not the same thing.
///
/// The score goes in front of the lot when `ADDSCORES` asked for it and nothing
/// in the pipeline named it, wherever in the argument list the word stood: a
/// `SORTBY` that put a property on the row first still answers the score first.
fn mapped(names: &[Box<[u8]>], row: &[Value], score: Option<f64>, out: &mut Out) {
    let held: Vec<(&Box<[u8]>, &Value)> = names
        .iter()
        .zip(row)
        .filter(|(_, value)| !matches!(value, Value::Missing))
        .collect();
    out.map(held.len() + usize::from(score.is_some()));
    if let Some(score) = score {
        out.bulk(b"__score");
        out.bulk(twelve(score).as_bytes());
    }
    for (name, value) in held {
        out.bulk(name);
        written(value, out);
    }
}

/// One value on the wire.
fn written(value: &Value, out: &mut Out) {
    match value {
        Value::Text(text) => out.bulk(text),
        Value::Number(number) => out.bulk(twelve(*number).as_bytes()),
        Value::Missing | Value::Nil => out.nil(),
        Value::List(list) => {
            out.array(list.len());
            for item in list {
                written(item, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reducer_with_no_name_gets_one_built_out_of_what_it_was_given() {
        assert_eq!(&*generated(b"COUNT", &[]), b"__generated_aliascount");
        assert_eq!(&*generated(b"Sum", &[b"@n"]), b"__generated_aliassumn");
        assert_eq!(
            &*generated(b"FIRST_VALUE", &[b"@n", b"BY", b"@n", b"Desc"]),
            b"__generated_aliasfirst_valuen,by,n,desc".as_slice()
        );
    }

    #[test]
    fn two_groups_that_run_together_without_their_lengths_do_not_run_together_with_them() {
        let one = [
            Value::Text(b"ab".to_vec().into()),
            Value::Text(b"c".to_vec().into()),
        ];
        let two = [
            Value::Text(b"a".to_vec().into()),
            Value::Text(b"bc".to_vec().into()),
        ];
        assert_ne!(tagged(&one), tagged(&two));
        // And a group holding nothing is not the group holding an empty value.
        assert_ne!(
            tagged(&[Value::Nil]),
            tagged(&[Value::Text(Box::default())])
        );
    }

    fn row(values: &[Value]) -> Held {
        Held {
            values: values.to_vec(),
            dropped: 0,
            from: None,
        }
    }

    #[test]
    fn a_row_without_the_property_sorts_last_whichever_way_the_key_runs() {
        let here = row(&[Value::Number(1.0)]);
        let gone = row(&[Value::Missing]);
        for down in [false, true] {
            assert_eq!(
                ordered(&[(0, down)], &here, &gone),
                core::cmp::Ordering::Less
            );
            assert_eq!(
                ordered(&[(0, down)], &gone, &here),
                core::cmp::Ordering::Greater
            );
        }
        assert_eq!(
            ordered(&[(0, false)], &gone, &gone),
            core::cmp::Ordering::Equal
        );
    }

    #[test]
    fn a_null_is_a_value_of_its_own_and_turns_round_with_the_key() {
        // Which is what tells a group key nothing filled in apart from a
        // property the document never had: one sorts under everything and the
        // other sorts last.
        let null = row(&[Value::Nil]);
        let held = row(&[Value::Number(-9.0)]);
        assert_eq!(
            ordered(&[(0, false)], &null, &held),
            core::cmp::Ordering::Less
        );
        assert_eq!(
            ordered(&[(0, true)], &null, &held),
            core::cmp::Ordering::Greater
        );
    }

    #[test]
    fn the_keys_are_read_in_turn_until_one_of_them_settles_it() {
        let first = row(&[Value::Text(b"aa".as_slice().into()), Value::Number(2.0)]);
        let second = row(&[Value::Text(b"aa".as_slice().into()), Value::Number(1.0)]);
        let keys = [(0, false), (1, false)];
        assert_eq!(
            ordered(&keys, &first, &second),
            core::cmp::Ordering::Greater
        );
        // The direction belongs to the key and not to the sort, so the second
        // one can run the other way while the first one does not.
        let keys = [(0, false), (1, true)];
        assert_eq!(ordered(&keys, &first, &second), core::cmp::Ordering::Less);
        assert_eq!(ordered(&[], &first, &second), core::cmp::Ordering::Equal);
    }

    #[test]
    fn two_values_are_ordered_the_way_a_comparison_orders_them() {
        let number = row(&[Value::Number(10.0)]);
        let words = row(&[Value::Text(b"9".as_slice().into())]);
        // Text beside a number is read as a number, so this is ten against
        // nine and not the two of them as bytes.
        assert_eq!(
            ordered(&[(0, false)], &number, &words),
            core::cmp::Ordering::Greater
        );
        let list = row(&[Value::List(vec![Value::Number(1.0)])]);
        assert_eq!(
            ordered(&[(0, false)], &list, &number),
            core::cmp::Ordering::Less
        );
    }

    #[test]
    fn a_sort_key_says_which_of_the_two_kinds_of_value_it_holds() {
        let row = [
            Value::Number(-4.0),
            Value::Text(b"blue".as_slice().into()),
            Value::Missing,
            Value::Nil,
        ];
        assert_eq!(keyed(&row, Some(0)).as_deref(), Some(b"#-4".as_slice()));
        assert_eq!(keyed(&row, Some(1)).as_deref(), Some(b"$blue".as_slice()));
        // A row the sort found nothing on, a group key nothing filled in, and a
        // pipeline that never sorted all answer the same null.
        assert_eq!(keyed(&row, Some(2)), None);
        assert_eq!(keyed(&row, Some(3)), None);
        assert_eq!(keyed(&row, Some(9)), None);
        assert_eq!(keyed(&row, None), None);
    }
}
