//! The half of JSONPath that names more than one place.
//!
//! [`Value::path`] answers one value and is the fast way to ask for one field,
//! and it refuses `[*]` and `..` on purpose because it has nowhere to put a
//! second answer. This is the other half. `$..price` and `$.items[*].sku` and
//! `$.a[0:10:2]` each name a set, and the `JSON.*` surface is written against
//! sets rather than against single values: `JSON.GET $..price` on a document
//! with four prices answers four numbers, and `JSON.SET` with the same path
//! writes four times.
//!
//! ```
//! use yo_doc::{Path, Value, from_json};
//!
//! let doc = from_json(br#"{"items":[{"sku":"a","price":3},{"sku":"b","price":5}]}"#)?;
//! let v = Value::new(&doc).expect("readable");
//!
//! let mut hits = Vec::new();
//! Path::parse(b"$..price")?.select(&v, &mut hits);
//! let prices: Vec<i64> = hits.iter().filter_map(Value::as_int).collect();
//! assert_eq!(prices, [3, 5]);
//! # Ok::<(), yo_common::Error>(())
//! ```
//!
//! # What is here
//!
//! The root `$`, a child by name written either way, `[*]` and `.*`, the
//! descent `..`, an index counting from either end, a union of indices or names
//! in one bracket, and a slice with an optional step. That is RFC 9535 without
//! its filter selector.
//!
//! # What is not here yet
//!
//! `?(@.price < 10)`. The grammar knows it exists and says so rather than
//! reading it as a name with punctuation in it, which is what a path parser
//! that has not thought about filters does, and it is worse than an error
//! because the client gets an empty answer instead of a complaint. Filters need
//! an expression language and a comparison order over mixed types, which is a
//! piece of work with its own decisions in it, so it is its own change.
//!
//! # Two orderings that are not Redis's
//!
//! Matches come back in document order, and for an object that is key order,
//! because that is the order members are stored in. RedisJSON walks an object
//! in the order the client wrote it. This is the same difference the JSON
//! writer has and it is the same one row in the register.
//!
//! A descent walks a node before its children, which is what every JSONPath
//! implementation does, so `$..a` on a document with an `a` inside an `a`
//! answers the outer one first.

use yo_common::{Code, Error, Result};

use crate::head::{DEPTH_MAX, Kind};
use crate::read::Value;

/// A parsed path.
///
/// Parsing is separate from matching because a path arrives once and is matched
/// against every document a command touches, and because a path that does not
/// parse should be an error before any document is read rather than an empty
/// answer after all of them.
#[derive(Debug, Clone)]
pub struct Path<'a> {
    sels: Vec<Sel<'a>>,
    legacy: bool,
}

/// One selector.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Sel<'a> {
    /// A member of an object, by name.
    Key(&'a [u8]),
    /// An element of an array, counting back from the end when negative.
    Index(i64),
    /// Every element of a container.
    Wild,
    /// This value and every value under it, which is what `..` is. The selector
    /// after it is what actually picks, so `$..a` is a descent and then a name.
    Descend,
    /// Several of the above in one bracket, applied in the order written.
    Union(Vec<Sel<'a>>),
    /// A run of an array. The bounds count back from the end when negative and
    /// the step may be negative, which walks it backwards.
    Slice {
        from: Option<i64>,
        to: Option<i64>,
        step: i64,
    },
}

impl<'a> Path<'a> {
    /// Parse `path`.
    ///
    /// A path that starts with `$` is a JSONPath and anything else is what
    /// RedisJSON calls a legacy path, which is the older syntax that answers one
    /// value. Both are matched the same way here and the difference is recorded
    /// in [`Path::legacy`], because what it changes is the shape of the reply
    /// and that is the dispatch layer's business rather than this one's.
    pub fn parse(path: &'a [u8]) -> Result<Path<'a>> {
        let (rest, legacy) = match path.strip_prefix(b"$") {
            Some(rest) => (rest, false),
            None => (path, true),
        };
        let mut p = Parse {
            rest,
            at: 0,
            legacy,
            sels: Vec::new(),
        };
        p.run()?;
        Ok(Path {
            sels: p.sels,
            legacy,
        })
    }

    /// Whether this path was written in the older syntax, without a leading
    /// `$`.
    #[must_use]
    pub fn legacy(&self) -> bool {
        self.legacy
    }

    /// Whether this path names at most one place, whatever document it is
    /// matched against.
    ///
    /// A path made only of names and indices does. Everything else can answer
    /// more than one value on some document even if it answers one on this one,
    /// and the difference decides what a command is allowed to do: `JSON.SET`
    /// creating a field that is not there yet only makes sense when the path
    /// says exactly where it goes.
    #[must_use]
    pub fn is_definite(&self) -> bool {
        self.sels
            .iter()
            .all(|s| matches!(s, Sel::Key(_) | Sel::Index(_)))
    }

    /// Every value this path names in `root`, in document order, appended to
    /// `out`.
    ///
    /// Appended rather than assigned, so that a command matching one path
    /// against many documents keeps one buffer. Nothing here allocates except
    /// that buffer and the frontier the walk carries.
    pub fn select<'d>(&self, root: &Value<'d>, out: &mut Vec<Value<'d>>) {
        let mut cur = vec![*root];
        let mut next = Vec::new();
        for sel in &self.sels {
            next.clear();
            for v in &cur {
                apply(sel, v, &mut next);
            }
            core::mem::swap(&mut cur, &mut next);
            if cur.is_empty() {
                return;
            }
        }
        out.append(&mut cur);
    }

    /// The one value this path names, for a caller that has already checked
    /// [`Path::is_definite`] or that only wants the first of several.
    ///
    /// RedisJSON's older syntax answers the first match, which is what this is
    /// for.
    #[must_use]
    pub fn first<'d>(&self, root: &Value<'d>) -> Option<Value<'d>> {
        let mut out = Vec::new();
        self.select(root, &mut out);
        out.into_iter().next()
    }
}

/// One selector against one value, with everything it names pushed onto `out`.
fn apply<'d>(sel: &Sel<'_>, v: &Value<'d>, out: &mut Vec<Value<'d>>) {
    match sel {
        Sel::Key(k) => out.extend(v.get(k)),
        Sel::Index(i) => out.extend(index(v, *i)),
        Sel::Wild => out.extend(v.iter()),
        Sel::Descend => descend(v, out, 0),
        Sel::Union(items) => {
            for item in items {
                apply(item, v, out);
            }
        }
        Sel::Slice { from, to, step } => slice(v, *from, *to, *step, out),
    }
}

/// An element of an array by an index that may count back from the end.
fn index<'d>(v: &Value<'d>, i: i64) -> Option<Value<'d>> {
    if v.kind() != Kind::Array {
        return None;
    }
    v.at(place(i, v.len())?)
}

/// Where `i` lands in a container of `n`, or `None` when it lands outside.
fn place(i: i64, n: usize) -> Option<usize> {
    if i < 0 {
        n.checked_sub(i.unsigned_abs() as usize)
    } else {
        let at = i as usize;
        (at < n).then_some(at)
    }
}

/// This value and everything under it, a node before its children.
///
/// The depth is counted here rather than left to the encoding, because this
/// walks a document that may have arrived from anywhere and a damaged one can
/// claim any shape it likes.
fn descend<'d>(v: &Value<'d>, out: &mut Vec<Value<'d>>, depth: usize) {
    out.push(*v);
    if depth >= DEPTH_MAX {
        return;
    }
    for child in v.iter() {
        descend(&child, out, depth + 1);
    }
}

/// The RFC 9535 slice, which is Python's slice with Python's defaults.
fn slice<'d>(
    v: &Value<'d>,
    from: Option<i64>,
    to: Option<i64>,
    step: i64,
    out: &mut Vec<Value<'d>>,
) {
    if v.kind() != Kind::Array || step == 0 {
        return;
    }
    let n = v.len() as i64;
    // A bound is clamped rather than wrapped, so `[0:1000]` is the whole array
    // and not an error, which is what every other slice in every other language
    // does and what a client writing one expects.
    let bound = |i: i64, lo: i64, hi: i64| {
        let i = if i < 0 { n + i } else { i };
        i.clamp(lo, hi)
    };
    if step > 0 {
        let mut at = bound(from.unwrap_or(0), 0, n);
        let end = bound(to.unwrap_or(n), 0, n);
        while at < end {
            out.extend(v.at(at as usize));
            at += step;
        }
    } else {
        let mut at = bound(from.unwrap_or(n - 1), -1, n - 1);
        let end = bound(to.unwrap_or(-n - 1), -1, n - 1);
        while at > end {
            out.extend(v.at(at as usize));
            at += step;
        }
    }
}

// ---------------------------------------------------------------- the grammar

struct Parse<'a> {
    rest: &'a [u8],
    at: usize,
    legacy: bool,
    sels: Vec<Sel<'a>>,
}

impl<'a> Parse<'a> {
    fn run(&mut self) -> Result<()> {
        // A path with no `$` may start with a bare name, so that `a.b` means
        // what `$.a.b` means. Only at the front of one of those: anywhere else,
        // and in a path that did start with a `$`, a missing separator is a typo
        // and reading it as a name would hide one.
        if self.legacy && self.at < self.rest.len() && !matches!(self.rest[self.at], b'.' | b'[') {
            let name = self.name()?;
            self.sels.push(Sel::Key(name));
        }
        while self.at < self.rest.len() {
            match self.rest[self.at] {
                b'.' if self.rest.get(self.at + 1) == Some(&b'.') => {
                    self.at += 2;
                    self.sels.push(Sel::Descend);
                    if self.at >= self.rest.len() {
                        return Err(self.bad("a `..` with nothing after it"));
                    }
                    // A descent picks with whatever follows it, and a bracket
                    // is handled by the next turn of this loop.
                    if self.rest.get(self.at) == Some(&b'[') {
                        continue;
                    }
                    let sel = self.after_dot()?;
                    self.sels.push(sel);
                }
                b'.' => {
                    self.at += 1;
                    let sel = self.after_dot()?;
                    self.sels.push(sel);
                }
                b'[' => {
                    let sel = self.bracket()?;
                    self.sels.push(sel);
                }
                _ => return Err(self.bad("a step that does not start with `.` or `[`")),
            }
        }
        Ok(())
    }

    /// What follows a `.` or a `..`, which is a name or a `*`.
    fn after_dot(&mut self) -> Result<Sel<'a>> {
        if self.rest.get(self.at) == Some(&b'*') {
            self.at += 1;
            return Ok(Sel::Wild);
        }
        Ok(Sel::Key(self.name()?))
    }

    /// A bare name, which runs to the next separator.
    fn name(&mut self) -> Result<&'a [u8]> {
        let from = self.at;
        while self.at < self.rest.len() && !matches!(self.rest[self.at], b'.' | b'[') {
            self.at += 1;
        }
        if self.at == from {
            return Err(self.bad("a `.` with no name after it"));
        }
        Ok(&self.rest[from..self.at])
    }

    /// Everything between one `[` and its `]`.
    fn bracket(&mut self) -> Result<Sel<'a>> {
        let body = &self.rest[self.at + 1..];
        let Some(close) = body.iter().position(|&c| c == b']') else {
            return Err(self.bad("a `[` with no `]` after it"));
        };
        let inner = &body[..close];
        self.at += close + 2;
        if inner.starts_with(b"?") {
            return Err(self.bad(
                "a filter, `[?(...)]`, is not read yet, so this path is refused rather than answering nothing",
            ));
        }
        if inner == b"*" {
            return Ok(Sel::Wild);
        }
        if inner.contains(&b':') {
            return self.slice(inner);
        }
        let mut items = Vec::new();
        for part in inner.split(|&c| c == b',') {
            items.push(self.one(trim(part))?);
        }
        match items.len() {
            0 => Err(self.bad("an empty `[]`")),
            1 => Ok(items.pop().expect("one item")),
            _ => Ok(Sel::Union(items)),
        }
    }

    /// One item of a bracket: a quoted name or an index.
    fn one(&self, part: &'a [u8]) -> Result<Sel<'a>> {
        if let Some(name) = quoted(part) {
            return Ok(Sel::Key(name));
        }
        Ok(Sel::Index(self.int(part)?))
    }

    fn slice(&self, inner: &[u8]) -> Result<Sel<'a>> {
        let mut parts = inner.split(|&c| c == b':');
        let from = self.maybe(parts.next().unwrap_or(b""))?;
        let to = self.maybe(parts.next().unwrap_or(b""))?;
        let step = self.maybe(parts.next().unwrap_or(b""))?.unwrap_or(1);
        if parts.next().is_some() {
            return Err(self.bad("a slice has at most a start, an end and a step"));
        }
        if step == 0 {
            return Err(self.bad("a slice with a step of zero"));
        }
        Ok(Sel::Slice { from, to, step })
    }

    /// A bound of a slice, which may be left out.
    fn maybe(&self, part: &[u8]) -> Result<Option<i64>> {
        let part = trim(part);
        if part.is_empty() {
            return Ok(None);
        }
        Ok(Some(self.int(part)?))
    }

    fn int(&self, part: &[u8]) -> Result<i64> {
        core::str::from_utf8(part)
            .ok()
            .and_then(|t| t.parse().ok())
            .ok_or_else(|| self.bad("an index that is not a number"))
    }

    /// An error that says where in the path it happened, since a path is short
    /// enough that the offset is the whole explanation.
    fn bad(&self, what: &str) -> Error {
        Error::fmt(
            Code::Invalid,
            format_args!("{what}, at byte {} of the path", self.at),
        )
    }
}

/// The bytes inside `"..."` or `'...'`, if that is what this is.
fn quoted(part: &[u8]) -> Option<&[u8]> {
    if part.len() >= 2 {
        let (first, last) = (part[0], part[part.len() - 1]);
        if (first == b'"' || first == b'\'') && last == first {
            return Some(&part[1..part.len() - 1]);
        }
    }
    None
}

/// Spaces off both ends, because `[0, 1]` is a path a person types.
fn trim(part: &[u8]) -> &[u8] {
    let from = part.iter().position(|&c| c != b' ').unwrap_or(part.len());
    let to = part
        .iter()
        .rposition(|&c| c != b' ')
        .map_or(from, |i| i + 1);
    &part[from..to]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::from_json;

    /// `{"store":{"book":[{"title":"a","price":8},{"title":"b","price":22}],
    ///   "bike":{"price":19}},"expensive":10}`
    ///
    /// The shape from the JSONPath paper, which is what every implementation is
    /// tested against, so a reader who knows the paper knows these documents.
    fn doc() -> Vec<u8> {
        from_json(
            br#"{"store":{"book":[{"title":"a","price":8},{"title":"b","price":22}],
                 "bike":{"price":19}},"expensive":10}"#,
        )
        .expect("the text parses")
    }

    /// What `path` names in [`doc`], as JSON text, so a whole answer fits on a
    /// line and reads like the thing it is.
    fn ask(bytes: &[u8], path: &str) -> String {
        let v = Value::new(bytes).expect("readable");
        let mut hits = Vec::new();
        Path::parse(path.as_bytes())
            .expect("the path parses")
            .select(&v, &mut hits);
        let mut out = Vec::new();
        out.push(b'[');
        for (i, hit) in hits.iter().enumerate() {
            if i > 0 {
                out.push(b',');
            }
            hit.write_json(&mut out).expect("writable");
        }
        out.push(b']');
        String::from_utf8(out).expect("UTF-8")
    }

    fn why(path: &str) -> String {
        Path::parse(path.as_bytes())
            .expect_err("this should not parse")
            .message()
            .to_string()
    }

    #[test]
    fn a_path_that_names_one_place_names_the_same_place_it_always_did() {
        let d = doc();
        assert_eq!(ask(&d, "$.expensive"), "[10]");
        assert_eq!(ask(&d, "$.store.bike.price"), "[19]");
        assert_eq!(ask(&d, "$.store.book[0].title"), r#"["a"]"#);
        assert_eq!(ask(&d, "$.store.book[-1].title"), r#"["b"]"#);
        assert_eq!(ask(&d, "$['store']['bike']['price']"), "[19]");
        assert_eq!(ask(&d, "store.bike.price"), "[19]", "the older syntax");
        // A name runs to the next separator and a space is not one, which is
        // the only way to reach a key with a space in it without quoting.
        assert_eq!(ask(&d, "$.store.no such key"), "[]");
        assert_eq!(ask(&d, "$"), ask(&d, ""), "the root, both ways of asking");
    }

    #[test]
    fn a_path_that_names_nothing_answers_nothing_rather_than_failing() {
        let d = doc();
        assert_eq!(ask(&d, "$.nope"), "[]");
        assert_eq!(ask(&d, "$.store.book[9]"), "[]");
        assert_eq!(ask(&d, "$.store.book[-9]"), "[]");
        assert_eq!(ask(&d, "$.expensive[0]"), "[]", "an index into a number");
        assert_eq!(ask(&d, "$.store.book.title"), "[]", "a name into an array");
        assert_eq!(ask(&d, "$.store[0]"), "[]", "an index into an object");
    }

    #[test]
    fn a_wildcard_names_every_child_and_a_descent_names_every_one_below() {
        let d = doc();
        assert_eq!(ask(&d, "$.store.book[*].price"), "[8,22]");
        assert_eq!(ask(&d, "$.store.book[*].title"), r#"["a","b"]"#);
        assert_eq!(
            ask(&d, "$..price"),
            "[19,8,22]",
            "the bike sorts before the books"
        );
        assert_eq!(
            ask(&d, "$.store.*.price"),
            "[19]",
            "the bike and not the books"
        );
        assert_eq!(ask(&d, "$..book[0].price"), "[8]");
        assert_eq!(ask(&d, "$..[0].title"), r#"["a"]"#);
        // A wildcard over a scalar names nothing, which is what stops
        // `$..*.price` from being an error on a document with a number in it.
        assert_eq!(ask(&d, "$.expensive.*"), "[]");
    }

    #[test]
    fn a_descent_walks_a_node_before_the_nodes_under_it() {
        let d = from_json(br#"{"a":{"b":1,"a":{"a":2}}}"#).expect("parses");
        assert_eq!(ask(&d, "$..a"), r#"[{"a":{"a":2},"b":1},{"a":2},2]"#);
    }

    #[test]
    fn a_union_names_what_it_lists_in_the_order_it_lists_it() {
        let d = from_json(br#"{"a":1,"b":2,"c":3,"xs":[10,11,12,13]}"#).expect("parses");
        assert_eq!(ask(&d, "$.xs[0,2]"), "[10,12]");
        assert_eq!(ask(&d, "$.xs[2,0]"), "[12,10]", "and not in index order");
        assert_eq!(ask(&d, "$.xs[0, 2]"), "[10,12]", "spaces are allowed");
        assert_eq!(ask(&d, "$['b','a']"), "[2,1]");
        assert_eq!(ask(&d, "$.xs[0,9]"), "[10]", "one of them names nothing");
    }

    #[test]
    fn a_slice_is_the_slice_every_other_language_has() {
        let d = from_json(br#"{"xs":[0,1,2,3,4,5]}"#).expect("parses");
        assert_eq!(ask(&d, "$.xs[1:3]"), "[1,2]");
        assert_eq!(ask(&d, "$.xs[:2]"), "[0,1]");
        assert_eq!(ask(&d, "$.xs[4:]"), "[4,5]");
        assert_eq!(ask(&d, "$.xs[:]"), "[0,1,2,3,4,5]");
        assert_eq!(ask(&d, "$.xs[-2:]"), "[4,5]");
        assert_eq!(ask(&d, "$.xs[:-4]"), "[0,1]");
        assert_eq!(ask(&d, "$.xs[0:6:2]"), "[0,2,4]");
        assert_eq!(ask(&d, "$.xs[::2]"), "[0,2,4]");
        assert_eq!(ask(&d, "$.xs[::-1]"), "[5,4,3,2,1,0]");
        assert_eq!(ask(&d, "$.xs[4:1:-1]"), "[4,3,2]");
        assert_eq!(
            ask(&d, "$.xs[0:1000]"),
            "[0,1,2,3,4,5]",
            "a bound is clamped"
        );
        assert_eq!(ask(&d, "$.xs[3:1]"), "[]", "an empty run is empty");
    }

    #[test]
    fn a_path_says_whether_it_could_ever_name_two_places() {
        let definite = |p: &str| Path::parse(p.as_bytes()).expect("parses").is_definite();
        assert!(definite("$.a.b[0]"));
        assert!(definite("$['a'][-1]"));
        assert!(definite(""), "the root is one place");
        assert!(!definite("$.a[*]"));
        assert!(!definite("$..a"));
        assert!(!definite("$.a[0,1]"));
        assert!(!definite("$.a[0:2]"));
        assert!(!definite("$.*"));
    }

    #[test]
    fn a_path_says_which_of_the_two_syntaxes_it_was_written_in() {
        let legacy = |p: &str| Path::parse(p.as_bytes()).expect("parses").legacy();
        assert!(!legacy("$.a"));
        assert!(!legacy("$"));
        assert!(legacy(".a"));
        assert!(legacy("a.b"));
    }

    #[test]
    fn the_first_match_is_the_first_one_in_document_order() {
        let d = doc();
        let v = Value::new(&d).expect("readable");
        let p = Path::parse(b"$..price").expect("parses");
        assert_eq!(p.first(&v).expect("there").as_int(), Some(19));
        assert!(Path::parse(b"$.nope").expect("parses").first(&v).is_none());
    }

    #[test]
    fn a_path_that_does_not_parse_says_so_and_says_where() {
        assert!(why("$.a[").contains("no `]`"));
        assert!(why("$.a[x]").contains("not a number"));
        assert!(why("$.a.").contains("no name after it"));
        assert!(why("$..").contains("nothing after it"));
        assert!(why("$.a[]").contains("not a number"));
        assert!(why("$.a[::0]").contains("step of zero"));
        assert!(why("$.a[1:2:3:4]").contains("at most a start"));
        assert!(why("$a").contains("does not start with"));
        assert!(why("$.a[?(@.b > 1)]").contains("is not read yet"));
        assert!(why("$.a[x]").contains("at byte "));
    }

    #[test]
    fn the_walk_stops_at_the_depth_limit_rather_than_running_out_of_stack() {
        // A document at the limit, so the descent goes all the way down it and
        // the guard is the thing that is not tripped rather than the thing that
        // saves it.
        let text = format!("{}1{}", "[".repeat(DEPTH_MAX), "]".repeat(DEPTH_MAX));
        let d = from_json(text.as_bytes()).expect("parses");
        let v = Value::new(&d).expect("readable");
        let mut hits = Vec::new();
        Path::parse(b"$..*").expect("parses").select(&v, &mut hits);
        assert_eq!(hits.len(), DEPTH_MAX, "every level below the root, once");
    }

    #[test]
    fn selecting_appends_so_that_one_buffer_serves_many_documents() {
        let one = from_json(br#"{"a":1}"#).expect("parses");
        let two = from_json(br#"{"a":2}"#).expect("parses");
        let p = Path::parse(b"$.a").expect("parses");
        let mut hits = Vec::new();
        p.select(&Value::new(&one).expect("readable"), &mut hits);
        p.select(&Value::new(&two).expect("readable"), &mut hits);
        let got: Vec<i64> = hits.iter().filter_map(Value::as_int).collect();
        assert_eq!(got, [1, 2]);
    }
}
