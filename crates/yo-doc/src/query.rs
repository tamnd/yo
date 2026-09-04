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
//! The filter, `[?(@.price < 10)]`, is here too. It is the only selector whose
//! answer depends on the document rather than only on the path, and the only one
//! that can look somewhere other than where it stands, because `$` inside an
//! expression is the whole document.
//!
//! ```
//! use yo_doc::{Path, Value, from_json};
//!
//! let doc = from_json(br#"{"items":[{"sku":"a","price":3},{"sku":"b","price":15}]}"#)?;
//! let v = Value::new(&doc).expect("readable");
//!
//! let mut hits = Vec::new();
//! Path::parse(b"$.items[?(@.price < 10)].sku")?.select(&v, &mut hits);
//! let cheap: Vec<&str> = hits.iter().filter_map(Value::as_text).collect();
//! assert_eq!(cheap, ["a"]);
//! # Ok::<(), yo_common::Error>(())
//! ```
//!
//! # What a filter's operators mean
//!
//! An operand is a path from the current node, a path from the root, or a
//! literal, and a path answers a set rather than one value. A comparison is true
//! when some pair drawn from the two sides satisfies it, so `@.tags[*] == "x"`
//! asks whether any tag is `x`. A side that answers nothing satisfies nothing,
//! which is why `@.missing == 1` and `@.missing < 1` are both false, and `!=` is
//! the negation of the whole comparison rather than a comparison of its own,
//! which is why `@.missing != 1` is true.
//!
//! An ordering comparison only arises between two values of the same sort.
//! Numbers order as numbers, strings order by their characters, `false` is below
//! `true`, and two nulls are equal, so `null >= null` is true and `null < null`
//! is not. Everything else is false: a string is not below a number, and an
//! array or an object is not below anything at all, not even an equal one.
//! Equality is the whole value, and it crosses the integer and float split so
//! `1 == 1.0`, but it crosses nothing else, so `0 == false` and `1 == "1"` are
//! both false. All of that was read off RedisJSON 8.10.1 rather than off the
//! RFC, which leaves most of it open.
//!
//! An expression with no operator in it asks whether the operand is there.
//! `[?(@.price)]` keeps the members that have a price, including the ones whose
//! price is `null` or `false`, because it is a question about the document and
//! not about the value. The one thing that is false on its own is the literal
//! `false`, so `[?(false)]` keeps nothing while `[?(0)]` and `[?(null)]` keep
//! everything.
//!
//! `=~` is a regular expression, and the flavour is the one in
//! [`yo_common::re`], which is what `ARGREP` uses. RedisJSON's is the Rust
//! `regex` crate's, so the two agree on everything anyone writes by hand and
//! part company on the corners, which is a row in the divergence register.
//!
//! The parentheses everyone writes around a filter are not part of it, so
//! `[?@.a == 1]` and `[? (@.a == 1)]` are the same filter. A filter iterates the
//! children of an array and of an object alike, and it works under a write as
//! well as a read: `JSON.SET`, `JSON.DEL` and `JSON.NUMINCRBY` all take one.
//!
//! # The operators past the comparisons
//!
//! `in` asks whether the left value is one of the elements of the array on the
//! right, and `nin` is its negation over both whole sides. `anyof` and `noneof`
//! ask whether two arrays share an element, and `subsetof` asks whether every
//! element of the left array is on the right, which makes `[]` a subset of
//! anything. `size`, which is also spelled `sizeof`, takes a bare number and is
//! the length of a string, an array or an object, and `empty` takes `true` or
//! `false` over the same three, so a number has neither and satisfies neither.
//! The right hand side of all five may be a path rather than a literal, and it
//! is the values that path answered that are the collection, so `3 in @.list`
//! and `@.tags anyof $.wanted` both read.
//!
//! The postfix methods are `.length()`, `.count()`, `.min()`, `.max()`, `.sum()`
//! and `.avg()`. `count()` is how many values the operand answered and is a
//! number even when that number is zero, so `@.nope.count() == 0` is true, and
//! it is the only one of the six that answers for an operand that answered
//! nothing. The four aggregates want an array of numbers and answer nothing for
//! an empty one or for an array with anything else in it. A name that is not one
//! of the six answers nothing rather than refusing the path, which is what
//! `@.p.size()` does.
//!
//! Arithmetic is `+ - * / %` over numbers, `*` and `/` and `%` bind tighter than
//! `+` and `-`, and parentheses group. Only `*` is an operator wherever it
//! stands, so `@.p*2 == 6` reads the way it looks. The other four are name
//! characters and need their spaces: `@.total-vat` is a key called `total-vat`,
//! `@.a+1` is a key called `a+1`, and `@.total - vat` is the subtraction. The
//! exception is straight after a `]`, where no name can be running, so
//! `@.list[0]-1` is a subtraction with no spaces in it at all. A leading `-` or
//! `+` is a sign, it answers a number or nothing so `-@.name` on a string
//! answers nothing, and one is as many as go in a row: `--@.p` is refused and
//! `-(-@.p)` is how a second one is written.
//!
//! Arithmetic and the methods want one node, which is the one place the set rule
//! above does not hold. An operand that answered two answers nothing rather than
//! a pair of sums, so `@.list[*] + 1` on a list of two is nothing and
//! `@.list[*].length()` is nothing as well. `count()` is outside that because
//! counting is what it is for. This is worth knowing before writing a wildcard
//! into an arithmetic operand, because what comes back is nothing rather than
//! what it looks like it asks for.
//!
//! The postfix `~` answers the key names of an object, one string each, and
//! nothing at all for an array or a scalar. It is a set rather than an array
//! value, so `@.p~ == "x"` is true of any object with an `x` in it, and the
//! operators that want a collection read the whole set as one: `@.p~ size 2` is
//! an object with two keys, and `@.p~ subsetof ["x","y"]` is an object with no
//! other key. It is a collection on the right of those operators too, so
//! `"x" in @.p~` asks whether the object has an `x`. `in` and `=~` do not take
//! it on the left and are false whatever is on the other side, which is the
//! reference's behaviour rather than a rule with a reason behind it.
//!
//! An object with no keys answers a set that is there and empty, and something
//! that is not an object answers no set at all, and every one of the collection
//! operators tells the two apart. On `{}` the tests `@.p~ subsetof ["x"]`,
//! `@.p~ empty true` and `@.p~ size 0` are all true, and on a number or a
//! missing key none of the three is.
//!
//! # A path that is an expression
//!
//! `JSON.GET`, `JSON.MGET` and `JSON.RESP` take a path that is not a way through
//! a document at all but a sum over one. `$.a + $.b`, `$.list.length()` and
//! `$.o~` are all projections, and what comes back is what the expression worked
//! out rather than where it was found. Nothing else about them is different from
//! the same expression inside a filter: the same operators, the same methods,
//! the same one node rule. [`Path::is_projection`] tells the two apart and
//! [`Path::project`] runs one.
//!
//! A projection never fails the way a path does. `$.nope + 1` answers `[]`
//! rather than raising the "path does not exist" a legacy path would, and it
//! answers an array even when it was written in the legacy syntax, so `.a + 1`
//! is `[4]` and not `4`.
//!
//! The first thing at the top level is a path however it is written, which is
//! how `.a + 1` and `2 + 3` both parse. The second is a member really called
//! `2`, plus three, and answers `[]`. Inside parentheses the ordinary rules are
//! back, so `(2)` is the number two. `@` is refused up here, since there is no
//! current node outside a filter.
//!
//! The kind of number that comes out follows the reference and is not always the
//! kind the arithmetic suggests. A whole number stays whole through `+`, `-`,
//! `*` and `%`, `/` is always a fraction even when it divides evenly, `length()`
//! and `count()` are whole, and `min()`, `max()`, `sum()` and `avg()` are
//! fractions. Dividing or taking a remainder by zero answers nothing.
//!
//! Every other `JSON.*` command refuses a projection rather than reading it as a
//! path, so `JSON.NUMINCRBY key "$.a + 1" 1` is an error and not a write
//! somewhere nobody asked for.
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

use crate::filter::{Arith, Expr, Fun, Item, Num, Op, Operand, Pattern};
use crate::head::{DEPTH_MAX, Kind};
use crate::path::Step;
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
    /// The expression this path is, when it is one rather than a way through
    /// the document. A path with one has no selectors and names nothing.
    proj: Option<Operand<'a>>,
}

/// One value a projection worked out.
///
/// A projection answers numbers and key names rather than places in a document,
/// which is why it is not a [`Value`]. The two number kinds are apart because
/// the reference writes `$.a + $.b` as `7` and `$.list.sum()` as `7.0`, and a
/// client reading the reply back as JSON sees the difference.
#[derive(Debug, Clone, Copy)]
pub enum Computed<'d> {
    /// A value the document holds.
    Value(Value<'d>),
    /// A key name, which is what `~` answers.
    Name(&'d [u8]),
    /// A whole number.
    Int(i64),
    /// A number that is not whole.
    Float(f64),
}

impl Computed<'_> {
    /// This as JSON text, appended to `out`, as if it sat `depth` levels inside
    /// whatever the caller has already opened.
    ///
    /// # Errors
    ///
    /// A document that arrived damaged, and a number that is not finite, which
    /// JSON has no way to write.
    pub fn write_json_at(
        &self,
        f: &crate::Format<'_>,
        out: &mut Vec<u8>,
        depth: usize,
    ) -> Result<()> {
        match self {
            Computed::Value(v) => v.write_json_at(f, out, depth),
            Computed::Name(k) => {
                crate::text::write_string(k, out);
                Ok(())
            }
            Computed::Int(i) => {
                crate::text::write_int(*i, out);
                Ok(())
            }
            Computed::Float(x) => crate::text::write_float(*x, out),
        }
    }
}

/// One selector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Sel<'a> {
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
    /// The children this expression is true of. Boxed because it is the one
    /// selector that is more than a few words and every other selector would
    /// otherwise pay for it.
    Filter(Box<Expr<'a>>),
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
        // A projection is tried first and falls through quietly when the path is
        // an ordinary one, so nothing about an ordinary path goes through the
        // expression grammar.
        if let Some(proj) = projection(path) {
            return Ok(Path {
                sels: Vec::new(),
                legacy: false,
                proj: Some(proj),
            });
        }
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
            proj: None,
        })
    }

    /// Whether this path is an expression rather than a way through the
    /// document.
    ///
    /// A projection answers values that are nowhere in the document, so there
    /// is nothing for a write to write to and nothing for `JSON.TYPE` to
    /// describe. Only `JSON.GET`, `JSON.MGET` and `JSON.RESP` take one and the
    /// rest refuse it, which is what this is for.
    #[must_use]
    pub fn is_projection(&self) -> bool {
        self.proj.is_some()
    }

    /// What this projection works out against `root`, and nothing at all when
    /// the path is not one.
    #[must_use]
    pub fn project<'d>(&'d self, root: &Value<'d>) -> Vec<Computed<'d>> {
        let Some(o) = &self.proj else {
            return Vec::new();
        };
        crate::filter::project(o, root)
            .into_iter()
            .map(|it| match it {
                Item::Ref(v) => Computed::Value(v),
                Item::Key(k) => Computed::Name(k),
                Item::Num(Num::Int(i)) => Computed::Int(i),
                Item::Num(Num::Float(x)) => Computed::Float(x),
            })
            .collect()
    }

    /// Whether this path was written in the older syntax, without a leading
    /// `$`.
    #[must_use]
    pub fn legacy(&self) -> bool {
        self.legacy
    }

    /// Whether this path is the root and nothing else.
    ///
    /// `$`, a bare `.` and the empty path are all it. `JSON.SET` needs to know,
    /// because the root is the only place a whole document can be written to a
    /// key that is not there yet, and `JSON.DEL` needs to know because deleting
    /// the root is deleting the key.
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.sels.is_empty()
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
        select_from(&self.sels, root, root, out);
    }

    /// This path without its last selector, and that selector, when the last
    /// one names one place.
    ///
    /// `JSON.SET` is what needs it. A path that matched nothing can still be a
    /// place to write, as long as what would hold the new value is there and the
    /// last step says exactly where in it the value goes, so `$.a.b` against a
    /// document with an `a` and no `b` splits into the parent `$.a` and the step
    /// `b`.
    ///
    /// `None` for the root, which has no last selector, and for a last selector
    /// that is a wildcard, a descent, a union or a slice, because none of those
    /// names a place that is not already there.
    #[must_use]
    pub fn split_last(&self) -> Option<(Path<'a>, Step<'a>)> {
        let step = match self.sels.last()? {
            Sel::Key(k) => Step::Key(k),
            Sel::Index(i) => Step::Index(*i),
            _ => return None,
        };
        let parent = Path {
            sels: self.sels[..self.sels.len() - 1].to_vec(),
            legacy: self.legacy,
            proj: None,
        };
        Some((parent, step))
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

/// Walk `sels` from `start`, with everything they name appended to `out`.
///
/// `root` is carried the whole way down because a filter may say `$`, which is
/// the document and not the value the filter stands on. It is the same value as
/// `start` for a path a client sent and a different one for a path inside a
/// filter.
pub(crate) fn select_from<'d>(
    sels: &[Sel<'_>],
    start: &Value<'d>,
    root: &Value<'d>,
    out: &mut Vec<Value<'d>>,
) {
    let mut cur = vec![*start];
    let mut next = Vec::new();
    for sel in sels {
        next.clear();
        for v in &cur {
            apply(sel, root, v, &mut next);
        }
        core::mem::swap(&mut cur, &mut next);
        if cur.is_empty() {
            return;
        }
    }
    out.append(&mut cur);
}

/// One selector against one value, with everything it names pushed onto `out`.
fn apply<'d>(sel: &Sel<'_>, root: &Value<'d>, v: &Value<'d>, out: &mut Vec<Value<'d>>) {
    match sel {
        Sel::Key(k) => out.extend(v.get(k)),
        Sel::Index(i) => out.extend(index(v, *i)),
        Sel::Wild => out.extend(v.iter()),
        Sel::Descend => descend(v, out, 0),
        Sel::Union(items) => {
            for item in items {
                apply(item, root, v, out);
            }
        }
        Sel::Slice { from, to, step } => slice(v, *from, *to, *step, out),
        // A filter is asked about the children and not about the value it
        // stands on, so an array is filtered element by element and an object
        // member by member, which is what `iter` walks for both.
        Sel::Filter(e) => out.extend(v.iter().filter(|child| e.holds(root, child))),
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
        // A legacy path of one dot is the root. It is the spelling the `JSON.*`
        // commands fall back to when the client gave no path at all, and it is
        // the only place a `.` is allowed to have nothing after it.
        if self.legacy && self.rest == b"." {
            return Ok(());
        }
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
    ///
    /// A name written without quotes may not have a space, a `~`, a `*` or a
    /// comparison character in it, because the top level of a path is an
    /// expression now and those are what it is made of. A member really called
    /// `my key` is reached by `$["my key"]`, which is what the reference wants
    /// too. The four arithmetic characters that are also name characters are
    /// still name characters, so `$.total-vat` and `$.a+1` are members.
    fn name(&mut self) -> Result<&'a [u8]> {
        let from = self.at;
        while self.at < self.rest.len()
            && !matches!(self.rest[self.at], b'.' | b'[')
            && !ends_name(self.rest[self.at])
        {
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
        let Some(close) = closer(body, b']') else {
            return Err(self.bad("a `[` with no `]` after it"));
        };
        let inner = &body[..close];
        self.at += close + 2;
        if let Some(rest) = inner.strip_prefix(b"?") {
            return self.filter(rest);
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

    /// Everything between `[?` and its `]`.
    ///
    /// The parentheses everyone writes around a filter are not part of it. They
    /// are a group like any other group, which is why `[?@.a == 1]` and
    /// `[? (@.a == 1)]` are both this and read the same.
    fn filter(&mut self, body: &'a [u8]) -> Result<Sel<'a>> {
        let mut f = Filter {
            body,
            at: 0,
            of: self.at,
            top: false,
        };
        let e = f.or()?;
        f.spaces();
        if f.at < f.body.len() {
            return Err(f.bad("a filter with something left over at the end of it"));
        }
        Ok(Sel::Filter(Box::new(e)))
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

// ------------------------------------------------------------------ a filter

/// The grammar inside `[?...]`.
///
/// Separate from [`Parse`] because it reads an expression rather than a run of
/// selectors, and because it works over the one bracket rather than over the
/// whole path. `of` is where that bracket started, so an error still says where
/// in the path the client should look.
struct Filter<'a> {
    body: &'a [u8],
    at: usize,
    of: usize,
    /// Whether the next atom read is the first one of a projection, where a
    /// bare name is a path rather than a value. It is cleared by the atom that
    /// reads it, so it only ever applies to one.
    top: bool,
}

impl<'a> Filter<'a> {
    /// `and` and then any number of `|| and`.
    fn or(&mut self) -> Result<Expr<'a>> {
        let mut e = self.and()?;
        while self.word(b"||") {
            e = Expr::Or(Box::new(e), Box::new(self.and()?));
        }
        Ok(e)
    }

    /// `unary` and then any number of `&& unary`, which is why `&&` binds
    /// tighter than `||`.
    fn and(&mut self) -> Result<Expr<'a>> {
        let mut e = self.unary()?;
        while self.word(b"&&") {
            e = Expr::And(Box::new(e), Box::new(self.unary()?));
        }
        Ok(e)
    }

    /// A `!`, a group, or a comparison.
    ///
    /// A `(` is ambiguous, because `(@.a || @.b)` groups an expression and
    /// `(1 + 2) * 3` groups arithmetic and the two are told apart only by what
    /// comes after the `)`. This reads it as an expression, and if what follows
    /// is an operator rather than the end of one, winds back and reads the whole
    /// thing as a comparison instead.
    fn unary(&mut self) -> Result<Expr<'a>> {
        self.spaces();
        if self.word(b"!") {
            return Ok(Expr::Not(Box::new(self.unary()?)));
        }
        let from = self.at;
        if self.word(b"(") {
            let e = self.or()?;
            if !self.word(b")") {
                return Err(self.bad("a `(` in a filter with no `)` after it"));
            }
            if !self.operator_next() {
                return Ok(e);
            }
            self.at = from;
        }
        self.cmp()
    }

    /// Whether what comes next is an operator, which is what says a `(...)` just
    /// read was arithmetic and not a group.
    fn operator_next(&mut self) -> bool {
        self.spaces();
        let rest = &self.body[self.at..];
        if rest.first().is_some_and(|c| {
            matches!(c, b'+' | b'-' | b'*' | b'/' | b'%' | b'<' | b'>' | b'=') || *c == b'!'
        }) {
            // A `!` is only an operator when it is a `!=`, since a `!` on its
            // own after a group is not something a filter can mean.
            return rest[0] != b'!' || rest.starts_with(b"!=");
        }
        WORD_OPS.iter().any(|(text, _)| word_at(rest, text))
    }

    /// One operand, and the other one when there is an operator between them.
    fn cmp(&mut self) -> Result<Expr<'a>> {
        let left = self.operand()?;
        let Some(op) = self.op() else {
            return Ok(Expr::Test(left));
        };
        let mut right = self.operand()?;
        if op == Op::Re {
            // A pattern is compiled here rather than once per value it is run
            // against, which is the whole reason a path is parsed at all. One
            // that will not compile is left as it was written, which makes it a
            // right hand side that matches nothing, because that is what the
            // reference does with `"["` rather than refusing the path.
            if let Operand::Lit(bytes) = &right
                && let Some(v) = Value::new(bytes)
                && let Some(text) = v.text_bytes()
                && let Ok(pat) = Pattern::new(text)
            {
                right = Operand::Re(pat);
            }
        }
        Ok(Expr::Cmp(left, op, right))
    }

    /// The operator between two operands, if there is one.
    ///
    /// The two character ones are tried first, so that `<=` is not read as a
    /// `<` with a stray `=` after it, and the ones written as words need a
    /// boundary after them so that a `size` in `sizes` is not one.
    fn op(&mut self) -> Option<Op> {
        self.spaces();
        for (text, op) in [
            (&b"=="[..], Op::Eq),
            (b"!=", Op::Ne),
            (b"<=", Op::Le),
            (b">=", Op::Ge),
            (b"=~", Op::Re),
            (b"<", Op::Lt),
            (b">", Op::Gt),
        ] {
            if self.word(text) {
                return Some(op);
            }
        }
        for (text, op) in WORD_OPS {
            if word_at(&self.body[self.at..], text) {
                self.at += text.len();
                return Some(*op);
            }
        }
        None
    }

    /// One side of a comparison, which is a sum of products of atoms.
    ///
    /// Arithmetic binds tighter than a comparison and `*` binds tighter than
    /// `+`, which is the ordering everybody expects and is the one the reference
    /// has.
    fn operand(&mut self) -> Result<Operand<'a>> {
        let mut e = self.product()?;
        loop {
            self.spaces();
            let op = match self.body.get(self.at) {
                Some(b'+') => Arith::Add,
                // A `-` is only ever subtraction when it is spaced, because a
                // key really called `total-vat` is reachable and an operator
                // nobody can write around is not worth it.
                Some(b'-') => Arith::Sub,
                _ => break,
            };
            self.at += 1;
            e = Operand::Math(Box::new(e), op, Box::new(self.product()?));
        }
        Ok(e)
    }

    /// A signed atom and then any number of `* / %` and another one.
    fn product(&mut self) -> Result<Operand<'a>> {
        let mut e = self.signed()?;
        loop {
            self.spaces();
            let op = match self.body.get(self.at) {
                Some(b'*') => Arith::Mul,
                Some(b'/') => Arith::Div,
                Some(b'%') => Arith::Rem,
                _ => break,
            };
            self.at += 1;
            e = Operand::Math(Box::new(e), op, Box::new(self.signed()?));
        }
        Ok(e)
    }

    /// An atom with a `-` or a `+` in front of it, or an atom.
    ///
    /// One sign and no more, which is what the reference does: `--@.a` is
    /// refused and `-(-@.a)` is not. A sign answers a number or nothing, so
    /// `-@.name` on a string answers nothing rather than the string.
    fn signed(&mut self) -> Result<Operand<'a>> {
        self.spaces();
        let neg = match self.body.get(self.at) {
            Some(b'-') => true,
            Some(b'+') => false,
            _ => return self.atom(),
        };
        self.at += 1;
        Ok(Operand::Sign(Box::new(self.atom()?), neg))
    }

    /// A path from `@` or from `$`, a value written into the path, or either of
    /// those in parentheses, and then any postfix on it.
    fn atom(&mut self) -> Result<Operand<'a>> {
        self.spaces();
        let Some(&c) = self.body.get(self.at) else {
            return Err(self.bad("a filter that stops where a value was expected"));
        };
        let mut e = if c == b'(' {
            // Only the first thing at the very top of a projection is a path
            // however it is written. Inside parentheses the ordinary rules are
            // back, so `(2)` is the number and `2 + 3` is a member called `2`.
            self.top = false;
            self.at += 1;
            let inner = self.operand()?;
            if !self.word(b")") {
                return Err(self.bad("a `(` in a filter with no `)` after it"));
            }
            inner
        } else if c == b'@' || c == b'$' {
            self.top = false;
            self.at += 1;
            self.path(c == b'@', false)?
        } else if core::mem::take(&mut self.top) {
            // The first thing in a projection is a path however it is written,
            // which is what makes `.a + 1` read and what makes `2 + 3` a member
            // really called `2` rather than five.
            self.path(false, true)?
        } else {
            self.literal()?
        };
        // One `~` and no more. Key names have no keys of their own, so `@.p~~`
        // is a path that means nothing and the reference refuses it.
        if self.body.get(self.at) == Some(&b'~') {
            self.at += 1;
            e = Operand::Keys(Box::new(e));
        }
        Ok(e)
    }

    /// The path that starts here, and the postfix method on the end of it when
    /// there is one.
    ///
    /// `at` says the path started at `@` rather than at `$`, and `legacy` says
    /// it started at neither, which only happens at the top of a projection.
    fn path(&mut self, at: bool, legacy: bool) -> Result<Operand<'a>> {
        let end = self.at + path_end(&self.body[self.at..]);
        // A postfix method looks like the last name of the path, because the
        // path stops at the `(` rather than at the `.` before it, so the name
        // comes back off the end here.
        let mut to = end;
        let mut fun = None;
        if self.body[end..].starts_with(b"()")
            && let Some(dot) = self.body[self.at..end].iter().rposition(|&b| b == b'.')
        {
            fun = Some(Fun::named(&self.body[self.at + dot + 1..end]));
            to = self.at + dot;
        }
        let mut p = Parse {
            rest: &self.body[self.at..to],
            at: 0,
            legacy,
            sels: Vec::new(),
        };
        p.run()?;
        self.at = if fun.is_some() { end + 2 } else { end };
        let path = Operand::Path { at, sels: p.sels };
        Ok(match fun {
            Some(f) => Operand::Call(Box::new(path), f),
            None => path,
        })
    }

    /// A number, a string, `true`, `false`, `null`, or a whole array or object.
    fn literal(&mut self) -> Result<Operand<'a>> {
        let from = self.at;
        let text = match self.body[self.at] {
            b'"' | b'\'' => self.string()?,
            open @ (b'[' | b'{') => {
                let body = &self.body[self.at + 1..];
                let close = if open == b'[' { b']' } else { b'}' };
                let Some(close) = closer(body, close) else {
                    return Err(self.bad("a value in a filter that is not closed"));
                };
                self.at += close + 2;
                self.body[from..self.at].to_vec()
            }
            _ => {
                while self.at < self.body.len() && !stops(self.body[self.at]) {
                    self.at += 1;
                }
                if self.at == from {
                    return Err(self.bad("a filter with an operator where a value goes"));
                }
                self.body[from..self.at].to_vec()
            }
        };
        let bytes = crate::from_json(&text)
            .map_err(|_| self.bad("a value in a filter that is not a value"))?;
        Ok(Operand::Lit(bytes))
    }

    /// A quoted string, as the JSON text of the same string.
    ///
    /// A filter may quote with either mark and JSON only knows the one, so a
    /// single quoted string is rewritten rather than parsed twice. What is
    /// inside it is left alone, escapes included, so `'A'` means what it
    /// means in JSON.
    fn string(&mut self) -> Result<Vec<u8>> {
        let quote = self.body[self.at];
        let mut out = vec![b'"'];
        let mut i = self.at + 1;
        while i < self.body.len() {
            let c = self.body[i];
            if c == b'\\' && i + 1 < self.body.len() {
                // A quote of the other kind was escaped to get past this
                // parser, and JSON has no escape for it, so the backslash goes
                // and the mark stays.
                let next = self.body[i + 1];
                if next == b'\'' {
                    out.push(b'\'');
                } else {
                    out.push(c);
                    out.push(next);
                }
                i += 2;
                continue;
            }
            if c == quote {
                out.push(b'"');
                self.at = i + 1;
                return Ok(out);
            }
            if c == b'"' {
                out.push(b'\\');
            }
            out.push(c);
            i += 1;
        }
        Err(self.bad("a string in a filter with no closing quote"))
    }

    /// Take `text` if it is next, after any spaces.
    fn word(&mut self, text: &[u8]) -> bool {
        self.spaces();
        if self.body[self.at..].starts_with(text) {
            self.at += text.len();
            return true;
        }
        false
    }

    fn spaces(&mut self) {
        while matches!(self.body.get(self.at), Some(b' ' | b'\t')) {
            self.at += 1;
        }
    }

    fn bad(&self, what: &str) -> Error {
        Error::fmt(
            Code::Invalid,
            format_args!("{what}, at byte {} of the path", self.of + self.at),
        )
    }
}

/// The expression a path is, when it is one rather than a way through the
/// document.
///
/// Redis 8.10 lets the top level of a path be arithmetic, a postfix method or a
/// `~`, so `$.a + $.b` answers a number that is nowhere in the document and
/// `$.o~` answers key names. It is told from an ordinary path by what it parses
/// into: a path on its own stays a path, and anything else is an expression. A
/// path in parentheses is an expression too, which is what the leading `(` here
/// is about.
///
/// Nothing here is an error. A body that does not read as an expression is left
/// to the ordinary parser, which is what says whether it is a path or a mistake.
fn projection(body: &[u8]) -> Option<Operand<'_>> {
    // The reference's top level starts at the first byte, so a leading space is
    // part of the first name rather than something to skip past, and nothing
    // that starts with one is an expression.
    if body.first().is_none_or(|c| matches!(c, b' ' | b'\t')) {
        return None;
    }
    let mut f = Filter {
        body,
        at: 0,
        of: 0,
        top: true,
    };
    let e = f.operand().ok()?;
    f.spaces();
    if f.at != body.len() || (body[0] != b'(' && matches!(e, Operand::Path { .. })) {
        return None;
    }
    // `@` is the node a filter stands on and there is no such node up here, so
    // a path that mentions one is not an expression. The reference refuses it
    // and this leaves it to the ordinary parser, which reads it as a member
    // really called `@` and answers nothing, which is the same thing to a
    // client.
    (!mentions_at(&e)).then_some(e)
}

/// Whether an operand reads `@` anywhere inside it.
fn mentions_at(o: &Operand<'_>) -> bool {
    match o {
        Operand::Path { at, .. } => *at,
        Operand::Lit(_) | Operand::Re(_) => false,
        Operand::Keys(inner) | Operand::Call(inner, _) | Operand::Sign(inner, _) => {
            mentions_at(inner)
        }
        Operand::Math(l, _, r) => mentions_at(l) || mentions_at(r),
    }
}

/// Where a path inside a filter ends.
///
/// A path there runs up against the expression around it, so it stops at the
/// first thing that cannot be part of one. Arithmetic is not read, so `+`, `-`
/// and the rest are not in that set and a member really called `total-vat` is
/// reachable, which matters more than an operator nobody can use.
fn path_end(body: &[u8]) -> usize {
    let mut depth = 0usize;
    let mut quote = 0u8;
    let mut i = 0;
    while i < body.len() {
        let c = body[i];
        if quote != 0 {
            if c == b'\\' {
                i += 2;
                continue;
            }
            if c == quote {
                quote = 0;
            }
        } else if depth == 0 && ends_path(body, i) {
            return i;
        } else {
            match c {
                b'"' | b'\'' => quote = c,
                b'[' => depth += 1,
                b']' => depth = depth.saturating_sub(1),
                _ => {}
            }
        }
        i += 1;
    }
    body.len()
}

/// Whether this byte ends a bare value inside a filter.
///
/// A `-` is not one of these, because a number's sign is read before this runs
/// and a `1-2` with no spaces is not a value anybody meant.
fn stops(c: u8) -> bool {
    matches!(
        c,
        b' ' | b'\t'
            | b'('
            | b')'
            | b'!'
            | b'<'
            | b'>'
            | b'='
            | b'&'
            | b'|'
            | b','
            | b'~'
            | b'+'
            | b'*'
            | b'/'
            | b'%'
    )
}

/// Whether the byte at `i` ends the path being read.
///
/// `+`, `-`, `/` and `%` are all characters a member name may have in it, so
/// `@.total-vat` and `@.p+1` are members and not arithmetic. They only end a
/// path where no name can be running, which is straight after a `]`, so
/// `@.list[0]-1` is a subtraction. A `*` is never a name character, because it
/// is the wildcard, so `@.p*2` is a product and a member really called `p*2` has
/// to be written `@["p*2"]`. All of that was read off RedisJSON 8.10.1.
fn ends_path(body: &[u8], i: usize) -> bool {
    if matches!(body[i], b'+' | b'-' | b'/' | b'%') {
        return i > 0 && body[i - 1] == b']';
    }
    ends_name(body[i])
}

/// Whether this byte ends a bare member name.
///
/// The same set that ends a path, less the four arithmetic characters that are
/// also name characters, since a name is exactly where those are allowed.
fn ends_name(c: u8) -> bool {
    stops(c) && !matches!(c, b'+' | b'/' | b'%')
}

/// The operators that are written as words rather than as symbols.
///
/// `nin` comes before `in` and `noneof` before `nin`, so that the longer one is
/// the one that matches.
const WORD_OPS: &[(&[u8], Op)] = &[
    (b"subsetof", Op::SubsetOf),
    (b"anyof", Op::AnyOf),
    (b"noneof", Op::NoneOf),
    (b"nin", Op::Nin),
    (b"in", Op::In),
    (b"sizeof", Op::Size),
    (b"size", Op::Size),
    (b"empty", Op::Empty),
];

/// Whether `body` starts with `text` and then something that is not more of a
/// word, so that the `in` in `into` is not the operator.
fn word_at(body: &[u8], text: &[u8]) -> bool {
    body.starts_with(text)
        && !body[text.len()..]
            .first()
            .is_some_and(|c| c.is_ascii_alphanumeric() || *c == b'_')
}

/// Where the `close` that matches an opener is, counting nesting and quotes.
///
/// The first `]` is the answer for `[0]` and for `['a']`, and it is the wrong
/// answer for `[?(@.a[0] == 1)]` and for `[?(@.a == "]")]`, which is why this
/// walks rather than searches. `close` is a parameter because a filter can hold
/// a whole value written out, and `{"a":1}` ends at a brace.
fn closer(body: &[u8], close: u8) -> Option<usize> {
    let mut depth = 0usize;
    let mut quote = 0u8;
    let mut i = 0;
    while i < body.len() {
        let c = body[i];
        if quote != 0 {
            if c == b'\\' {
                i += 2;
                continue;
            }
            if c == quote {
                quote = 0;
            }
        } else if c == close && depth == 0 {
            return Some(i);
        } else {
            match c {
                b'"' | b'\'' => quote = c,
                b'[' | b'{' => depth += 1,
                b']' | b'}' => depth = depth.saturating_sub(1),
                _ => {}
            }
        }
        i += 1;
    }
    None
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

    /// The root has three spellings and one of them is a bare dot, which is
    /// what the `JSON.*` commands use when the client gave no path at all.
    #[test]
    fn a_bare_dot_is_the_root_and_is_the_only_dot_with_nothing_after_it() {
        for spelling in ["$", ".", ""] {
            let p = Path::parse(spelling.as_bytes()).expect("the path parses");
            assert!(p.is_root(), "{spelling} should be the root");
            assert!(p.is_definite(), "{spelling} names one place");
        }
        assert!(!Path::parse(b"$").expect("parses").legacy());
        assert!(Path::parse(b".").expect("parses").legacy());
        let d = doc();
        assert_eq!(
            ask(&d, "."),
            format!(
                "[{}]",
                String::from_utf8_lossy(
                    &Value::new(&d)
                        .expect("readable")
                        .to_json()
                        .expect("writable")
                )
            )
        );
        // Every other dot still has to be followed by something.
        assert!(why("..").contains("`..` with nothing after it"));
        assert!(why(".a.").contains("no name after it"));
        assert!(why("$.").contains("no name after it"));
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
        // A bare name stops at a space, because the top level of a path is an
        // expression and a space is where one operand ends. A key with a space
        // in it is reached by quoting it.
        assert!(why("$.store.no such key").contains("does not start with"));
        assert_eq!(ask(&d, r#"$.store["no such key"]"#), "[]");
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
        assert!(why("$.a[x]").contains("at byte "));
        assert!(why("$.a[?(@.b > 1]").contains("no `)`"));
        assert!(why("$.a[?(@.b >)]").contains("where a value goes"));
        assert!(why("$.a[?(@.b > 1) 2]").contains("left over"));
        assert!(why("$.a[?(@.b == nope)]").contains("not a value"));
        assert!(why("$.a[?]").contains("where a value was expected"));
        // An operator with nothing after it, and a `~` on something that has no
        // keys to answer, which the reference refuses as well.
        assert!(why("$.a[?(@.b in)]").contains("where a value goes"));
        assert!(why("$.a[?(@.b size)]").contains("where a value goes"));
        assert!(why("$.a[?(@.b + )]").contains("where a value goes"));
        assert!(why("$.a[?(@.b~~)]").contains("no `)`"));
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

    /// One member per type a `p` can be, each with a name that says which, so
    /// that an answer reads as the list of types that survived rather than as a
    /// list of values.
    ///
    /// The last one has no `p` at all, which is the case most of the operators
    /// treat differently from the rest.
    fn types() -> Vec<u8> {
        from_json(
            br#"[{"p":1,"id":"i"},{"p":2.5,"id":"f"},{"p":"s","id":"t"},
                 {"p":null,"id":"n"},{"p":false,"id":"b"},{"p":[1],"id":"a"},
                 {"p":{"x":1},"id":"o"},{"q":9,"id":"m"}]"#,
        )
        .expect("the text parses")
    }

    /// The ids of the members of [`types`] a filter keeps, as one string.
    fn kept(filter: &str) -> String {
        let d = types();
        ask(&d, &format!("$[?{filter}].id"))
            .replace(['[', ']', '"'], "")
            .replace(',', "")
    }

    #[test]
    fn a_filter_keeps_the_children_its_expression_is_true_of() {
        let d = doc();
        assert_eq!(ask(&d, "$.store.book[?(@.price < 10)].title"), r#"["a"]"#);
        assert_eq!(ask(&d, "$.store.book[?(@.price > 10)].title"), r#"["b"]"#);
        // `$` inside a filter is the whole document rather than the member, so a
        // member can be compared against something somewhere else entirely.
        assert_eq!(
            ask(&d, "$.store.book[?(@.price < $.expensive)].title"),
            r#"["a"]"#
        );
        // The parentheses are a group like any other, so leaving them out and
        // padding them with spaces both read the same.
        assert_eq!(ask(&d, "$.store.book[?@.price<10].title"), r#"["a"]"#);
        assert_eq!(ask(&d, "$.store.book[? (@.price < 10) ].title"), r#"["a"]"#);
    }

    /// An ordering comparison only arises between two values of the same sort,
    /// and where it does not arise the answer is no.
    #[test]
    fn ordering_is_within_a_type_and_not_across_one() {
        assert_eq!(kept("(@.p < 2)"), "i");
        assert_eq!(kept("(@.p > 2)"), "f");
        assert_eq!(kept("(@.p >= 1)"), "if");
        assert_eq!(kept("(@.p <= 2.5)"), "if");
        // Strings order by their characters, and a string is not below a number
        // however the two would sort if they were written out.
        assert_eq!(kept(r#"(@.p > "")"#), "t");
        assert_eq!(kept(r#"(@.p < "s")"#), "");
        assert_eq!(kept(r#"(@.p <= "s")"#), "t");
        assert_eq!(kept(r#"(@.p > "1")"#), "t");
        assert_eq!(kept(r#"(@.p < "1")"#), "");
        // `false` is below `true`, and two nulls are equal without either being
        // below the other.
        assert_eq!(kept("(@.p > false)"), "");
        assert_eq!(kept("(@.p >= false)"), "b");
        assert_eq!(kept("(@.p < true)"), "b");
        assert_eq!(kept("(@.p > null)"), "");
        assert_eq!(kept("(@.p >= null)"), "n");
        // An array and an object have no order at all, so even an equal one is
        // not below or above itself.
        assert_eq!(kept("(@.p >= [1])"), "");
        assert_eq!(kept("(@.p <= [1])"), "");
        assert_eq!(kept(r#"(@.p >= {"x":1})"#), "");
        assert_eq!(kept("(@.p > [])"), "");
        assert_eq!(kept("(@.p > {})"), "");
    }

    /// Equality is the whole value, and the only line it crosses is the one
    /// between the two ways a number is held.
    #[test]
    fn equality_crosses_the_number_split_and_no_other() {
        assert_eq!(kept("(@.p == 1)"), "i");
        assert_eq!(kept("(@.p == 1.0)"), "i");
        assert_eq!(kept("(@.p == 2.5)"), "f");
        assert_eq!(kept(r#"(@.p == "s")"#), "t");
        assert_eq!(kept("(@.p == null)"), "n");
        assert_eq!(kept("(@.p == false)"), "b");
        assert_eq!(kept("(@.p == [1])"), "a");
        assert_eq!(kept(r#"(@.p == {"x":1})"#), "o");
        // A zero is not a false, a one is not a `"1"`, and an array of one is
        // not the thing it holds.
        assert_eq!(kept("(@.p == 0)"), "");
        assert_eq!(kept(r#"(@.p == "1")"#), "");
        assert_eq!(kept("(@.p == [2])"), "");
        assert_eq!(kept(r#"(@.p == {"x":2})"#), "");
    }

    /// `!=` negates the comparison rather than being one, which is the whole
    /// difference: a member with no `p` satisfies it and satisfies nothing else.
    #[test]
    fn not_equal_is_the_negation_of_the_whole_comparison() {
        assert_eq!(kept("(@.p != 1)"), "ftnbaom");
        assert_eq!(kept("(@.p != 9)"), "iftnbaom");
        // Two sides that both answer nothing are equal to nothing, so this keeps
        // every member and its opposite keeps none.
        assert_eq!(kept("(@.zz == @.yy)"), "");
        assert_eq!(kept("(@.zz != @.yy)"), "iftnbaom");
    }

    /// An operand on its own asks whether the document has it, so a `p` that is
    /// there is true whatever it holds.
    #[test]
    fn a_bare_operand_asks_whether_it_is_there() {
        assert_eq!(kept("(@.p)"), "iftnbao");
        assert_eq!(kept("(!@.p)"), "m");
        assert_eq!(kept("!@.p"), "m");
        assert_eq!(kept("(@.q)"), "m");
        // A literal is itself, and the only one that is false is the one that
        // says so.
        assert_eq!(kept("(false)"), "");
        assert_eq!(kept("(0)"), "iftnbaom");
        assert_eq!(kept(r#"("")"#), "iftnbaom");
        assert_eq!(kept("(null)"), "iftnbaom");
        assert_eq!(kept("(true)"), "iftnbaom");
    }

    #[test]
    fn and_binds_tighter_than_or() {
        // Read the other way round this would keep nothing, because no member is
        // both an `i` and dearer than a hundred.
        assert_eq!(kept(r#"(@.id == "i" || @.id == "f" && @.p > 100)"#), "i");
        assert_eq!(kept(r#"(@.id == "i" && @.p == 1 || @.id == "t")"#), "it");
        assert_eq!(kept(r#"((@.id == "i" || @.id == "f") && @.p > 2)"#), "f");
        assert_eq!(kept(r#"(!(@.id == "i") && @.p == 2.5)"#), "f");
    }

    /// A path answers a set, so a comparison asks whether any pair out of the
    /// two sets satisfies it.
    #[test]
    fn a_comparison_holds_when_any_pair_of_answers_does() {
        let d = from_json(br#"[{"t":["x","y"]},{"t":["z"]},{"t":[]}]"#).expect("parses");
        assert_eq!(ask(&d, r#"$[?(@.t[*] == "y")].t"#), r#"[["x","y"]]"#);
        assert_eq!(ask(&d, r#"$[?(@.t[*] == "q")].t"#), "[]");
        // An empty set satisfies nothing, and `!=` is the one operator that
        // reads that as true.
        assert_eq!(ask(&d, r#"$[?(@.t[*] != "z")].t"#), r#"[["x","y"],[]]"#);
        // A path under `@` is a path like any other, `*` and `..` included.
        assert_eq!(kept("(@..x == 1)"), "o");
        // `[*]` is every child of a container and an object is a container, so
        // the object whose only member is a one is kept alongside the array.
        assert_eq!(kept("(@.p[*] == 1)"), "ao");
    }

    #[test]
    fn a_pattern_is_unanchored_and_minds_its_case() {
        assert_eq!(kept(r#"(@.p =~ "s")"#), "t");
        assert_eq!(kept(r#"(@.p =~ "^s$")"#), "t");
        assert_eq!(kept(r#"(@.p =~ "S")"#), "");
        assert_eq!(kept(r#"(@.p =~ "^x")"#), "");
        // A pattern that is not a string, and a pattern that is not a pattern,
        // both answer no rather than refusing the path.
        assert_eq!(kept("(@.p =~ 1)"), "");
        assert_eq!(kept("(@.p =~ null)"), "");
        assert_eq!(kept(r#"(@.p =~ "[")"#), "");
    }

    /// A filter walks the children of whatever it is applied to, and an object
    /// has children too.
    #[test]
    fn a_filter_reads_an_object_the_same_way_it_reads_an_array() {
        let d = from_json(br#"{"one":{"p":1},"two":{"p":9}}"#).expect("parses");
        assert_eq!(ask(&d, "$[?(@.p < 5)]"), r#"[{"p":1}]"#);
        assert_eq!(ask(&d, "$.*[?(@.p < 5)]"), "[]");
        // Nothing that is not a container has children, so a filter over one
        // answers nothing rather than answering it.
        let flat = from_json(br#"[1,"a",null]"#).expect("parses");
        assert_eq!(ask(&flat, "$[*][?(@.p)]"), "[]");
    }

    /// A filter is a selector, so it composes with the rest of them and a path
    /// can go on after it or hold more than one.
    #[test]
    fn a_filter_is_a_selector_like_the_others() {
        let d = from_json(
            br#"{"runs":[{"ok":true,"steps":[{"ms":9},{"ms":31}]},
                        {"ok":false,"steps":[{"ms":2}]}]}"#,
        )
        .expect("parses");
        assert_eq!(
            ask(&d, "$.runs[?(@.ok == true)].steps[?(@.ms > 10)].ms"),
            "[31]"
        );
        assert_eq!(ask(&d, "$..steps[?(@.ms < 10)].ms"), "[9,2]");
        // A filter is not one place, so it is not somewhere a value can be
        // grown, which is what `JSON.SET` asks about.
        assert!(
            !Path::parse(b"$.runs[?(@.ok)]")
                .expect("parses")
                .is_definite()
        );
    }

    /// `in` is membership in the elements of the right side, and `anyof`,
    /// `noneof` and `subsetof` are all about two arrays rather than about a
    /// value and an array.
    #[test]
    fn the_membership_operators_read_an_array_on_the_right() {
        assert_eq!(kept("(@.p in [1,2])"), "i");
        assert_eq!(kept("(@.p nin [1,2])"), "ftnbaom");
        // Membership is the same equality as `==`, so it reaches every type.
        assert_eq!(kept(r#"(@.p in [[1],{"x":1},null,false,"s"])"#), "tnbao");
        // These three want an array on the left as well, which only `a` has.
        assert_eq!(kept("(@.p anyof [1,9])"), "a");
        assert_eq!(kept("(@.p noneof [1,9])"), "iftnbom");
        assert_eq!(kept("(@.p subsetof [1,2,3])"), "a");
        // `nin` and `noneof` negate the whole comparison, so the member with no
        // `p` at all satisfies them and satisfies nothing else here.
        assert_eq!(kept("(@.p subsetof [])"), "");
    }

    /// `size` and `empty` are the one length a string, an array and an object
    /// each have, and nothing else has one.
    #[test]
    fn size_and_empty_are_about_the_three_types_with_a_length() {
        assert_eq!(kept("(@.p size 1)"), "tao");
        assert_eq!(kept("(@.p size 0)"), "");
        assert_eq!(kept("(@.p empty false)"), "tao");
        assert_eq!(kept("(@.p empty true)"), "");
        let d = from_json(br#"[{"p":"","id":"s"},{"p":[],"id":"a"},{"p":{},"id":"o"}]"#)
            .expect("parses");
        assert_eq!(ask(&d, "$[?(@.p empty true)].id"), r#"["s","a","o"]"#);
        assert_eq!(ask(&d, "$[?(@.p size 0)].id"), r#"["s","a","o"]"#);
    }

    /// The six postfix methods, and the one thing that separates `count()` from
    /// the rest of them.
    #[test]
    fn a_method_answers_something_the_document_does_not_hold() {
        assert_eq!(kept("(@.p.length() == 1)"), "tao");
        // `count()` is how many values the operand answered, so it answers a
        // number for an operand that answered nothing, which none of the others
        // do.
        assert_eq!(kept("(@.p.count() == 1)"), "iftnbao");
        assert_eq!(kept("(@.p.count() == 0)"), "m");
        // The aggregates want an array of numbers, and `[1]` is the only one.
        assert_eq!(kept("(@.p.min() == 1)"), "a");
        assert_eq!(kept("(@.p.max() == 1)"), "a");
        assert_eq!(kept("(@.p.sum() == 1)"), "a");
        assert_eq!(kept("(@.p.avg() == 1)"), "a");
        // A name that is not one of the six answers nothing rather than being a
        // path that will not parse.
        assert_eq!(kept("(@.p.size() == 1)"), "");
        assert_eq!(kept("(@.p.nope() == 1)"), "");
    }

    /// Arithmetic is over numbers, it binds the way it does everywhere else,
    /// and it composes with a method on either side.
    #[test]
    fn arithmetic_is_numbers_and_the_usual_precedence() {
        assert_eq!(kept("(@.p + 1 == 2)"), "i");
        assert_eq!(kept("(@.p - 1 == 0)"), "i");
        assert_eq!(kept("(@.p * 2 == 5)"), "f");
        assert_eq!(kept("(@.p / 2 == 0.5)"), "i");
        assert_eq!(kept("(@.p % 2 == 1)"), "i");
        assert_eq!(kept("(@.p + @.p == 2)"), "i");
        assert_eq!(kept("(@.p.length() + 1 == 2)"), "tao");
        // Everything is kept when the expression has no operand in it at all,
        // which is what makes these two about precedence and nothing else.
        assert_eq!(kept("(1 + 2 * 3 == 7)"), "iftnbaom");
        assert_eq!(kept("((1 + 2) * 3 == 9)"), "iftnbaom");
        assert_eq!(kept("(1 + 2 * 3 == 9)"), "");
        // `*` is the one that does not need its spaces. The other four are
        // characters a key name can hold, so without spaces they are part of the
        // name and the member they name is not there.
        assert_eq!(kept("(@.p*2==2)"), "i");
        assert_eq!(kept("(@.p+1==2)"), "");
        assert_eq!(kept("(@.p/2==0.5)"), "");
        assert_eq!(kept("(@.p%2==1)"), "");
        let d = from_json(br#"[{"a-b":1,"a+b":2,"a/b":3,"id":"k"}]"#).expect("parses");
        assert_eq!(ask(&d, r#"$[?(@.a-b == 1)].id"#), r#"["k"]"#);
        assert_eq!(ask(&d, r#"$[?(@.a+b == 2)].id"#), r#"["k"]"#);
        assert_eq!(ask(&d, r#"$[?(@.a/b == 3)].id"#), r#"["k"]"#);
        // Straight after a `]` no name can be running, so there the four are
        // operators with no spaces around them.
        let d = from_json(br#"[{"l":[4],"id":"k"}]"#).expect("parses");
        assert_eq!(ask(&d, r#"$[?(@.l[0]-1 == 3)].id"#), r#"["k"]"#);
        assert_eq!(ask(&d, r#"$[?(@.l[0]+1 == 5)].id"#), r#"["k"]"#);
    }

    /// Arithmetic and the methods want one node and answer nothing for two,
    /// which is the one place a filter is not written against sets. `count()`
    /// is outside the rule.
    #[test]
    fn arithmetic_and_the_methods_want_one_node() {
        let d = from_json(
            br#"[{"l":[1,2],"s":["ab","cd"],"id":"two"},{"l":[1],"s":["a"],"id":"one"}]"#,
        )
        .expect("parses");
        assert_eq!(ask(&d, "$[?(@.l[*] + 1 == 2)].id"), r#"["one"]"#);
        assert_eq!(ask(&d, "$[?(@.s[*].length() == 1)].id"), r#"["one"]"#);
        assert_eq!(ask(&d, "$[?(-@.l[*] == -1)].id"), r#"["one"]"#);
        assert_eq!(ask(&d, "$[?(@.l[*].count() == 2)].id"), r#"["two"]"#);
    }

    /// `~` answers the key names of an object, as a set of strings rather than
    /// as an array value.
    #[test]
    fn the_keys_operator_answers_a_name_at_a_time() {
        assert_eq!(kept("(@.p~)"), "o");
        assert_eq!(kept(r#"(@.p~ == "x")"#), "o");
        assert_eq!(kept(r#"(@.p~ != "x")"#), "iftnbam");
        // The operators that want a collection read the whole set as one, so
        // `size` is how many keys there are and not how long a key is.
        assert_eq!(kept("(@.p~ size 1)"), "o");
        assert_eq!(kept(r#"(@.p~ subsetof ["x"])"#), "o");
        assert_eq!(kept(r#"(@.p~ anyof ["x"])"#), "o");
        assert_eq!(kept(r#"(@.p~ noneof ["x"])"#), "iftnbam");
        assert_eq!(kept("(@.p~ empty false)"), "o");
        assert_eq!(kept("(@.p~ empty true)"), "");
        // `in` and `=~` do not take a key name, which is the reference's
        // behaviour and not a rule with a reason behind it.
        assert_eq!(kept(r#"(@.p~ in ["x"])"#), "");
        assert_eq!(kept(r#"(@.p~ =~ "x")"#), "");
        assert_eq!(kept(r#"(@.p~ nin ["x"])"#), "iftnbaom");
        // A two key object counts as two, and nothing that is not an object
        // answers at all.
        let d = from_json(br#"[{"p":{"abc":1},"id":"one"},{"p":{"a":1,"b":2},"id":"two"}]"#)
            .expect("parses");
        assert_eq!(ask(&d, "$[?(@.p~ size 1)].id"), r#"["one"]"#);
        assert_eq!(ask(&d, "$[?(@.p~ size 2)].id"), r#"["two"]"#);
        assert_eq!(ask(&d, "$[?(@.p~ size 3)].id"), "[]");
    }

    /// A set of key names is a collection wherever a collection goes, including
    /// on the right of one of the collection operators.
    #[test]
    fn a_key_set_reads_as_a_collection_on_either_side() {
        assert_eq!(kept(r#"("x" in @.p~)"#), "o");
        assert_eq!(kept(r#"("q" in @.p~)"#), "");
        assert_eq!(kept("(@.p~ anyof @.p~)"), "o");
        assert_eq!(kept("(@.p~ subsetof @.p~)"), "o");
        assert_eq!(kept("(@.p~ noneof @.p~)"), "iftnbam");
        assert_eq!(kept(r#"(["x"] subsetof @.p~)"#), "o");
        // A path is a collection on the right too, so a value can be looked for
        // in an array the member itself holds.
        assert_eq!(kept("(1 in @.p)"), "a");
        assert_eq!(kept("(2 in @.p)"), "");
    }

    /// An object with no keys answers a set that is there and empty, and
    /// something that is not an object answers no set at all. Every collection
    /// operator can tell the two apart.
    #[test]
    fn an_empty_object_has_a_key_set_and_a_scalar_has_none() {
        let d = from_json(br#"[{"p":{},"id":"e"},{"p":1,"id":"s"},{"id":"m"}]"#).expect("parses");
        for (path, want) in [
            (r#"$[?(@.p~ subsetof ["x"])].id"#, r#"["e"]"#),
            (r#"$[?(@.p~ anyof ["x"])].id"#, "[]"),
            // `noneof` is the negation of the whole comparison, so a side that
            // answers no set at all makes it true.
            (r#"$[?(@.p~ noneof ["x"])].id"#, r#"["e","s","m"]"#),
            ("$[?(@.p~ empty true)].id", r#"["e"]"#),
            ("$[?(@.p~ empty false)].id", "[]"),
            ("$[?(@.p~ size 0)].id", r#"["e"]"#),
            // The bare test is about answering a name, and an empty set answers
            // none, so it is the one place the two read alike.
            ("$[?(@.p~)].id", "[]"),
        ] {
            assert_eq!(ask(&d, path), want, "{path}");
        }
    }

    /// `sizeof` is another spelling of `size`, and a leading `-` or `+` is a
    /// number or nothing at all.
    #[test]
    fn the_alias_and_the_signs_read_the_way_the_reference_reads_them() {
        assert_eq!(kept("(@.p sizeof 1)"), "tao");
        assert_eq!(kept("(@.p size 1)"), "tao");
        assert_eq!(kept("(-@.p == -1)"), "i");
        assert_eq!(kept("(+@.p == 1)"), "i");
        assert_eq!(kept("(@.p == +1)"), "i");
        assert_eq!(kept("(@.p > -1)"), "if");
        assert_eq!(kept("(@.p - -1 == 2)"), "i");
        // A sign is a number and nothing else, so it drops a string rather than
        // passing it along, and the bare test on it is that number's own test.
        assert_eq!(kept("(-@.p)"), "if");
        // One sign and no more, and a group is how a second one is written.
        assert_eq!(kept("(-(-@.p) == 1)"), "i");
        assert!(why("$[?(--@.p == 1)]").contains("not a value"));
    }

    /// What a projection answers over [`doc`], as JSON text, the same way
    /// [`ask`] reads a path.
    fn sum(bytes: &[u8], path: &str) -> String {
        let v = Value::new(bytes).expect("readable");
        let p = Path::parse(path.as_bytes()).expect("the path parses");
        assert!(p.is_projection(), "{path} should be a projection");
        let mut out = Vec::new();
        out.push(b'[');
        for (i, got) in p.project(&v).iter().enumerate() {
            if i > 0 {
                out.push(b',');
            }
            got.write_json_at(&crate::Format::default(), &mut out, 0)
                .expect("writable");
        }
        out.push(b']');
        String::from_utf8(out).expect("UTF-8")
    }

    /// A path that is an expression rather than a way through the document.
    #[test]
    fn a_projection_works_something_out_rather_than_naming_a_place() {
        let d = doc();
        assert_eq!(sum(&d, "$.expensive + 1"), "[11]");
        assert_eq!(sum(&d, "$.expensive * 2"), "[20]");
        assert_eq!(sum(&d, "-$.expensive"), "[-10]");
        assert_eq!(sum(&d, "$.store.book.length()"), "[2]");
        assert_eq!(sum(&d, "$.store.book[*].count()"), "[2]");
        assert_eq!(sum(&d, "$.store.bike~"), r#"["price"]"#);
        // A division is a fraction however evenly it divides, and the four
        // aggregates are fractions too, which is the reference's doing rather
        // than anything the arithmetic asks for.
        assert_eq!(sum(&d, "$.expensive / 1"), "[10.0]");
        assert_eq!(sum(&d, "$.store.book[*].price.sum()"), "[]");
        // Nothing at all rather than the error a path would raise, and an array
        // even when the path was written the legacy way.
        assert_eq!(sum(&d, "$.nope + 1"), "[]");
        assert_eq!(sum(&d, "$.nope.count()"), "[0]");
        assert_eq!(sum(&d, "$.expensive / 0"), "[]");
        assert_eq!(sum(&d, ".expensive + 1"), "[11]");
        assert_eq!(sum(&d, ".store.book.length()"), "[2]");
        // The first thing up here is a path however it is written, so this is a
        // member really called `2` and not the number. Inside a group the
        // ordinary rules are back.
        assert_eq!(sum(&d, "2 + 3"), "[]");
        assert_eq!(sum(&d, "(2) + 3"), "[5]");
        // A path is a path and not a projection, which is what keeps every
        // other `JSON.*` command working.
        for path in ["$.expensive", "$..price", "$.store.book[?(@.price < 10)]"] {
            let p = Path::parse(path.as_bytes()).expect("parses");
            assert!(!p.is_projection(), "{path} is a path");
        }
        // `@` has no meaning outside a filter, so a projection that mentions one
        // is not a projection and does not parse as a path either.
        assert!(why("@.expensive + 1").contains("does not start with"));
    }
}
