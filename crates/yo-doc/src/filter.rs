//! The filter selector, `[?(@.price < 10)]`.
//!
//! A filter is a selector like any other: it is applied to a value and it
//! answers a subset of that value's children, so `$.items[?(@.price < 10)]`
//! reads as the items, then the ones the expression is true for. It is the only
//! selector whose answer depends on the document rather than only on the path,
//! and the only one that can look somewhere other than where it stands, because
//! `$` inside an expression is the whole document.
//!
//! What the operators mean is written up in [`query`](crate::query), because it
//! is what a client sees rather than what this module does. This is the tree
//! they parse into and the walk that answers them.
//!
//! Everything here works over a set on either side, since a path operand answers
//! a set, and the empty set is the ordinary case rather than an error: a member
//! that has no `price` at all is a member `@.price < 10` is false for. The one
//! operator that reads it the other way round is `!=`, which negates the whole
//! comparison, so it is answered before the pairs are walked rather than inside
//! the walk.
//!
//! Nothing here allocates per value except the two answer buffers, and a pattern
//! is compiled once while the path is parsed rather than once per value it is
//! run against.

use core::cmp::Ordering;
use std::sync::Arc;

use yo_common::re::{Matcher, Regex};
use yo_common::{Code, Error, Result};

use crate::head::Kind;
use crate::query::{Sel, select_from};
use crate::read::Value;

/// One filter expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Expr<'a> {
    /// Either side, and the first one that is true wins.
    Or(Box<Expr<'a>>, Box<Expr<'a>>),
    /// Both sides. Binds tighter than `Or`, the same as everywhere else.
    And(Box<Expr<'a>>, Box<Expr<'a>>),
    /// The opposite of what is inside it.
    Not(Box<Expr<'a>>),
    /// An operand on its own, which asks whether it is there.
    Test(Operand<'a>),
    /// Two operands and what is being asked about them.
    Cmp(Operand<'a>, Op, Operand<'a>),
}

/// One side of a comparison.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Operand<'a> {
    /// A path, from the current node when `at` is set and from the root when it
    /// is not. It answers a set, which may be empty.
    Path {
        /// Whether the path started at `@` rather than at `$`.
        at: bool,
        /// The selectors after that.
        sels: Vec<Sel<'a>>,
    },
    /// A value written into the path itself, held encoded so that comparing it
    /// against a value out of a document is the same code either way.
    Lit(Vec<u8>),
    /// A compiled regular expression, which is the right hand side of `=~` and
    /// is nothing anywhere else.
    Re(Pattern),
}

/// A compiled pattern and the text it came from.
///
/// The text is kept because two paths are equal when they read the same and a
/// compiled program is not a thing to compare. The `Arc` is because a path is
/// cloned by [`Path::split_last`](crate::Path::split_last) and a program is not
/// worth copying.
#[derive(Debug, Clone)]
pub(crate) struct Pattern {
    text: Vec<u8>,
    re: Arc<Regex>,
}

impl PartialEq for Pattern {
    fn eq(&self, other: &Pattern) -> bool {
        self.text == other.text
    }
}

impl Eq for Pattern {}

impl Pattern {
    /// Compile `text`, once, while the path is being parsed.
    pub(crate) fn new(text: &[u8]) -> Result<Pattern> {
        let re = Regex::new(text, false).map_err(|e| {
            Error::new(
                Code::Invalid,
                format!("a filter pattern that is not a pattern: {e}"),
            )
        })?;
        Ok(Pattern {
            text: text.to_vec(),
            re: Arc::new(re),
        })
    }
}

/// What is being asked about two operands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Op {
    /// `==`.
    Eq,
    /// `!=`.
    Ne,
    /// `<`.
    Lt,
    /// `<=`.
    Le,
    /// `>`.
    Gt,
    /// `>=`.
    Ge,
    /// `=~`, a regular expression against a string.
    Re,
}

impl Expr<'_> {
    /// Whether this expression is true of `cur`, inside `root`.
    pub(crate) fn holds<'v>(&'v self, root: &Value<'v>, cur: &Value<'v>) -> bool {
        match self {
            Expr::Or(l, r) => l.holds(root, cur) || r.holds(root, cur),
            Expr::And(l, r) => l.holds(root, cur) && r.holds(root, cur),
            Expr::Not(e) => !e.holds(root, cur),
            // A path is a question about the document, so a member that is
            // there is true whatever it holds. A literal is itself, and the one
            // value that is false is the one that says so.
            Expr::Test(o) => match o {
                Operand::Path { .. } => !values(o, root, cur).is_empty(),
                Operand::Lit(_) => values(o, root, cur)
                    .first()
                    .is_none_or(|v| v.as_bool() != Some(false)),
                // Never built: a pattern is only ever the right hand side of a
                // `=~`, and the parser will not put one anywhere else.
                Operand::Re(_) => false,
            },
            Expr::Cmp(l, op, r) => cmp(l, *op, r, root, cur),
        }
    }
}

/// Everything one operand answers.
fn values<'v>(o: &'v Operand<'_>, root: &Value<'v>, cur: &Value<'v>) -> Vec<Value<'v>> {
    let mut out = Vec::new();
    match o {
        Operand::Path { at, sels } => {
            select_from(sels, if *at { cur } else { root }, root, &mut out);
        }
        // The bytes belong to the path rather than to the document, which is
        // fine: the path outlives the walk, and that is what the lifetime here
        // says.
        Operand::Lit(bytes) => out.extend(Value::new(bytes)),
        Operand::Re(_) => {}
    }
    out
}

/// One comparison, which is true when some pair out of the two sides satisfies
/// it.
fn cmp<'v>(
    l: &'v Operand<'_>,
    op: Op,
    r: &'v Operand<'_>,
    root: &Value<'v>,
    cur: &Value<'v>,
) -> bool {
    if op == Op::Re {
        // A pattern that is not a string literal is not a pattern, and the
        // reference answers nothing rather than complaining.
        let Operand::Re(pat) = r else {
            return false;
        };
        let mut m = Matcher::new();
        m.reserve(&pat.re);
        return values(l, root, cur)
            .iter()
            .filter_map(Value::text_bytes)
            .any(|s| m.is_match(&pat.re, s));
    }
    let left = values(l, root, cur);
    let right = values(r, root, cur);
    // `!=` is the negation of `==` over the whole of both sides and not a
    // comparison in its own right, so a side that answers nothing makes it true
    // where it makes every other operator false.
    if op == Op::Ne {
        return !left.iter().any(|a| right.iter().any(|b| same(a, b)));
    }
    left.iter().any(|a| {
        right.iter().any(|b| match op {
            Op::Eq => same(a, b),
            Op::Lt => order(a, b) == Some(Ordering::Less),
            Op::Le => matches!(order(a, b), Some(Ordering::Less | Ordering::Equal)),
            Op::Gt => order(a, b) == Some(Ordering::Greater),
            Op::Ge => matches!(order(a, b), Some(Ordering::Greater | Ordering::Equal)),
            Op::Ne | Op::Re => unreachable!("both are answered above"),
        })
    })
}

/// Where two values stand relative to each other, or `None` when the question
/// does not arise.
///
/// Numbers order as numbers whichever of the two ways each is held, strings
/// order by their bytes, which for UTF-8 is by their characters, and `false` is
/// below `true`. Two nulls are equal and neither is below the other, which is
/// why `null >= null` is true and `null < null` is not.
///
/// Everything else is `None`, and `None` makes every ordering comparison false.
/// A string is not below a number, and an array or an object is not below
/// anything at all, not even an equal one: `[] >= []` is false in the reference
/// and false here.
fn order(a: &Value<'_>, b: &Value<'_>) -> Option<Ordering> {
    if let (Some(x), Some(y)) = (number(a), number(b)) {
        return x.partial_cmp(&y);
    }
    if a.kind() != b.kind() {
        return None;
    }
    match a.kind() {
        Kind::Null => Some(Ordering::Equal),
        Kind::Bool => Some(a.as_bool().cmp(&b.as_bool())),
        Kind::Text => Some(a.text_bytes().cmp(&b.text_bytes())),
        Kind::Int | Kind::Float | Kind::Array | Kind::Object => None,
    }
}

/// A value as a number, whichever of the two ways it is held.
fn number(v: &Value<'_>) -> Option<f64> {
    #[expect(
        clippy::cast_precision_loss,
        reason = "a comparison and not a round trip"
    )]
    v.as_int().map(|i| i as f64).or_else(|| v.as_float())
}

/// Whether two values are the same value.
///
/// Numbers are compared as numbers, so `1` and `1.0` are the same, and nothing
/// else crosses a type: a `0` is not a `false` and a `1` is not a `"1"`. An
/// array is its elements in order, and an object is its members looked up by
/// name so that two objects written in different orders still come out equal.
fn same(a: &Value<'_>, b: &Value<'_>) -> bool {
    if let (Some(x), Some(y)) = (number(a), number(b)) {
        return x == y;
    }
    if a.kind() != b.kind() {
        return false;
    }
    match a.kind() {
        Kind::Null => true,
        Kind::Bool => a.as_bool() == b.as_bool(),
        Kind::Text => a.text_bytes() == b.text_bytes(),
        Kind::Array => {
            a.len() == b.len()
                && (0..a.len()).all(|i| match (a.at(i), b.at(i)) {
                    (Some(x), Some(y)) => same(&x, &y),
                    _ => false,
                })
        }
        Kind::Object => object(a, b),
        // The two number kinds are answered at the top, whichever way round
        // they were written.
        Kind::Int | Kind::Float => false,
    }
}

/// Two objects, member by member.
///
/// An interned object holds key ids rather than key bytes, so it yields no
/// members here and is only ever equal to another empty object. Nothing that
/// reaches a filter is interned: a document a `JSON.*` command stored was built
/// from JSON text and interning is the columnar side's.
fn object(a: &Value<'_>, b: &Value<'_>) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut seen = 0;
    for (key, x) in a.members() {
        let Some(y) = b.get(key) else {
            return false;
        };
        if !same(&x, &y) {
            return false;
        }
        seen += 1;
    }
    seen == a.len()
}
