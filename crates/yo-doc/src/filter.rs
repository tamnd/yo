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
//! that has no `price` at all is a member `@.price < 10` is false for. The
//! operators that read it the other way round are `!=`, `nin` and `noneof`,
//! which negate the whole comparison, so they are answered before the pairs are
//! walked rather than inside the walk.
//!
//! An operand is not always something in the document. `@.p + 1`, `@.p.sum()`
//! and `@.p~` all answer values that are nowhere in it, which is what [`Item`]
//! is for: a hit is either a value the document holds, a number that was worked
//! out, or a key name. A comparison between two document values is the whole
//! value, and one where either side was worked out is a number against a number
//! or a string against a string, because those are the only two kinds anything
//! here can make.
//!
//! Nothing here allocates per value except the answer buffers, and a pattern is
//! compiled once while the path is parsed rather than once per value it is run
//! against.

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
    /// The key names of whatever is under it, which is the postfix `~`.
    Keys(Box<Operand<'a>>),
    /// A postfix method, `@.tags.length()`.
    Call(Box<Operand<'a>>, Fun),
    /// Arithmetic, which is only ever a number and only ever over numbers.
    Math(Box<Operand<'a>>, Arith, Box<Operand<'a>>),
    /// A leading `-` or `+`. It answers a number and nothing else, so `-@.name`
    /// on a string answers nothing rather than answering the string.
    Sign(Box<Operand<'a>>, bool),
}

/// A postfix method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fun {
    /// How long a string, an array or an object is. Nothing else has a length.
    Length,
    /// How many values the operand answered, which is a number even when that
    /// number is zero, so `@.nope.count() == 0` is true.
    Count,
    /// The smallest, largest, total and mean of an array of numbers. An array
    /// with anything else in it, and an empty one, answer nothing.
    Min,
    Max,
    Sum,
    Avg,
    /// A name that is not one of the above. It answers nothing, which is what
    /// the reference does with `@.p.size()` rather than refusing the path.
    Unknown,
}

impl Fun {
    /// The method `name` spells, or [`Fun::Unknown`] when it spells none.
    pub(crate) fn named(name: &[u8]) -> Fun {
        match name {
            b"length" => Fun::Length,
            b"count" => Fun::Count,
            b"min" => Fun::Min,
            b"max" => Fun::Max,
            b"sum" => Fun::Sum,
            b"avg" => Fun::Avg,
            _ => Fun::Unknown,
        }
    }
}

/// An arithmetic operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Arith {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
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
    /// `in`, whether the left value is one of the elements on the right.
    In,
    /// `nin`, the negation of `in` over both whole sides.
    Nin,
    /// `anyof`, whether two arrays share an element.
    AnyOf,
    /// `noneof`, the negation of `anyof` over both whole sides.
    NoneOf,
    /// `subsetof`, whether every element on the left is on the right.
    SubsetOf,
    /// `size`, whether the left is a string, an array or an object of this
    /// length.
    Size,
    /// `empty`, whether the left is a string, an array or an object and whether
    /// it is empty, against the `true` or `false` on the right.
    Empty,
}

/// One thing an operand answered.
///
/// A path answers values the document holds and a literal answers its own
/// bytes, which are both a [`Value`]. Arithmetic and the aggregates answer a
/// number that is nowhere in the document, and `~` answers a key name, which is
/// bytes in the document but not a value in it.
#[derive(Debug, Clone, Copy)]
enum Item<'v> {
    Ref(Value<'v>),
    Num(f64),
    Key(&'v [u8]),
}

impl<'v> Item<'v> {
    /// This as a number, whichever way it is held.
    fn number(&self) -> Option<f64> {
        match self {
            Item::Ref(v) => number(v),
            Item::Num(n) => Some(*n),
            Item::Key(_) => None,
        }
    }

    /// This as a string.
    fn text(&self) -> Option<&'v [u8]> {
        match self {
            Item::Ref(v) => v.text_bytes(),
            Item::Num(_) => None,
            Item::Key(k) => Some(k),
        }
    }

    /// How long this is, for the things that have a length.
    ///
    /// A number that was worked out has none, and neither has a key name, since
    /// neither is something a client could have asked the length of.
    fn size(&self) -> Option<usize> {
        match self {
            Item::Ref(v) => match v.kind() {
                Kind::Text => Some(v.text_bytes().unwrap_or_default().len()),
                Kind::Array | Kind::Object => Some(v.len()),
                _ => None,
            },
            Item::Num(_) | Item::Key(_) => None,
        }
    }

    /// The elements of this, when it is an array.
    fn elements(&self) -> impl Iterator<Item = Item<'v>> {
        let v = match self {
            Item::Ref(v) if v.kind() == Kind::Array => Some(*v),
            _ => None,
        };
        v.into_iter().flat_map(|v| v.iter().map(Item::Ref))
    }
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
                Operand::Lit(_) => values(o, root, cur)
                    .first()
                    .is_none_or(|v| v.as_bool() != Some(false)),
                _ => !values(o, root, cur).is_empty(),
            },
            Expr::Cmp(l, op, r) => cmp(l, *op, r, root, cur),
        }
    }
}

impl Item<'_> {
    /// This as a boolean, for the literal that a bare operand tests.
    fn as_bool(&self) -> Option<bool> {
        match self {
            Item::Ref(v) => v.as_bool(),
            _ => None,
        }
    }
}

/// Everything one operand answers.
fn values<'v>(o: &'v Operand<'_>, root: &Value<'v>, cur: &Value<'v>) -> Vec<Item<'v>> {
    let mut out = Vec::new();
    match o {
        Operand::Path { at, sels } => {
            let mut hits = Vec::new();
            select_from(sels, if *at { cur } else { root }, root, &mut hits);
            out.extend(hits.into_iter().map(Item::Ref));
        }
        // The bytes belong to the path rather than to the document, which is
        // fine: the path outlives the walk, and that is what the lifetime here
        // says.
        Operand::Lit(bytes) => out.extend(Value::new(bytes).map(Item::Ref)),
        Operand::Re(_) => {}
        Operand::Keys(inner) => out.extend(keyset(inner, root, cur).unwrap_or_default()),
        Operand::Call(inner, fun) => {
            let got = values(inner, root, cur);
            call(&got, *fun, &mut out);
        }
        Operand::Math(l, op, r) => {
            let (left, right) = (values(l, root, cur), values(r, root, cur));
            for a in left.iter().filter_map(Item::number) {
                for b in right.iter().filter_map(Item::number) {
                    out.push(Item::Num(match op {
                        Arith::Add => a + b,
                        Arith::Sub => a - b,
                        Arith::Mul => a * b,
                        Arith::Div => a / b,
                        Arith::Rem => a % b,
                    }));
                }
            }
        }
        Operand::Sign(inner, neg) => out.extend(
            values(inner, root, cur)
                .iter()
                .filter_map(Item::number)
                .map(|n| Item::Num(if *neg { -n } else { n })),
        ),
    }
    out
}

/// The key names an operand answers, and `None` when nothing under it was an
/// object.
///
/// The two are not the same thing, and every collection operator can tell them
/// apart: `{}~` is an empty set that is there, so `{}~ subsetof ["x"]` is true
/// and `{}~ empty true` is true, while `1~` and a path that matched nothing
/// answer no set at all and satisfy neither.
fn keyset<'v>(of: &'v Operand<'_>, root: &Value<'v>, cur: &Value<'v>) -> Option<Vec<Item<'v>>> {
    let mut out = None;
    for it in values(of, root, cur) {
        let Item::Ref(v) = it else { continue };
        if v.kind() != Kind::Object {
            continue;
        }
        let into: &mut Vec<Item<'v>> = out.get_or_insert_default();
        into.extend((0..v.len()).filter_map(|i| v.key_at(i)).map(Item::Key));
    }
    out
}

/// One postfix method over what the operand under it answered.
fn call<'v>(got: &[Item<'v>], fun: Fun, out: &mut Vec<Item<'v>>) {
    if fun == Fun::Count {
        // The one method that answers a number whatever it was given, which is
        // why an operand that matched nothing still compares against zero.
        #[expect(
            clippy::cast_precision_loss,
            reason = "a count that reaches the precision of a double is not a document"
        )]
        out.push(Item::Num(got.len() as f64));
        return;
    }
    for it in got {
        match fun {
            Fun::Length => {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "a length that reaches the precision of a double is not a document"
                )]
                out.extend(it.size().map(|n| Item::Num(n as f64)));
            }
            // An array with anything but numbers in it, and an empty one, answer
            // nothing rather than answering over what is left, because a mean
            // over the numeric half of a mixed array is not a number anybody
            // asked for.
            Fun::Min | Fun::Max | Fun::Sum | Fun::Avg => {
                let ns: Option<Vec<f64>> = it.elements().map(|e| e.number()).collect();
                let Some(ns) = ns.filter(|ns| !ns.is_empty()) else {
                    continue;
                };
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "a count that reaches the precision of a double is not a document"
                )]
                let n = match fun {
                    Fun::Min => ns.iter().copied().fold(f64::INFINITY, f64::min),
                    Fun::Max => ns.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                    Fun::Sum => ns.iter().sum(),
                    _ => ns.iter().sum::<f64>() / ns.len() as f64,
                };
                out.push(Item::Num(n));
            }
            Fun::Count | Fun::Unknown => {}
        }
    }
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
    // `~` answers a set of key names, and the collection operators read that set
    // as the collection rather than looking inside whatever the keys name. It is
    // also the one operand that can answer nothing and still be there, which is
    // what `Side::live` is about.
    let (left, right) = (Side::of(l, root, cur), Side::of(r, root, cur));
    if op == Op::Re {
        // A pattern that is not a string literal is not a pattern, and the
        // reference answers nothing rather than complaining. A key name is not a
        // string as far as this operator is concerned either.
        let Operand::Re(pat) = r else {
            return false;
        };
        if left.keys {
            return false;
        }
        let mut m = Matcher::new();
        m.reserve(&pat.re);
        return left
            .items
            .iter()
            .filter_map(Item::text)
            .any(|s| m.is_match(&pat.re, s));
    }
    // These three are the negation of a comparison over the whole of both sides
    // and not comparisons in their own right, so a side that answers nothing
    // makes them true where it makes every other operator false.
    match op {
        Op::Ne => {
            return !left
                .items
                .iter()
                .any(|a| right.items.iter().any(|b| same(a, b)));
        }
        Op::Nin => return !within(&left, &right),
        Op::NoneOf => return !shares(&left, &right),
        _ => {}
    }
    match op {
        Op::In => return within(&left, &right),
        Op::AnyOf => return shares(&left, &right),
        Op::SubsetOf => {
            if !left.live || !right.live {
                return false;
            }
            let inside = |a: &Item<'v>| right.bag().any(|x| same(a, &x));
            if left.keys {
                return left.items.iter().all(inside);
            }
            return left.items.iter().any(|a| {
                matches!(a, Item::Ref(v) if v.kind() == Kind::Array)
                    && a.elements().all(|e| inside(&e))
            });
        }
        Op::Size => {
            let Some(want) = right.items.iter().find_map(|b| b.number()) else {
                return false;
            };
            #[expect(
                clippy::cast_precision_loss,
                reason = "a length that reaches the precision of a double is not a document"
            )]
            return if left.keys {
                left.live && left.items.len() as f64 == want
            } else {
                left.items
                    .iter()
                    .any(|a| a.size().is_some_and(|n| n as f64 == want))
            };
        }
        Op::Empty => {
            let Some(want) = right.items.iter().find_map(|b| b.as_bool()) else {
                return false;
            };
            return if left.keys {
                left.live && left.items.is_empty() == want
            } else {
                left.items
                    .iter()
                    .any(|a| a.size().is_some_and(|n| (n == 0) == want))
            };
        }
        _ => {}
    }
    left.items.iter().any(|a| {
        right.items.iter().any(|b| match op {
            Op::Eq => same(a, b),
            Op::Lt => order(a, b) == Some(Ordering::Less),
            Op::Le => matches!(order(a, b), Some(Ordering::Less | Ordering::Equal)),
            Op::Gt => order(a, b) == Some(Ordering::Greater),
            Op::Ge => matches!(order(a, b), Some(Ordering::Greater | Ordering::Equal)),
            Op::Ne
            | Op::Re
            | Op::In
            | Op::Nin
            | Op::AnyOf
            | Op::NoneOf
            | Op::SubsetOf
            | Op::Size
            | Op::Empty => unreachable!("all of these are answered above"),
        })
    })
}

/// What one side of a comparison answered.
///
/// `keys` and `live` are both about `~`, which is the only operand that answers
/// a collection rather than a value and the only one that can answer an empty
/// collection that is nonetheless there.
struct Side<'v> {
    items: Vec<Item<'v>>,
    keys: bool,
    live: bool,
}

impl<'v> Side<'v> {
    fn of(o: &'v Operand<'_>, root: &Value<'v>, cur: &Value<'v>) -> Side<'v> {
        if let Operand::Keys(inner) = o {
            let got = keyset(inner, root, cur);
            return Side {
                live: got.is_some(),
                items: got.unwrap_or_default(),
                keys: true,
            };
        }
        Side {
            items: values(o, root, cur),
            keys: false,
            live: true,
        }
    }

    /// The collection this side is, for the operators that want one. That is
    /// the answers themselves for a `~` and the elements of an array otherwise,
    /// so anything that is not either has no collection and matches nothing.
    fn bag(&self) -> impl Iterator<Item = Item<'v>> {
        let (mine, elems) = if self.keys {
            (Some(self.items.iter().copied()), None)
        } else {
            (None, Some(self.items.iter().flat_map(Item::elements)))
        };
        mine.into_iter()
            .flatten()
            .chain(elems.into_iter().flatten())
    }
}

/// Whether anything on the left is an element of the collection on the right,
/// which is `in`. A key name on the left is not asked, which is the reference's
/// behaviour and not a rule with a reason behind it.
fn within(left: &Side<'_>, right: &Side<'_>) -> bool {
    !left.keys && right.live && left.items.iter().any(|a| right.bag().any(|e| same(a, &e)))
}

/// Whether the two collections share an element, which is `anyof`.
fn shares(left: &Side<'_>, right: &Side<'_>) -> bool {
    left.live && right.live && left.bag().any(|e| right.bag().any(|x| same(&e, &x)))
}

/// Where two answers stand relative to each other, or `None` when the question
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
fn order(a: &Item<'_>, b: &Item<'_>) -> Option<Ordering> {
    if let (Item::Ref(x), Item::Ref(y)) = (a, b) {
        return order_value(x, y);
    }
    if let (Some(x), Some(y)) = (a.number(), b.number()) {
        return x.partial_cmp(&y);
    }
    match (a.text(), b.text()) {
        (Some(x), Some(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

/// [`order`] for two values the document holds, which is the only case where a
/// boolean or a null can turn up.
fn order_value(a: &Value<'_>, b: &Value<'_>) -> Option<Ordering> {
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

/// Whether two answers are the same.
fn same(a: &Item<'_>, b: &Item<'_>) -> bool {
    if let (Item::Ref(x), Item::Ref(y)) = (a, b) {
        return same_value(x, y);
    }
    if let (Some(x), Some(y)) = (a.number(), b.number()) {
        return x == y;
    }
    match (a.text(), b.text()) {
        (Some(x), Some(y)) => x == y,
        _ => false,
    }
}

/// Whether two values are the same value.
///
/// Numbers are compared as numbers, so `1` and `1.0` are the same, and nothing
/// else crosses a type: a `0` is not a `false` and a `1` is not a `"1"`. An
/// array is its elements in order, and an object is its members looked up by
/// name so that two objects written in different orders still come out equal.
fn same_value(a: &Value<'_>, b: &Value<'_>) -> bool {
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
                    (Some(x), Some(y)) => same_value(&x, &y),
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
        if !same_value(&x, &y) {
            return false;
        }
        seen += 1;
    }
    seen == a.len()
}
