//! The query language, from the bytes a client sends to the tree that answers.
//!
//! This is the half of search that is not storage. A client sends
//! `@title:hello world -@year:[2000 2010]` and something has to turn that into a
//! shape a reader can walk, and the shape is what `FT.EXPLAIN` prints. Getting
//! the shape right is most of getting the answer right, so the tree is built and
//! printed before anything reads a document, and it is checked against a real
//! server's own printout rather than against a reading of the grammar.
//!
//! # It is parsed straight off the bytes
//!
//! There is no token stream in between. Two things push that way. The errors
//! carry a byte offset into the query, `Syntax error at offset 7 near the`, and
//! an offset that survives a token stream is an offset that has been carried
//! through it by hand. And the grammar is not the same everywhere: the letters
//! inside `[` `]` are numbers, the letters inside `{` `}` are tags with their
//! own separator rules, and `w'...'` holds a pattern where `*` is not an
//! operator. A scanner would need a mode for each of those and the modes are
//! exactly what the recursive descent already knows.
//!
//! # There is more than one language here
//!
//! `DIALECT` is a version number on the grammar, and the one a client gets when
//! it does not ask is still the first one. The two are not cosmetically
//! different, they parse the same bytes into different trees:
//!
//! ```text
//! @a:hello world      dialect 1   @a:INTERSECT { @a:hello  @a:world }
//!                     dialect 2   INTERSECT { @a:hello  world }
//! -hello world        dialect 1   NOT{ INTERSECT { hello  world } }
//!                     dialect 2   INTERSECT { NOT{ hello }  world }
//! ```
//!
//! In the first dialect a field modifier and a negation both reach as far to the
//! right as they can, and in the second they take one thing each. Both are here,
//! because the loose one is the default and therefore the one most clients are
//! on, and the tight one is what anybody writing a new client will pick.
//!
//! # Fields are a bitmask, not a name
//!
//! `@a|b:hello` asks two fields at once, so what a node carries is a set of
//! fields rather than one. It is a set with an ANDing rule on top: a modifier
//! applies itself to everything below it, so `@a:(hello|@b:world)` leaves
//! `world` asking for a field that is in `a` and in `b` at once, which is no
//! field at all. A real server prints that as `@NULL:` and answers nothing, and
//! so does this.
//!
//! # Stopwords come out here, not later
//!
//! A term that is a stopword is dropped while the tree is built, so
//! `hello the world` is an intersection of two and `the` on its own is nothing
//! at all. A phrase is the exception: `"hello the"` keeps the stopword under the
//! second dialect and is a syntax error under the first, which is a real
//! difference between them and not a detail of this implementation.

pub mod explain;
pub mod parse;

pub use explain::explain;
pub use parse::{Ask, Bad, parse};

/// Which fields a node asks, as one bit per text field in schema order.
///
/// Sixty four bits because a real server stops at a hundred and twenty eight and
/// at thirty two without `MAXTEXTFIELDS`, so anything past sixty four is already
/// past the point where an index is doing something else wrong.
pub type Mask = u64;

/// Every field, which is what a node asks for when nobody narrowed it down.
///
/// It is all ones rather than the fields the schema happens to have, so it stays
/// right when a field is added later, and so that a modifier naming every field
/// one at a time still prints itself out rather than disappearing.
pub const EVERY: Mask = Mask::MAX;

/// One node of a parsed query.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    /// What kind of node it is and what it holds.
    pub what: What,
    /// Which fields it asks.
    pub mask: Mask,
    /// What the client asked to weigh it by, when that was not one.
    pub weight: Option<f64>,
    /// How many words a phrase may have between its own, when the client said.
    ///
    /// A negative one means no limit and is what a phrase has when nobody asked,
    /// so it is kept rather than dropped and simply is not printed.
    pub slop: Option<i64>,
    /// Whether the words of a phrase have to come in the order they were
    /// written, which the printout shows next to a slop.
    pub inorder: bool,
}

/// What a node is.
///
/// The shape is a real server's shape rather than the one a fresh design would
/// pick, because `FT.EXPLAIN` prints it and clients read the printout. The one
/// that looks most redundant is the union a single word turns into: `hello`
/// parses to a union of `hello` and the stem of `hello`, and that union is a
/// real node with a real cost rather than a way of talking about one.
#[derive(Debug, Clone, PartialEq)]
pub enum What {
    /// Nothing at all, which is what a query of pure stopwords comes to.
    Empty,
    /// Every document, which is what a bare `*` asks for.
    Wildcard,
    /// One word.
    Term(Word),
    /// Any of these.
    Union(Vec<Node>),
    /// All of these.
    Intersect(Vec<Node>),
    /// These words next to each other, in this order.
    Exact(Vec<Node>),
    /// None of these.
    Not(Box<Node>),
    /// These, but a document without them still matches.
    Optional(Box<Node>),
    /// Words starting with this.
    Prefix(Box<[u8]>),
    /// Words ending with this.
    Suffix(Box<[u8]>),
    /// Words with this inside them.
    Infix(Box<[u8]>),
    /// Words within an edit distance of this.
    Fuzzy(Box<[u8]>, u8),
    /// Words matching this pattern, where `*` and `?` are the pattern.
    Pattern(Box<[u8]>),
    /// A number in a range.
    Numeric(Range),
    /// A tag field and the values asked of it.
    Tag(Box<[u8]>, Vec<Node>),
    /// A point and a radius.
    Geo(Circle),
    /// The nearest vectors, or the ones inside a radius.
    Vector(Box<Vector>),
}

/// One word of a query, and where it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Word {
    /// The word, folded to lower case.
    pub word: Box<[u8]>,
    /// Whether it came out of the stemmer rather than out of the query, which
    /// the printout marks and a scorer weighs differently.
    pub expanded: bool,
    /// Whether it is the stem itself rather than another form of it, which the
    /// printout marks again on top of that.
    pub stem: bool,
}

/// A numeric range, with the ends and whether each one is included.
#[derive(Debug, Clone, PartialEq)]
pub struct Range {
    /// The field being compared, as a query names it.
    pub field: Box<[u8]>,
    /// The bottom, which may be negative infinity.
    pub min: f64,
    /// The top, which may be infinity.
    pub max: f64,
    /// Whether the bottom is excluded.
    pub min_open: bool,
    /// Whether the top is excluded.
    pub max_open: bool,
}

/// A circle on the earth, which is what a geo filter is.
#[derive(Debug, Clone, PartialEq)]
pub struct Circle {
    /// The field holding the points.
    pub field: Box<[u8]>,
    /// Degrees east.
    pub lon: f64,
    /// Degrees north.
    pub lat: f64,
    /// How far out, in the unit below.
    pub radius: f64,
    /// `m`, `km`, `mi` or `ft`, in the spelling the client used.
    pub unit: Box<[u8]>,
}

/// A name and the bytes it stands for, which is the shape `PARAMS` arrives in
/// and the shape the runtime options after a vector clause arrive in.
pub type Pair = (Box<[u8]>, Box<[u8]>);

/// A vector query, which asks for the nearest few or for everything inside a
/// radius.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    /// The field holding the vectors.
    pub field: Box<[u8]>,
    /// The name of the parameter holding the vector to compare against, without
    /// its dollar sign.
    pub param: Box<[u8]>,
    /// How many to return, for a nearest neighbour query.
    pub k: Option<u64>,
    /// How far out, for a range query.
    pub radius: Option<f64>,
    /// What the distance is called in the reply, when the client renamed it.
    pub alias: Option<Box<[u8]>>,
    /// The runtime options, in the order the client gave them.
    pub options: Vec<Pair>,
    /// What the nearest neighbours are being taken from, when the client
    /// narrowed it down first.
    pub over: Option<Box<Node>>,
}

/// Whether a list of children is a word and the forms of it, rather than
/// something the client wrote as a union.
///
/// The difference matters three times: such a union never swallows a sibling, a
/// weight goes on the word inside it rather than on the union, and a union
/// narrowed to no field at all folds back to the word because a real server does
/// not expand a node that can never match.
pub(crate) fn expansion(list: &[Node]) -> bool {
    let Some((first, rest)) = list.split_first() else {
        return false;
    };
    matches!(
        first.what,
        What::Term(Word {
            expanded: false,
            ..
        })
    ) && !rest.is_empty()
        && rest
            .iter()
            .all(|n| matches!(n.what, What::Term(Word { expanded: true, .. })))
}

impl Node {
    /// A node of this kind asking every field at weight one.
    #[must_use]
    pub const fn new(what: What) -> Node {
        Node {
            what,
            mask: EVERY,
            weight: None,
            slop: None,
            inorder: false,
        }
    }

    /// The node a dropped stopword and an emptied group both come to.
    #[must_use]
    pub const fn empty() -> Node {
        Node::new(What::Empty)
    }

    /// One word exactly as the client typed it.
    #[must_use]
    pub fn term(word: &[u8]) -> Node {
        Node::new(What::Term(Word {
            word: word.into(),
            expanded: false,
            stem: false,
        }))
    }

    /// Whether this is the node that matches nothing and prints as `<empty>`.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        matches!(self.what, What::Empty)
    }

    /// Narrows this node and everything under it to a set of fields.
    ///
    /// It narrows rather than sets, because a modifier inside a modifier means
    /// both. Asking for a field that the enclosing modifier did not allow leaves
    /// a node that asks for no field at all, which is a thing a client can write
    /// and a real server keeps rather than rejects.
    pub fn narrow(&mut self, mask: Mask) {
        self.mask &= mask;
        match &mut self.what {
            What::Union(list) | What::Intersect(list) | What::Exact(list) | What::Tag(_, list) => {
                for child in list {
                    child.narrow(mask);
                }
            }
            What::Not(child) | What::Optional(child) => child.narrow(mask),
            What::Vector(v) => {
                if let Some(over) = &mut v.over {
                    over.narrow(mask);
                }
            }
            _ => {}
        }
    }

    /// Whether this node would swallow another one of the same kind rather than
    /// nest it.
    ///
    /// `a b c` is one intersection of three and `(a) b c` is an intersection of
    /// two, the second of which is an intersection of two. The difference is
    /// whether the left hand side is already a plain node of the kind being
    /// built: a group that has been narrowed to a field is not plain any more
    /// and gets a parent of its own.
    ///
    /// An attribute clause does not stop it. `(a b) => {$weight: 2;} c` is one
    /// intersection of three carrying the weight, which reads oddly and is what
    /// a real server does, so a client that meant to weigh only the pair has to
    /// write another pair of brackets around it.
    fn absorbs(&self, all: bool) -> bool {
        if self.mask != EVERY {
            return false;
        }
        match &self.what {
            What::Intersect(_) => all,
            // The union a word expanded into is not a union the client wrote,
            // so `hello|world` is two of them side by side rather than `world`
            // landing in among the forms of `hello`.
            What::Union(list) => !all && !expansion(list),
            _ => false,
        }
    }

    /// Folds one more thing into a chain of `|` or of juxtaposition.
    ///
    /// The fold is to the left, which is what makes `(a|b)|c` a union of three
    /// and `a|(b|c)` a union of two under the first dialect. That asymmetry is
    /// not a nicety: the printed tree is what a client reads to work out what
    /// its query cost, so a shape that differs from a real server's is a wrong
    /// answer even when the set of documents would have come out the same.
    ///
    /// The second dialect flattens harder. When the left hand side will not
    /// take another child it tries the right hand side, and what comes out is
    /// `a|(b|c)` as one union of three with `a` last in it, because the side
    /// that could hold the children is the one that kept them.
    fn fold(acc: Node, next: Node, either: bool, all: bool) -> Node {
        if next.is_empty() {
            return acc;
        }
        if acc.is_empty() {
            return next;
        }
        let mut acc = acc;
        if acc.absorbs(all) {
            match &mut acc.what {
                What::Intersect(list) | What::Union(list) => list.push(next),
                _ => unreachable!("absorbs said yes to a node that holds nothing"),
            }
            return acc;
        }
        let mut next = next;
        if either && next.absorbs(all) {
            match &mut next.what {
                What::Intersect(list) | What::Union(list) => list.push(acc),
                _ => unreachable!("absorbs said yes to a node that holds nothing"),
            }
            return next;
        }
        let list = vec![acc, next];
        Node::new(if all {
            What::Intersect(list)
        } else {
            What::Union(list)
        })
    }

    /// Whether folding these two puts the right hand side in front of the left.
    ///
    /// The answer matters outside the shape of the tree, because a parameter
    /// nobody passed is named in the order the tree is walked rather than the
    /// order it was written, so `$B $C $D` complains about `C`.
    fn swaps(acc: &Node, next: &Node, either: bool, all: bool) -> bool {
        either && !next.is_empty() && !acc.is_empty() && !acc.absorbs(all) && next.absorbs(all)
    }
}
