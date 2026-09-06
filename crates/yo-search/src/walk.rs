//! Running a parsed query over what an index holds.
//!
//! [`crate::query`] turns the bytes a client wrote into a tree and this walks
//! it, so a tree goes in one end and every document that answers comes out the
//! other, in number order, each with the shape of how it answered.
//!
//! ```
//! use yo_search::query::{Ask, parse};
//! use yo_search::{Definition, English, Field, Index, Kind, Text, walk};
//!
//! let mut index = Index::new(b"ix", Definition::default(), vec![
//!     Field::new(b"t", Kind::Text(Text::default())),
//! ]);
//! let mut english = English::new();
//! index.write(&mut english, b"d:1", &[(b"t", b"hello world")])?;
//! index.write(&mut english, b"d:2", &[(b"t", b"goodbye world")])?;
//!
//! let node = parse(b"hello", &index, &Ask::default()).expect("a query that parses");
//! let hits = walk::run(&index.held, &node);
//! assert_eq!(hits.len(), 1);
//! assert_eq!(index.held.docs.key(hits[0].id), Some(b"d:1".as_slice()));
//! # Ok::<(), yo_search::held::Failed>(())
//! ```
//!
//! # One method, and everything is built out of it
//!
//! Every node turns into something that answers one question: the first
//! document at or after this number that matches you, and what it matched with.
//! Stepping on is asking for the one after the last answer, an intersection is
//! asking each side for the same number until they agree on one, and a negation
//! is walking the document table asking whether the thing underneath answers
//! each number. So there is one method rather than a next and a seek that have
//! to be kept telling the same story.
//!
//! Each of them remembers the answer it last gave, and asking again for a number
//! it has already passed gives that answer back rather than moving. That is what
//! lets an intersection ask its children in any order it likes and lets a union
//! ask twice, once to find the smallest number anybody has and once to collect
//! everybody who has it.
//!
//! # A union adds up and an expansion does not
//!
//! `hello|world` scores a document holding both on both, and this is measured:
//! the score of the union is the sum of what the two terms are worth on their
//! own, to the last digit. `hel*` does not. A document holding two terms a
//! prefix stands for is scored on the first of them in byte order and the other
//! one is not counted at all, which is also measured, on an index built twice
//! over with the rarer of the two words on either side so that the answer could
//! not be a coincidence of which was worth more.
//!
//! So there are two unions here and they differ only in that. The one a client
//! writes with a `|` adds its branches up, and the one an expansion turns into
//! takes the first branch that answered.
//!
//! # A bare `*` is a term whose idf is one
//!
//! A wildcard has no term in it, so there is nothing to weigh by how rare it is,
//! and what a real server scores it with is the rest of BM25 with the rarity
//! left out: one occurrence, corrected for how long the document is. That is
//! measured to the last digit as well, which is why [`crate::score::Found`] has
//! a shape of its own for it rather than borrowing the shape of a term.
//!
//! # A phrase is an intersection that asks where
//!
//! `"hello world"` is every document holding both words with the second one
//! next to the first, which means asking each word not only whether it is in a
//! document but where, and the places are what the posting lists carry beside
//! the frequency for exactly this.
//!
//! There is one rule underneath the phrase, the `SLOP` a client can ask for and
//! the `INORDER` beside it, and it is measured rather than reasoned about. Give
//! each word one of the places it was found at. The words are close enough when
//! the last of those places is no further from the first than the slop plus one
//! less than the number of words, so a phrase, which is a slop of nothing, wants
//! them in a run with no room to spare. In order means the places have to climb,
//! though not strictly, which is the part nobody would guess: `"aa aa"` answers
//! a document holding one `aa` because both words are allowed to stand on it. In
//! any order they may not all stand on the same place, which is why the same
//! query with a slop and no order answers nothing at all unless the word really
//! is in there twice.
//!
//! A word that stands for several terms brings the places of all of them, so a
//! stem counts where the word it came from would. One tag value, a number, a
//! negation and an optional stay out of the rule, so they neither fail it nor
//! count towards how many words there are, which is why a range or a single tag
//! beside two words under a slop changes nothing about which documents answer.
//! Two tag values written as a union do take part, and a union has no places of
//! its own to give, so `(@g:{aa|bb} alpha)=>{$slop:0}` answers nothing where
//! `(@g:{aa} alpha)=>{$slop:0}` answers. All of that is measured on 8.10.1.
//!
//! # The rarest branch of an intersection goes first
//!
//! An intersection asks its children in order until one of them is further
//! along, and the fewer documents the first child has the sooner that happens,
//! so the children are sorted by how many documents they are likely to answer
//! before any of them is asked anything. A word's guess is how many documents
//! hold it, a union's is its branches added up, an intersection's is the
//! smallest of its children, and a negation, an optional and a range are guessed
//! at the whole index because that is roughly what they answer.
//!
//! The sort is stable, so two branches that guess the same stay in the order
//! they were written. That is not only a speed matter: it is the order a real
//! server explains a score in, and `fox dog` and `dog fox` both explain
//! themselves as dog and then fox. A phrase and an intersection asked for in
//! order are left alone, because there the order is the question.
//!
//! # How far apart the words landed
//!
//! Three of the nine scorers divide what a document is worth by how spread out
//! the query's words are in it, and the number they divide by is measured:
//! take the branches of the top of the query in the order above, drop the ones
//! with no places to give, take the smallest gap between each pair left next to
//! each other, and the answer is the whole number part of the square root of
//! the squared gaps added up, or one if there is nothing to add up.
//!
//! So `w1 fox v50` over a document with `w1` first, `fox` in the middle and
//! `v50` last comes to a hundred and eleven, which is the square root of a
//! hundred squared plus fifty squared with the tail cut off, and that one query
//! is what pins down both the sort and the square root at once. A tag, a range,
//! a negation and a wildcard have no places, so they are lifted out of the chain
//! rather than breaking it, and an optional hands over the places of whatever it
//! found even though it stays out of a phrase check.
//!
//! Working it out costs a pass over the places of every top branch, so it is
//! only done when the scorer that was asked for divides by it, which the default
//! one does not.
//!
//! # A vector clause
//!
//! Measured rather than walked. Every other step here answers where the next
//! document at or after a number is, and a nearest neighbour question has no
//! such answer until every candidate has been measured, so the whole clause is
//! worked out at once and handed back in document number order like everything
//! else. The distance itself is not carried out of here: a reply that wants to
//! show it asks the field for it again, because a query can ask to see the
//! distance from a clause that never ordered anything.
//!
//! A clause that narrowed the index down first is walked first and only what it
//! answered is measured. That is what the query means as well as what is quick:
//! `alpha=>[KNN 5 @v $B]` is the five nearest of the documents that say alpha,
//! not the alpha ones among the five nearest overall.
//!
//! # What is not walked yet
//!
//! A geo shape, which needs a field the document reader does not read yet, so
//! there is nothing in the index to walk even when there is a node for it. It
//! answers nothing rather than answering wrongly.

use std::collections::BTreeMap;

use crate::docs::Docs;
use crate::expand;
use crate::held::Held;
use crate::nums::Ends;
use crate::posts::{Id, Posts, Reader, stemmed};
use crate::query::explain::fixed;
use crate::query::{Circle, Node, Range, What, Word};
use crate::score::{Found, Term};
use crate::tags::Tags;

/// One document that answered, and the shape of how it answered.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit<'a> {
    /// Which document.
    pub id: Id,
    /// What matched in it, which is what a scorer walks.
    pub found: Found<'a>,
    /// How far apart the query's words landed, which three scorers divide by.
    ///
    /// One unless [`spaced`] was the way in, because working it out costs a
    /// pass over the places and most queries never look at it.
    pub slop: u32,
}

impl<'a> Hit<'a> {
    /// A document that answered, with nothing worked out about where.
    fn new(id: Id, found: Found<'a>) -> Hit<'a> {
        Hit { id, found, slop: 1 }
    }
}

/// Every document that answers a query, in number order.
#[must_use]
pub fn run<'a>(held: &'a Held, node: &'a Node) -> Vec<Hit<'a>> {
    gather(held, node, false)
}

/// The same, with how far apart the query's words landed worked out as well.
#[must_use]
pub fn spaced<'a>(held: &'a Held, node: &'a Node) -> Vec<Hit<'a>> {
    gather(held, node, true)
}

/// What one step of a walk turned out to be, which is what a profile reports.
///
/// Built after the walk rather than during it, so nothing here costs anything
/// until somebody asks. The counts are the exception: they are kept as the walk
/// runs because there is no way to work out afterwards how many times a branch
/// was asked.
#[derive(Debug, Clone, PartialEq)]
pub struct Ran {
    /// The word a profile calls this kind of step.
    pub kind: &'static str,
    /// What a union is a union of, which is the word `UNION` when a client
    /// wrote the bar and the name of the expansion when one made it.
    pub about: Option<Box<[u8]>>,
    /// The term, the tag value or the range this answered off, which the leaves
    /// have and the branches do not.
    pub term: Option<Box<[u8]>>,
    /// How many different documents this answered.
    pub reads: u64,
    /// The guess an intersection sorts its branches on, which a profile reports
    /// for a leaf and leaves off a branch.
    pub size: Option<u32>,
    /// What is under it.
    pub under: Vec<Ran>,
    /// Whether a profile writes the one thing under this as a single child
    /// rather than as a list, which a negation and an optional do.
    pub alone: bool,
    /// Whether `LIMITED` folds the children away into a count, which it does
    /// for a union an expansion made and not for one a client wrote.
    pub folds: bool,
}

impl Ran {
    /// A step with nothing under it and nothing to say about a term.
    fn leaf(kind: &'static str, reads: u64) -> Ran {
        Ran {
            kind,
            about: None,
            term: None,
            reads,
            size: None,
            under: Vec::new(),
            alone: false,
            folds: false,
        }
    }

    /// The same, naming what it answered off and how much of it there is.
    fn named(kind: &'static str, reads: u64, term: &[u8], size: u32) -> Ran {
        Ran {
            term: Some(term.into()),
            size: Some(size),
            ..Ran::leaf(kind, reads)
        }
    }
}

/// Every document that answers, and what the walk that found them was.
#[must_use]
pub fn profiled<'a>(held: &'a Held, node: &'a Node, measure: bool) -> (Vec<Hit<'a>>, Ran) {
    let mut step = build(held, node);
    let out = drain(&mut step, measure);
    let ran = step.ran();
    (out, ran)
}

/// Every document that answers, with the places measured or not.
fn gather<'a>(held: &'a Held, node: &'a Node, measure: bool) -> Vec<Hit<'a>> {
    let mut step = build(held, node);
    drain(&mut step, measure)
}

/// Asks a built step for every document it has, from the first number up.
fn drain<'a>(step: &mut Box<dyn Step<'a> + 'a>, measure: bool) -> Vec<Hit<'a>> {
    let mut out = Vec::new();
    let mut want = 1;
    while let Some(mut hit) = step.seek(want) {
        if measure {
            hit.slop = step.slop(hit.id);
        }
        let next = hit.id.checked_add(1);
        out.push(hit);
        match next {
            Some(next) => want = next,
            None => break,
        }
    }
    out
}

/// Something that answers where the next document it matches is.
trait Step<'a> {
    /// The first document at or after this number that matches, if there is
    /// one.
    ///
    /// Asking for a number already passed gives the last answer back, so the
    /// same question may be put twice and the second time is free.
    fn seek(&mut self, id: Id) -> Option<Hit<'a>>;

    /// Roughly how many documents this will answer, for sorting on.
    ///
    /// A guess and not a count, and it only has to be right enough to put the
    /// rarest branch of an intersection first. It is also the order a real
    /// server explains a score in, so it is measured rather than tuned.
    fn size(&self) -> u32;

    /// Whether this can never answer anything, which lets a negation over it be
    /// dropped where it stands rather than kept as a branch that always
    /// answers and is always worth nothing.
    fn empty(&self) -> bool {
        false
    }

    /// Whether this answers every document and adds nothing to a score, which
    /// is what a negation over something no document holds comes to.
    fn always(&self) -> bool {
        false
    }

    /// The places this matched a document at, added to what is there already,
    /// and whether it takes part in a position check at all.
    ///
    /// Only asked right after this answered the number, so what it adds is what
    /// the last answer was found at. A word takes part and so does anything
    /// built out of other things, which is where its places come from. A tag
    /// value, a number, a negation, an optional and a wildcard stay out of it,
    /// so they neither fail a position check nor count towards how many words
    /// there are.
    ///
    /// Both halves of that are measured. A range beside two words under a slop
    /// changes nothing about which documents answer, and neither does one tag
    /// value, where two tag values written as a union answer nothing at all
    /// under the same slop, because a union takes part and has no places to give.
    fn places(&mut self, id: Id, into: &mut Vec<u32>) -> bool {
        let _ = (id, into);
        false
    }

    /// How far apart this node's branches landed in a document.
    ///
    /// One for anything with nothing to measure, which is every leaf and every
    /// branch whose children have no places between them.
    fn slop(&mut self, id: Id) -> u32 {
        let _ = id;
        1
    }

    /// What this turned out to be, for a profile to report.
    fn ran(&self) -> Ran;
}

/// What one node of the tree turns into.
///
/// The node's own weight goes on whatever this makes and is not pushed down
/// into the children, because a real server prints the weight on the branch a
/// client hung it on and prints one on every leaf underneath.
fn build<'a>(held: &'a Held, node: &'a Node) -> Box<dyn Step<'a> + 'a> {
    let weight = node.weight.unwrap_or(1.0);
    let mask = mask(node.mask);
    match &node.what {
        What::Empty => Box::new(Never),
        What::Wildcard => Box::new(Every::new(&held.docs)),
        What::Term(word) => Box::new(term(held, word, mask, weight)),
        What::Union(list) => {
            // A branch no document holds is dropped before the union is built,
            // and a union left with one branch is that branch. It matters
            // because the parser makes a union out of every word to hold the
            // stem behind it, so `fox` arrives here as `fox` and `+fox`, the
            // second of which is never written because the stem and the word
            // are the same, and a real server explains the whole thing as the
            // bare word. A union that keeps two branches keeps its wrapper
            // even when only one of them answered this document, which is
            // measured on `running`: `+run` alone answers `h:1` and the
            // wrapper is still printed, because `running` itself is a word
            // some other document has.
            let mut under: Vec<Box<dyn Step<'a> + 'a>> = under(held, list)
                .into_iter()
                .filter(|child| !child.empty())
                .collect();
            match under.len() {
                0 => Box::new(Never),
                1 if weight == 1.0 => under.pop().unwrap_or_else(|| Box::new(Never)),
                _ => Box::new(Any::new(under, true, weight)),
            }
        }
        What::Intersect(list) => {
            let mut under = narrowed(held, list);
            // A slop of less than nothing is no limit at all, so an intersection
            // with one and no order asked for is an ordinary intersection and
            // does not go looking for places it will not read.
            let slop = node.slop.unwrap_or(-1);
            if slop >= 0 || node.inorder {
                return Box::new(Near::new(under, slop, node.inorder, weight));
            }
            // One branch is that branch and not an intersection of one, which
            // is measured twice over: `* FILTER n 1 5` drops the wildcard and
            // explains itself as the filter alone, and `fox -zebra` drops the
            // negation and explains itself as the fox alone.
            match under.len() {
                1 if weight == 1.0 => under.pop().unwrap_or_else(|| Box::new(Never)),
                _ => Box::new(All::new(under, weight)),
            }
        }
        // Measured: the slop and the order a client hangs on a phrase are
        // printed back and change nothing, so a phrase is a run in order
        // whatever was asked of it.
        What::Exact(list) => Box::new(Near::new(under(held, list), 0, true, weight)),
        What::Not(child) => Box::new(Unless {
            docs: &held.docs,
            under: build(held, child),
            reads: 0,
            gave: None,
        }),
        What::Optional(child) => Box::new(Maybe {
            docs: &held.docs,
            under: build(held, child),
            reads: 0,
            gave: None,
        }),
        What::Prefix(prefix) => spread(
            held,
            expand::under(held.dictionary(), prefix),
            mask,
            weight,
            &told("PREFIX", prefix),
        ),
        What::Suffix(suffix) => spread(
            held,
            expand::ending(held.dictionary(), suffix),
            mask,
            weight,
            &told("SUFFIX", suffix),
        ),
        What::Infix(part) => spread(
            held,
            expand::inside(held.dictionary(), part),
            mask,
            weight,
            &told("INFIX", part),
        ),
        What::Pattern(pattern) => spread(
            held,
            expand::like(held.dictionary(), pattern),
            mask,
            weight,
            &told("WILDCARD", pattern),
        ),
        What::Fuzzy(word, distance) => spread(
            held,
            expand::near(held.dictionary(), word, *distance),
            mask,
            weight,
            &told("FUZZY", word),
        ),
        What::Numeric(range) => Box::new(numbers(held, range)),
        What::Geo(circle) => Box::new(places(held, circle)),
        What::Tag(field, list) => tagged(held, field, list, weight),
        What::Vector(vector) => Box::new(nearby(held, vector)),
    }
}

/// The documents whose vector is nearest a query, or is within a range of it.
///
/// A clause that narrowed the index down first is walked first and what it
/// answered is what gets measured, which is both faster and what the answer
/// means: `alpha=>[KNN 5 @v $B]` is the five nearest of the documents that say
/// alpha rather than the alpha ones among the five nearest overall.
fn nearby<'a>(held: &'a Held, vector: &'a crate::query::Vector) -> Nearby<'a> {
    let Some(vecs) = held.vecs(&vector.field) else {
        return Nearby::new(Vec::new(), held.docs.len() as u32);
    };
    let Some(asked) = &vector.asked else {
        return Nearby::new(Vec::new(), held.docs.len() as u32);
    };
    let over = vector.over.as_ref().map(|node| {
        let mut step = build(held, node);
        drain(&mut step, false)
    });
    let narrow = over
        .as_ref()
        .map(|hits| hits.iter().map(|hit| hit.id).collect::<Vec<Id>>());
    let found = match (vector.k, vector.radius) {
        (Some(k), _) => vecs.nearest(asked, k as usize, narrow.as_deref()),
        (None, Some(radius)) => vecs.within(asked, radius as f32, narrow.as_deref()),
        (None, None) => Vec::new(),
    };
    // Answered in document number order like every other step, which is what a
    // real server answers a vector clause in: five documents written furthest
    // first come back in the order they were written.
    let mut near: Vec<Id> = found
        .into_iter()
        .filter(|near| held.docs.get(near.id).is_some())
        .map(|near| near.id)
        .collect();
    near.sort_unstable();
    let guess = near.len() as u32;
    // What the clause in front matched is carried through rather than dropped,
    // because a vector clause narrows an answer and does not rescore it: the
    // three nearest of `(alpha|beta)` come back in the order `alpha|beta` put
    // them in, with the scores that query gave them, which is measured. With
    // no clause in front there is nothing to carry and a vector clause scores
    // the way a range does, which is as a filter.
    let ids = match over {
        Some(hits) => {
            let mut carried: BTreeMap<Id, Found<'a>> =
                hits.into_iter().map(|hit| (hit.id, hit.found)).collect();
            near.into_iter()
                .filter_map(|id| carried.remove(&id).map(|found| (id, found)))
                .collect()
        }
        None => near.into_iter().map(|id| (id, Found::filter())).collect(),
    };
    Nearby::new(ids, guess)
}

/// The documents a vector clause found, with what the clause in front of it
/// matched in them.
struct Nearby<'a> {
    ids: Vec<(Id, Found<'a>)>,
    at: usize,
    reads: u64,
    gave: Option<Id>,
    guess: u32,
}

impl<'a> Nearby<'a> {
    fn new(ids: Vec<(Id, Found<'a>)>, guess: u32) -> Nearby<'a> {
        Nearby {
            ids,
            at: 0,
            reads: 0,
            gave: None,
            guess,
        }
    }
}

impl<'a> Step<'a> for Nearby<'a> {
    fn seek(&mut self, id: Id) -> Option<Hit<'a>> {
        while self.at < self.ids.len() && self.ids[self.at].0 < id {
            self.at += 1;
        }
        let (found, what) = self.ids.get(self.at)?;
        let found = *found;
        let what = what.clone();
        if self.gave != Some(found) {
            self.gave = Some(found);
            self.reads += 1;
        }
        Some(Hit::new(found, what))
    }

    fn size(&self) -> u32 {
        self.guess
    }

    fn empty(&self) -> bool {
        self.ids.is_empty()
    }

    fn ran(&self) -> Ran {
        Ran::leaf("VECTOR", self.reads)
    }
}

/// The fields a node asks, as the posting lists carry them.
///
/// A node asks for its fields in sixty four bits and a posting list records the
/// ones it was found in in thirty two, which is the same set until an index has
/// more than thirty two text fields in it. Everything above the thirty second
/// shares the last bit, both here and where a document is read, so the two
/// agree with each other and both are wrong in the same way about an index
/// nobody has.
fn mask(mask: crate::query::Mask) -> u32 {
    let low = mask as u32;
    // Any field past the thirty second is asked for by the top bit, which is
    // the bit those fields were indexed under.
    if mask >> 32 == 0 { low } else { low | 1 << 31 }
}

/// Every child of a node, built.
fn under<'a>(held: &'a Held, list: &'a [Node]) -> Vec<Box<dyn Step<'a> + 'a>> {
    list.iter().map(|node| build(held, node)).collect()
}

/// The same, with a branch that narrows nothing left out.
///
/// A negation over a word no document holds answers every document and is worth
/// nothing, so an intersection is the same query without it. Measured through
/// the explanation, which a real server prints without the branch rather than
/// with a branch worth zero. Dropping every branch would turn a query that
/// answers everything into one that answers nothing, so that one case keeps a
/// wildcard in their place.
fn narrowed<'a>(held: &'a Held, list: &'a [Node]) -> Vec<Box<dyn Step<'a> + 'a>> {
    let mut out: Vec<Box<dyn Step<'a> + 'a>> = Vec::with_capacity(list.len());
    for node in list {
        let step = build(held, node);
        if step.always() {
            continue;
        }
        out.push(step);
    }
    if out.is_empty() && !list.is_empty() {
        out.push(Box::new(Every::new(&held.docs)));
    }
    out
}

/// One term's posting list, or nothing when no document holds it.
fn term<'a>(held: &'a Held, word: &Word, mask: u32, weight: f64) -> One<'a> {
    let stem;
    let name: &[u8] = if word.stem {
        stem = stemmed(&word.word);
        &stem
    } else {
        &word.word
    };
    One::new(held.entry(name), &held.docs, mask, weight)
}

/// How a profile names the expansion a union came out of, which is the kind of
/// expansion and then what was written.
fn told(kind: &str, written: &[u8]) -> Vec<u8> {
    let mut out = kind.as_bytes().to_vec();
    out.extend_from_slice(b" - ");
    out.extend_from_slice(written);
    out
}

/// A union over every term one expansion stands for, in dictionary order.
fn spread<'a>(
    held: &'a Held,
    found: Vec<(&'a [u8], &'a Posts)>,
    mask: u32,
    weight: f64,
    about: &[u8],
) -> Box<dyn Step<'a> + 'a> {
    // One term is that term and not a union of one, which is measured: `qu*`
    // stands for one word and explains itself as a bare leaf, where `w1*`
    // stands for eleven and explains itself as a union even on the document
    // that answered only one of them.
    if let [entry] = found[..] {
        return Box::new(One::new(Some(entry), &held.docs, mask, weight));
    }
    let under = found
        .into_iter()
        .map(|entry| {
            Box::new(One::new(Some(entry), &held.docs, mask, 1.0)) as Box<dyn Step<'a> + 'a>
        })
        .collect();
    Box::new(Any::new(under, false, weight).tells(about.to_vec()))
}

/// The documents whose number in a field is inside a range.
fn numbers<'a>(held: &'a Held, range: &Range) -> List<'a> {
    let ends = Ends {
        min: range.min,
        max: range.max,
        min_open: range.min_open,
        max_open: range.max_open,
    };
    let ids = held
        .numbers(&range.field)
        .map(|nums| nums.range(ends))
        .unwrap_or_default();
    // A range is a filter and nothing else, and what that is worth depends on
    // the scorer rather than on the range, so the shape says which it is and
    // the scorer decides. [`Found::Blank`] has the measurement.
    let ends = fixed(range.min) + " - " + &fixed(range.max);
    List::new(live(&held.docs, ids), Found::filter())
        .about(held.docs.len() as u32)
        .tells("NUMERIC", ends.into_bytes())
}

/// The documents whose point is inside the circle.
fn places<'a>(held: &'a Held, circle: &Circle) -> List<'a> {
    let ids = held
        .places(&circle.field)
        .and_then(|geos| geos.circle(circle.lon, circle.lat, circle.radius, &circle.unit))
        .unwrap_or_default();
    // Nothing at all for a field with no points in it, which is what a field of
    // another kind is, and the parser has already refused a unit that is not
    // one of the four rather than leaving it to answer nothing here.
    let round = format!(
        "{},{} - {} {}",
        fixed(circle.lon),
        fixed(circle.lat),
        fixed(circle.radius),
        String::from_utf8_lossy(&circle.unit),
    );
    List::new(live(&held.docs, ids), Found::filter()).tells("GEO", round.into_bytes())
}

/// The documents a tag field's values were asked for.
fn tagged<'a>(
    held: &'a Held,
    field: &[u8],
    list: &'a [Node],
    weight: f64,
) -> Box<dyn Step<'a> + 'a> {
    let Some(tags) = held.values(field) else {
        return Box::new(Never);
    };
    // One value is that value and not a union of one, which is measured twice:
    // through a position check, where `(@g:{aa} alpha)=>{$slop:0}` answers and
    // `(@g:{aa|bb} alpha)=>{$slop:0}` answers nothing because one value stays
    // out of the check and a union takes part with no places to give, and
    // through the explanation, where `@g:{re*}` prints a bare leaf and
    // `@g:{red|blue}` prints a union.
    let alone = list.len() == 1;
    let inner = if alone { weight } else { 1.0 };
    let mut under: Vec<Box<dyn Step<'a> + 'a>> = list
        .iter()
        .map(|node| value(held, tags, node, inner))
        .collect();
    if alone {
        return under.pop().unwrap_or_else(|| Box::new(Never));
    }
    Box::new(Any::new(under, true, weight).tells(b"TAG".to_vec()))
}

/// One value asked of a tag field.
fn value<'a>(
    held: &'a Held,
    tags: &'a Tags,
    node: &'a Node,
    weight: f64,
) -> Box<dyn Step<'a> + 'a> {
    match &node.what {
        // A value written as several words is one value with a space in it
        // rather than several values, which is measured: an index holding the
        // tag `aa bb` answers `{aa bb}` and one holding the two tags `aa` and
        // `bb` does not.
        What::Intersect(list) => {
            let mut joined = Vec::new();
            for word in list {
                if let What::Term(word) = &word.what {
                    if !joined.is_empty() {
                        joined.push(b' ');
                    }
                    joined.extend_from_slice(&word.word);
                }
            }
            Box::new(held_tag(held, tags, &joined, weight))
        }
        What::Term(word) => Box::new(held_tag(held, tags, &word.word, weight)),
        What::Prefix(prefix) => sweep(held, tags, weight, prefix.len(), |value| {
            value.starts_with(prefix)
        }),
        What::Suffix(suffix) => sweep(held, tags, weight, suffix.len(), |value| {
            value.ends_with(suffix)
        }),
        What::Infix(part) => sweep(held, tags, weight, part.len(), |value| {
            value.windows(part.len()).any(|window| *window == **part)
        }),
        What::Pattern(pattern) => {
            let pattern = crate::token::fold(pattern);
            sweep(held, tags, weight, expand::SHORTEST, move |value| {
                expand::glob(&pattern, value)
            })
        }
        _ => Box::new(Never),
    }
}

/// One value of a tag field, with the documents holding it.
fn held_tag<'a>(held: &'a Held, tags: &'a Tags, value: &[u8], weight: f64) -> List<'a> {
    let (name, ids) = tags.entry(value).unwrap_or((&[], &[]));
    // Measured: a tag that answered is scored the way a term that answered is,
    // with one occurrence and the rarity of the value, and named the way the
    // index spells it rather than the way the query wrote it.
    let found = Found::Term(Term::new(1, weight, ids.len() as u32).about(name));
    List::new(live(&held.docs, ids.to_vec()), found).tells("TAG", name.to_vec())
}

/// Every value of a tag field that fits, as a union in byte order.
fn sweep<'a>(
    held: &'a Held,
    tags: &'a Tags,
    weight: f64,
    written: usize,
    fits: impl Fn(&[u8]) -> bool,
) -> Box<dyn Step<'a> + 'a> {
    if written < expand::SHORTEST {
        return Box::new(Never);
    }
    let values: Vec<(&'a [u8], &'a [Id])> = tags
        .all()
        .filter(|(value, _)| fits(value))
        .take(expand::MOST)
        .collect();
    let alone = values.len() == 1;
    let mut under: Vec<Box<dyn Step<'a> + 'a>> = values
        .into_iter()
        .map(|(name, ids)| {
            let each = if alone { weight } else { 1.0 };
            let found = Found::Term(Term::new(1, each, ids.len() as u32).about(name));
            Box::new(List::new(live(&held.docs, ids.to_vec()), found).tells("TAG", name.to_vec()))
                as Box<dyn Step<'a> + 'a>
        })
        .collect();
    if alone {
        return under.pop().unwrap_or_else(|| Box::new(Never));
    }
    Box::new(Any::new(under, false, weight).tells(b"TAG".to_vec()))
}

/// The numbers among these that still mean something.
fn live(docs: &Docs, ids: Vec<Id>) -> Vec<Id> {
    let mut ids = ids;
    ids.retain(|id| docs.get(*id).is_some());
    ids
}

/// Nothing at all, which is what a query of pure stopwords comes to and what a
/// node nobody walks yet answers.
struct Never;

impl Step<'_> for Never {
    fn seek(&mut self, _: Id) -> Option<Hit<'static>> {
        None
    }

    fn size(&self) -> u32 {
        0
    }

    fn empty(&self) -> bool {
        true
    }

    fn ran(&self) -> Ran {
        Ran::leaf("EMPTY", 0)
    }
}

/// One term's posting list.
struct One<'a> {
    /// The list, or nothing when no document holds the term.
    reader: Option<Reader<'a>>,
    /// What the dictionary calls the term, for saying so in an explanation.
    name: &'a [u8],
    /// The document table, for skipping numbers that stopped meaning anything.
    docs: &'a Docs,
    /// The fields the query asked for.
    mask: u32,
    /// What the query said the term is worth.
    weight: f64,
    /// How many documents hold the term, which is the rarity a scorer wants.
    df: u32,
    /// The answer last given, for giving it again.
    at: Option<Hit<'a>>,
    /// How many different documents this has answered.
    reads: u64,
    /// Room for the places of one document, which a reader hands back into a
    /// buffer of its own rather than onto the end of somebody else's.
    room: Vec<u32>,
}

impl<'a> One<'a> {
    fn new(
        entry: Option<(&'a [u8], &'a Posts)>,
        docs: &'a Docs,
        mask: u32,
        weight: f64,
    ) -> One<'a> {
        One {
            reader: entry.map(|(_, posts)| posts.read()),
            name: entry.map_or(&[], |(name, _)| name),
            docs,
            mask,
            weight,
            df: entry.map_or(0, |(_, posts)| posts.len()),
            at: None,
            reads: 0,
            room: Vec::new(),
        }
    }
}

impl<'a> Step<'a> for One<'a> {
    fn seek(&mut self, id: Id) -> Option<Hit<'a>> {
        if let Some(hit) = &self.at
            && hit.id >= id
        {
            return Some(hit.clone());
        }
        let reader = self.reader.as_mut()?;
        let mut want = id;
        loop {
            let post = reader.seek(want)?;
            if post.fields & self.mask != 0 && self.docs.get(post.id).is_some() {
                let term = Term::new(post.freq, self.weight, self.df).about(self.name);
                let hit = Hit::new(post.id, Found::Term(term));
                self.at = Some(hit.clone());
                self.reads += 1;
                return Some(hit);
            }
            want = post.id.checked_add(1)?;
        }
    }

    fn size(&self) -> u32 {
        self.df
    }

    fn empty(&self) -> bool {
        self.reader.is_none()
    }

    fn places(&mut self, id: Id, into: &mut Vec<u32>) -> bool {
        if self.at.as_ref().is_some_and(|hit| hit.id == id)
            && let Some(reader) = self.reader.as_ref()
        {
            reader.places(&mut self.room);
            into.extend_from_slice(&self.room);
        }
        true
    }

    fn ran(&self) -> Ran {
        Ran::named("TEXT", self.reads, self.name, self.df)
    }
}

/// A list of numbers worked out in advance, which is what a range and a tag
/// value come to.
struct List<'a> {
    ids: Vec<Id>,
    at: usize,
    /// Which of the three kinds of list this is, for a profile to name it.
    kind: &'static str,
    /// What a profile reports this answered off, which is a value for a tag and
    /// the ends of the range for a number or a circle.
    term: Vec<u8>,
    /// How many different documents this has answered.
    reads: u64,
    /// The number last given, so that giving it again is not counted twice.
    gave: Option<Id>,
    found: Found<'a>,
    /// How many documents this is worth guessing at when an intersection sorts
    /// its branches, which is not always how many it actually holds.
    guess: u32,
}

impl<'a> List<'a> {
    fn new(ids: Vec<Id>, found: Found<'a>) -> List<'a> {
        let guess = ids.len() as u32;
        List {
            ids,
            at: 0,
            kind: "TAG",
            term: Vec::new(),
            reads: 0,
            gave: None,
            found,
            guess,
        }
    }

    /// The same list, saying what kind of thing it answered off and which one,
    /// which is what a profile prints and nothing else looks at.
    fn tells(mut self, kind: &'static str, term: Vec<u8>) -> List<'a> {
        self.kind = kind;
        self.term = term;
        self
    }

    /// The same list, guessed at as this many documents rather than as its own
    /// length.
    ///
    /// A range answers off a sorted run of values rather than off a posting
    /// list, and a real server does not count the run before it decides where
    /// the range goes in an intersection, it takes the whole index as the
    /// guess. Measured: `@n:[1 5] fox` prints the fox first even though four
    /// documents are in the range and seven hold the word, and `@n:[1 5]
    /// @g:{red}` prints the tag first even though the range was written first.
    fn about(mut self, guess: u32) -> List<'a> {
        self.guess = guess;
        self
    }
}

impl<'a> Step<'a> for List<'a> {
    fn seek(&mut self, id: Id) -> Option<Hit<'a>> {
        while self.at < self.ids.len() && self.ids[self.at] < id {
            self.at += 1;
        }
        let found = *self.ids.get(self.at)?;
        if self.gave != Some(found) {
            self.gave = Some(found);
            self.reads += 1;
        }
        Some(Hit::new(found, self.found.clone()))
    }

    fn size(&self) -> u32 {
        self.guess
    }

    fn empty(&self) -> bool {
        self.ids.is_empty()
    }

    fn ran(&self) -> Ran {
        Ran::named(self.kind, self.reads, &self.term, self.guess)
    }
}

/// Every document there is, which is what a bare `*` asks for.
struct Every<'a> {
    docs: &'a Docs,
    /// How many different documents this has answered.
    reads: u64,
    /// The number last given, so that giving it again is not counted twice.
    gave: Option<Id>,
}

impl<'a> Every<'a> {
    fn new(docs: &'a Docs) -> Every<'a> {
        Every {
            docs,
            reads: 0,
            gave: None,
        }
    }
}

impl<'a> Step<'a> for Every<'a> {
    fn seek(&mut self, id: Id) -> Option<Hit<'a>> {
        let mut want = id.max(1);
        while want <= self.docs.last() {
            if self.docs.get(want).is_some() {
                if self.gave != Some(want) {
                    self.gave = Some(want);
                    self.reads += 1;
                }
                return Some(Hit::new(want, Found::Every));
            }
            want = want.checked_add(1)?;
        }
        None
    }

    fn size(&self) -> u32 {
        self.docs.len() as u32
    }

    fn empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// No guess at how many, which is the one leaf a real server leaves the
    /// estimate off.
    fn ran(&self) -> Ran {
        Ran::leaf("WILDCARD", self.reads)
    }
}

/// Any of these, either adding up what answered or taking the first.
struct Any<'a> {
    under: Vec<Box<dyn Step<'a> + 'a>>,
    /// Whether every branch that answered counts, which a `|` does and an
    /// expansion does not.
    sum: bool,
    /// What the query said the union as a whole is worth.
    weight: f64,
    /// What a profile calls this a union of, which is the bare word for one a
    /// client wrote and the name of the expansion for one that made itself.
    about: Vec<u8>,
    /// How many different documents this has answered.
    reads: u64,
    /// The number last given, so that giving it again is not counted twice.
    gave: Option<Id>,
}

impl<'a> Any<'a> {
    fn new(under: Vec<Box<dyn Step<'a> + 'a>>, sum: bool, weight: f64) -> Any<'a> {
        Any {
            under,
            sum,
            weight,
            about: b"UNION".to_vec(),
            reads: 0,
            gave: None,
        }
    }

    /// The same union, saying what made it, which is what a profile prints and
    /// what decides whether `LIMITED` folds the branches away.
    fn tells(mut self, about: Vec<u8>) -> Any<'a> {
        self.about = about;
        self
    }
}

impl<'a> Step<'a> for Any<'a> {
    fn seek(&mut self, id: Id) -> Option<Hit<'a>> {
        let mut first: Option<Id> = None;
        for child in &mut self.under {
            if let Some(hit) = child.seek(id) {
                first = Some(first.map_or(hit.id, |at| at.min(hit.id)));
            }
        }
        let id = first?;
        let mut found = Vec::new();
        for child in &mut self.under {
            let Some(hit) = child.seek(id) else { continue };
            if hit.id != id {
                continue;
            }
            found.push(hit.found);
            if !self.sum {
                break;
            }
        }
        if self.gave != Some(id) {
            self.gave = Some(id);
            self.reads += 1;
        }
        Some(Hit::new(id, Found::any(self.weight, found)))
    }

    /// Every branch added up, which overshoots when the branches overlap and is
    /// the guess a real server makes.
    fn size(&self) -> u32 {
        self.under
            .iter()
            .fold(0_u32, |sum, child| sum.saturating_add(child.size()))
    }

    fn empty(&self) -> bool {
        self.under.iter().all(|child| child.empty())
    }

    /// Every branch that answered this document, whichever one was scored.
    ///
    /// A word and the stem it came from are two branches of one union and a
    /// phrase counts the places of both, so this is the whole union and not the
    /// branch a score was taken from.
    fn places(&mut self, id: Id, into: &mut Vec<u32>) -> bool {
        for child in &mut self.under {
            if child.seek(id).is_some_and(|hit| hit.id == id) {
                child.places(id, into);
            }
        }
        true
    }

    /// An expansion counts as one branch however many terms it stands for, so
    /// there is nothing to measure between, which is measured: `w1*` reaches
    /// eleven words of one document and still comes back with a slop of one.
    fn slop(&mut self, id: Id) -> u32 {
        if !self.sum {
            return 1;
        }
        let mut lists = Vec::with_capacity(self.under.len());
        for child in &mut self.under {
            if child.seek(id).is_some_and(|hit| hit.id == id) {
                spot(child, id, &mut lists);
            }
        }
        apart(&lists)
    }

    /// A union an expansion made is the one a `LIMITED` profile folds into a
    /// count, and a union a client wrote with a bar is not.
    fn ran(&self) -> Ran {
        Ran {
            about: Some(self.about.as_slice().into()),
            under: self.under.iter().map(|child| child.ran()).collect(),
            folds: self.about != b"UNION",
            ..Ran::leaf("UNION", self.reads)
        }
    }
}

/// All of these, which is what a space between two words means.
struct All<'a> {
    under: Vec<Box<dyn Step<'a> + 'a>>,
    /// What the query said the intersection as a whole is worth.
    weight: f64,
    /// How many different documents this has answered.
    reads: u64,
    /// The number last given, so that giving it again is not counted twice.
    gave: Option<Id>,
}

impl<'a> All<'a> {
    fn new(mut under: Vec<Box<dyn Step<'a> + 'a>>, weight: f64) -> All<'a> {
        // The rarest branch first, so the leapfrog has the fewest documents to
        // land on, and stable so that two branches that guess the same stay in
        // the order they were written, which is the order a real server
        // explains them in.
        under.sort_by_key(|child| child.size());
        All {
            under,
            weight,
            reads: 0,
            gave: None,
        }
    }
}

impl<'a> Step<'a> for All<'a> {
    fn seek(&mut self, id: Id) -> Option<Hit<'a>> {
        if self.under.is_empty() {
            return None;
        }
        let mut want = id;
        loop {
            let mut found = Vec::with_capacity(self.under.len());
            let mut past = None;
            for child in &mut self.under {
                let hit = child.seek(want)?;
                if hit.id != want {
                    past = Some(hit.id);
                    break;
                }
                found.push(hit.found);
            }
            match past {
                // Somebody is further along, so everybody is asked again from
                // there, which is what makes this a leapfrog rather than a walk.
                Some(at) => want = at,
                None => {
                    if self.gave != Some(want) {
                        self.gave = Some(want);
                        self.reads += 1;
                    }
                    return Some(Hit::new(want, Found::all(self.weight, found)));
                }
            }
        }
    }

    /// The smallest branch, because nothing answers this that does not answer
    /// every branch of it.
    fn size(&self) -> u32 {
        self.under
            .iter()
            .map(|child| child.size())
            .min()
            .unwrap_or(0)
    }

    fn empty(&self) -> bool {
        self.under.is_empty() || self.under.iter().any(|child| child.empty())
    }

    fn places(&mut self, id: Id, into: &mut Vec<u32>) -> bool {
        for child in &mut self.under {
            child.places(id, into);
        }
        true
    }

    fn slop(&mut self, id: Id) -> u32 {
        let mut lists = Vec::with_capacity(self.under.len());
        for child in &mut self.under {
            spot(child, id, &mut lists);
        }
        apart(&lists)
    }

    fn ran(&self) -> Ran {
        Ran {
            under: self.under.iter().map(|child| child.ran()).collect(),
            ..Ran::leaf("INTERSECT", self.reads)
        }
    }
}

/// All of these, near enough to each other, which is what a phrase is and what
/// a slop asks for.
struct Near<'a> {
    under: Vec<Box<dyn Step<'a> + 'a>>,
    /// How much room there is beyond a run, where less than nothing is no limit
    /// at all and only the order is being asked for.
    slop: i64,
    /// Whether the places have to climb, which they may do without moving.
    inorder: bool,
    /// What the query said the whole thing is worth.
    weight: f64,
    /// Where each word that takes part was found, kept between documents so the
    /// room is taken once rather than per document.
    at: Vec<Vec<u32>>,
    /// The answer last given, for giving it again.
    last: Option<Hit<'a>>,
    /// How many different documents this has answered.
    reads: u64,
}

impl<'a> Near<'a> {
    fn new(
        mut under: Vec<Box<dyn Step<'a> + 'a>>,
        slop: i64,
        inorder: bool,
        weight: f64,
    ) -> Near<'a> {
        // Asked for in order the order is the question, so it is left alone.
        // Asked for in any order it is sorted the way an ordinary intersection
        // is, which is measured: `fox dog => { $slop: 100 }` explains itself as
        // dog and then fox.
        if !inorder {
            under.sort_by_key(|child| child.size());
        }
        Near {
            under,
            slop,
            inorder,
            weight,
            at: Vec::new(),
            last: None,
            reads: 0,
        }
    }

    /// Whether the words of this document sit close enough together.
    fn close(&mut self, id: Id) -> bool {
        self.at.clear();
        for child in &mut self.under {
            let mut mine = Vec::new();
            if child.places(id, &mut mine) {
                // A union hands over the places of every branch that answered,
                // so they arrive in branch order and a word and its stem hand
                // over the same place twice.
                mine.sort_unstable();
                mine.dedup();
                self.at.push(mine);
            }
        }
        close(&self.at, self.slop, self.inorder)
    }
}

impl<'a> Step<'a> for Near<'a> {
    fn seek(&mut self, id: Id) -> Option<Hit<'a>> {
        if let Some(hit) = &self.last
            && hit.id >= id
        {
            return Some(hit.clone());
        }
        if self.under.is_empty() {
            return None;
        }
        let mut want = id;
        loop {
            let mut found = Vec::with_capacity(self.under.len());
            let mut past = None;
            for child in &mut self.under {
                let hit = child.seek(want)?;
                if hit.id != want {
                    past = Some(hit.id);
                    break;
                }
                found.push(hit.found);
            }
            if let Some(at) = past {
                want = at;
                continue;
            }
            if self.close(want) {
                let hit = Hit::new(want, Found::all(self.weight, found));
                self.last = Some(hit.clone());
                self.reads += 1;
                return Some(hit);
            }
            // Everybody is here and they are too far apart, so the next document
            // to try is the one after this and not the one after anybody's list.
            want = want.checked_add(1)?;
        }
    }

    fn size(&self) -> u32 {
        self.under
            .iter()
            .map(|child| child.size())
            .min()
            .unwrap_or(0)
    }

    fn empty(&self) -> bool {
        self.under.is_empty() || self.under.iter().any(|child| child.empty())
    }

    fn places(&mut self, id: Id, into: &mut Vec<u32>) -> bool {
        for child in &mut self.under {
            child.places(id, into);
        }
        true
    }

    fn slop(&mut self, id: Id) -> u32 {
        let mut lists = Vec::with_capacity(self.under.len());
        for child in &mut self.under {
            spot(child, id, &mut lists);
        }
        apart(&lists)
    }

    /// A phrase is an intersection that also checks where the words landed, and
    /// a real server profiles it as one.
    fn ran(&self) -> Ran {
        Ran {
            under: self.under.iter().map(|child| child.ran()).collect(),
            ..Ran::leaf("INTERSECT", self.reads)
        }
    }
}

/// One branch's places, sorted and with the repeats taken out, added to the
/// chain unless the branch has none to give.
fn spot(child: &mut Box<dyn Step<'_> + '_>, id: Id, into: &mut Vec<Vec<u32>>) {
    let mut mine = Vec::new();
    child.places(id, &mut mine);
    if mine.is_empty() {
        return;
    }
    mine.sort_unstable();
    mine.dedup();
    into.push(mine);
}

/// How far apart a chain of branches landed, as a real server counts it.
///
/// The smallest gap between each pair of branches next to each other, squared
/// and added up, and the whole number part of the square root of that. Fewer
/// than two branches with places between them is nothing to measure, and so is
/// a chain that has every branch on the same place, and both come back as one
/// because the answer is something three of the scorers divide by.
fn apart(at: &[Vec<u32>]) -> u32 {
    if at.len() < 2 {
        return 1;
    }
    let mut sum = 0_u64;
    for pair in at.windows(2) {
        let gap = u64::from(closest(&pair[0], &pair[1]));
        sum = sum.saturating_add(gap * gap);
    }
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    let root = (sum as f64).sqrt() as u32;
    root.max(1)
}

/// The smallest gap between a place in one list and a place in the other.
///
/// Both lists are in order, so this is one walk up the two of them together
/// rather than every pair.
fn closest(a: &[u32], b: &[u32]) -> u32 {
    let (mut i, mut j) = (0, 0);
    let mut best = u32::MAX;
    while i < a.len() && j < b.len() {
        best = best.min(a[i].abs_diff(b[j]));
        if a[i] < b[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
    best
}

/// Whether one place can be given to each of these so that they sit close
/// enough together.
///
/// The measured rule, on 8.10.1. Fewer than two words are always close enough,
/// whatever they are and wherever they are, which is why a phrase of one word is
/// that word. Otherwise the room is the slop plus one less than the number of
/// words, so a phrase of three words spans three places and a slop of two spans
/// five. In order the places may repeat, and in any order they may not all be
/// the same place, which is the whole of the difference between the two.
fn close(at: &[Vec<u32>], slop: i64, inorder: bool) -> bool {
    if at.len() < 2 {
        return true;
    }
    // A slop of less than nothing is no room limit at all, so only the order is
    // being asked for and the words may be as far apart as they like.
    let room = (slop >= 0)
        .then(|| slop.checked_add(at.len() as i64 - 1).unwrap_or(i64::MAX))
        .and_then(|room| u32::try_from(room).ok());
    if inorder {
        return climbing(at, room);
    }
    // No order and no limit is an ordinary intersection, which everybody here
    // has already answered.
    room.is_none_or(|room| window(at, room))
}

/// Whether the places can be made to climb, within the room there is.
///
/// Greedy from each place the first word was found at: the smallest place of the
/// next word that is not behind where the last one landed is the best one to
/// take, because taking a later one only makes the run longer.
fn climbing(at: &[Vec<u32>], room: Option<u32>) -> bool {
    for first in &at[0] {
        let mut last = *first;
        let mut fits = true;
        for list in &at[1..] {
            let Some(next) = list[list.partition_point(|place| *place < last)..].first() else {
                fits = false;
                break;
            };
            last = *next;
        }
        if fits && room.is_none_or(|room| last - first <= room) {
            return true;
        }
    }
    false
}

/// Whether the places fit inside a window of this much, in any order, without
/// every word standing on the same place.
///
/// Every window worth trying starts at a place somebody was found at, so this
/// tries each of those in turn and asks whether everybody has something inside
/// it. The two places rule falls out of the same walk: a window holding one
/// place and nothing else can only be answered by everybody standing on it.
fn window(at: &[Vec<u32>], room: u32) -> bool {
    for list in at {
        for start in list {
            let stop = start.saturating_add(room);
            let mut every = true;
            let mut two = false;
            for other in at {
                let inside = &other[other.partition_point(|place| place < start)..];
                let inside = &inside[..inside.partition_point(|place| *place <= stop)];
                if inside.is_empty() {
                    every = false;
                    break;
                }
                two |= inside.iter().any(|place| place != start);
            }
            if every && two {
                return true;
            }
        }
    }
    false
}

/// None of these, which is a walk over the documents asking each one.
struct Unless<'a> {
    docs: &'a Docs,
    under: Box<dyn Step<'a> + 'a>,
    /// How many different documents this has answered.
    reads: u64,
    /// The number last given, so that giving it again is not counted twice.
    gave: Option<Id>,
}

impl<'a> Step<'a> for Unless<'a> {
    fn seek(&mut self, id: Id) -> Option<Hit<'a>> {
        let mut want = id.max(1);
        while want <= self.docs.last() {
            if self.docs.get(want).is_some()
                && self.under.seek(want).is_none_or(|hit| hit.id > want)
            {
                // Measured: a document that answered a negation and nothing else
                // scores zero, which is a match counted no times at all.
                if self.gave != Some(want) {
                    self.gave = Some(want);
                    self.reads += 1;
                }
                return Some(Hit::new(want, Found::missing()));
            }
            want = want.checked_add(1)?;
        }
        None
    }

    /// Roughly everything, because a negation answers what it does not find and
    /// most words are in most documents no more than a few times.
    fn size(&self) -> u32 {
        self.docs.len() as u32
    }

    fn always(&self) -> bool {
        self.under.empty()
    }

    fn ran(&self) -> Ran {
        Ran {
            under: vec![self.under.ran()],
            alone: true,
            ..Ran::leaf("NOT", self.reads)
        }
    }
}

/// This, but a document without it answers anyway.
struct Maybe<'a> {
    docs: &'a Docs,
    under: Box<dyn Step<'a> + 'a>,
    /// How many different documents this has answered.
    reads: u64,
    /// The number last given, so that giving it again is not counted twice.
    gave: Option<Id>,
}

impl<'a> Step<'a> for Maybe<'a> {
    fn seek(&mut self, id: Id) -> Option<Hit<'a>> {
        let mut want = id.max(1);
        while want <= self.docs.last() {
            if self.docs.get(want).is_some() {
                let found = match self.under.seek(want) {
                    Some(hit) if hit.id == want => hit.found,
                    // Measured: a document an optional did not answer is worth
                    // nothing, which is a match at a weight of nothing.
                    _ => Found::skipped(),
                };
                if self.gave != Some(want) {
                    self.gave = Some(want);
                    self.reads += 1;
                }
                return Some(Hit::new(want, found));
            }
            want = want.checked_add(1)?;
        }
        None
    }

    fn size(&self) -> u32 {
        self.docs.len() as u32
    }

    /// The places of whatever it found, while still saying it takes no part.
    ///
    /// Both halves are measured and they pull opposite ways. An optional stays
    /// out of a position check, so `(alpha ~zulu)=>{$slop:0}` answers a document
    /// with a word between the two, and it hands its places over to the slop, so
    /// `w1 ~fox` comes back with the distance from the one to the other.
    fn places(&mut self, id: Id, into: &mut Vec<u32>) -> bool {
        if self.under.seek(id).is_some_and(|hit| hit.id == id) {
            self.under.places(id, into);
        }
        false
    }

    fn ran(&self) -> Ran {
        Ran {
            under: vec![self.under.ran()],
            alone: true,
            ..Ran::leaf("OPTIONAL", self.reads)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::english::English;
    use crate::field::{Field, Kind, Tag, Text};
    use crate::index::{Definition, Index};
    use crate::query::{Ask, parse};
    use crate::score::{Facts, Scorer};

    /// One document on its way in: a key and the fields under it.
    type Written<'a> = (&'a [u8], &'a [(&'a [u8], &'a [u8])]);

    /// An index over one text field, one tag field and one number, with the
    /// documents written into it.
    fn indexed(docs: &[Written<'_>]) -> Index {
        let mut index = Index::new(
            b"ix",
            Definition::default(),
            vec![
                Field::new(b"t", Kind::Text(Text::default())),
                Field::new(b"g", Kind::Tag(Tag::default())),
                Field::new(b"n", Kind::Numeric),
            ],
        );
        let mut english = English::new();
        for (key, fields) in docs {
            index
                .write(&mut english, key, fields)
                .expect("a document that indexes");
        }
        index.held.settle();
        index
    }

    /// A text index with no stemming, which is the shape most of the measured
    /// numbers were taken on.
    fn plain(docs: &[(&[u8], &[u8])]) -> Index {
        let text = Text {
            nostem: true,
            ..Text::default()
        };
        let mut index = Index::new(
            b"ix",
            Definition::default(),
            vec![Field::new(b"t", Kind::Text(text))],
        );
        let mut english = English::new();
        for (key, value) in docs {
            index
                .write(&mut english, key, &[(b"t", *value)])
                .expect("a document that indexes");
        }
        index
    }

    /// The keys a query answers under the second dialect, which is the one an
    /// attribute like `=>{$slop:0}` is read under.
    fn near(index: &Index, query: &[u8]) -> Vec<Vec<u8>> {
        let ask = Ask {
            dialect: 2,
            ..Ask::default()
        };
        let node = parse(query, index, &ask).expect("a query that parses");
        run(&index.held, &node)
            .into_iter()
            .map(|hit| {
                index
                    .held
                    .docs
                    .key(hit.id)
                    .expect("a hit is a live document")
                    .to_vec()
            })
            .collect()
    }

    /// The keys a query answers, in the order the walk gives them.
    fn keys(index: &Index, query: &[u8]) -> Vec<Vec<u8>> {
        let node = parse(query, index, &Ask::default()).expect("a query that parses");
        run(&index.held, &node)
            .into_iter()
            .map(|hit| {
                index
                    .held
                    .docs
                    .key(hit.id)
                    .expect("a hit is a live document")
                    .to_vec()
            })
            .collect()
    }

    /// What a query scores each document it answers, by key.
    fn scores(index: &Index, query: &[u8]) -> Vec<(Vec<u8>, f64)> {
        let node = parse(query, index, &Ask::default()).expect("a query that parses");
        let facts = index.held.facts();
        run(&index.held, &node)
            .into_iter()
            .map(|hit| {
                let doc = index.held.docs.get(hit.id).expect("a live document");
                let score = Scorer::default_scorer().of(&facts, doc, &hit.found, None, 1);
                (doc.key.to_vec(), score)
            })
            .collect()
    }

    fn same(got: f64, want: f64) {
        assert!(
            (got - want).abs() <= f64::EPSILON * want.abs().max(1.0) * 4.0,
            "got {got:?} want {want:?}"
        );
    }

    fn score(index: &Index, query: &[u8], key: &[u8]) -> f64 {
        scores(index, query)
            .into_iter()
            .find(|(k, _)| k == key)
            .map(|(_, score)| score)
            .unwrap_or_else(|| panic!("{} did not answer", String::from_utf8_lossy(query)))
    }

    #[test]
    fn a_word_answers_the_documents_that_hold_it() {
        let index = plain(&[(b"d:1", b"aa bb"), (b"d:2", b"aa cc"), (b"d:3", b"dd ee")]);
        assert_eq!(keys(&index, b"aa"), [b"d:1", b"d:2"]);
        assert_eq!(keys(&index, b"bb"), [b"d:1".to_vec()]);
        assert!(keys(&index, b"zz").is_empty());
    }

    #[test]
    fn a_union_answers_either_and_an_intersection_answers_both() {
        let index = plain(&[(b"d:1", b"aa bb"), (b"d:2", b"aa cc"), (b"d:3", b"dd ee")]);
        assert_eq!(keys(&index, b"aa|dd"), [b"d:1", b"d:2", b"d:3"]);
        assert_eq!(keys(&index, b"aa bb"), [b"d:1".to_vec()]);
        assert!(keys(&index, b"aa dd").is_empty());
    }

    /// Measured on an index of four documents holding two tokens each, where
    /// `aa` is in three of them and `bb` in one. The union of the two on the
    /// document that holds both is the sum of the two on their own, which is
    /// what says a union adds up rather than picking.
    #[test]
    fn a_union_is_worth_the_sum_of_what_answered() {
        let index = plain(&[
            (b"d:1", b"aa bb"),
            (b"d:2", b"aa cc"),
            (b"d:3", b"dd ee"),
            (b"d:4", b"aa aa"),
        ]);
        same(score(&index, b"aa", b"d:1"), 0.3566749439387324);
        same(score(&index, b"bb", b"d:1"), 1.2039728043259361);
        same(score(&index, b"aa|bb", b"d:1"), 1.5606477482646686);
        same(score(&index, b"aa bb", b"d:1"), 1.5606477482646686);
        // Twice the word is twice the frequency and not twice the score.
        same(score(&index, b"aa", b"d:4"), 0.49042805123755123);
    }

    /// Measured on the same corpus: a document a negation answers scores
    /// nothing, and one an optional does not answer scores nothing but is still
    /// in the answer.
    #[test]
    fn a_negation_and_an_optional_are_worth_nothing_on_their_own() {
        let index = plain(&[
            (b"d:1", b"aa bb"),
            (b"d:2", b"aa cc"),
            (b"d:3", b"dd ee"),
            (b"d:4", b"aa aa"),
        ]);
        assert_eq!(keys(&index, b"-bb"), [b"d:2", b"d:3", b"d:4"]);
        same(score(&index, b"-bb", b"d:2"), 0.0);
        assert_eq!(keys(&index, b"~bb"), [b"d:1", b"d:2", b"d:3", b"d:4"]);
        same(score(&index, b"~bb", b"d:1"), 1.2039728043259361);
        same(score(&index, b"~bb", b"d:3"), 0.0);
        // An optional beside a word narrows nothing and adds what it found.
        assert_eq!(keys(&index, b"aa ~bb"), [b"d:1", b"d:2", b"d:4"]);
        same(score(&index, b"aa ~bb", b"d:1"), 1.5606477482646686);
    }

    /// Measured: a bare `*` scores every document with the length correction
    /// and no rarity at all, on an index of three documents holding seven
    /// tokens between them.
    #[test]
    fn a_wildcard_answers_everything_and_scores_by_length() {
        let index = plain(&[
            (b"d:1", b"aa bb"),
            (b"d:2", b"cc dd ee"),
            (b"d:3", b"ff gg"),
        ]);
        assert_eq!(keys(&index, b"*"), [b"d:1", b"d:2", b"d:3"]);
        same(score(&index, b"*", b"d:1"), 1.0620689667079168);
        same(score(&index, b"*", b"d:2"), 0.8953488355169045);
    }

    /// Measured twice over, with the rarer of the two words on either side of
    /// the other in byte order, and the answer is the first in byte order both
    /// times rather than the rarer or the commoner.
    #[test]
    fn an_expansion_counts_the_first_term_it_finds_and_not_the_rest() {
        let first = plain(&[
            (b"d:1", b"aba abc"),
            (b"d:2", b"abc zz"),
            (b"d:3", b"abc yy"),
            (b"d:4", b"qqq ww"),
            (b"d:5", b"rrr ss"),
            (b"d:6", b"ttt uu"),
        ]);
        assert_eq!(keys(&first, b"ab*"), [b"d:1", b"d:2", b"d:3"]);
        same(score(&first, b"aba", b"d:1"), 1.5404450409471488);
        // The measured number is a natural log of two, which is what an idf
        // comes to when a term is in half the index and a hair over.
        same(score(&first, b"abc", b"d:1"), std::f64::consts::LN_2);
        // `aba` comes first and it is the rarer of the two.
        same(score(&first, b"ab*", b"d:1"), 1.5404450409471488);

        let second = plain(&[
            (b"d:1", b"abc abd"),
            (b"d:2", b"abc zz"),
            (b"d:3", b"abc yy"),
            (b"d:4", b"qqq ww"),
            (b"d:5", b"rrr ss"),
            (b"d:6", b"ttt uu"),
        ]);
        // `abc` comes first and it is the commoner of the two.
        same(score(&second, b"ab*", b"d:1"), std::f64::consts::LN_2);
    }

    #[test]
    fn a_suffix_an_infix_and_a_pattern_answer_what_they_stand_for() {
        let index = plain(&[(b"d:1", b"abc"), (b"d:2", b"zbc"), (b"d:3", b"qqq")]);
        assert_eq!(keys(&index, b"*bc"), [b"d:1", b"d:2"]);
        assert_eq!(keys(&index, b"*b*"), Vec::<Vec<u8>>::new());
        assert_eq!(keys(&index, b"*bc*"), [b"d:1", b"d:2"]);
        let ask = Ask {
            dialect: 2,
            ..Ask::default()
        };
        let node = parse(b"w'a?c'", &index, &ask).expect("a pattern that parses");
        assert_eq!(run(&index.held, &node).len(), 1);
    }

    /// A fuzzy word reaches the words near it, and a document holding two of
    /// them is counted on the first the way any other expansion is.
    #[test]
    fn a_fuzzy_word_reaches_what_is_near_it() {
        let index = plain(&[(b"d:1", b"aa bb"), (b"d:2", b"cc"), (b"d:3", b"zzzz")]);
        assert_eq!(keys(&index, b"%%aa%%"), [b"d:1", b"d:2"]);
        assert_eq!(keys(&index, b"%aa%"), [b"d:1".to_vec()]);
    }

    /// Measured: `dog` answers a document holding `dogs` through the stem, and
    /// both documents are worth one term of the same rarity, because a stem is
    /// only written when it differs from the word.
    #[test]
    fn a_word_reaches_a_document_through_the_stem() {
        let index = indexed(&[
            (b"d:1", &[(b"t", b"dog runs")]),
            (b"d:2", &[(b"t", b"dogs run fast")]),
            (b"d:3", &[(b"t", b"cat naps")]),
        ]);
        assert_eq!(keys(&index, b"dog"), [b"d:1", b"d:2"]);
        same(score(&index, b"dog", b"d:1"), 1.041708311263062);
        same(score(&index, b"dog", b"d:2"), 0.8781843295249644);
        // The written word and the stem both answer the document that has
        // both, and there the two add up.
        same(score(&index, b"dogs", b"d:2"), 1.7563686590499288);
        same(score(&index, b"dogs", b"d:1"), 1.041708311263062);
    }

    /// Measured: a tag scores the way a term does, a value written as two words
    /// is one value with a space in it, and a tag prefix of one letter answers
    /// nothing.
    #[test]
    fn a_tag_is_matched_whole_and_scored_like_a_term() {
        let index = indexed(&[
            (
                b"d:1",
                &[(b"t", b"dog runs"), (b"g", b"aa,bb"), (b"n", b"1")],
            ),
            (
                b"d:2",
                &[(b"t", b"dogs run fast"), (b"g", b"aa bb"), (b"n", b"2")],
            ),
            (b"d:3", &[(b"t", b"cat naps"), (b"g", b"cc"), (b"n", b"3")]),
        ]);
        assert_eq!(keys(&index, b"@g:{aa}"), [b"d:1".to_vec()]);
        same(score(&index, b"@g:{aa}", b"d:1"), 1.041708311263062);
        assert_eq!(keys(&index, b"@g:{aa bb}"), [b"d:2".to_vec()]);
        same(score(&index, b"@g:{aa bb}", b"d:2"), 0.8781843295249644);
        assert_eq!(keys(&index, b"@g:{aa|cc}"), [b"d:1", b"d:3"]);
        assert_eq!(keys(&index, b"@g:{a*}"), Vec::<Vec<u8>>::new());
        assert_eq!(keys(&index, b"@g:{aa*}"), [b"d:1", b"d:2"]);
    }

    /// Measured: a range is a filter and the documents it answers score zero.
    #[test]
    fn a_range_answers_the_numbers_inside_it_and_scores_nothing() {
        let index = indexed(&[
            (b"d:1", &[(b"t", b"dog runs"), (b"g", b"aa"), (b"n", b"1")]),
            (b"d:2", &[(b"t", b"dogs run"), (b"g", b"bb"), (b"n", b"2")]),
            (b"d:3", &[(b"t", b"cat naps"), (b"g", b"cc"), (b"n", b"3")]),
        ]);
        assert_eq!(keys(&index, b"@n:[1 2]"), [b"d:1", b"d:2"]);
        same(score(&index, b"@n:[1 2]", b"d:1"), 0.0);
        assert_eq!(keys(&index, b"@n:[(1 3]"), [b"d:2", b"d:3"]);
        assert_eq!(keys(&index, b"@n:[-inf +inf]"), [b"d:1", b"d:2", b"d:3"]);
    }

    /// A field modifier narrows a word to the fields it names, and asking for
    /// no field at all answers nothing.
    #[test]
    fn a_field_modifier_narrows_a_word_to_the_field_it_names() {
        let mut index = Index::new(
            b"ix",
            Definition::default(),
            vec![
                Field::new(b"a", Kind::Text(Text::default())),
                Field::new(b"b", Kind::Text(Text::default())),
            ],
        );
        let mut english = English::new();
        index
            .write(&mut english, b"d:1", &[(b"a", b"hello"), (b"b", b"world")])
            .expect("a document that indexes");
        assert_eq!(keys(&index, b"@a:hello"), [b"d:1".to_vec()]);
        assert!(keys(&index, b"@b:hello").is_empty());
        assert_eq!(keys(&index, b"@a|b:world"), [b"d:1".to_vec()]);
    }

    /// A number that stopped meaning anything is skipped rather than answered,
    /// which is what keeps a rewritten document out of its own old answer.
    #[test]
    fn a_document_written_again_answers_under_its_new_number_only() {
        let mut index = plain(&[(b"d:1", b"aa bb"), (b"d:2", b"aa cc")]);
        let mut english = English::new();
        index
            .write(&mut english, b"d:1", &[(b"t", b"dd")])
            .expect("a document that indexes");
        assert_eq!(keys(&index, b"aa"), [b"d:2".to_vec()]);
        assert_eq!(keys(&index, b"dd"), [b"d:1".to_vec()]);
        assert_eq!(keys(&index, b"*"), [b"d:2", b"d:1"]);
    }

    /// A query of nothing but stopwords answers nothing at all, rather than
    /// answering everything the way an empty filter would.
    #[test]
    fn a_query_of_stopwords_answers_nothing() {
        let index = plain(&[(b"d:1", b"aa bb")]);
        assert!(keys(&index, b"the").is_empty());
    }

    /// An empty index answers nothing without going looking for anything.
    #[test]
    fn an_empty_index_answers_nothing() {
        let index = plain(&[]);
        assert!(keys(&index, b"aa").is_empty());
        assert!(keys(&index, b"*").is_empty());
        assert!(keys(&index, b"-aa").is_empty());
        assert_eq!(Facts::new(0, 0).average(), 0.0);
    }

    /// The corpus the phrase rule was measured on, nine documents over one text
    /// field with no stemming, so a word is only ever itself.
    fn spaced() -> Index {
        plain(&[
            (b"r:1", b"alpha beta"),
            (b"r:2", b"alpha zulu alpha beta"),
            (b"r:3", b"beta alpha gamma"),
            (b"r:4", b"alpha zulu beta zulu gamma"),
            (b"r:5", b"alpha beta gamma"),
            (b"r:6", b"gamma beta alpha"),
            (b"r:7", b"alpha"),
            (b"r:8", b"beta alpha"),
            (b"r:9", b"alpha zulu beta"),
        ])
    }

    /// A phrase is the words in a run and nothing between them, which is every
    /// document holding `alpha` with `beta` straight after it and no others.
    ///
    /// Measured on 8.10.1 over the same nine documents.
    #[test]
    fn a_phrase_answers_the_words_in_a_run() {
        let index = spaced();
        assert_eq!(near(&index, b"\"alpha beta\""), [b"r:1", b"r:2", b"r:5"]);
        assert_eq!(near(&index, b"\"alpha zulu\""), [b"r:2", b"r:4", b"r:9"]);
        assert_eq!(near(&index, b"\"alpha beta gamma\""), [b"r:5".to_vec()]);
        assert!(near(&index, b"\"alpha beta alpha\"").is_empty());
        assert!(near(&index, b"\"beta alpha beta\"").is_empty());
    }

    /// Two words of a phrase may stand on the same place, so a phrase of one
    /// word written twice answers every document holding it once.
    ///
    /// This is the part of the rule nobody would guess and it is measured: the
    /// nine documents all answer `"alpha alpha"`, and the five holding a `beta`
    /// after an `alpha` answer `"alpha alpha beta"`.
    #[test]
    fn a_phrase_lets_two_words_stand_on_one_place() {
        let index = spaced();
        assert_eq!(near(&index, b"\"alpha alpha\"").len(), 9);
        assert_eq!(
            near(&index, b"\"alpha alpha beta\""),
            [b"r:1", b"r:2", b"r:4", b"r:5", b"r:9"]
        );
    }

    /// A slop is how much room there is beyond the run, and in any order the
    /// words may not all stand on one place.
    ///
    /// Measured: `alpha beta` under a slop of nothing answers the six documents
    /// holding the two words next to each other either way round, a slop of one
    /// adds the two holding them a word apart, and `alpha alpha` under a slop
    /// answers only the one document holding the word twice.
    #[test]
    fn a_slop_is_room_and_any_order_wants_two_places() {
        let index = spaced();
        assert_eq!(
            near(&index, b"(alpha beta)=>{$slop:0}"),
            [b"r:1", b"r:2", b"r:3", b"r:5", b"r:6", b"r:8"]
        );
        assert_eq!(
            near(&index, b"(alpha beta)=>{$slop:1}"),
            [
                b"r:1", b"r:2", b"r:3", b"r:4", b"r:5", b"r:6", b"r:8", b"r:9"
            ]
        );
        assert_eq!(near(&index, b"(alpha alpha)=>{$slop:1}"), [b"r:2".to_vec()]);
        assert_eq!(
            near(&index, b"(alpha beta gamma)=>{$slop:2}"),
            [b"r:3", b"r:4", b"r:5", b"r:6"]
        );
    }

    /// In order the places have to climb, and asking for the order without a
    /// slop asks for nothing else.
    ///
    /// Measured: `alpha gamma` in order answers the three documents holding a
    /// gamma after an alpha however far away, where the same query without the
    /// order also answers the one holding them the other way round.
    #[test]
    fn in_order_with_no_slop_is_the_order_and_nothing_else() {
        let index = spaced();
        assert_eq!(
            near(&index, b"(alpha gamma)=>{$inorder:true}"),
            [b"r:3", b"r:4", b"r:5"]
        );
        assert_eq!(
            near(&index, b"alpha gamma"),
            [b"r:3", b"r:4", b"r:5", b"r:6"]
        );
        assert_eq!(
            near(&index, b"(alpha beta)=>{$slop:0;$inorder:true}"),
            [b"r:1", b"r:2", b"r:5"]
        );
        assert_eq!(
            near(&index, b"(alpha beta)=>{$slop:1;$inorder:true}"),
            [b"r:1", b"r:2", b"r:4", b"r:5", b"r:9"]
        );
    }

    /// One word at controlled distances, which is what pins the threshold down.
    fn apart() -> Index {
        plain(&[
            (b"s:1", b"alpha"),
            (b"s:2", b"alpha alpha"),
            (b"s:3", b"alpha zulu alpha"),
            (b"s:4", b"alpha zulu zulu alpha"),
            (b"s:5", b"alpha alpha alpha"),
            (b"s:6", b"alpha zulu alpha zulu alpha"),
            (b"s:7", b"alpha zulu zulu zulu alpha"),
        ])
    }

    /// The room is the slop plus one less than the number of words, which is
    /// the whole of the threshold and is measured a step at a time.
    ///
    /// Two alphas reach two places apart at a slop of one and four apart at a
    /// slop of three, and three alphas reach a document holding them two apart
    /// at a slop of nothing, because the third word buys another place of room.
    #[test]
    fn the_room_is_the_slop_plus_one_less_than_the_words() {
        let index = apart();
        assert_eq!(near(&index, b"(alpha alpha)=>{$slop:0}"), [b"s:2", b"s:5"]);
        assert_eq!(
            near(&index, b"(alpha alpha)=>{$slop:1}"),
            [b"s:2", b"s:3", b"s:5", b"s:6"]
        );
        assert_eq!(
            near(&index, b"(alpha alpha)=>{$slop:2}"),
            [b"s:2", b"s:3", b"s:4", b"s:5", b"s:6"]
        );
        assert_eq!(
            near(&index, b"(alpha alpha)=>{$slop:3}"),
            [b"s:2", b"s:3", b"s:4", b"s:5", b"s:6", b"s:7"]
        );
        assert_eq!(
            near(&index, b"(alpha alpha alpha)=>{$slop:0}"),
            [b"s:2", b"s:3", b"s:5", b"s:6"]
        );
        assert_eq!(near(&index, b"(alpha alpha alpha)=>{$slop:2}").len(), 6);
    }

    /// In order the same query answers every document holding the word at all,
    /// because the places may repeat, and that holds at any slop.
    #[test]
    fn in_order_lets_the_places_repeat_at_any_slop() {
        let index = apart();
        assert_eq!(
            near(&index, b"(alpha alpha)=>{$slop:0;$inorder:true}").len(),
            7
        );
        assert_eq!(
            near(&index, b"(alpha alpha alpha)=>{$slop:0;$inorder:true}").len(),
            7
        );
    }

    /// What takes part in a position check and what stays out of it.
    ///
    /// Measured on an index of two documents, `alpha zulu beta` tagged `aa` and
    /// `alpha beta` tagged `bb`. A range, a negation and one tag value change
    /// nothing about which documents answer, and two tag values written as a
    /// union answer nothing at all, because a union takes part and has no places
    /// of its own to give.
    #[test]
    fn a_range_and_one_tag_stay_out_of_a_position_check() {
        let index = indexed(&[
            (
                b"u:1",
                &[
                    (b"t".as_slice(), b"alpha zulu beta".as_slice()),
                    (b"g", b"aa"),
                    (b"n", b"5"),
                ][..],
            ),
            (
                b"u:2",
                &[
                    (b"t".as_slice(), b"alpha beta".as_slice()),
                    (b"g", b"bb"),
                    (b"n", b"7"),
                ][..],
            ),
        ]);
        assert_eq!(near(&index, b"(alpha beta)=>{$slop:0}"), [b"u:2".to_vec()]);
        assert_eq!(
            near(&index, b"(@n:[1 10] alpha beta)=>{$slop:0}"),
            [b"u:2".to_vec()]
        );
        assert_eq!(
            near(&index, b"(@g:{aa} alpha)=>{$slop:0}"),
            [b"u:1".to_vec()]
        );
        assert!(near(&index, b"(@g:{aa} alpha beta)=>{$slop:0}").is_empty());
        assert!(near(&index, b"(@g:{aa|bb} alpha)=>{$slop:0}").is_empty());
        assert_eq!(near(&index, b"(alpha -zulu)=>{$slop:0}"), [b"u:2".to_vec()]);
        assert_eq!(near(&index, b"(alpha ~zulu)=>{$slop:0}"), [b"u:1", b"u:2"]);
    }

    /// A word that stands for several terms brings the places of all of them,
    /// which is measured through a prefix and through a union.
    #[test]
    fn an_expansion_brings_the_places_of_every_term_it_stands_for() {
        let index = indexed(&[
            (
                b"u:1",
                &[
                    (b"t".as_slice(), b"alpha zulu beta".as_slice()),
                    (b"g", b"aa"),
                ][..],
            ),
            (
                b"u:2",
                &[(b"t".as_slice(), b"alpha beta".as_slice()), (b"g", b"bb")][..],
            ),
        ]);
        assert_eq!(near(&index, b"(alp* beta)=>{$slop:0}"), [b"u:2".to_vec()]);
        assert_eq!(near(&index, b"(alp* zulu)=>{$slop:0}"), [b"u:1".to_vec()]);
        assert_eq!(
            near(&index, b"((alpha|gamma) beta)=>{$slop:0}"),
            [b"u:2".to_vec()]
        );
    }

    /// A query parsed against an index, for the tests below to hold on to.
    ///
    /// `shape` hands back a borrow of the tree it walked, so the tree has to
    /// outlive the call and cannot be a local inside it. Parsing here and
    /// keeping the result at the call site is one line more to read and leaves
    /// nothing behind, which the Miri run in `deep.yml` checks for.
    fn parsed(index: &Index, query: &[u8]) -> Node {
        parse(query, index, &Ask::default()).expect("a query that parses")
    }

    /// The shape one query gave one document, which is what an explanation is
    /// printed off and what the tests below are about.
    fn shape<'a>(index: &'a Index, node: &'a Node, key: &[u8]) -> Found<'a> {
        run(&index.held, node)
            .into_iter()
            .find(|hit| index.held.docs.key(hit.id) == Some(key))
            .map(|hit| hit.found)
            .unwrap_or_else(|| panic!("nothing answered for {}", String::from_utf8_lossy(key)))
    }

    /// Measured: `FT.SEARCH hz fox EXPLAINSCORE` prints the one leaf on its own
    /// and `FT.SEARCH hz running EXPLAINSCORE` prints a branch over it, even on
    /// a document only the stem answered. The parser gives both of them a union,
    /// so what tells them apart is that nothing is ever written under `+fox`
    /// because the stem and the word are the same, while `running` and `+run`
    /// are both words some document holds.
    #[test]
    fn a_union_left_with_one_branch_is_that_branch() {
        let index = indexed(&[
            (
                b"u:1",
                &[(b"t".as_slice(), b"fox running".as_slice()), (b"g", b"aa")][..],
            ),
            (
                b"u:2",
                &[(b"t".as_slice(), b"fox runs".as_slice()), (b"g", b"bb")][..],
            ),
        ]);
        let fox = parsed(&index, b"fox");
        let running = parsed(&index, b"running");
        assert!(matches!(shape(&index, &fox, b"u:1"), Found::Term(_)));
        assert!(matches!(shape(&index, &running, b"u:2"), Found::Any { .. }));
    }

    /// Measured: `@n:[1 5] fox` explains the word first and the range second,
    /// though four documents are in the range and seven hold the word, because
    /// a real server guesses a range at the whole index rather than counting it.
    #[test]
    fn a_range_is_guessed_at_the_whole_index_when_an_intersection_sorts() {
        let index = indexed(&[
            (
                b"u:1",
                &[
                    (b"t".as_slice(), b"fox".as_slice()),
                    (b"g", b"aa"),
                    (b"n", b"1"),
                ][..],
            ),
            (
                b"u:2",
                &[
                    (b"t".as_slice(), b"cat".as_slice()),
                    (b"g", b"bb"),
                    (b"n", b"2"),
                ][..],
            ),
        ]);
        let node = parsed(&index, b"@n:[1 5] fox");
        let Found::All { under, .. } = shape(&index, &node, b"u:1") else {
            panic!("an intersection of a range and a word");
        };
        assert!(matches!(under.first(), Some(Found::Term(_))));
        assert!(matches!(under.get(1), Some(Found::Blank { .. })));
    }

    /// What a walk of this query turned out to be.
    fn ran(index: &Index, query: &[u8]) -> Ran {
        let node = parse(query, index, &Ask::default()).expect("a query that parses");
        profiled(&index.held, &node, false).1
    }

    /// Three documents, two of which hold the first word and two the second.
    fn three() -> Index {
        plain(&[
            (b"k:1", b"alpha"),
            (b"k:2", b"alpha beta"),
            (b"k:3", b"beta"),
        ])
    }

    #[test]
    fn a_word_says_what_it_is_and_how_many_hold_it() {
        let index = three();
        let ran = ran(&index, b"alpha");
        assert_eq!(ran.kind, "TEXT");
        assert_eq!(ran.term.as_deref(), Some(&b"alpha"[..]));
        assert_eq!(ran.size, Some(2));
        assert_eq!(ran.reads, 2);
        assert!(ran.under.is_empty());
    }

    #[test]
    fn a_union_counts_a_document_once_however_many_branches_answered_it() {
        let index = three();
        let ran = ran(&index, b"alpha|beta");
        assert_eq!(ran.kind, "UNION");
        assert_eq!(ran.about.as_deref(), Some(&b"UNION"[..]));
        assert_eq!(ran.reads, 3);
        // Nothing guesses how many a branch will answer, which is the one thing
        // a real server prints on a leaf and leaves off a branch.
        assert_eq!(ran.size, None);
        let counts: Vec<u64> = ran.under.iter().map(|child| child.reads).collect();
        assert_eq!(counts, vec![2, 2]);
    }

    #[test]
    fn an_intersection_counts_what_answered_and_not_what_it_stepped_over() {
        let index = three();
        let ran = ran(&index, b"alpha beta");
        assert_eq!(ran.kind, "INTERSECT");
        assert_eq!(ran.reads, 1);
        // The second word was asked twice and answered the same document both
        // times, which counts once. Measured on a real server over these three
        // documents, which answers two and one for the same two words.
        let counts: Vec<u64> = ran.under.iter().map(|child| child.reads).collect();
        assert_eq!(counts, vec![2, 1]);
    }

    #[test]
    fn a_negation_and_an_optional_each_hold_one_thing() {
        let index = three();
        let no = ran(&index, b"-alpha");
        assert_eq!(no.kind, "NOT");
        assert!(no.alone);
        assert_eq!(no.reads, 1);
        assert_eq!(no.under.len(), 1);
        assert_eq!(no.under[0].reads, 2);
        let maybe = ran(&index, b"~alpha");
        assert_eq!(maybe.kind, "OPTIONAL");
        assert!(maybe.alone);
        assert_eq!(maybe.reads, 3);
        assert_eq!(maybe.under[0].reads, 2);
    }

    #[test]
    fn a_wildcard_and_a_query_of_nothing_are_told_apart() {
        let index = three();
        let every = ran(&index, b"*");
        assert_eq!(every.kind, "WILDCARD");
        assert_eq!(every.reads, 3);
        assert_eq!(every.size, None);
        let none = ran(&index, b"the");
        assert_eq!(none.kind, "EMPTY");
        assert_eq!(none.reads, 0);
        assert_eq!(none.size, None);
    }

    #[test]
    fn an_expansion_says_what_made_it_and_a_bar_does_not() {
        let index = plain(&[(b"k:1", b"alpha"), (b"k:2", b"alps"), (b"k:3", b"beta")]);
        let spread = ran(&index, b"alp*");
        assert_eq!(spread.kind, "UNION");
        assert_eq!(spread.about.as_deref(), Some(&b"PREFIX - alp"[..]));
        // Which is the whole of what `LIMITED` looks at when it decides to fold
        // the branches away into a count.
        assert!(spread.folds);
        assert_eq!(spread.under.len(), 2);
        assert!(!ran(&index, b"alpha|beta").folds);
    }

    #[test]
    fn a_range_a_circle_and_a_tag_each_name_what_they_answered_off() {
        let index = indexed(&[(
            b"u:1",
            &[
                (b"t".as_slice(), b"fox".as_slice()),
                (b"g", b"aa"),
                (b"n", b"1"),
            ][..],
        )]);
        let range = ran(&index, b"@n:[1 5]");
        assert_eq!(range.kind, "NUMERIC");
        assert_eq!(range.term.as_deref(), Some(&b"1.000000 - 5.000000"[..]));
        let tag = ran(&index, b"@g:{aa}");
        assert_eq!(tag.kind, "TAG");
        assert_eq!(tag.term.as_deref(), Some(&b"aa"[..]));
        assert_eq!(tag.reads, 1);
    }

    #[test]
    fn a_phrase_is_profiled_as_an_intersection() {
        let index = plain(&[(b"k:1", b"alpha beta"), (b"k:2", b"beta alpha")]);
        let ran = ran(&index, b"\"alpha beta\"");
        assert_eq!(ran.kind, "INTERSECT");
        assert_eq!(ran.reads, 1);
        assert_eq!(ran.under.len(), 2);
    }
}
