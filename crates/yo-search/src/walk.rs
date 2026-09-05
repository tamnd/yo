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
//! # What is not walked yet
//!
//! A geo filter and a vector query, which need fields the document reader does
//! not read yet, so there is nothing in the index to walk even when there is a
//! node for it. Both answer nothing rather than answering wrongly.

use crate::docs::Docs;
use crate::expand;
use crate::held::Held;
use crate::nums::Ends;
use crate::posts::{Id, Posts, Reader, stemmed};
use crate::query::{Node, Range, What, Word};
use crate::score::{Found, Term};
use crate::tags::Tags;

/// One document that answered, and the shape of how it answered.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    /// Which document.
    pub id: Id,
    /// What matched in it, which is what a scorer walks.
    pub found: Found,
}

/// Every document that answers a query, in number order.
#[must_use]
pub fn run(held: &Held, node: &Node) -> Vec<Hit> {
    let mut step = build(held, node, 1.0);
    let mut out = Vec::new();
    let mut want = 1;
    while let Some(hit) = step.seek(want) {
        let Some(next) = hit.id.checked_add(1) else {
            out.push(hit);
            break;
        };
        want = next;
        out.push(hit);
    }
    out
}

/// Something that answers where the next document it matches is.
trait Step {
    /// The first document at or after this number that matches, if there is
    /// one.
    ///
    /// Asking for a number already passed gives the last answer back, so the
    /// same question may be put twice and the second time is free.
    fn seek(&mut self, id: Id) -> Option<Hit>;

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
}

/// What one node of the tree turns into.
fn build<'a>(held: &'a Held, node: &'a Node, weight: f64) -> Box<dyn Step + 'a> {
    let weight = weight * node.weight.unwrap_or(1.0);
    let mask = mask(node.mask);
    match &node.what {
        What::Empty => Box::new(Never),
        What::Wildcard => Box::new(Every::new(&held.docs)),
        What::Term(word) => Box::new(term(held, word, mask, weight)),
        What::Union(list) => Box::new(Any::new(under(held, list, weight), true)),
        What::Intersect(list) => {
            let under = under(held, list, weight);
            // A slop of less than nothing is no limit at all, so an intersection
            // with one and no order asked for is an ordinary intersection and
            // does not go looking for places it will not read.
            let slop = node.slop.unwrap_or(-1);
            if slop >= 0 || node.inorder {
                Box::new(Near::new(under, slop, node.inorder))
            } else {
                Box::new(All::new(under))
            }
        }
        // Measured: the slop and the order a client hangs on a phrase are
        // printed back and change nothing, so a phrase is a run in order
        // whatever was asked of it.
        What::Exact(list) => Box::new(Near::new(under(held, list, weight), 0, true)),
        What::Not(child) => Box::new(Unless {
            docs: &held.docs,
            under: build(held, child, weight),
        }),
        What::Optional(child) => Box::new(Maybe {
            docs: &held.docs,
            under: build(held, child, weight),
        }),
        What::Prefix(prefix) => {
            spread(held, expand::under(held.dictionary(), prefix), mask, weight)
        }
        What::Suffix(suffix) => spread(
            held,
            expand::ending(held.dictionary(), suffix),
            mask,
            weight,
        ),
        What::Infix(part) => spread(held, expand::inside(held.dictionary(), part), mask, weight),
        What::Pattern(pattern) => {
            spread(held, expand::like(held.dictionary(), pattern), mask, weight)
        }
        What::Fuzzy(word, distance) => spread(
            held,
            expand::near(held.dictionary(), word, *distance),
            mask,
            weight,
        ),
        What::Numeric(range) => Box::new(numbers(held, range)),
        What::Tag(field, list) => tagged(held, field, list, weight),
        // Measured against nothing yet, so they answer nothing.
        What::Geo(_) | What::Vector(_) => Box::new(Never),
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
fn under<'a>(held: &'a Held, list: &'a [Node], weight: f64) -> Vec<Box<dyn Step + 'a>> {
    list.iter().map(|node| build(held, node, weight)).collect()
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
    One::new(held.posts(name), &held.docs, mask, weight)
}

/// A union over every term one expansion stands for, in dictionary order.
fn spread<'a>(
    held: &'a Held,
    found: Vec<(&'a [u8], &'a Posts)>,
    mask: u32,
    weight: f64,
) -> Box<dyn Step + 'a> {
    let under = found
        .into_iter()
        .map(|(_, posts)| {
            Box::new(One::new(Some(posts), &held.docs, mask, weight)) as Box<dyn Step + 'a>
        })
        .collect();
    Box::new(Any::new(under, false))
}

/// The documents whose number in a field is inside a range.
fn numbers(held: &Held, range: &Range) -> List {
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
    // A range is a filter and nothing else: a real server scores a document that
    // answered one and nothing else at zero, which is what an empty match adds
    // up to.
    List::new(live(&held.docs, ids), Found::All(Vec::new()))
}

/// The documents a tag field's values were asked for.
fn tagged<'a>(held: &'a Held, field: &[u8], list: &'a [Node], weight: f64) -> Box<dyn Step + 'a> {
    let Some(tags) = held.values(field) else {
        return Box::new(Never);
    };
    let mut under: Vec<Box<dyn Step + 'a>> = list
        .iter()
        .map(|node| value(held, tags, node, weight))
        .collect();
    // One value is that value and not a union of one, which is measured through
    // a position check: `(@g:{aa} alpha)=>{$slop:0}` answers where
    // `(@g:{aa|bb} alpha)=>{$slop:0}` answers nothing, because one value stays
    // out of the check and a union takes part in it with no places to give.
    if under.len() == 1 {
        return under.pop().unwrap_or_else(|| Box::new(Never));
    }
    Box::new(Any::new(under, true))
}

/// One value asked of a tag field.
fn value<'a>(held: &'a Held, tags: &'a Tags, node: &'a Node, weight: f64) -> Box<dyn Step + 'a> {
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
fn held_tag(held: &Held, tags: &Tags, value: &[u8], weight: f64) -> List {
    let ids = tags.get(value);
    // Measured: a tag that answered is scored the way a term that answered is,
    // with one occurrence and the rarity of the value.
    let found = Found::Term(Term::new(1, weight, ids.len() as u32));
    List::new(live(&held.docs, ids.to_vec()), found)
}

/// Every value of a tag field that fits, as a union in byte order.
fn sweep<'a>(
    held: &'a Held,
    tags: &'a Tags,
    weight: f64,
    written: usize,
    fits: impl Fn(&[u8]) -> bool,
) -> Box<dyn Step + 'a> {
    if written < expand::SHORTEST {
        return Box::new(Never);
    }
    let under = tags
        .all()
        .filter(|(value, _)| fits(value))
        .take(expand::MOST)
        .map(|(_, ids)| {
            let found = Found::Term(Term::new(1, weight, ids.len() as u32));
            Box::new(List::new(live(&held.docs, ids.to_vec()), found)) as Box<dyn Step + 'a>
        })
        .collect();
    Box::new(Any::new(under, false))
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

impl Step for Never {
    fn seek(&mut self, _: Id) -> Option<Hit> {
        None
    }
}

/// One term's posting list.
struct One<'a> {
    /// The list, or nothing when no document holds the term.
    reader: Option<Reader<'a>>,
    /// The document table, for skipping numbers that stopped meaning anything.
    docs: &'a Docs,
    /// The fields the query asked for.
    mask: u32,
    /// What the query said the term is worth.
    weight: f64,
    /// How many documents hold the term, which is the rarity a scorer wants.
    df: u32,
    /// The answer last given, for giving it again.
    at: Option<Hit>,
    /// Room for the places of one document, which a reader hands back into a
    /// buffer of its own rather than onto the end of somebody else's.
    room: Vec<u32>,
}

impl<'a> One<'a> {
    fn new(posts: Option<&'a Posts>, docs: &'a Docs, mask: u32, weight: f64) -> One<'a> {
        One {
            reader: posts.map(Posts::read),
            docs,
            mask,
            weight,
            df: posts.map_or(0, Posts::len),
            at: None,
            room: Vec::new(),
        }
    }
}

impl Step for One<'_> {
    fn seek(&mut self, id: Id) -> Option<Hit> {
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
                let found = Found::Term(Term::new(post.freq, self.weight, self.df));
                let hit = Hit { id: post.id, found };
                self.at = Some(hit.clone());
                return Some(hit);
            }
            want = post.id.checked_add(1)?;
        }
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
}

/// A list of numbers worked out in advance, which is what a range and a tag
/// value come to.
struct List {
    ids: Vec<Id>,
    at: usize,
    found: Found,
}

impl List {
    fn new(ids: Vec<Id>, found: Found) -> List {
        List { ids, at: 0, found }
    }
}

impl Step for List {
    fn seek(&mut self, id: Id) -> Option<Hit> {
        while self.at < self.ids.len() && self.ids[self.at] < id {
            self.at += 1;
        }
        let found = self.ids.get(self.at)?;
        Some(Hit {
            id: *found,
            found: self.found.clone(),
        })
    }
}

/// Every document there is, which is what a bare `*` asks for.
struct Every<'a> {
    docs: &'a Docs,
}

impl<'a> Every<'a> {
    fn new(docs: &'a Docs) -> Every<'a> {
        Every { docs }
    }
}

impl Step for Every<'_> {
    fn seek(&mut self, id: Id) -> Option<Hit> {
        let mut want = id.max(1);
        while want <= self.docs.last() {
            if self.docs.get(want).is_some() {
                return Some(Hit {
                    id: want,
                    found: Found::Every,
                });
            }
            want = want.checked_add(1)?;
        }
        None
    }
}

/// Any of these, either adding up what answered or taking the first.
struct Any<'a> {
    under: Vec<Box<dyn Step + 'a>>,
    /// Whether every branch that answered counts, which a `|` does and an
    /// expansion does not.
    sum: bool,
}

impl<'a> Any<'a> {
    fn new(under: Vec<Box<dyn Step + 'a>>, sum: bool) -> Any<'a> {
        Any { under, sum }
    }
}

impl Step for Any<'_> {
    fn seek(&mut self, id: Id) -> Option<Hit> {
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
        let found = if self.sum {
            Found::Any(found)
        } else {
            found.pop().unwrap_or_else(|| Found::All(Vec::new()))
        };
        Some(Hit { id, found })
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
}

/// All of these, which is what a space between two words means.
struct All<'a> {
    under: Vec<Box<dyn Step + 'a>>,
}

impl<'a> All<'a> {
    fn new(under: Vec<Box<dyn Step + 'a>>) -> All<'a> {
        All { under }
    }
}

impl Step for All<'_> {
    fn seek(&mut self, id: Id) -> Option<Hit> {
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
                    return Some(Hit {
                        id: want,
                        found: Found::All(found),
                    });
                }
            }
        }
    }

    fn places(&mut self, id: Id, into: &mut Vec<u32>) -> bool {
        for child in &mut self.under {
            child.places(id, into);
        }
        true
    }
}

/// All of these, near enough to each other, which is what a phrase is and what
/// a slop asks for.
struct Near<'a> {
    under: Vec<Box<dyn Step + 'a>>,
    /// How much room there is beyond a run, where less than nothing is no limit
    /// at all and only the order is being asked for.
    slop: i64,
    /// Whether the places have to climb, which they may do without moving.
    inorder: bool,
    /// Where each word that takes part was found, kept between documents so the
    /// room is taken once rather than per document.
    at: Vec<Vec<u32>>,
    /// The answer last given, for giving it again.
    last: Option<Hit>,
}

impl<'a> Near<'a> {
    fn new(under: Vec<Box<dyn Step + 'a>>, slop: i64, inorder: bool) -> Near<'a> {
        Near {
            under,
            slop,
            inorder,
            at: Vec::new(),
            last: None,
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

impl Step for Near<'_> {
    fn seek(&mut self, id: Id) -> Option<Hit> {
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
                let hit = Hit {
                    id: want,
                    found: Found::All(found),
                };
                self.last = Some(hit.clone());
                return Some(hit);
            }
            // Everybody is here and they are too far apart, so the next document
            // to try is the one after this and not the one after anybody's list.
            want = want.checked_add(1)?;
        }
    }

    fn places(&mut self, id: Id, into: &mut Vec<u32>) -> bool {
        for child in &mut self.under {
            child.places(id, into);
        }
        true
    }
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
    under: Box<dyn Step + 'a>,
}

impl Step for Unless<'_> {
    fn seek(&mut self, id: Id) -> Option<Hit> {
        let mut want = id.max(1);
        while want <= self.docs.last() {
            if self.docs.get(want).is_some()
                && self.under.seek(want).is_none_or(|hit| hit.id > want)
            {
                // Measured: a document that answered a negation and nothing else
                // scores zero, which is what nothing having matched adds up to.
                return Some(Hit {
                    id: want,
                    found: Found::All(Vec::new()),
                });
            }
            want = want.checked_add(1)?;
        }
        None
    }
}

/// This, but a document without it answers anyway.
struct Maybe<'a> {
    docs: &'a Docs,
    under: Box<dyn Step + 'a>,
}

impl Step for Maybe<'_> {
    fn seek(&mut self, id: Id) -> Option<Hit> {
        let mut want = id.max(1);
        while want <= self.docs.last() {
            if self.docs.get(want).is_some() {
                let found = match self.under.seek(want) {
                    Some(hit) if hit.id == want => hit.found,
                    _ => Found::All(Vec::new()),
                };
                return Some(Hit { id: want, found });
            }
            want = want.checked_add(1)?;
        }
        None
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
                let score = Scorer::default_scorer().of(&facts, doc, &hit.found, None);
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
}
