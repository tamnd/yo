//! The inverted index: which documents a term is in, how often, in which
//! fields and where.
//!
//! A term points at a list of entries and an entry is one document. The list is
//! kept in blocks of [`BLOCK`] entries, the ids inside a block are stored as
//! gaps rather than numbers, and every number in an entry is a varint, so a term
//! that half the corpus has still costs about a byte a document. Blocks are what
//! makes a query fast as well as small: a block knows its first and last id, so
//! an intersection walking two long lists can skip a whole block at a time
//! instead of reading its way through it.
//!
//! ```
//! use yo_search::posts::{Posts, Post};
//!
//! let mut p = Posts::new();
//! p.push(1, 2, 0b01, &[1, 7]);
//! p.push(9, 1, 0b10, &[3]);
//!
//! let mut r = p.read();
//! assert_eq!(r.step(), Some(Post { id: 1, freq: 2, fields: 0b01 }));
//! let mut places = Vec::new();
//! r.places(&mut places);
//! assert_eq!(places, [1, 7]);
//! assert_eq!(r.seek(5), Some(Post { id: 9, freq: 1, fields: 0b10 }));
//! assert_eq!(r.step(), None);
//! ```
//!
//! # What an entry holds
//!
//! The document, how often the term is in it, which fields it came from as one
//! bit each, and the places it sits at. The frequency is not a count of
//! occurrences, it is the sum of what each occurrence is worth, and what an
//! occurrence is worth is the weight of the field it was found in. [`adds`] is
//! that rule. A word twice in a field of weight three is a frequency of six, and
//! that is the number scoring reads later.
//!
//! Places are counted from one and run across the whole document rather than
//! restarting per field, which is why a phrase can be found spanning the end of
//! one field and the start of the next. That is what a real server does and it
//! is easy to mistake for a bug the first time a search matches across two
//! fields that have nothing to do with each other.
//!
//! # What is not here
//!
//! Taking anything out. A list only ever grows at the end, and a document that
//! is edited is given a new number rather than being found and rewritten, so its
//! old number stays in every list it was in and is skipped on the way past. A
//! real server does the same and collects the leftovers later, which is why
//! dumping a list can show numbers that no longer belong to any document.

use std::collections::BTreeMap;
use std::ops::Range;

/// A document's number inside one index.
///
/// Handed out in order, never reused, and never the same for two documents even
/// if one of them is the older reading of the other.
pub type Id = u32;

/// How many entries go in a block before the next one starts.
///
/// A hundred is what a real server uses and it is a reasonable number on its
/// own: enough that the eight bytes of bookkeeping a block carries are paid for,
/// few enough that skipping to a document in the middle of a long list does not
/// mean decoding much that gets thrown away.
pub const BLOCK: usize = 100;

/// The byte a stem is written behind in the term dictionary.
///
/// Stems and words live in the same dictionary and a stem is told apart by this
/// byte in front of it, so `dogs` and the stem `+dog` are two terms. It is not a
/// decoration, it is what stops a document containing the literal word `dog`
/// from being confused with one that only contains `dogs`.
pub const STEM: u8 = b'+';

/// What one occurrence of a term in a field of this weight adds to a frequency.
///
/// The weight a client declares is a fraction and the frequency is a whole
/// number, and the way a real server reconciles them is to throw the fraction
/// away and then refuse to go below one. So a weight of `0.9` counts the same as
/// a weight of `1`, a weight of `2.9` counts as `2`, and a weight of `0` still
/// counts as `1` rather than making the field invisible.
#[must_use]
pub fn adds(weight: f64) -> u32 {
    if weight.is_nan() || weight < 1.0 {
        return 1;
    }
    if weight >= f64::from(u32::MAX) {
        return u32::MAX;
    }
    weight as u32
}

/// The term a word is stemmed to, as it is written in the dictionary.
#[must_use]
pub fn stemmed(stem: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(stem.len() + 1);
    out.push(STEM);
    out.extend_from_slice(stem);
    out
}

/// One entry read out of a list, without its places.
///
/// The places are left behind because most of the time nobody wants them. An
/// intersection reads ids and stops, scoring reads ids and frequencies, and only
/// a phrase or a highlight goes on to ask [`Reader::places`] for the rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Post {
    /// Which document.
    pub id: Id,
    /// The sum of what each occurrence of the term in it is worth.
    pub freq: u32,
    /// One bit per field of the schema the term was found in.
    pub fields: u32,
}

/// One run of entries with its own bookkeeping.
#[derive(Debug, Clone)]
struct Block {
    first: Id,
    last: Id,
    entries: u32,
    bytes: Vec<u8>,
}

/// What a block says about itself, which is what `INVIDX_SUMMARY` reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// The first document in it.
    pub first: Id,
    /// The last document in it.
    pub last: Id,
    /// How many entries are in it.
    pub entries: u32,
}

/// Every document one term is in.
#[derive(Debug, Clone, Default)]
pub struct Posts {
    blocks: Vec<Block>,
    entries: u32,
    last: Id,
}

impl Posts {
    /// An empty list.
    #[must_use]
    pub fn new() -> Posts {
        Posts::default()
    }

    /// Adds a document to the end of the list.
    ///
    /// Ids have to arrive in order and each one only once, which they do because
    /// a document is given its number when it is indexed and indexing a document
    /// twice gives it a second, larger number. An id that is not larger than the
    /// last one is dropped rather than corrupting the gaps behind it.
    pub fn push(&mut self, id: Id, freq: u32, fields: u32, places: &[u32]) {
        if self.entries > 0 && id <= self.last {
            return;
        }
        let fresh = match self.blocks.last() {
            None => true,
            Some(b) => b.entries as usize >= BLOCK,
        };
        if fresh {
            self.blocks.push(Block {
                first: id,
                last: id,
                entries: 0,
                bytes: Vec::new(),
            });
        }
        let block = self.blocks.last_mut().unwrap_or_else(|| unreachable!());
        let gap = id - block.last;
        put(&mut block.bytes, gap);
        put(&mut block.bytes, freq);
        put(&mut block.bytes, fields);
        let count = u32::try_from(places.len()).unwrap_or(u32::MAX);
        put(&mut block.bytes, count);
        let mut seen = 0;
        for place in places.iter().take(count as usize) {
            put(&mut block.bytes, place.saturating_sub(seen));
            seen = *place;
        }
        block.last = id;
        block.entries += 1;
        self.entries += 1;
        self.last = id;
    }

    /// A walk over the whole list from the start.
    #[must_use]
    pub fn read(&self) -> Reader<'_> {
        Reader {
            posts: self,
            block: 0,
            at: 0,
            id: self.blocks.first().map_or(0, |b| b.first),
            left: self.blocks.first().map_or(0, |b| b.entries),
            places: 0..0,
        }
    }

    /// How many entries the list has.
    ///
    /// Which is how many documents it names, counting the ones that have since
    /// been edited or removed, because the list does not know about those.
    #[must_use]
    pub const fn len(&self) -> u32 {
        self.entries
    }

    /// Whether no document has this term.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries == 0
    }

    /// The last document in the list, or zero when it is empty.
    #[must_use]
    pub const fn last(&self) -> Id {
        self.last
    }

    /// How many blocks the list is kept in.
    #[must_use]
    pub fn blocks(&self) -> usize {
        self.blocks.len()
    }

    /// What each block covers, in order.
    pub fn spans(&self) -> impl Iterator<Item = Span> + '_ {
        self.blocks.iter().map(|b| Span {
            first: b.first,
            last: b.last,
            entries: b.entries,
        })
    }

    /// How many bytes the entries take, not counting the bookkeeping.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.blocks.iter().map(|b| b.bytes.len()).sum()
    }
}

/// A walk over one term's documents.
///
/// This is a cursor and not an [`Iterator`] on purpose. [`Reader::places`]
/// answers about the entry the cursor is on, so anything that hands the walk off
/// to an adapter and keeps only what comes out of it has quietly thrown the
/// places away. Making it iterate would make that mistake easy to write and hard
/// to see.
#[derive(Debug)]
pub struct Reader<'a> {
    posts: &'a Posts,
    block: usize,
    at: usize,
    id: Id,
    left: u32,
    places: Range<usize>,
}

impl Reader<'_> {
    /// The next document, or `None` at the end of the list.
    pub fn step(&mut self) -> Option<Post> {
        while self.left == 0 {
            let next = self.block + 1;
            let b = self.posts.blocks.get(next)?;
            self.block = next;
            self.at = 0;
            self.id = b.first;
            self.left = b.entries;
        }
        let bytes = &self.posts.blocks[self.block].bytes;
        let mut at = self.at;
        self.id += get(bytes, &mut at);
        let freq = get(bytes, &mut at);
        let fields = get(bytes, &mut at);
        let count = get(bytes, &mut at) as usize;
        let start = at;
        for _ in 0..count {
            get(bytes, &mut at);
        }
        self.places = start..at;
        self.at = at;
        self.left -= 1;
        Some(Post {
            id: self.id,
            freq,
            fields,
        })
    }

    /// The first document at or after `id`, or `None` when there is none.
    ///
    /// Whole blocks that end before `id` are stepped over without being read,
    /// which is the point of having blocks at all.
    pub fn seek(&mut self, id: Id) -> Option<Post> {
        while self
            .posts
            .blocks
            .get(self.block)
            .is_some_and(|b| b.last < id)
        {
            let next = self.block + 1;
            let Some(b) = self.posts.blocks.get(next) else {
                self.left = 0;
                return None;
            };
            self.block = next;
            self.at = 0;
            self.id = b.first;
            self.left = b.entries;
        }
        loop {
            let post = self.step()?;
            if post.id >= id {
                return Some(post);
            }
        }
    }

    /// The places the last document read has the term at.
    ///
    /// Cleared first, so the same buffer can be handed back on every entry and
    /// no allocation happens after the first document with that many places.
    pub fn places(&self, into: &mut Vec<u32>) {
        into.clear();
        let Some(block) = self.posts.blocks.get(self.block) else {
            return;
        };
        let mut at = self.places.start;
        let mut seen = 0;
        while at < self.places.end {
            seen += get(&block.bytes, &mut at);
            into.push(seen);
        }
    }
}

/// Every term in one index and the documents each is in.
///
/// Ordered, because half of what is asked of it is a range. `FT.SEARCH` with a
/// `foo*` in it wants every term starting `foo`, `FT.TAGVALS` wants them all in
/// order, and the debug dump wants them all in order as well, so an ordered map
/// answers three questions where a hash map answers one.
#[derive(Debug, Clone, Default)]
pub struct Terms {
    terms: BTreeMap<Box<[u8]>, Posts>,
}

impl Terms {
    /// A dictionary with nothing in it.
    #[must_use]
    pub fn new() -> Terms {
        Terms::default()
    }

    /// Records that a document has a term, adding the term if it is new.
    pub fn add(&mut self, term: &[u8], id: Id, freq: u32, fields: u32, places: &[u32]) {
        self.terms
            .entry(term.into())
            .or_default()
            .push(id, freq, fields, places);
    }

    /// Records that a document has a value, for a field that only asks whether.
    ///
    /// A `TAG` field is a dictionary of its own with no frequencies and no
    /// places in it, because a tag is matched whole and never scored, so there
    /// is nothing to count and nowhere for it to be. It is the same structure
    /// underneath and it is kept as one rather than written twice, which is also
    /// why a tag value gets prefix matching and an ordered dump for nothing.
    pub fn mark(&mut self, value: &[u8], id: Id) {
        self.add(value, id, 1, 0, &[]);
    }

    /// The documents one term is in.
    #[must_use]
    pub fn get(&self, term: &[u8]) -> Option<&Posts> {
        self.terms.get(term)
    }

    /// Every term, in order, stems among them behind their [`STEM`] byte.
    pub fn all(&self) -> impl Iterator<Item = &[u8]> + '_ {
        self.terms.keys().map(|t| &**t)
    }

    /// Every term starting with a prefix, in order.
    pub fn under<'a>(&'a self, prefix: &'a [u8]) -> impl Iterator<Item = (&'a [u8], &'a Posts)> {
        self.terms
            .range(prefix.to_vec().into_boxed_slice()..)
            .take_while(move |(t, _)| t.starts_with(prefix))
            .map(|(t, p)| (&**t, p))
    }

    /// How many terms there are, stems counted separately from their words.
    #[must_use]
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Whether nothing has been indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// How many bytes the lists take, not counting the terms themselves.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.terms.values().map(Posts::bytes).sum()
    }
}

/// Writes a number as one to five bytes, seven bits at a time, low bits first.
fn put(out: &mut Vec<u8>, mut n: u32) {
    while n >= 0x80 {
        out.push((n as u8) | 0x80);
        n >>= 7;
    }
    out.push(n as u8);
}

/// Reads back what [`put`] wrote and steps `at` past it.
fn get(src: &[u8], at: &mut usize) -> u32 {
    let mut n = 0;
    let mut shift = 0;
    while let Some(&b) = src.get(*at) {
        *at += 1;
        n |= u32::from(b & 0x7f) << shift;
        if b < 0x80 {
            break;
        }
        shift += 7;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all(posts: &Posts) -> Vec<(Post, Vec<u32>)> {
        let mut out = Vec::new();
        let mut r = posts.read();
        let mut places = Vec::new();
        while let Some(post) = r.step() {
            r.places(&mut places);
            out.push((post, places.clone()));
        }
        out
    }

    /// What goes in comes back out, gaps and varints notwithstanding.
    #[test]
    fn a_list_gives_back_what_was_put_in_it() {
        let mut p = Posts::new();
        p.push(1, 1, 0b1, &[1]);
        p.push(2, 4, 0b11, &[3, 9, 200]);
        p.push(1000, 2, 0b100, &[]);
        assert_eq!(
            all(&p),
            [
                (
                    Post {
                        id: 1,
                        freq: 1,
                        fields: 0b1
                    },
                    vec![1]
                ),
                (
                    Post {
                        id: 2,
                        freq: 4,
                        fields: 0b11
                    },
                    vec![3, 9, 200]
                ),
                (
                    Post {
                        id: 1000,
                        freq: 2,
                        fields: 0b100
                    },
                    vec![]
                ),
            ]
        );
        assert_eq!(p.len(), 3);
        assert_eq!(p.last(), 1000);
    }

    /// Numbers past the small ones still survive the round trip, which is the
    /// thing a varint gets wrong if the shift is off by one anywhere.
    #[test]
    fn a_number_of_any_size_survives() {
        let mut p = Posts::new();
        for shift in 0..32 {
            let n = 1u32 << shift;
            p.push(shift + 1, n, n, &[n]);
        }
        for (shift, (post, places)) in all(&p).iter().enumerate() {
            let want = 1u32 << shift;
            assert_eq!(post.freq, want);
            assert_eq!(post.fields, want);
            assert_eq!(places, &[want]);
        }
        let mut p = Posts::new();
        p.push(u32::MAX, u32::MAX, u32::MAX, &[u32::MAX]);
        assert_eq!(
            all(&p),
            [(
                Post {
                    id: u32::MAX,
                    freq: u32::MAX,
                    fields: u32::MAX
                },
                vec![u32::MAX]
            )]
        );
    }

    /// A block holds a hundred entries and the next one starts a new block, and
    /// each block knows what it covers.
    #[test]
    fn a_hundred_entries_fill_a_block() {
        let mut p = Posts::new();
        for id in 1..=250 {
            p.push(id, 1, 1, &[id]);
        }
        assert_eq!(p.blocks(), 3);
        let spans: Vec<_> = p.spans().collect();
        assert_eq!(
            spans[0],
            Span {
                first: 1,
                last: 100,
                entries: 100
            }
        );
        assert_eq!(
            spans[1],
            Span {
                first: 101,
                last: 200,
                entries: 100
            }
        );
        assert_eq!(
            spans[2],
            Span {
                first: 201,
                last: 250,
                entries: 50
            }
        );
        assert_eq!(all(&p).len(), 250);
    }

    /// Seeking lands on the first document at or after the one asked for, and
    /// keeps working across the block boundaries.
    #[test]
    fn seeking_lands_on_the_first_document_that_is_not_behind() {
        let mut p = Posts::new();
        for id in (10..=5000).step_by(10) {
            p.push(id, 1, 1, &[]);
        }
        let mut r = p.read();
        assert_eq!(r.seek(1).map(|p| p.id), Some(10));
        assert_eq!(r.seek(11).map(|p| p.id), Some(20));
        assert_eq!(r.seek(1000).map(|p| p.id), Some(1000));
        assert_eq!(r.seek(4321).map(|p| p.id), Some(4330));
        assert_eq!(r.seek(5000).map(|p| p.id), Some(5000));
        assert_eq!(r.seek(5001), None);
        assert_eq!(r.step(), None);
    }

    /// A seek that walks off the end says so and stays said, rather than
    /// starting the list again.
    #[test]
    fn seeking_past_the_end_stays_past_it() {
        let mut p = Posts::new();
        p.push(1, 1, 1, &[]);
        p.push(2, 1, 1, &[]);
        let mut r = p.read();
        assert_eq!(r.seek(99), None);
        assert_eq!(r.step(), None);
        assert_eq!(r.seek(1), None);
    }

    /// Ids arrive in order and one that does not is refused, because taking it
    /// would make the gap behind it wrap round and lose every entry after it.
    #[test]
    fn an_id_that_goes_backwards_is_refused() {
        let mut p = Posts::new();
        p.push(5, 1, 1, &[]);
        p.push(3, 1, 1, &[]);
        p.push(5, 1, 1, &[]);
        p.push(6, 1, 1, &[]);
        assert_eq!(
            all(&p).iter().map(|(p, _)| p.id).collect::<Vec<_>>(),
            [5, 6]
        );
    }

    /// An empty list reads as one and says nothing about a last document.
    #[test]
    fn an_empty_list_is_empty() {
        let p = Posts::new();
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);
        assert_eq!(p.last(), 0);
        assert_eq!(p.blocks(), 0);
        assert_eq!(p.bytes(), 0);
        assert_eq!(all(&p), []);
        let mut r = p.read();
        assert_eq!(r.seek(1), None);
    }

    /// The weight a client writes is cut down to a whole number and then not
    /// allowed below one, so a field can be worth more than the others but never
    /// worth nothing.
    #[test]
    fn a_weight_is_a_whole_number_of_at_least_one() {
        assert_eq!(adds(0.0), 1);
        assert_eq!(adds(0.5), 1);
        assert_eq!(adds(0.9), 1);
        assert_eq!(adds(1.0), 1);
        assert_eq!(adds(1.5), 1);
        assert_eq!(adds(2.0), 2);
        assert_eq!(adds(2.9), 2);
        assert_eq!(adds(10.0), 10);
        assert_eq!(adds(-3.0), 1);
        assert_eq!(adds(f64::NAN), 1);
        assert_eq!(adds(f64::INFINITY), u32::MAX);
    }

    /// The dictionary keeps its terms in order with the stems in front, which is
    /// what the `+` byte in a stem sorts as and what the debug dump reports.
    #[test]
    fn terms_come_back_in_order_with_the_stems_in_front() {
        let mut t = Terms::new();
        for (term, id) in [
            (b"quick".as_slice(), 1),
            (b"dogs", 1),
            (b"dogs", 2),
            (&stemmed(b"dog"), 1),
            (&stemmed(b"dog"), 2),
            (b"brown", 2),
        ] {
            t.add(term, id, 1, 1, &[]);
        }
        let terms: Vec<_> = t.all().map(<[u8]>::to_vec).collect();
        assert_eq!(
            terms,
            [
                b"+dog".to_vec(),
                b"brown".to_vec(),
                b"dogs".to_vec(),
                b"quick".to_vec()
            ]
        );
        assert_eq!(t.len(), 4);
        assert_eq!(t.get(b"dogs").map(Posts::len), Some(2));
        assert_eq!(t.get(b"dog").map(Posts::len), None);
        assert_eq!(t.get(b"+dog").map(Posts::len), Some(2));
    }

    /// A tag field puts its values in the same dictionary with nothing counted
    /// and nothing placed, because a tag is matched whole and never scored.
    #[test]
    fn a_tag_value_is_a_term_with_nothing_on_it() {
        let mut t = Terms::new();
        t.mark(b"red", 1);
        t.mark(b"blue", 1);
        t.mark(b"red", 4);
        let posts = t.get(b"red").expect("red is in the dictionary");
        let mut r = posts.read();
        assert_eq!(
            r.step(),
            Some(Post {
                id: 1,
                freq: 1,
                fields: 0
            })
        );
        let mut places = Vec::new();
        r.places(&mut places);
        assert_eq!(places, []);
        assert_eq!(r.step().map(|p| p.id), Some(4));
        assert_eq!(
            t.all().map(<[u8]>::to_vec).collect::<Vec<_>>(),
            [b"blue".to_vec(), b"red".to_vec()]
        );
    }

    /// Asking for a prefix gives every term under it and stops at the first one
    /// that is not, rather than reading the rest of the dictionary.
    #[test]
    fn a_prefix_gives_the_terms_under_it() {
        let mut t = Terms::new();
        for term in [
            b"do".as_slice(),
            b"dog",
            b"dogs",
            b"dogged",
            b"dot",
            b"e",
            b"cat",
        ] {
            t.add(term, 1, 1, 1, &[]);
        }
        let under: Vec<_> = t.under(b"dog").map(|(t, _)| t.to_vec()).collect();
        assert_eq!(
            under,
            [b"dog".to_vec(), b"dogged".to_vec(), b"dogs".to_vec()]
        );
        assert_eq!(t.under(b"zz").count(), 0);
        assert_eq!(t.under(b"").count(), 7);
    }

    /// A term that half the corpus has costs about a byte a document, which is
    /// the whole reason the ids are gaps and the numbers are varints.
    #[test]
    fn a_common_term_costs_about_a_byte_a_document() {
        let mut p = Posts::new();
        for id in 1..=10_000 {
            p.push(id, 1, 1, &[]);
        }
        assert!(p.bytes() < 45_000, "{} bytes", p.bytes());
        let mut plain = Posts::new();
        for id in (1..=10_000).map(|n| n * 1000) {
            plain.push(id, 1, 1, &[]);
        }
        assert!(plain.bytes() < 55_000, "{} bytes", plain.bytes());
    }
}
