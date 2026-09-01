//! Equality indexes over a path into a document (`09` section 4).
//!
//! A collection can find a document by its id already, because the primary
//! table is keyed by it. An index is what makes it findable by what is inside
//! it: one element table per indexed path, keyed by the value at that path,
//! holding the ids of the documents that have it.
//!
//! ```
//! use yo_doc::{Builder, Docs, Key};
//!
//! let mut docs = Docs::new();
//! docs.create_index("$.status")?;
//! for (id, status) in [("a", "open"), ("b", "shut"), ("c", "open")] {
//!     let mut b = Builder::new();
//!     b.begin_object()?;
//!     b.key(b"status")?;
//!     b.text(status)?;
//!     b.end_object()?;
//!     let bytes = b.finish()?.to_vec();
//!     docs.put_bytes(id.as_bytes(), &bytes)?;
//! }
//!
//! let mut found = Vec::new();
//! docs.find("$.status", &Key::text("open"), |id, _| found.push(id.to_vec()))?;
//! found.sort();
//! assert_eq!(found, [b"a".to_vec(), b"c".to_vec()]);
//! # Ok::<(), yo_common::Error>(())
//! ```
//!
//! # It is the same code again
//!
//! The table from key to posting list is [`Elements`], which is a hash's field
//! table. The posting list is [`Set`], which is a Redis set, so a key that one
//! document has costs a listpack entry rather than a hash table, a key that a
//! million documents have is a partitioned element table, and a collection
//! whose ids are numbers gets an intset and eight bytes a posting. None of that
//! was written for this.
//!
//! It also means intersecting two indexes is `SINTER`, on the same sets, with
//! the same code, at the same speed. That is the whole of `09` section 5's
//! "probe each equality index, intersect the smallest result first", and there
//! is nothing to build for it.
//!
//! # What a key is
//!
//! [`Key`] is the value at the path with a tag byte in front of it, so a
//! document with the string `"7"` at a path and one with the number seven do
//! not land on the same key. Numbers are stored big endian with the sign bit
//! flipped, which orders them correctly as bytes. Nothing here needs that
//! ordering, but the ordered index does and it costs nothing to have it now.
//!
//! An integer and a float that names the same integer, `7` and `7.0`, get the
//! same key. A caller asking for seven means seven, and JSON has one number
//! type, so the alternative is a query that misses documents for a reason
//! nobody can see.
//!
//! Where this encoding is not enough is ordering across the two number tags: a
//! float that is not a whole number sorts after every integer rather than among
//! them. Equality does not care and the ordered index is not an element table,
//! so it will carry its own key encoding rather than stretch this one.
//!
//! # What is not indexed
//!
//! A path that lands on an object or an array puts nothing in an equality
//! index, and a document that has no value at the path puts nothing in either.
//! Both are absences rather than errors: an equality index answers which
//! documents have a given scalar there, and neither of those documents does.
//! Indexing each element of an array is the separate `array` kind.

use yo_common::num::i64_digits;
use yo_common::small::Small;
use yo_common::{Code, Error, Result};
use yo_kv::{Elements, Set, SetLimits, Slab};

use crate::head::Kind;
use crate::read::Value;

/// The longest an index key may be, which is the longest name an element table
/// takes.
///
/// A text value past this cannot be filed, and a write that would have to file
/// one fails rather than storing a document the index will never find. A silent
/// absence from an index is a query that returns the wrong answer with no way
/// to tell, and that is worse than a write that says no.
pub const KEY_MAX: usize = yo_kv::NAME_MAX - 1;

/// How much of a key sits in the caller's frame before it needs the allocator.
///
/// A tag and eight bytes covers every number, every boolean and null, and a tag
/// and thirty one bytes covers the short strings that get indexed in practice:
/// a status, a country, an identifier.
const KEY_INLINE: usize = 32;

const TAG_NULL: u8 = 0;
const TAG_FALSE: u8 = 1;
const TAG_TRUE: u8 = 2;
const TAG_INT: u8 = 3;
const TAG_FLOAT: u8 = 4;
const TAG_TEXT: u8 = 5;

/// A value as an index looks it up.
///
/// Built from what the caller is searching for, or from what was found at a
/// path in a document being written. The two go through the same code on
/// purpose, because a query that encodes its argument differently from the way
/// the write encoded the document is a query that finds nothing and says
/// nothing about why.
#[derive(Clone)]
pub struct Key(Small<u8, KEY_INLINE>);

impl Key {
    /// The key for `null`.
    #[must_use]
    pub fn null() -> Key {
        Key(Small::collect([TAG_NULL]))
    }

    /// The key for a boolean.
    #[must_use]
    pub fn bool(v: bool) -> Key {
        Key(Small::collect([if v { TAG_TRUE } else { TAG_FALSE }]))
    }

    /// The key for an integer.
    #[must_use]
    pub fn int(v: i64) -> Key {
        let mut k = Small::collect([TAG_INT]);
        for b in order_int(v) {
            k.push(b);
        }
        Key(k)
    }

    /// The key for a float.
    ///
    /// A float that names a whole number gets the same key as that integer, so
    /// `7.0` and `7` are one key and a search for either finds both.
    #[must_use]
    pub fn float(v: f64) -> Key {
        if let Some(n) = whole(v) {
            return Key::int(n);
        }
        let mut k = Small::collect([TAG_FLOAT]);
        for b in order_float(v) {
            k.push(b);
        }
        Key(k)
    }

    /// The key for a string.
    #[must_use]
    pub fn text(v: &str) -> Key {
        Key::text_bytes(v.as_bytes())
    }

    /// The key for a string that is already bytes.
    #[must_use]
    pub fn text_bytes(v: &[u8]) -> Key {
        let mut k = Small::collect([TAG_TEXT]);
        for &b in v {
            k.push(b);
        }
        Key(k)
    }

    /// The key for a value found in a document, or `None` if it is a container
    /// and so has no equality key.
    #[must_use]
    pub fn of(v: Value<'_>) -> Option<Key> {
        match v.kind() {
            Kind::Null => Some(Key::null()),
            Kind::Bool => Some(Key::bool(v.as_bool()?)),
            Kind::Int => Some(Key::int(v.as_int()?)),
            Kind::Float => Some(Key::float(v.as_float()?)),
            Kind::Text => Some(Key::text_bytes(v.text_bytes()?)),
            Kind::Array | Kind::Object => None,
        }
    }

    /// The bytes this is filed under.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        self.0.as_slice()
    }

    /// Whether this key is too long to file.
    #[must_use]
    pub fn is_too_long(&self) -> bool {
        self.as_bytes().len() > KEY_MAX
    }
}

impl PartialEq for Key {
    fn eq(&self, other: &Key) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl Eq for Key {}

impl core::fmt::Debug for Key {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let b = self.as_bytes();
        match b.first() {
            Some(&TAG_NULL) => f.write_str("null"),
            Some(&TAG_FALSE) => f.write_str("false"),
            Some(&TAG_TRUE) => f.write_str("true"),
            Some(&TAG_TEXT) => write!(f, "{:?}", String::from_utf8_lossy(&b[1..])),
            Some(&TAG_INT) | Some(&TAG_FLOAT) => write!(f, "{}", Hex(&b[1..])),
            _ => f.write_str("<no key>"),
        }
    }
}

/// Bytes as hex, for the numbers a key holds in an order preserving form that
/// is not worth decoding back just to print it.
struct Hex<'a>(&'a [u8]);

impl core::fmt::Display for Hex<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for b in self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

/// An integer as bytes that sort the way the integer does.
///
/// Big endian puts the most significant byte first, and flipping the sign bit
/// moves the negatives below the positives, which is what two's complement gets
/// wrong when it is read as unsigned.
fn order_int(v: i64) -> [u8; 8] {
    ((v as u64) ^ (1 << 63)).to_be_bytes()
}

/// A float as bytes that sort the way the float does.
///
/// IEEE 754 is already ordered correctly for the positives once the sign bit is
/// set, and the negatives are ordered backwards, so they get every bit flipped
/// instead. This is the standard trick and it is exact: no value maps onto
/// another one.
fn order_float(v: f64) -> [u8; 8] {
    let bits = v.to_bits();
    let flipped = if bits & (1 << 63) != 0 {
        !bits
    } else {
        bits | (1 << 63)
    };
    flipped.to_be_bytes()
}

/// The integer a float names exactly, if it names one.
///
/// `as` saturates rather than wrapping, so the check on the way back catches
/// anything out of range as well as anything with a fraction.
fn whole(v: f64) -> Option<i64> {
    if !v.is_finite() {
        return None;
    }
    let n = v as i64;
    if n as f64 == v { Some(n) } else { None }
}

/// One index, over one path.
#[derive(Debug)]
pub struct PathIndex {
    /// The path as it was written, kept whole so it can be parsed again per
    /// lookup. Parsing is a scan of a dozen bytes and it saves an owned step
    /// type that would have to be kept in step with [`crate::Steps`].
    path: Box<[u8]>,
    /// The key to the slab slot its posting list sits in.
    keys: Elements<u32>,
    /// The posting lists. A slab rather than a payload beside the row, because
    /// [`Elements`] moves its last row into the hole on a removal and a posting
    /// list is not `Copy`.
    posts: Slab<Set>,
    /// How many document ids are filed altogether, over every key.
    postings: usize,
}

impl PathIndex {
    /// An empty index over `path`, which has already been checked to parse.
    pub(crate) fn new(path: &[u8]) -> PathIndex {
        PathIndex {
            path: path.into(),
            keys: Elements::new(),
            posts: Slab::new(),
            postings: 0,
        }
    }

    /// The path this indexes.
    #[must_use]
    pub fn path(&self) -> &[u8] {
        &self.path
    }

    /// How many distinct values are filed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether nothing is filed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// How many document ids are filed altogether.
    ///
    /// One per document that has a scalar at this path, so the difference
    /// between this and the collection's length is how many documents the index
    /// does not cover.
    #[must_use]
    pub fn postings(&self) -> usize {
        self.postings
    }

    /// The documents filed under `key`.
    ///
    /// A [`Set`], so it can be intersected with another one by the same code
    /// `SINTER` uses.
    #[must_use]
    pub fn get(&self, key: &Key) -> Option<&Set> {
        self.posts.get(*self.keys.get(key.as_bytes())?)
    }

    /// How many documents are filed under `key`.
    ///
    /// The number a query planner sorts its filters by, and it is a probe
    /// rather than a walk.
    #[must_use]
    pub fn count(&self, key: &Key) -> usize {
        self.get(key).map_or(0, Set::len)
    }

    /// File `id` under `key`.
    pub(crate) fn add(&mut self, key: &[u8], id: &[u8]) -> Result<()> {
        if let Some(&slot) = self.keys.get(key) {
            let set = self.posts.get_mut(slot).expect("a row points at its list");
            if set.add(id, &SetLimits::DEFAULT) {
                self.postings += 1;
            }
            return Ok(());
        }
        let mut set = Set::new();
        set.add(id, &SetLimits::DEFAULT);
        let slot = self.posts.insert(set);
        if self.keys.insert(key, slot).is_err() {
            self.posts.remove(slot);
            return Err(Error::new(
                Code::Full,
                "the index cannot hold another distinct value",
            ));
        }
        self.postings += 1;
        Ok(())
    }

    /// Take `id` out from under `key`, and drop the key if it was the last one.
    pub(crate) fn take(&mut self, key: &[u8], id: &[u8]) {
        let Some(&slot) = self.keys.get(key) else {
            return;
        };
        let set = self.posts.get_mut(slot).expect("a row points at its list");
        if !set.remove(id) {
            return;
        }
        self.postings -= 1;
        if set.is_empty() {
            self.posts.remove(slot);
            self.keys.remove(key);
        }
    }

    /// Throw everything filed away and keep the path.
    pub(crate) fn clear(&mut self) {
        self.keys.clear();
        self.posts.clear();
        self.postings = 0;
    }

    /// What the index costs, posting lists included.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.keys.memory_bytes()
            + self.posts.slot_bytes()
            + self.posts.iter().map(Set::memory_bytes).sum::<usize>()
    }
}

/// Hand every id in `set` to `f` as bytes.
///
/// A posting list of numeric ids is an intset, so the ids come back as integers
/// and have to be written out again to be probed with. The digits go in a
/// buffer on this frame, so a walk over a million postings allocates nothing.
pub(crate) fn each_id(set: &Set, mut f: impl FnMut(&[u8])) -> usize {
    let mut digits = [0u8; yo_common::num::DIGITS_MAX];
    let mut n = 0usize;
    for member in set.iter() {
        match member {
            yo_kv::listpack::Entry::Str(s) => f(s),
            yo_kv::listpack::Entry::Int(v) => f(i64_digits(&mut digits, v)),
        }
        n += 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_number_and_the_string_of_it_are_different_keys() {
        assert_ne!(Key::int(7), Key::text("7"));
        assert_ne!(Key::null(), Key::text(""));
        assert_ne!(Key::bool(true), Key::int(1));
    }

    #[test]
    fn a_float_that_names_a_whole_number_is_that_number() {
        assert_eq!(Key::float(7.0), Key::int(7));
        assert_eq!(Key::float(-0.0), Key::int(0));
        assert_eq!(Key::float(-3.0), Key::int(-3));
        assert_ne!(Key::float(7.5), Key::int(7));
        assert_ne!(Key::float(1e30), Key::int(i64::MAX));
        assert_ne!(Key::float(f64::NAN), Key::float(0.0));
    }

    #[test]
    fn numbers_sort_as_bytes_the_way_they_sort_as_numbers() {
        let mut ns = [0i64, -1, i64::MIN, i64::MAX, 7, -7, 1 << 40];
        let mut keys: Vec<Key> = ns.iter().map(|&n| Key::int(n)).collect();
        ns.sort_unstable();
        keys.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        let want: Vec<Key> = ns.iter().map(|&n| Key::int(n)).collect();
        assert_eq!(keys, want);

        let mut fs = [0.5f64, -0.5, -1.5, 1e300, -1e300, f64::MIN_POSITIVE];
        let mut keys: Vec<Key> = fs.iter().map(|&f| Key::float(f)).collect();
        fs.sort_by(f64::total_cmp);
        keys.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        let want: Vec<Key> = fs.iter().map(|&f| Key::float(f)).collect();
        assert_eq!(keys, want);
    }

    #[test]
    fn a_short_key_stays_off_the_heap() {
        assert!(Key::int(i64::MIN).0.is_inline());
        assert!(Key::text("a-fairly-ordinary-status").0.is_inline());
        assert!(!Key::text(&"x".repeat(64)).0.is_inline());
    }

    #[test]
    fn a_key_prints_as_what_it_is() {
        assert_eq!(format!("{:?}", Key::null()), "null");
        assert_eq!(format!("{:?}", Key::bool(true)), "true");
        assert_eq!(format!("{:?}", Key::text("open")), "\"open\"");
        assert_eq!(format!("{:?}", Key::int(0)), "8000000000000000");
    }

    #[test]
    fn the_last_document_under_a_key_takes_the_key_with_it() {
        let mut index = PathIndex::new(b"$.status");
        let open = Key::text("open");
        index.add(open.as_bytes(), b"a").expect("room");
        index.add(open.as_bytes(), b"b").expect("room");
        assert_eq!(index.len(), 1);
        assert_eq!(index.postings(), 2);
        assert_eq!(index.count(&open), 2);

        index.take(open.as_bytes(), b"a");
        assert_eq!(index.postings(), 1);
        assert_eq!(index.len(), 1);
        index.take(open.as_bytes(), b"b");
        assert_eq!(index.postings(), 0);
        assert!(index.is_empty(), "an empty posting list is not a key");
        assert_eq!(index.count(&open), 0);
    }

    #[test]
    fn filing_the_same_document_twice_files_it_once() {
        let mut index = PathIndex::new(b"$.status");
        let open = Key::text("open");
        index.add(open.as_bytes(), b"a").expect("room");
        index.add(open.as_bytes(), b"a").expect("room");
        assert_eq!(index.postings(), 1);
        index.take(open.as_bytes(), b"a");
        assert_eq!(index.postings(), 0);
    }

    #[test]
    fn taking_out_something_that_was_never_filed_changes_nothing() {
        let mut index = PathIndex::new(b"$.status");
        let open = Key::text("open");
        index.add(open.as_bytes(), b"a").expect("room");
        index.take(open.as_bytes(), b"never");
        index.take(Key::text("shut").as_bytes(), b"a");
        assert_eq!(index.postings(), 1);
        assert_eq!(index.count(&open), 1);
    }

    #[test]
    fn a_posting_list_of_numbers_reads_back_as_bytes() {
        let mut index = PathIndex::new(b"$.customer");
        let key = Key::int(4);
        for id in ["11", "2", "333"] {
            index.add(key.as_bytes(), id.as_bytes()).expect("room");
        }
        let mut got = Vec::new();
        let n = each_id(index.get(&key).expect("filed"), |id| {
            got.push(String::from_utf8_lossy(id).into_owned());
        });
        assert_eq!(n, 3);
        got.sort();
        assert_eq!(got, ["11", "2", "333"]);
    }
}
