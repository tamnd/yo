//! Indexes over a path into a document (`09` sections 4 and 5).
//!
//! A collection can find a document by its id already, because the primary
//! table is keyed by it. An index is what makes it findable by what is inside
//! it: one element table per indexed path, keyed by the value at that path,
//! holding the ids of the documents that have it.
//!
//! An index answers equality, and an ordered one answers ranges as well. The
//! array and text kinds file a document under more than one key at a time,
//! every element of an array or every word of a string, and are asked the same
//! question an equality index is.
//!
//! ```
//! use std::ops::Bound;
//! use yo_doc::{Builder, Docs, Key};
//!
//! let mut docs = Docs::new();
//! docs.create_index("$.status")?;
//! docs.create_ordered_index("$.price")?;
//! docs.create_text_index("$.name")?;
//! for (id, status, price, name) in [
//!     ("a", "open", 30, "A red bicycle"),
//!     ("b", "shut", 10, "a blue kite"),
//!     ("c", "open", 20, "A red kite"),
//! ] {
//!     let mut b = Builder::new();
//!     b.begin_object()?;
//!     b.key(b"status")?;
//!     b.text(status)?;
//!     b.key(b"price")?;
//!     b.int(price)?;
//!     b.key(b"name")?;
//!     b.text(name)?;
//!     b.end_object()?;
//!     let bytes = b.finish()?.to_vec();
//!     docs.put_bytes(id.as_bytes(), &bytes)?;
//! }
//!
//! let mut found = Vec::new();
//! docs.find("$.status", &Key::text("open"), |id, _| found.push(id.to_vec()))?;
//! found.sort();
//! assert_eq!(found, [b"a".to_vec(), b"c".to_vec()]);
//!
//! // Cheapest first, up to and including twenty.
//! let mut upto = Vec::new();
//! docs.range("$.price", Bound::Unbounded, Bound::Included(&Key::int(20)), |id, _| {
//!     upto.push(id.to_vec())
//! })?;
//! assert_eq!(upto, [b"b".to_vec(), b"c".to_vec()]);
//!
//! // One word out of the name, with the case folded on both sides.
//! let red = Key::word("RED").expect("one word");
//! let mut kites = Vec::new();
//! docs.find("$.name", &red, |id, _| kites.push(id.to_vec()))?;
//! kites.sort();
//! assert_eq!(kites, [b"a".to_vec(), b"c".to_vec()]);
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
//! The order an ordered index walks is the counted B+ tree from `08` section 5,
//! which is what a sorted set ranks with. That tree holds row numbers and asks
//! the caller to compare, so it took no changes at all to put index keys under
//! it instead of zset members. It costs about three bytes per distinct value,
//! and a range is one descent and then a link hop per leaf, so the cost of a
//! range is the size of the answer rather than the size of the collection.
//!
//! The key table stays unordered either way, because it is a hash's field table
//! and a hash is not ordered. The order is a separate structure over its row
//! numbers, which is the same split the sorted set makes rather than a second
//! design.
//!
//! # What a key is
//!
//! [`Key`] is the value at the path with a tag byte in front of it, so a
//! document with the string `"7"` at a path and one with the number seven do
//! not land on the same key. Every key is written so that comparing two of them
//! as bytes gives the same answer comparing the values would. Equality does not
//! need that, but it means the tree can compare keys with `memcmp` and never
//! decode one.
//!
//! There is one tag for numbers rather than one for integers and one for
//! floats, because a range over a path holding both has to put them in one
//! order, and two tags cannot. Every finite number is a mantissa times a power
//! of two, so a numeric key is written as where its leading bit sits, which is
//! `floor(log2(|v|)) + 1` and is called the place here, followed by the mantissa
//! shifted up to the top of eight bytes. Two numbers with different places are
//! ordered by the place alone, and two with the same place are ordered by the
//! mantissa read from the leading bit down, which is what the shift lines up.
//! The place is biased by 32768 so that the whole range a f64 can reach sorts as
//! an unsigned number, and everything after the class byte is flipped for a
//! negative, because a bigger magnitude there is a smaller number.
//!
//! The byte in front of the place is the class, which is one of negative
//! infinity, negative, zero, positive, positive infinity and NaN. Those five
//! that are not an ordinary finite value have no size worth writing, so they
//! carry a place and a mantissa of zero and are ordered by the class alone. NaN
//! sorts above everything rather than being refused, so a range never has to
//! think about it.
//!
//! An integer and a float that names the same integer, `7` and `7.0`, get the
//! same key, because both normalise to a leading `111` and a place of three. A
//! caller asking for seven means seven, and JSON has one number type, so the
//! alternative is a query that misses documents for a reason nobody can see.
//!
//! Types do not interleave either: everything with a smaller tag sorts before
//! everything with a larger one, so nulls, then booleans, then numbers, then
//! strings. That one is on purpose. A range over a path is a range over one
//! type, and a total order across types has to pick an arbitrary answer to
//! whether a string is above or below a number.
//!
//! # What a probe costs
//!
//! G15 is that finding a document by a value at an indexed path costs what
//! `HGET` costs, and the reason to expect that is the section above: the index
//! is a hash's field table and a probe of it is a probe of one. `benches/yojb`
//! runs both against the same records so the claim is a ratio. On a 13900K with
//! nothing else running, nanoseconds:
//!
//! ```text
//!                        1024 docs    16384 docs
//!   HGET                      23.5          28.1
//!   HGET, 12 byte field       27.2          29.3
//!   probe                     38.9          41.0
//!     of which the key        10.2          12.1
//!     so the lookup           28.7          28.8
//!   find                      65.5          93.0
//! ```
//!
//! The two `HGET` rows are the same call against a different field width, and
//! the second one is there because the first is not a fair comparison. An index
//! key for an integer is twelve bytes, and the hash fields in this bench are the
//! decimal of the loop counter, so at most five. Measuring the index against the
//! short one charges it for the longer key and calls the difference index
//! overhead, which it is not, because a hash asked for a twelve byte field pays
//! exactly the same thing.
//!
//! `probe` is [`PathIndex::count`] with the path already resolved, which is the
//! index lookup plus the key encoding, and `key` is the encoding on its own.
//! Take one off the other and the lookup is 28.7 and 28.8 against 27.2 and 29.3
//! for the hash at the same key width. That is the gate: the probe is one probe
//! and it costs what a probe costs. It is also flat where `HGET` is not, which
//! is the table growing from 1024 to 16384 entries showing up on one side and
//! not the other, and at the larger size the index is the faster of the two.
//!
//! `find` is the whole call a caller makes and it is more than one probe by
//! construction: it looks the path up by name, encodes the key, probes the
//! index, and then probes the primary table with the id it got back. The second
//! probe is a document read, and `HGET` does not need one because the value it
//! returns is in the slot it just found. So an indexed equality that hands back
//! the document is two probes, and only the first of them is the index.
//!
//! The key encoding used to be the largest single piece of this, at 15.6 ns on
//! an M-series laptop where the whole probe was 28.2, because it pushed twelve
//! bytes into a [`Small`] one at a time and every push re-reads which variant
//! the list is on. It is built in a local array and copied in one go now, which
//! took that to 3.1 ns and the probe to 10.6.
//!
//! # More than one key at a time
//!
//! An array index files a document under every element of the array at the
//! path, and a text index under every word of the string. Both answer the same
//! question an equality index does, so [`PathIndex::find`] and
//! [`PathIndex::count`] do not know which kind they are on, and the only thing
//! that changes is how many keys a document has.
//!
//! A scalar at the path of an array index is an array of one. A collection
//! where some documents carry a list of tags and some carry a single tag is a
//! real collection, and an index that filed one and not the other would miss
//! documents for a reason nobody can see.
//!
//! A text index folds case, so a search has to fold it too, and [`Key::word`]
//! is what does that on the query side. Everything that is not a letter or a
//! digit is a separator. That is a word index and not a search engine: there is
//! no ranking, no stemming and no phrase matching, and the ranking that belongs
//! on top of it is `10`. Splitting on bytes is also wrong for a language that
//! does not put spaces between words, and the answer there is a real tokeniser
//! rather than a rule here that is subtly wrong in another way.
//!
//! # What is not indexed
//!
//! A path that lands on an object or an array puts nothing in an equality
//! index, and a document that has no value at the path puts nothing in any
//! kind. Both are absences rather than errors: an index answers which documents
//! have a given value there, and neither of those documents does.
//!
//! A path that lands on something other than a string puts nothing in a text
//! index. A number has no words in it, and filing `7` under the key `7` in a
//! text index would make one kind quietly behave like another.

use core::cmp::Ordering;
use core::ops::Bound;

use yo_common::num::i64_digits;
use yo_common::small::Small;
use yo_common::{Code, Error, Result};
use yo_kv::{Elements, Rank, Set, SetLimits, Slab, rank};

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
/// Twelve bytes covers every number, one byte covers a boolean or a null, and a
/// tag and thirty one bytes covers the short strings that get indexed in
/// practice: a status, a country, an identifier.
const KEY_INLINE: usize = 32;

const TAG_NULL: u8 = 0;
const TAG_FALSE: u8 = 1;
const TAG_TRUE: u8 = 2;
const TAG_NUM: u8 = 3;
const TAG_TEXT: u8 = 4;

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
        let (neg, mant) = if v < 0 {
            (true, v.unsigned_abs())
        } else {
            (false, v as u64)
        };
        number(Class::of(neg, mant == 0), mant, i32::from(bits(mant)))
    }

    /// The key for a float.
    ///
    /// A float and an integer that name the same number get the same key, so
    /// `7.0` and `7` are one key and a search for either finds both.
    #[must_use]
    pub fn float(v: f64) -> Key {
        if v.is_nan() {
            return number(Class::Nan, 0, 0);
        }
        if v.is_infinite() {
            return number(
                if v.is_sign_negative() {
                    Class::NegInf
                } else {
                    Class::PosInf
                },
                0,
                0,
            );
        }
        let raw = v.to_bits();
        let neg = raw >> 63 == 1;
        let exponent = ((raw >> 52) & 0x7ff) as i32;
        let fraction = raw & ((1 << 52) - 1);
        // A subnormal has no implied leading one and a fixed exponent, and
        // everything else has both.
        let (mant, scale) = if exponent == 0 {
            (fraction, -1074)
        } else {
            (fraction | (1 << 52), exponent - 1075)
        };
        number(
            Class::of(neg, mant == 0),
            mant,
            scale + i32::from(bits(mant)),
        )
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
        k.extend_from_slice(v);
        Key(k)
    }

    /// The key one word is filed under in a text index, or `None` if this is
    /// not one word.
    ///
    /// A search against a text index goes through this rather than
    /// [`Key::text`], because a text index folds case when it files a document
    /// and a search that does not fold it finds nothing and says nothing about
    /// why. Anything that is not letters and digits is a separator, so a phrase
    /// is two words and answers `None`: matching one is a search this index
    /// cannot answer on its own, rather than a search for the first word.
    #[must_use]
    pub fn word(v: &str) -> Option<Key> {
        let mut rest = v.as_bytes();
        let word = next_word(&mut rest)?;
        if next_word(&mut rest).is_some() {
            return None;
        }
        Some(fold(word))
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
            Some(&TAG_NUM) => write!(f, "{}", Hex(&b[1..])),
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

/// Where a number sits in the order, before its size is looked at.
///
/// The class is the first byte of a numeric key, so the five kinds of number
/// that are not an ordinary finite value each land somewhere fixed rather than
/// being encoded into the same bytes the finite ones use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Class {
    NegInf = 0,
    Negative = 1,
    Zero = 2,
    Positive = 3,
    PosInf = 4,
    /// Not a number, which JSON has no way to write and a document can only get
    /// from a program that put one there. It sorts above everything rather than
    /// being refused, so a range never has to think about it.
    Nan = 5,
}

impl Class {
    fn of(neg: bool, zero: bool) -> Class {
        match (zero, neg) {
            (true, _) => Class::Zero,
            (false, true) => Class::Negative,
            (false, false) => Class::Positive,
        }
    }
}

/// How many bits a magnitude takes.
fn bits(mant: u64) -> u16 {
    (64 - mant.leading_zeros()) as u16
}

/// A number as bytes that sort the way the number does, whether it arrived as
/// an integer or as a float.
///
/// Every finite number is `mantissa * 2^k` for some odd mantissa, so `place` is
/// where its leading bit sits, which is `floor(log2(|v|)) + 1`. Two numbers with
/// different `place` are ordered by it alone, and two with the same `place` are
/// ordered by their mantissas read from the leading bit down. Lining the
/// mantissa up to the top of eight bytes is what makes that a byte comparison,
/// and it is also what makes `7` and `7.0` the same bytes: both normalise to a
/// leading `111` and a `place` of three, whatever they looked like on the way
/// in.
///
/// Negatives get the ten bytes after the class flipped, because a bigger
/// magnitude is a smaller number.
fn number(class: Class, mant: u64, place: i32) -> Key {
    let (place, mant) = match class {
        // The size of an infinity, a zero or a NaN is not a question, and
        // writing it as zero keeps every numeric key the same width.
        Class::Negative | Class::Positive => (place, mant << mant.leading_zeros()),
        _ => (0, 0),
    };
    // Biased so that the whole range a f64 can reach, which is roughly -1074 to
    // 1025, is an unsigned number that sorts the way the signed one does.
    let place = ((place + 32768) as u16).to_be_bytes();
    let flip = if class == Class::Negative { 0xff } else { 0 };
    // Written into a local array and copied in one go rather than pushed a byte
    // at a time. Every numeric key is these twelve bytes, the length is known
    // before the first one is written, and a push has to re-read which variant
    // the list is on each time round.
    let mut k = [TAG_NUM, class as u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    k[2..4].copy_from_slice(&place);
    k[4..].copy_from_slice(&mant.to_be_bytes());
    for b in &mut k[2..] {
        *b ^= flip;
    }
    Key(Small::from_slice(&k))
}

/// A float as bytes that sort the way the float does.
/// What an index can be asked, and how many keys a document gets at its path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexKind {
    /// One value at a time. A table from key to posting list and nothing else.
    Equality,
    /// One value at a time, or every value between two of them. The same table
    /// with a counted B+ tree over its rows.
    Ordered,
    /// One element of an array at a time. A document with `["red", "blue"]` at
    /// the path is filed under both, so a search for either finds it.
    Array,
    /// One word of a string at a time, folded to lower case. A document with
    /// `"A red bicycle"` at the path is filed under `a`, `red` and `bicycle`.
    Text,
}

impl IndexKind {
    /// Whether this kind can be asked for a range as well as for a value.
    #[must_use]
    pub fn is_ordered(self) -> bool {
        self == IndexKind::Ordered
    }

    /// Whether a document can be filed under more than one key at a time.
    #[must_use]
    pub fn is_multi(self) -> bool {
        matches!(self, IndexKind::Array | IndexKind::Text)
    }
}

/// A value at an indexed path that cannot be a key, because it is longer than
/// [`KEY_MAX`].
///
/// Carried back rather than turned into an error here, so the layer that knows
/// which path and which document it was can say so.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TooLong;

/// Append every key `at` files under, as a list of one length byte pair and
/// then that many bytes.
///
/// Length prefixed rather than one buffer per key, because an array index files
/// a document under as many keys as the array is long and a write is not
/// allowed to allocate per element.
///
/// A path that lands on nothing this kind can use puts nothing in the list.
/// That is an absence and not an error: an index answers which documents have a
/// given value at the path, and a document with an object there does not have
/// one.
pub(crate) fn keys_at(
    kind: IndexKind,
    at: Value<'_>,
    out: &mut Vec<u8>,
) -> core::result::Result<(), TooLong> {
    match kind {
        IndexKind::Equality | IndexKind::Ordered => {
            if let Some(key) = Key::of(at) {
                push_key(&key, out)?;
            }
        }
        IndexKind::Array => match at.kind() {
            // A scalar at the path is an array of one. A caller that files
            // `["red"]` on one document and `"red"` on the next means the same
            // thing by both, and an index that disagreed would be a query that
            // misses documents for a reason nobody can see.
            Kind::Array => {
                for elem in at.iter() {
                    if let Some(key) = Key::of(elem) {
                        push_key(&key, out)?;
                    }
                }
            }
            Kind::Object => {}
            _ => {
                if let Some(key) = Key::of(at) {
                    push_key(&key, out)?;
                }
            }
        },
        IndexKind::Text => {
            if let Some(text) = at.text_bytes() {
                let mut rest = text;
                while let Some(word) = next_word(&mut rest) {
                    push_key(&fold(word), out)?;
                }
            }
        }
    }
    Ok(())
}

/// Put one key on the end of a key list.
fn push_key(key: &Key, out: &mut Vec<u8>) -> core::result::Result<(), TooLong> {
    let bytes = key.as_bytes();
    if key.is_too_long() {
        return Err(TooLong);
    }
    // The length fits two bytes because KEY_MAX does, and the check above ran.
    let n = bytes.len() as u16;
    out.extend_from_slice(&n.to_le_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}

/// Walk a key list back out again.
pub(crate) fn each_key(mut list: &[u8], mut f: impl FnMut(&[u8])) {
    while list.len() >= 2 {
        let n = usize::from(u16::from_le_bytes([list[0], list[1]]));
        let Some(key) = list.get(2..2 + n) else {
            return;
        };
        f(key);
        list = &list[2 + n..];
    }
}

/// The next run of letters and digits in `rest`, with `rest` left after it.
///
/// Everything else is a separator, so punctuation, spaces and the bytes of a
/// multi byte character all split. Splitting inside a word of a language that
/// does not use spaces is wrong, and a real tokeniser is the answer rather than
/// a rule here that is subtly wrong in a different way, so `10` will bring one.
/// For the ASCII text that gets a text index today this is what a caller means.
/// One word as the key a text index files it under, folded to lower case.
fn fold(word: &[u8]) -> Key {
    Key(Small::collect(
        core::iter::once(TAG_TEXT).chain(word.iter().map(u8::to_ascii_lowercase)),
    ))
}

fn next_word<'a>(rest: &mut &'a [u8]) -> Option<&'a [u8]> {
    let start = rest.iter().position(|b| b.is_ascii_alphanumeric())?;
    let after = rest[start..]
        .iter()
        .position(|b| !b.is_ascii_alphanumeric())
        .map_or(rest.len(), |n| start + n);
    let word = &rest[start..after];
    *rest = &rest[after..];
    Some(word)
}

/// One index, over one path.
#[derive(Debug)]
pub struct PathIndex {
    /// The path as it was written, kept whole so it can be parsed again per
    /// lookup. Parsing is a scan of a dozen bytes and it saves an owned step
    /// type that would have to be kept in step with [`crate::Steps`].
    path: Box<[u8]>,
    /// What this index can be asked and how many keys a document gets.
    kind: IndexKind,
    /// The key to the slab slot its posting list sits in.
    keys: Elements<u32>,
    /// The rows of `keys` in key order, for an ordered index, and nothing at all
    /// for an equality one.
    ///
    /// The table above is unordered, because it is a hash's field table. The
    /// order lives here instead of being a property of the table, which is the
    /// same split a sorted set makes: the members are in an element table and
    /// the rank is a separate tree over its row numbers.
    order: Option<Rank>,
    /// The posting lists. A slab rather than a payload beside the row, because
    /// [`Elements`] moves its last row into the hole on a removal and a posting
    /// list is not `Copy`.
    posts: Slab<Set>,
    /// How many document ids are filed altogether, over every key.
    postings: usize,
}

impl PathIndex {
    /// An empty index over `path`, which has already been checked to parse.
    pub(crate) fn new(path: &[u8], kind: IndexKind) -> PathIndex {
        PathIndex {
            path: path.into(),
            kind,
            keys: Elements::new(),
            order: kind.is_ordered().then(Rank::new),
            posts: Slab::new(),
            postings: 0,
        }
    }

    /// The path this indexes.
    #[must_use]
    pub fn path(&self) -> &[u8] {
        &self.path
    }

    /// What this index can be asked.
    #[must_use]
    pub fn kind(&self) -> IndexKind {
        self.kind
    }

    /// Every key `at` files under in this index.
    pub(crate) fn keys_at(
        &self,
        at: Value<'_>,
        out: &mut Vec<u8>,
    ) -> core::result::Result<(), TooLong> {
        keys_at(self.kind, at, out)
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

    /// Every key between `lo` and `hi` with the documents filed under it, in
    /// order.
    ///
    /// One descent of the tree and then a link per leaf, so a range of a
    /// thousand keys costs one search and a handful of hops. An equality index
    /// has no order to walk and answers nothing at all rather than pretending to
    /// have a range; the layer above turns that into an error, because a range
    /// query that silently finds nothing is worse than one that says no.
    #[must_use]
    pub fn range(&self, lo: Bound<&Key>, hi: Bound<&Key>) -> Ranged<'_> {
        let Some((order, start, left)) = self.span(lo, hi) else {
            return Ranged {
                index: self,
                walk: None,
                left: 0,
            };
        };
        Ranged {
            index: self,
            walk: Some(order.iter_from(start)),
            left,
        }
    }

    /// [`PathIndex::range`] backwards, largest key first.
    #[must_use]
    pub fn range_rev(&self, lo: Bound<&Key>, hi: Bound<&Key>) -> RangedRev<'_> {
        let Some((order, start, left)) = self.span(lo, hi) else {
            return RangedRev {
                index: self,
                walk: None,
                left: 0,
            };
        };
        RangedRev {
            index: self,
            walk: Some(order.iter_back_from(start + left - 1)),
            left,
        }
    }

    /// How many documents are filed under any key between `lo` and `hi`.
    ///
    /// This reads the keys in the range and not the documents, so it costs the
    /// number of distinct values rather than the number of postings.
    #[must_use]
    pub fn count_in(&self, lo: Bound<&Key>, hi: Bound<&Key>) -> usize {
        self.range(lo, hi).map(|(_, set)| set.len()).sum()
    }

    /// Where a range starts and how many keys are in it, or `None` if there is
    /// no order to walk or nothing in the range.
    fn span(&self, lo: Bound<&Key>, hi: Bound<&Key>) -> Option<(&Rank, usize, usize)> {
        let order = self.order.as_ref()?;
        let keys = &self.keys;
        let start = match lo {
            Bound::Unbounded => 0,
            Bound::Included(k) => rank_of(order, keys, k.as_bytes()),
            Bound::Excluded(k) => rank_after(order, keys, k.as_bytes()),
        };
        let end = match hi {
            Bound::Unbounded => keys.len(),
            Bound::Included(k) => rank_after(order, keys, k.as_bytes()),
            Bound::Excluded(k) => rank_of(order, keys, k.as_bytes()),
        };
        if end <= start {
            return None;
        }
        Some((order, start, end - start))
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
        let row = self.keys.len() as u32;
        if self.keys.insert(key, slot).is_err() {
            self.posts.remove(slot);
            return Err(Error::new(
                Code::Full,
                "the index cannot hold another distinct value",
            ));
        }
        let PathIndex { keys, order, .. } = self;
        if let Some(order) = order {
            // The key is in the table already and not in the tree, so the search
            // compares it against every other key and lands where it belongs.
            let at = rank_of(order, keys, key);
            order.insert_at(at, row);
        }
        self.postings += 1;
        Ok(())
    }

    /// Take `id` out from under `key`, and drop the key if it was the last one.
    pub(crate) fn take(&mut self, key: &[u8], id: &[u8]) {
        let Some(row) = self.keys.index_of(key) else {
            return;
        };
        let slot = *self.keys.at(row).expect("a row that was just found").1;
        let set = self.posts.get_mut(slot).expect("a row points at its list");
        if !set.remove(id) {
            return;
        }
        self.postings -= 1;
        if !set.is_empty() {
            return;
        }
        self.posts.remove(slot);
        self.untrack(key, row);
        self.keys.remove_at(row);
    }

    /// Take `row` out of the tree, and tell the tree about the row the element
    /// table is about to renumber.
    ///
    /// The table is dense, so taking a row out moves the last row into the hole
    /// and one key nobody asked about gets a new number. Where that key sits has
    /// to be found before anything moves, because afterwards the tree is holding
    /// a number that means something else. This is the same dance a sorted set
    /// does, for the same reason.
    ///
    /// An equality index has no tree and nothing to do here.
    fn untrack(&mut self, key: &[u8], row: usize) {
        let PathIndex { keys, order, .. } = self;
        let Some(order) = order else {
            return;
        };
        let rank = rank_of(order, keys, key);
        let last = keys.len() - 1;
        let moved = if last == row {
            None
        } else {
            let name = keys.at(last).expect("the last row").0;
            Some(order.seek(|other| {
                let (other_name, _) = keys.at(other as usize).expect("a row the tree holds");
                name.cmp(other_name)
            }))
        };
        order.remove_at(rank);
        if let Some(at) = moved {
            // Everything above the hole shifted down by one when the row came
            // out of the tree.
            let at = if at > rank { at - 1 } else { at };
            order.set_at(at, row as u32);
        }
    }

    /// Throw everything filed away and keep the path and the kind.
    pub(crate) fn clear(&mut self) {
        self.keys.clear();
        self.posts.clear();
        self.postings = 0;
        if let Some(order) = &mut self.order {
            *order = Rank::new();
        }
    }

    /// What the index costs, posting lists and the order included.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.keys.memory_bytes()
            + self.posts.slot_bytes()
            + self.posts.iter().map(Set::memory_bytes).sum::<usize>()
            + self.order.as_ref().map_or(0, Rank::bytes)
    }
}

/// Keys in order with their posting lists, from [`PathIndex::range`].
pub struct Ranged<'a> {
    index: &'a PathIndex,
    walk: Option<rank::Walk<'a>>,
    left: usize,
}

impl<'a> Iterator for Ranged<'a> {
    type Item = (&'a [u8], &'a Set);

    fn next(&mut self) -> Option<(&'a [u8], &'a Set)> {
        if self.left == 0 {
            return None;
        }
        let row = self.walk.as_mut()?.next()?;
        self.left -= 1;
        entry(self.index, row)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left, Some(self.left))
    }
}

impl ExactSizeIterator for Ranged<'_> {}

/// Keys in reverse order with their posting lists, from
/// [`PathIndex::range_rev`].
pub struct RangedRev<'a> {
    index: &'a PathIndex,
    walk: Option<rank::Back<'a>>,
    left: usize,
}

impl<'a> Iterator for RangedRev<'a> {
    type Item = (&'a [u8], &'a Set);

    fn next(&mut self) -> Option<(&'a [u8], &'a Set)> {
        if self.left == 0 {
            return None;
        }
        let row = self.walk.as_mut()?.next()?;
        self.left -= 1;
        entry(self.index, row)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left, Some(self.left))
    }
}

impl ExactSizeIterator for RangedRev<'_> {}

/// The rank `key` sits at in `order`, or would sit at.
///
/// A free function rather than a method because every caller has the tree and
/// the table split out of the index already, either because it is about to
/// write to the tree while reading the table or because it is holding a borrow
/// of the tree it means to keep.
fn rank_of(order: &Rank, keys: &Elements<u32>, key: &[u8]) -> usize {
    order.seek(|row| {
        let (name, _) = keys.at(row as usize).expect("a row the tree holds");
        key.cmp(name)
    })
}

/// The rank one past `key`, which is where it sits when it is not there and one
/// to the right of it when it is.
fn rank_after(order: &Rank, keys: &Elements<u32>, key: &[u8]) -> usize {
    order.seek(|row| {
        let (name, _) = keys.at(row as usize).expect("a row the tree holds");
        match key.cmp(name) {
            Ordering::Less => Ordering::Less,
            Ordering::Equal | Ordering::Greater => Ordering::Greater,
        }
    })
}

/// The key and the posting list a tree row names.
fn entry(index: &PathIndex, row: u32) -> Option<(&[u8], &Set)> {
    let (name, &slot) = index.keys.at(row as usize)?;
    Some((name, index.posts.get(slot)?))
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

    /// The keys a kind takes from one value, as printable strings.
    fn taken(kind: IndexKind, build: impl FnOnce(&mut crate::Builder)) -> Vec<String> {
        let mut b = crate::Builder::new();
        build(&mut b);
        let bytes = b.finish().expect("built").to_vec();
        let value = Value::new(&bytes).expect("readable");
        let mut list = Vec::new();
        keys_at(kind, value, &mut list).expect("short enough");
        let mut out = Vec::new();
        each_key(&list, |key| {
            out.push(format!("{:?}", Key(Small::collect(key.iter().copied()))))
        });
        out
    }

    #[test]
    fn an_array_index_takes_one_key_per_element() {
        let keys = taken(IndexKind::Array, |b| {
            b.begin_array().expect("open");
            b.text("red").expect("value");
            b.int(7).expect("value");
            b.begin_object().expect("open");
            b.end_object().expect("close");
            b.end_array().expect("close");
        });
        assert_eq!(keys.len(), 2, "the object inside is not a key: {keys:?}");
        assert_eq!(keys[0], "\"red\"");

        // A scalar is a list of one, and an object is a list of none.
        assert_eq!(
            taken(IndexKind::Array, |b| b.text("red").expect("v")).len(),
            1
        );
        assert_eq!(
            taken(IndexKind::Array, |b| {
                b.begin_object().expect("open");
                b.end_object().expect("close");
            })
            .len(),
            0
        );
    }

    #[test]
    fn a_text_index_splits_on_everything_that_is_not_a_letter_or_a_digit() {
        let keys = taken(IndexKind::Text, |b| {
            b.text("  The RED car, model 3! ").expect("value")
        });
        assert_eq!(
            keys,
            ["\"the\"", "\"red\"", "\"car\"", "\"model\"", "\"3\""]
        );

        assert!(taken(IndexKind::Text, |b| b.text("!!! ...").expect("v")).is_empty());
        assert!(taken(IndexKind::Text, |b| b.int(7).expect("v")).is_empty());
    }

    #[test]
    fn a_word_key_is_what_a_text_index_filed_and_a_phrase_is_not_one() {
        assert_eq!(Key::word("RED"), Key::word("red"));
        assert_eq!(Key::word("red!"), Key::word("red"));
        assert!(Key::word("red car").is_none(), "a phrase is two words");
        assert!(Key::word("").is_none());
        assert!(Key::word("!!!").is_none());
        assert_eq!(
            Key::word("red").expect("a word"),
            Key::text("red"),
            "a word that needs no folding is the string key, and there is no \
             second text tag to keep them apart"
        );
        assert_ne!(Key::word("RED").expect("a word"), Key::text("RED"));
    }

    #[test]
    fn a_key_list_reads_back_exactly_what_went_into_it() {
        let mut list = Vec::new();
        push_key(&Key::text("red"), &mut list).expect("short");
        push_key(&Key::int(7), &mut list).expect("short");
        push_key(&Key::null(), &mut list).expect("short");
        let mut out = Vec::new();
        each_key(&list, |key| out.push(key.to_vec()));
        assert_eq!(
            out,
            [
                Key::text("red").as_bytes().to_vec(),
                Key::int(7).as_bytes().to_vec(),
                Key::null().as_bytes().to_vec(),
            ]
        );

        let long = "x".repeat(KEY_MAX);
        assert!(push_key(&Key::text(&long), &mut list).is_err());
    }

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
    fn an_integer_and_a_float_sort_among_each_other() {
        // The order this has to produce is the numeric one, and the two ways of
        // writing a number are mixed on purpose so that nothing can pass by
        // keeping the integers on one side and the floats on the other.
        let mut mixed: Vec<Key> = [
            Key::float(12.5),
            Key::int(99),
            Key::int(-3),
            Key::float(-2.5),
            Key::int(0),
            Key::float(0.25),
            Key::int(13),
        ]
        .to_vec();
        mixed.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        let want = [
            Key::int(-3),
            Key::float(-2.5),
            Key::int(0),
            Key::float(0.25),
            Key::float(12.5),
            Key::int(13),
            Key::int(99),
        ];
        assert_eq!(mixed, want);
    }

    #[test]
    fn seven_and_seven_point_zero_are_one_key() {
        assert_eq!(Key::int(7), Key::float(7.0));
        assert_eq!(Key::int(-7), Key::float(-7.0));
        assert_eq!(Key::int(0), Key::float(0.0));
        // A negative zero is a zero. Nothing else would let a caller who asks
        // for zero find a document that has one.
        assert_eq!(Key::int(0), Key::float(-0.0));
        assert_eq!(Key::int(1 << 53), Key::float((1u64 << 53) as f64));
        // And two numbers that are close are still two numbers. `i64::MAX` is
        // one below a power of two and the nearest f64 to it is that power of
        // two, so these are not the same value and do not get the same key.
        assert_ne!(Key::int(i64::MAX), Key::float(i64::MAX as f64));
    }

    #[test]
    fn the_ends_of_the_number_line_sort_where_they_belong() {
        let mut ends = [
            Key::float(f64::NAN),
            Key::float(f64::INFINITY),
            Key::int(1),
            Key::float(f64::NEG_INFINITY),
            Key::int(-1),
            Key::float(f64::MIN),
            Key::float(f64::MAX),
        ]
        .to_vec();
        ends.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
        let want = [
            Key::float(f64::NEG_INFINITY),
            Key::float(f64::MIN),
            Key::int(-1),
            Key::int(1),
            Key::float(f64::MAX),
            Key::float(f64::INFINITY),
            // Above everything, so a range never has to think about it.
            Key::float(f64::NAN),
        ];
        assert_eq!(ends, want);
    }

    #[test]
    fn every_number_is_the_same_width() {
        for k in [
            Key::int(0),
            Key::int(i64::MIN),
            Key::float(1e300),
            Key::float(f64::MIN_POSITIVE),
            Key::float(f64::NAN),
            Key::float(f64::NEG_INFINITY),
        ] {
            assert_eq!(k.as_bytes().len(), 12, "{k:?}");
        }
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
        // The class byte for a zero, then a place and a mantissa that are both
        // written as zero because the size of a zero is not a question.
        assert_eq!(format!("{:?}", Key::int(0)), "0280000000000000000000");
    }

    /// An ordered index over `$.n` holding the integers given, one document per
    /// integer, named after it.
    fn ordered(ns: impl IntoIterator<Item = i64>) -> PathIndex {
        let mut index = PathIndex::new(b"$.n", IndexKind::Ordered);
        for n in ns {
            index
                .add(Key::int(n).as_bytes(), n.to_string().as_bytes())
                .expect("room");
        }
        index
    }

    /// The keys a range walks, decoded back to the integers they came from.
    fn walked(index: &PathIndex, lo: Bound<&Key>, hi: Bound<&Key>) -> Vec<i64> {
        let out: Vec<i64> = index.range(lo, hi).map(|(k, _)| unorder_int(k)).collect();
        let mut back: Vec<i64> = index
            .range_rev(lo, hi)
            .map(|(k, _)| unorder_int(k))
            .collect();
        back.reverse();
        assert_eq!(out, back, "backwards is forwards read the other way");
        out
    }

    /// The number a numeric key was made from, for the whole numbers these
    /// tests file.
    fn unorder_int(key: &[u8]) -> i64 {
        assert_eq!(key[0], TAG_NUM, "these tests only file numbers");
        let class = key[1];
        if class == Class::Zero as u8 {
            return 0;
        }
        let flip = if class == Class::Negative as u8 {
            0xffu8
        } else {
            0
        };
        let place = u16::from_be_bytes([key[2] ^ flip, key[3] ^ flip]) as i32 - 32768;
        let mut mant = [0u8; 8];
        for (out, b) in mant.iter_mut().zip(&key[4..12]) {
            *out = b ^ flip;
        }
        // The mantissa sits at the top of the eight bytes, so shifting it back
        // down by however far its leading bit is from `place` gives the integer.
        let n = (u64::from_be_bytes(mant) >> (64 - place)) as i64;
        if flip == 0 { n } else { -n }
    }

    #[test]
    fn an_ordered_index_walks_its_keys_in_order() {
        // Written in an order that is neither sorted nor reverse sorted, and
        // over enough keys to push the tree past one leaf.
        let index = ordered((0..500i64).map(|i| (i * 137) % 500 - 250));
        assert_eq!(index.len(), 500);
        assert_eq!(index.kind(), IndexKind::Ordered);

        let all = walked(&index, Bound::Unbounded, Bound::Unbounded);
        assert_eq!(all, (-250..250).collect::<Vec<i64>>());

        let (lo, hi) = (Key::int(-3), Key::int(4));
        assert_eq!(
            walked(&index, Bound::Included(&lo), Bound::Excluded(&hi)),
            [-3, -2, -1, 0, 1, 2, 3]
        );
        assert_eq!(
            walked(&index, Bound::Excluded(&lo), Bound::Included(&hi)),
            [-2, -1, 0, 1, 2, 3, 4]
        );
        assert_eq!(
            walked(&index, Bound::Unbounded, Bound::Excluded(&Key::int(-247))),
            [-250, -249, -248]
        );
        assert_eq!(
            walked(&index, Bound::Included(&Key::int(247)), Bound::Unbounded),
            [247, 248, 249]
        );
    }

    #[test]
    fn a_range_that_names_nothing_is_empty_rather_than_wrong() {
        let index = ordered([10i64, 20, 30]);
        let (lo, hi) = (Key::int(20), Key::int(20));
        assert!(walked(&index, Bound::Excluded(&lo), Bound::Excluded(&hi)).is_empty());
        assert_eq!(
            walked(&index, Bound::Included(&lo), Bound::Included(&hi)),
            [20]
        );
        // Backwards bounds, which a caller can hand over by accident.
        assert!(
            walked(
                &index,
                Bound::Included(&Key::int(30)),
                Bound::Excluded(&Key::int(10))
            )
            .is_empty()
        );
        // Between two keys that are there, and past both ends.
        assert!(
            walked(
                &index,
                Bound::Included(&Key::int(21)),
                Bound::Excluded(&Key::int(29))
            )
            .is_empty()
        );
        assert!(walked(&index, Bound::Included(&Key::int(31)), Bound::Unbounded).is_empty());
        assert!(walked(&index, Bound::Unbounded, Bound::Excluded(&Key::int(10))).is_empty());
        assert_eq!(index.count_in(Bound::Unbounded, Bound::Unbounded), 3);
    }

    #[test]
    fn an_equality_index_has_no_range_and_says_so_by_being_empty() {
        let mut index = PathIndex::new(b"$.n", IndexKind::Equality);
        index.add(Key::int(1).as_bytes(), b"a").expect("room");
        assert_eq!(index.kind(), IndexKind::Equality);
        assert_eq!(index.range(Bound::Unbounded, Bound::Unbounded).count(), 0);
        assert_eq!(index.count_in(Bound::Unbounded, Bound::Unbounded), 0);
        assert_eq!(index.count(&Key::int(1)), 1, "equality still works");
    }

    #[test]
    fn removing_keys_from_an_ordered_index_keeps_the_rest_in_order() {
        // Every removal moves the element table's last row into the hole, so the
        // tree is holding a row number that has come to mean a different key.
        // This is the test that the renumbering is told to it.
        let mut index = ordered(0..200i64);
        for n in (0..200i64).step_by(3) {
            index.take(Key::int(n).as_bytes(), n.to_string().as_bytes());
        }
        let left: Vec<i64> = (0..200i64).filter(|n| n % 3 != 0).collect();
        assert_eq!(index.len(), left.len());
        assert_eq!(walked(&index, Bound::Unbounded, Bound::Unbounded), left);

        // And the keys still find their own posting lists after all that.
        for n in &left {
            assert_eq!(index.count(&Key::int(*n)), 1, "{n} lost its list");
        }
        for n in (0..200i64).step_by(3) {
            assert_eq!(index.count(&Key::int(n)), 0, "{n} kept one");
        }
    }

    #[test]
    fn an_ordered_index_that_is_emptied_and_refilled_is_still_ordered() {
        let mut index = ordered(0..64i64);
        for n in 0..64i64 {
            index.take(Key::int(n).as_bytes(), n.to_string().as_bytes());
        }
        assert!(index.is_empty());
        assert_eq!(index.postings(), 0);
        assert!(walked(&index, Bound::Unbounded, Bound::Unbounded).is_empty());

        for n in (0..32i64).rev() {
            index
                .add(Key::int(n).as_bytes(), n.to_string().as_bytes())
                .expect("room");
        }
        assert_eq!(
            walked(&index, Bound::Unbounded, Bound::Unbounded),
            (0..32).collect::<Vec<i64>>()
        );

        index.clear();
        assert_eq!(index.kind(), IndexKind::Ordered, "a clear keeps the kind");
        assert!(index.is_empty());
        index.add(Key::int(9).as_bytes(), b"9").expect("room");
        assert_eq!(walked(&index, Bound::Unbounded, Bound::Unbounded), [9]);
    }

    #[test]
    fn a_key_with_many_documents_counts_once_in_the_order() {
        let mut index = PathIndex::new(b"$.n", IndexKind::Ordered);
        for i in 0..100 {
            index
                .add(
                    Key::int(i64::from(i % 5)).as_bytes(),
                    format!("d{i}").as_bytes(),
                )
                .expect("room");
        }
        assert_eq!(index.len(), 5, "five distinct values");
        assert_eq!(index.postings(), 100);
        assert_eq!(
            walked(&index, Bound::Unbounded, Bound::Unbounded),
            [0, 1, 2, 3, 4]
        );
        assert_eq!(index.count_in(Bound::Unbounded, Bound::Unbounded), 100);
        assert_eq!(
            index.count_in(Bound::Included(&Key::int(1)), Bound::Included(&Key::int(2))),
            40
        );
    }

    #[test]
    fn the_last_document_under_a_key_takes_the_key_with_it() {
        let mut index = PathIndex::new(b"$.status", IndexKind::Equality);
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
        let mut index = PathIndex::new(b"$.status", IndexKind::Equality);
        let open = Key::text("open");
        index.add(open.as_bytes(), b"a").expect("room");
        index.add(open.as_bytes(), b"a").expect("room");
        assert_eq!(index.postings(), 1);
        index.take(open.as_bytes(), b"a");
        assert_eq!(index.postings(), 0);
    }

    #[test]
    fn taking_out_something_that_was_never_filed_changes_nothing() {
        let mut index = PathIndex::new(b"$.status", IndexKind::Equality);
        let open = Key::text("open");
        index.add(open.as_bytes(), b"a").expect("room");
        index.take(open.as_bytes(), b"never");
        index.take(Key::text("shut").as_bytes(), b"a");
        assert_eq!(index.postings(), 1);
        assert_eq!(index.count(&open), 1);
    }

    #[test]
    fn a_posting_list_of_numbers_reads_back_as_bytes() {
        let mut index = PathIndex::new(b"$.customer", IndexKind::Equality);
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
