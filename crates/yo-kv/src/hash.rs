//! A hash, in whichever of the two representations currently fits it.
//!
//! A hash is a listpack of alternating fields and values, or an element table
//! with the values in a blob beside it. Which one is not a free choice: `OBJECT
//! ENCODING` has to answer `listpack` or `hashtable` at exactly the sizes a real
//! server answers them, so the rule here is `hash_max_listpack_entries` and
//! `hash_max_listpack_value` read off `t_hash.c` in the 8.10.1 tarball.
//!
//! ```text
//!   small, any bytes                    everything else
//! +---------------------------+   +---------------------------------+
//! | f | v | f | v | f | v ... |-->| element table + a value blob    |
//! | ~2 B a side, walked        |   | one probe, no cap               |
//! +---------------------------+   +---------------------------------+
//!   to 128 fields, 64 B a side
//! ```
//!
//! Promotion is one-way and upward, which is Y4. The set has three bands because
//! an all integer set has an intset to be; a hash has no equivalent, because
//! there is no representation that is cheaper for a hash whose fields happen to
//! be numbers.
//!
//! # Where the values live
//!
//! In the listpack they are simply the odd elements, which is why
//! [`Listpack::find`] takes a step: a field is at an even index and its value is
//! the next one along, and searching with a step of two never matches a value by
//! accident. `HSET h a b` followed by `HGET h b` finds nothing, which is right,
//! and a search with a step of one would have found the `b` that is a value.
//!
//! In the table band a row's payload is a [`Span`] into a [`Blob`] the hash owns.
//! That is `05` section 4.2's element per row: a value is bytes in a shared
//! stretch, not an allocation of its own, and rewriting one appends and abandons
//! rather than moving everything after it. The abandoned bytes are counted and
//! come back when they outnumber the live ones.
//!
//! Field names are interned by the element table, which is the point of the
//! split. `HSET h field v1` and then `HSET h field v2` writes eight bytes of row
//! and the new value, and touches the field's name not at all.
//!
//! # What this does not have yet
//!
//! Field TTL. [`crate::ttl::Deadlines`] is built and waiting for it, and when it
//! lands the listpack band gains a third element per field and reports
//! `listpackex`, the way Redis does. Doing it now would mean a step of three
//! through a structure that has no deadlines in it.

use yo_common::num::parse_i64;

use crate::blob::{Blob, Span};
use crate::elem::Elements;
use crate::listpack::{self, Listpack};
use crate::scan::Cursor;

/// A field name or a value, as it is stored.
///
/// Both sides of a pair are the same thing to a listpack, which stores something
/// that looks like an integer as an integer. `HSET h f 42` and `HSET h f 042`
/// hold different bytes and both answer with what went in, and the formatting
/// happens once, into the reply buffer, the way Y18 asks.
pub type Text<'a> = listpack::Entry<'a>;

/// Where the encoding changes over.
///
/// These are `hash-max-listpack-entries` and `hash-max-listpack-value`, runtime
/// configuration in Redis, so they are passed in rather than being constants.
/// The value limit applies to a field name and to a value alike, which is what
/// `hashTypeTryConversion` does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// At this many fields a hash stops being a listpack.
    pub max_listpack_entries: usize,
    /// A field or a value longer than this cannot go in a listpack.
    pub max_listpack_value: usize,
}

impl Limits {
    /// Redis's defaults: 128 and 64.
    pub const DEFAULT: Limits = Limits {
        max_listpack_entries: 128,
        max_listpack_value: 64,
    };
}

impl Default for Limits {
    fn default() -> Limits {
        Limits::DEFAULT
    }
}

/// Which of the two a hash is in, which is what `OBJECT ENCODING` reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// One packed blob of alternating fields and values, walked linearly.
    Listpack,
    /// The element table, with the values in a blob beside it.
    Hashtable,
}

impl Encoding {
    /// The word `OBJECT ENCODING` replies with.
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Encoding::Listpack => "listpack",
            Encoding::Hashtable => "hashtable",
        }
    }
}

/// The native band: interned field names, values in a blob of their own.
#[derive(Debug)]
struct Table {
    fields: Elements<Span>,
    values: Blob,
}

impl Table {
    fn new(hint: usize) -> Table {
        Table {
            fields: Elements::with_capacity(hint),
            // Sixteen bytes a value is a guess and being wrong about it costs a
            // realloc, which is what a guess is allowed to cost.
            values: Blob::with_capacity(hint.saturating_mul(16)),
        }
    }

    #[inline]
    fn get(&self, field: &[u8]) -> Option<&[u8]> {
        self.fields.get(field).map(|&span| self.values.span(span))
    }

    /// Store `value` against `field` and say whether the field is new.
    fn set(&mut self, field: &[u8], value: &[u8]) -> bool {
        let span = self.values.push_span(value);
        if let Some(slot) = self.fields.get_mut(field) {
            let old = std::mem::replace(slot, span);
            self.values.release_span(old);
            self.settle();
            return false;
        }
        match self.fields.insert(field, span) {
            Ok(_) => true,
            Err(_) => {
                // A field name over NAME_MAX or a table at MAX_ROWS. The value
                // bytes are already in the blob, so they are given back rather
                // than left as a leak nothing accounts for.
                self.values.release_span(span);
                self.settle();
                false
            }
        }
    }

    fn remove(&mut self, field: &[u8]) -> bool {
        match self.fields.remove(field) {
            Some(span) => {
                self.values.release_span(span);
                self.settle();
                true
            }
            None => false,
        }
    }

    /// Give the dead value bytes back once there are more of them than live.
    fn settle(&mut self) {
        if !self.values.worth_compacting() {
            return;
        }
        let fields = &mut self.fields;
        self.values.compact(|keep| {
            for span in fields.payloads_mut() {
                keep.moved_span(span);
            }
        });
    }
}

/// The two representations.
#[derive(Debug)]
enum Body {
    Packed(Listpack),
    Table(Table),
}

/// A hash of fields to values.
#[derive(Debug)]
pub struct Hash {
    body: Body,
}

impl Default for Hash {
    fn default() -> Hash {
        Hash::new()
    }
}

impl Hash {
    /// An empty hash, which starts as a listpack.
    #[must_use]
    pub fn new() -> Hash {
        Hash {
            body: Body::Packed(Listpack::new()),
        }
    }

    /// An empty hash sized for what is about to go in it.
    ///
    /// `HSET k f1 v1 f2 v2 ...` with a thousand pairs builds a table once rather
    /// than converting on the way there. The hint is only a hint and being wrong
    /// costs a conversion and no correctness.
    #[must_use]
    pub fn with_hint(hint: usize, limits: &Limits) -> Hash {
        if hint <= limits.max_listpack_entries {
            Hash::new()
        } else {
            Hash {
                body: Body::Table(Table::new(hint)),
            }
        }
    }

    /// Which representation this is in.
    #[inline]
    #[must_use]
    pub const fn encoding(&self) -> Encoding {
        match self.body {
            Body::Packed(_) => Encoding::Listpack,
            Body::Table(_) => Encoding::Hashtable,
        }
    }

    /// How many fields. This is `HLEN`.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        match &self.body {
            // Two elements a field, and a listpack that held an odd number of
            // them would be a bug somewhere above rather than a half field.
            Body::Packed(lp) => lp.len() / 2,
            Body::Table(t) => t.fields.len(),
        }
    }

    /// Whether there are none.
    ///
    /// An empty hash does not exist in Redis, so the caller deletes the key when
    /// this turns true rather than storing an empty one.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// What is stored against `field`. This is `HGET`.
    #[must_use]
    pub fn get(&self, field: &[u8]) -> Option<Text<'_>> {
        match &self.body {
            Body::Packed(lp) => {
                let at = lp.find(field, 2)?;
                lp.get(at + 1)
            }
            Body::Table(t) => t.get(field).map(Text::Str),
        }
    }

    /// Whether `field` is here at all. This is `HEXISTS`.
    #[must_use]
    pub fn contains(&self, field: &[u8]) -> bool {
        match &self.body {
            Body::Packed(lp) => lp.find(field, 2).is_some(),
            Body::Table(t) => t.fields.contains(field),
        }
    }

    /// How long the value against `field` is. This is `HSTRLEN`.
    ///
    /// A missing field is zero to Redis and `None` here, because the layer that
    /// knows it is answering `HSTRLEN` is the one that should decide that a
    /// missing field and an empty value give the same number.
    #[must_use]
    pub fn value_len(&self, field: &[u8]) -> Option<usize> {
        match &self.body {
            Body::Packed(_) => self.get(field).map(|v| v.byte_len()),
            Body::Table(t) => t.fields.get(field).map(|s| s.len as usize),
        }
    }

    /// The pair at `index`, in whatever order the representation holds them.
    ///
    /// Insertion order in both bands, and neither is a promise. `HRANDFIELD`
    /// needs positions and this is what gives it them, the same way `SPOP` uses
    /// the set's.
    #[must_use]
    pub fn at(&self, index: usize) -> Option<(Text<'_>, Text<'_>)> {
        match &self.body {
            Body::Packed(lp) => {
                let field = lp.get(index * 2)?;
                let value = lp.get(index * 2 + 1)?;
                Some((field, value))
            }
            Body::Table(t) => {
                let (name, span) = t.fields.at(index)?;
                Some((Text::Str(name), Text::Str(t.values.span(*span))))
            }
        }
    }

    /// Every field and its value, in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (Text<'_>, Text<'_>)> {
        (0..self.len()).map(|i| self.at(i).expect("index is under the length"))
    }

    /// Walk part of the hash and say where to resume. This is `HSCAN`.
    ///
    /// Only the table band walks in windows, for the reason [`crate::set::Set`]
    /// gives: a hundred and twenty eight fields is smaller than the arithmetic
    /// to split them up, and a hash that small cannot hold the loop long enough
    /// for splitting to buy anything. A listpack hands back everything and
    /// [`Cursor::END`], ignoring the cursor it was given, which is safe because
    /// promotion is one way.
    pub fn scan<F>(&self, cursor: Cursor, count: usize, mut f: F) -> Cursor
    where
        F: FnMut(Text<'_>, Text<'_>),
    {
        match &self.body {
            Body::Table(t) => {
                let values = &t.values;
                t.fields.scan(cursor, count, |name, span| {
                    f(Text::Str(name), Text::Str(values.span(*span)));
                })
            }
            Body::Packed(_) => {
                for (field, value) in self.iter() {
                    f(field, value);
                }
                Cursor::END
            }
        }
    }

    /// Store `value` against `field`, promoting if it no longer fits.
    ///
    /// Answers whether the field is new, which is the number `HSET` reports.
    pub fn set(&mut self, field: &[u8], value: &[u8], limits: &Limits) -> bool {
        if let Body::Packed(lp) = &mut self.body {
            // Redis checks both sides against the value limit before it writes,
            // in hashTypeTryConversion, so a pair too long for the band converts
            // the hash and is never briefly stored in a listpack that should not
            // hold it.
            if field.len() > limits.max_listpack_value || value.len() > limits.max_listpack_value {
                self.become_table(1);
            } else if let Some(at) = lp.find(field, 2) {
                lp.replace(at + 1, value);
                return false;
            } else {
                lp.push(field);
                lp.push(value);
                // Strictly greater, so the 128th field is still a listpack and
                // the 129th is not.
                if lp.len() / 2 > limits.max_listpack_entries {
                    self.become_table(0);
                }
                return true;
            }
        }
        match &mut self.body {
            Body::Table(t) => t.set(field, value),
            Body::Packed(_) => unreachable!("the conversion above left a table"),
        }
    }

    /// Take `field` out. Answers whether it was there. This is `HDEL`.
    ///
    /// Never demotes, which is Y4's one-way rule and Redis's behaviour.
    pub fn remove(&mut self, field: &[u8]) -> bool {
        match &mut self.body {
            Body::Packed(lp) => match lp.find(field, 2) {
                // The field and its value go together, and they are adjacent,
                // which is the whole reason the pair is stored this way round.
                Some(at) => lp.delete(at, 2),
                None => false,
            },
            Body::Table(t) => t.remove(field),
        }
    }

    /// Bytes held by whichever representation this is.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        match &self.body {
            Body::Packed(lp) => lp.byte_len(),
            Body::Table(t) => t.fields.memory_bytes() + t.values.memory_bytes(),
        }
    }

    /// Value bytes no field points at any more.
    ///
    /// Reported rather than hidden, the same as the element table's dead name
    /// bytes, because a hash that has been rewritten holds them and `INFO
    /// memory` should be able to say so.
    #[must_use]
    pub fn dead_value_bytes(&self) -> usize {
        match &self.body {
            Body::Packed(_) => 0,
            Body::Table(t) => t.values.dead(),
        }
    }

    /// Move to the table band, with room for `extra` more fields than are here.
    fn become_table(&mut self, extra: usize) {
        let Body::Packed(lp) = &self.body else {
            return;
        };
        let mut t = Table::new(lp.len() / 2 + extra);
        let mut pair = lp.iter();
        while let (Some(field), Some(value)) = (pair.next(), pair.next()) {
            // A listpack holds a field that looks like a number as a number, and
            // the table holds names as bytes, so this is where the digits get
            // written. Once, on promotion, and never again.
            let f = field.to_vec();
            let v = value.to_vec();
            t.set(&f, &v);
        }
        self.body = Body::Table(t);
    }
}

/// Whether these bytes would be stored as an integer, for a caller deciding
/// what `OBJECT ENCODING` or an RDB writer should say about them.
#[must_use]
#[inline]
pub fn stores_as_int(bytes: &[u8]) -> bool {
    parse_i64(bytes).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hash that never leaves the listpack band.
    const SMALL: Limits = Limits::DEFAULT;
    /// A hash that is a table from its second field.
    const AS_TABLE: Limits = Limits {
        max_listpack_entries: 1,
        max_listpack_value: 64,
    };

    fn text(t: Text<'_>) -> Vec<u8> {
        t.to_vec()
    }

    fn pairs(h: &Hash) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = h
            .iter()
            .map(|(f, v)| {
                (
                    String::from_utf8(text(f)).expect("utf8"),
                    String::from_utf8(text(v)).expect("utf8"),
                )
            })
            .collect();
        out.sort();
        out
    }

    #[test]
    fn a_field_written_comes_back() {
        for limits in [&SMALL, &AS_TABLE] {
            let mut h = Hash::new();
            assert!(h.set(b"a", b"1", limits), "the field is new");
            assert!(h.set(b"b", b"2", limits));
            assert!(!h.set(b"a", b"3", limits), "and now it is not");

            assert_eq!(h.len(), 2);
            assert_eq!(h.get(b"a").map(text), Some(b"3".to_vec()));
            assert_eq!(h.get(b"b").map(text), Some(b"2".to_vec()));
            assert_eq!(h.get(b"c"), None);
            assert!(h.contains(b"a") && !h.contains(b"c"));
        }
    }

    #[test]
    fn a_value_is_never_mistaken_for_a_field() {
        // The listpack band searches with a step of two, and this is the shape
        // that catches a step of one: b is a value and never a field.
        let mut h = Hash::new();
        h.set(b"a", b"b", &SMALL);
        assert_eq!(h.get(b"b"), None, "b is a value, not a field");
        assert!(!h.contains(b"b"));
        assert!(!h.remove(b"b"), "and it cannot be deleted as one");
        assert_eq!(h.len(), 1);

        assert!(h.set(b"b", b"c", &SMALL), "so writing b is a new field");
        assert_eq!(h.get(b"a").map(text), Some(b"b".to_vec()));
        assert_eq!(h.get(b"b").map(text), Some(b"c".to_vec()));
    }

    #[test]
    fn deleting_takes_the_value_with_the_field() {
        for limits in [&SMALL, &AS_TABLE] {
            let mut h = Hash::new();
            for (f, v) in [("a", "1"), ("b", "2"), ("c", "3")] {
                h.set(f.as_bytes(), v.as_bytes(), limits);
            }
            assert!(h.remove(b"b"));
            assert!(!h.remove(b"b"), "twice is once");

            assert_eq!(h.len(), 2);
            assert_eq!(
                pairs(&h),
                [
                    ("a".to_owned(), "1".to_owned()),
                    ("c".to_owned(), "3".to_owned())
                ],
                "and nothing shifted into the wrong pairing"
            );
        }
    }

    #[test]
    fn it_promotes_on_the_count_and_on_the_length() {
        let mut h = Hash::new();
        for i in 0..128u32 {
            h.set(format!("f{i}").as_bytes(), b"v", &SMALL);
        }
        assert_eq!(h.encoding(), Encoding::Listpack, "128 is still a listpack");
        h.set(b"one more", b"v", &SMALL);
        assert_eq!(h.encoding(), Encoding::Hashtable, "and 129 is not");
        assert_eq!(h.len(), 129);

        // Either side being too long converts on its own, at any count.
        let long = vec![b'x'; 65];
        let mut by_value = Hash::new();
        by_value.set(b"f", &long, &SMALL);
        assert_eq!(by_value.encoding(), Encoding::Hashtable);
        assert_eq!(by_value.get(b"f").map(text), Some(long.clone()));

        let mut by_field = Hash::new();
        by_field.set(&long, b"v", &SMALL);
        assert_eq!(by_field.encoding(), Encoding::Hashtable);
        assert_eq!(by_field.get(&long).map(text), Some(b"v".to_vec()));
    }

    #[test]
    fn promotion_carries_every_pair_over_intact() {
        let mut h = Hash::new();
        // Numbers, so the listpack holds them as integers and the promotion has
        // to write the digits out on the way to the table.
        for i in 0..128u32 {
            h.set(
                format!("{i}").as_bytes(),
                format!("{}", i * 2).as_bytes(),
                &SMALL,
            );
        }
        assert_eq!(h.encoding(), Encoding::Listpack);
        let before = pairs(&h);

        h.set(b"last", b"one", &SMALL);
        assert_eq!(h.encoding(), Encoding::Hashtable);

        let mut after = pairs(&h);
        after.retain(|(f, _)| f != "last");
        assert_eq!(after, before, "the pairs survived the conversion");
        for i in 0..128u32 {
            assert_eq!(
                h.get(format!("{i}").as_bytes()).map(text),
                Some(format!("{}", i * 2).into_bytes()),
                "field {i} is findable by its digits"
            );
        }
    }

    #[test]
    fn it_never_demotes() {
        let mut h = Hash::new();
        for i in 0..200u32 {
            h.set(format!("f{i}").as_bytes(), b"v", &SMALL);
        }
        assert_eq!(h.encoding(), Encoding::Hashtable);
        for i in 0..199u32 {
            h.remove(format!("f{i}").as_bytes());
        }
        assert_eq!(h.len(), 1);
        assert_eq!(
            h.encoding(),
            Encoding::Hashtable,
            "one field left and still a table"
        );
    }

    #[test]
    fn a_length_is_answered_without_writing_the_digits() {
        for limits in [&SMALL, &AS_TABLE] {
            let mut h = Hash::new();
            h.set(b"n", b"1234567", limits);
            h.set(b"s", b"hello", limits);
            h.set(b"e", b"", limits);

            assert_eq!(h.value_len(b"n"), Some(7));
            assert_eq!(h.value_len(b"s"), Some(5));
            assert_eq!(h.value_len(b"e"), Some(0));
            assert_eq!(h.value_len(b"missing"), None);
        }
    }

    #[test]
    fn a_rewritten_value_gives_its_bytes_back_eventually() {
        let mut h = Hash::with_hint(1000, &SMALL);
        assert_eq!(h.encoding(), Encoding::Hashtable);
        let big = vec![b'z'; 200];
        for _ in 0..200 {
            h.set(b"one", &big, &SMALL);
        }
        assert_eq!(h.len(), 1);
        assert_eq!(h.get(b"one").map(text), Some(big.clone()));
        assert!(
            h.dead_value_bytes() < 4096,
            "{} bytes left dead",
            h.dead_value_bytes()
        );
    }

    #[test]
    fn compacting_the_values_moves_every_field_to_the_right_bytes() {
        let mut h = Hash::with_hint(1000, &SMALL);
        // Each field's value is its own name repeated, so a reference that moved
        // to the wrong place is visible rather than merely wrong.
        let want: Vec<(Vec<u8>, Vec<u8>)> = (0..300u32)
            .map(|i| {
                let f = format!("field{i}").into_bytes();
                let v = f.repeat(20);
                (f, v)
            })
            .collect();
        for (f, v) in &want {
            h.set(f, v, &SMALL);
        }
        // Rewrite every one of them, which abandons the whole first copy and is
        // far over both the floor and the ratio.
        for (f, v) in &want {
            h.set(f, v, &SMALL);
        }
        for (f, v) in &want {
            assert_eq!(
                h.get(f).map(text).as_deref(),
                Some(&v[..]),
                "field moved wrongly"
            );
        }
        assert_eq!(h.len(), 300);
    }

    #[test]
    fn a_scan_walks_a_hash_of_any_size_exactly_once() {
        for hint in [0usize, 2000] {
            let mut h = Hash::with_hint(hint, &SMALL);
            for i in 0..100u32 {
                h.set(
                    format!("f{i}").as_bytes(),
                    format!("v{i}").as_bytes(),
                    &SMALL,
                );
            }
            let mut seen: Vec<(String, String)> = Vec::new();
            let mut cursor = Cursor::START;
            loop {
                cursor = h.scan(cursor, 7, |f, v| {
                    seen.push((
                        String::from_utf8(text(f)).expect("utf8"),
                        String::from_utf8(text(v)).expect("utf8"),
                    ));
                });
                if cursor.is_end() {
                    break;
                }
            }
            seen.sort();
            assert_eq!(seen.len(), 100, "at hint {hint}");
            assert_eq!(seen, pairs(&h), "at hint {hint}");
        }
    }

    #[test]
    fn a_draw_reaches_every_pair_and_pairs_them_right() {
        for limits in [&SMALL, &AS_TABLE] {
            let mut h = Hash::new();
            for i in 0..50u32 {
                h.set(
                    format!("f{i}").as_bytes(),
                    format!("v{i}").as_bytes(),
                    limits,
                );
            }
            for i in 0..h.len() {
                let (f, v) = h.at(i).expect("under the length");
                let f = String::from_utf8(text(f)).expect("utf8");
                let v = String::from_utf8(text(v)).expect("utf8");
                assert_eq!(v, f.replace('f', "v"), "row {i} paired wrongly");
            }
            assert_eq!(h.at(h.len()), None, "and there is nothing past the end");
        }
    }

    #[test]
    fn a_hint_that_is_wrong_costs_a_conversion_and_no_answers() {
        // Sized for a table and given three fields, which is a waste and not a
        // bug, and sized for a listpack and given two hundred, which converts.
        let mut big = Hash::with_hint(5000, &SMALL);
        big.set(b"a", b"1", &SMALL);
        assert_eq!(big.encoding(), Encoding::Hashtable);
        assert_eq!(big.get(b"a").map(text), Some(b"1".to_vec()));

        let mut small = Hash::with_hint(2, &SMALL);
        for i in 0..200u32 {
            small.set(format!("f{i}").as_bytes(), b"v", &SMALL);
        }
        assert_eq!(small.encoding(), Encoding::Hashtable);
        assert_eq!(small.len(), 200);
    }

    #[test]
    fn an_empty_hash_has_allocated_almost_nothing() {
        let h = Hash::new();
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
        assert_eq!(h.get(b"a"), None);
        assert!(h.memory_bytes() < 64, "{} bytes", h.memory_bytes());
    }
}
