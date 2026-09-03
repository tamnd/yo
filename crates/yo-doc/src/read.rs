//! Reading a YOJB value without decoding it.
//!
//! Every accessor here is bounds checked and answers `None` rather than
//! panicking, because these bytes come off a disk and a corrupt document is a
//! thing that happens. Nothing here allocates and nothing here copies: a child
//! is a slice of its parent and a string is a slice of the document.

use core::cmp::Ordering;

use crate::head::{self, ARRAY, COUNT_MAX, DEPTH_MAX, INTERNED, Kind, OFFSETS, SORTED, Tag};

/// A value, borrowed from the bytes it is stored in.
///
/// The slice starts at the value's header and may run past its end, which is
/// what makes a child free: it is the parent's slice from the child's offset,
/// with no length to compute. Use [`Value::encoded_len`] when the exact end
/// matters, which is when the value is being copied somewhere else.
#[derive(Clone, Copy)]
pub struct Value<'a> {
    b: &'a [u8],
}

impl<'a> Value<'a> {
    /// A value over `bytes`, if the header at the front is one this version
    /// understands and its payload is there.
    ///
    /// This is a header check and not a walk. It is what a read does, because a
    /// read touches one path and checking the whole document to answer one
    /// field would cost more than the read. [`Value::validate`] is the walk,
    /// for the caller that is about to trust the whole thing.
    #[must_use]
    pub fn new(bytes: &'a [u8]) -> Option<Value<'a>> {
        let v = Value { b: bytes };
        let h = head::read(bytes, 0)?;
        let tag = Tag::of(h)?;
        if matches!(tag, Tag::Container) {
            if h & OFFSETS == 0 || head::count(h) > COUNT_MAX {
                return None;
            }
            // The entry table has to be there before anything can be indexed.
            // The value region is checked per element, on the way in.
            let end = v.entries_end()?;
            if bytes.len() < end {
                return None;
            }
        } else if bytes.len() < 4 + head::count(h) {
            return None;
        }
        Some(v)
    }

    /// What this value is.
    #[must_use]
    pub fn kind(&self) -> Kind {
        match self.tag() {
            Tag::Null => Kind::Null,
            Tag::False | Tag::True => Kind::Bool,
            Tag::Int => Kind::Int,
            Tag::Float => Kind::Float,
            Tag::Text => Kind::Text,
            Tag::Container if self.head() & ARRAY == 0 => Kind::Object,
            Tag::Container => Kind::Array,
        }
    }

    /// Whether this is `null`.
    #[must_use]
    pub fn is_null(&self) -> bool {
        matches!(self.tag(), Tag::Null)
    }

    /// The boolean this holds, if it holds one.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self.tag() {
            Tag::False => Some(false),
            Tag::True => Some(true),
            _ => None,
        }
    }

    /// The integer this holds, if it holds one.
    ///
    /// The payload is as narrow as the number allows, so a document full of
    /// small numbers costs five bytes each rather than twelve, and reading one
    /// back is a sign extending load of one, two, four or eight bytes.
    #[must_use]
    pub fn as_int(&self) -> Option<i64> {
        if !matches!(self.tag(), Tag::Int) {
            return None;
        }
        let raw = self.payload()?;
        Some(match raw.len() {
            1 => i64::from(raw[0] as i8),
            2 => i64::from(i16::from_le_bytes(raw.try_into().ok()?)),
            4 => i64::from(i32::from_le_bytes(raw.try_into().ok()?)),
            8 => i64::from_le_bytes(raw.try_into().ok()?),
            _ => return None,
        })
    }

    /// The float this holds, if it holds one.
    #[must_use]
    pub fn as_float(&self) -> Option<f64> {
        if !matches!(self.tag(), Tag::Float) {
            return None;
        }
        let raw = self.payload()?;
        Some(f64::from_le_bytes(raw.try_into().ok()?))
    }

    /// The string this holds, if it holds one and it is UTF-8.
    #[must_use]
    pub fn as_text(&self) -> Option<&'a str> {
        core::str::from_utf8(self.text_bytes()?).ok()
    }

    /// The string this holds as it is stored, without the UTF-8 check.
    ///
    /// A string written through this crate is UTF-8 by construction, so the
    /// check only ever catches a damaged file. A caller that is going to hand
    /// the bytes straight back out over RESP does not need it.
    #[must_use]
    pub fn text_bytes(&self) -> Option<&'a [u8]> {
        if !matches!(self.tag(), Tag::Text) {
            return None;
        }
        self.payload()
    }

    /// How many elements a container holds. Zero for anything else.
    #[must_use]
    pub fn len(&self) -> usize {
        if matches!(self.tag(), Tag::Container) {
            head::count(self.head())
        } else {
            0
        }
    }

    /// Whether this is a container with nothing in it.
    ///
    /// A scalar is not empty, it is not a container, so this is false for one.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        matches!(self.tag(), Tag::Container) && self.len() == 0
    }

    /// Whether this object's keys are ids from a collection's intern table
    /// rather than bytes stored with the document.
    ///
    /// Nothing else about reading changes, except that a lookup is by id and
    /// getting a key's name back needs the table.
    #[must_use]
    pub fn is_interned(&self) -> bool {
        self.is_container() && self.head() & INTERNED != 0
    }

    /// The value of element `i`, counting in the container's own order.
    ///
    /// For an array that is the order the elements were written in. For an
    /// object it is key order, which is not the order the document was written
    /// in, and it is the order [`Value::members`] walks.
    #[must_use]
    pub fn at(&self, i: usize) -> Option<Value<'a>> {
        let (_, off) = self.entry(i)?;
        let child = self.b.get(off..)?;
        Value::new(child)
    }

    /// The key of member `i` of an object, if the object stores its keys as
    /// bytes.
    #[must_use]
    pub fn key_at(&self, i: usize) -> Option<&'a [u8]> {
        if !self.is_object() || self.is_interned() {
            return None;
        }
        let at = self.key_off(i)?;
        let end = self.key_end(i)?;
        self.b.get(at..end)
    }

    /// The intern table id of member `i` of an object, if the object stores its
    /// keys as ids.
    #[must_use]
    pub fn key_id_at(&self, i: usize) -> Option<u16> {
        if !self.is_interned() {
            return None;
        }
        let at = 4 + i * 2;
        let raw = self.b.get(at..at + 2)?;
        Some(u16::from_le_bytes(raw.try_into().expect("two bytes")))
    }

    /// The value stored under `key`, by binary search over the entry table.
    ///
    /// Keys are ordered by length and then by bytes, so the search compares a
    /// length before it compares anything else and most steps never touch the
    /// key region at all. This is the lookup G15 is about: for a document whose
    /// keys are interned it is not even this, it is [`Value::get_id`].
    #[must_use]
    pub fn get(&self, key: &[u8]) -> Option<Value<'a>> {
        self.at(self.find(key)?)
    }

    /// The index of `key` among this object's members.
    #[must_use]
    pub fn find(&self, key: &[u8]) -> Option<usize> {
        if !self.is_object() || self.is_interned() {
            return None;
        }
        let n = self.len();
        if self.head() & SORTED == 0 {
            return (0..n).find(|&i| self.key_at(i) == Some(key));
        }
        let (mut lo, mut hi) = (0usize, n);
        while lo < hi {
            let mid = (lo + hi) / 2;
            match key_order(self.key_at(mid)?, key) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => return Some(mid),
            }
        }
        None
    }

    /// The value stored under intern table id `id`.
    #[must_use]
    pub fn get_id(&self, id: u16) -> Option<Value<'a>> {
        self.at(self.find_id(id)?)
    }

    /// The index of intern table id `id` among this object's members.
    #[must_use]
    pub fn find_id(&self, id: u16) -> Option<usize> {
        if !self.is_interned() {
            return None;
        }
        let n = self.len();
        if self.head() & SORTED == 0 {
            return (0..n).find(|&i| self.key_id_at(i) == Some(id));
        }
        let (mut lo, mut hi) = (0usize, n);
        while lo < hi {
            let mid = (lo + hi) / 2;
            match self.key_id_at(mid)?.cmp(&id) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => return Some(mid),
            }
        }
        None
    }

    /// Every element of a container, in the container's own order.
    #[must_use]
    pub fn iter(&self) -> Elems<'a> {
        Elems { v: *self, i: 0 }
    }

    /// Every member of an object, key first, in key order.
    ///
    /// An interned object yields nothing here, because the names are not in the
    /// document. Walk it with [`Value::key_id_at`] and [`Value::at`].
    #[must_use]
    pub fn members(&self) -> Members<'a> {
        Members { v: *self, i: 0 }
    }

    /// How many bytes this value occupies, header included.
    ///
    /// A container works this out from its last element, which recurses down
    /// the right hand edge of the document and so costs one step per level
    /// rather than one per element. That is the price of not spending four
    /// bytes a container on a length nothing else needs.
    #[must_use]
    pub fn encoded_len(&self) -> Option<usize> {
        self.encoded_len_at(0)
    }

    fn encoded_len_at(&self, depth: usize) -> Option<usize> {
        if depth > DEPTH_MAX {
            return None;
        }
        let h = self.head();
        if !matches!(Tag::of(h)?, Tag::Container) {
            return Some(4 + head::count(h));
        }
        let n = head::count(h);
        if n == 0 {
            return self.entries_end();
        }
        let (_, off) = self.entry(n - 1)?;
        let last = Value::new(self.b.get(off..)?)?;
        off.checked_add(last.encoded_len_at(depth + 1)?)
    }

    /// Where this value begins inside `root`, in bytes.
    ///
    /// A child is its parent's slice from the child's offset, so the offset is
    /// still there to be read back off the slice itself and nothing has to be
    /// carried alongside it. This is how a write identifies the places a path
    /// matched: [`Path::select`](crate::Path::select) answers values, and a
    /// value plus the document it came out of is an offset, which is what
    /// [`edit`](crate::edit()) takes.
    ///
    /// `None` if this value did not come out of `root`.
    #[must_use]
    pub fn offset_in(&self, root: &Value<'_>) -> Option<usize> {
        let here = self.b.as_ptr() as usize;
        let base = root.b.as_ptr() as usize;
        let off = here.checked_sub(base)?;
        (off < root.b.len()).then_some(off)
    }

    /// This value's bytes and nothing after them.
    #[must_use]
    pub fn as_bytes(&self) -> Option<&'a [u8]> {
        self.b.get(..self.encoded_len()?)
    }

    /// Walk the whole value and check that every part of it is there.
    ///
    /// This is what a caller runs over bytes it did not write: a record read
    /// back from a file that failed its checksum in an interesting way, or a
    /// document handed in over a socket. Everything it checks, the accessors
    /// also check one at a time, so a document that fails here still cannot
    /// make a read panic. It is O(the document).
    #[must_use]
    pub fn validate(&self) -> bool {
        self.validate_at(0)
    }

    fn validate_at(&self, depth: usize) -> bool {
        if depth > DEPTH_MAX {
            return false;
        }
        let Some(h) = head::read(self.b, 0) else {
            return false;
        };
        let Some(tag) = Tag::of(h) else {
            return false;
        };
        if !matches!(tag, Tag::Container) {
            let n = head::count(h);
            if matches!(tag, Tag::Int) && !matches!(n, 1 | 2 | 4 | 8) {
                return false;
            }
            if matches!(tag, Tag::Float) && n != 8 {
                return false;
            }
            if matches!(tag, Tag::Null | Tag::False | Tag::True) && n != 0 {
                return false;
            }
            return self.b.len() >= 4 + n;
        }
        if h & OFFSETS == 0 {
            return false;
        }
        let n = head::count(h);
        let Some(mut want) = self.entries_end() else {
            return false;
        };
        if self.b.len() < want {
            return false;
        }
        if self.is_object() && !self.is_interned() {
            // The key region runs from the end of the entry table to the first
            // value, and the keys inside it have to tile it in order.
            for i in 0..n {
                let (Some(at), Some(end)) = (self.key_off(i), self.key_end(i)) else {
                    return false;
                };
                if at != want || end < at || self.b.len() < end {
                    return false;
                }
                want = end;
            }
        }
        for i in 0..n {
            let Some((copy, off)) = self.entry(i) else {
                return false;
            };
            // Values are stored in entry order and they tile the value region,
            // which is what lets a length be a difference of two offsets.
            if off != want {
                return false;
            }
            let Some(child) = self.b.get(off..).and_then(Value::new) else {
                return false;
            };
            if child.head() != copy || !child.validate_at(depth + 1) {
                return false;
            }
            let Some(len) = child.encoded_len_at(depth + 1) else {
                return false;
            };
            want = off + len;
        }
        if self.is_object() && h & SORTED != 0 && !self.keys_ascend(n) {
            return false;
        }
        true
    }

    /// The header word.
    fn head(&self) -> u32 {
        head::read(self.b, 0).unwrap_or(0)
    }

    fn tag(&self) -> Tag {
        Tag::of(self.head()).unwrap_or(Tag::Null)
    }

    fn is_container(&self) -> bool {
        matches!(self.tag(), Tag::Container)
    }

    fn is_object(&self) -> bool {
        self.is_container() && self.head() & ARRAY == 0
    }

    /// A scalar's bytes, after the header.
    fn payload(&self) -> Option<&'a [u8]> {
        let n = head::count(self.head());
        self.b.get(4..4 + n)
    }

    /// Where the entry table starts, which is after the key entries.
    fn entries_at(&self) -> usize {
        4 + crate::layout::keys_area(self.head(), self.len())
    }

    /// Where the key region starts, which is after the entry table.
    fn entries_end(&self) -> Option<usize> {
        self.entries_at().checked_add(self.len().checked_mul(8)?)
    }

    /// Element `i`'s header copy and where its value starts.
    fn entry(&self, i: usize) -> Option<(u32, usize)> {
        if !self.is_container() || i >= self.len() {
            return None;
        }
        let at = self.entries_at() + i * 8;
        let copy = head::read(self.b, at)?;
        let off = head::read(self.b, at + 4)? as usize;
        // A child starts after its parent's entry table, always. Checking it
        // here rather than only in [`Value::validate`] is what keeps a damaged
        // offset from making a child that contains its own parent, and so keeps
        // every walk over a document finite.
        if off < self.entries_end()? {
            return None;
        }
        Some((copy, off))
    }

    /// Where member `i`'s key starts.
    fn key_off(&self, i: usize) -> Option<usize> {
        if i >= self.len() {
            return None;
        }
        Some(head::read(self.b, 4 + i * 4)? as usize)
    }

    /// Where member `i`'s key ends.
    ///
    /// Keys are stored in member order and tile the key region, so a key ends
    /// where the next one starts, and the last one ends where the first value
    /// starts. That is why the region costs four bytes a key rather than eight.
    fn key_end(&self, i: usize) -> Option<usize> {
        if i + 1 < self.len() {
            self.key_off(i + 1)
        } else {
            self.entry(0).map(|(_, off)| off)
        }
    }

    fn keys_ascend(&self, n: usize) -> bool {
        for i in 1..n {
            let ord = if self.is_interned() {
                match (self.key_id_at(i - 1), self.key_id_at(i)) {
                    (Some(a), Some(b)) => a.cmp(&b),
                    _ => return false,
                }
            } else {
                match (self.key_at(i - 1), self.key_at(i)) {
                    (Some(a), Some(b)) => key_order(a, b),
                    _ => return false,
                }
            };
            if ord != Ordering::Less {
                return false;
            }
        }
        true
    }
}

impl core::fmt::Debug for Value<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.kind() {
            Kind::Null => f.write_str("null"),
            Kind::Bool => write!(f, "{}", self.as_bool().unwrap_or(false)),
            Kind::Int => write!(f, "{}", self.as_int().unwrap_or(0)),
            Kind::Float => write!(f, "{}", self.as_float().unwrap_or(0.0)),
            Kind::Text => write!(f, "{:?}", self.as_text().unwrap_or("")),
            Kind::Array => f.debug_list().entries(self.iter()).finish(),
            Kind::Object => {
                let mut m = f.debug_map();
                for (k, v) in self.members() {
                    m.entry(&String::from_utf8_lossy(k), &v);
                }
                m.finish()
            }
        }
    }
}

/// How two object keys compare: shorter first, then by bytes.
///
/// Length first is not arbitrary. It puts the cheapest comparison at the front
/// of the search, so most steps of a lookup are an integer compare against a
/// number the reader already has, and it keeps keys of one length together in
/// the key region.
#[must_use]
pub fn key_order(a: &[u8], b: &[u8]) -> Ordering {
    a.len().cmp(&b.len()).then_with(|| a.cmp(b))
}

/// Every element of a container, from [`Value::iter`].
#[derive(Clone)]
pub struct Elems<'a> {
    v: Value<'a>,
    i: usize,
}

impl<'a> Iterator for Elems<'a> {
    type Item = Value<'a>;

    fn next(&mut self) -> Option<Value<'a>> {
        let out = self.v.at(self.i)?;
        self.i += 1;
        Some(out)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.v.len().saturating_sub(self.i);
        (left, Some(left))
    }
}

/// Every member of an object, from [`Value::members`].
#[derive(Clone)]
pub struct Members<'a> {
    v: Value<'a>,
    i: usize,
}

impl<'a> Iterator for Members<'a> {
    type Item = (&'a [u8], Value<'a>);

    fn next(&mut self) -> Option<(&'a [u8], Value<'a>)> {
        let key = self.v.key_at(self.i)?;
        let val = self.v.at(self.i)?;
        self.i += 1;
        Some((key, val))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.v.len().saturating_sub(self.i);
        (left, Some(left))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Builder;
    use yo_common::Rng;

    /// A document with one of everything in it.
    fn sample() -> Vec<u8> {
        let mut b = Builder::new();
        b.begin_object().expect("open");
        for (k, write) in [
            ("nil", 0),
            ("yes", 1),
            ("n", 2),
            ("big", 3),
            ("f", 4),
            ("s", 5),
            ("arr", 6),
            ("obj", 7),
        ] {
            b.key(k.as_bytes()).expect("key");
            match write {
                0 => b.null().expect("value"),
                1 => b.bool(true).expect("value"),
                2 => b.int(-3).expect("value"),
                3 => b.int(i64::MAX).expect("value"),
                4 => b.float(0.125).expect("value"),
                5 => b.text("a string with some length to it").expect("value"),
                6 => {
                    b.begin_array().expect("open");
                    b.int(1).expect("value");
                    b.text("two").expect("value");
                    b.end_array().expect("close");
                }
                _ => {
                    b.begin_object().expect("open");
                    b.key(b"deep").expect("key");
                    b.int(9).expect("value");
                    b.end_object().expect("close");
                }
            }
        }
        b.end_object().expect("close");
        b.finish().expect("finished").to_vec()
    }

    /// Touch every accessor on every part of `bytes`, however damaged it is.
    ///
    /// The point is that nothing here panics and nothing here runs forever. A
    /// corrupt count can claim sixteen million elements over four bytes, so the
    /// walk stops after a few, and a corrupt offset cannot point backwards
    /// because [`Value::entry`] refuses one that does.
    fn walk(bytes: &[u8]) {
        fn go(v: Value<'_>, depth: usize) {
            if depth > 8 {
                return;
            }
            let _ = v.kind();
            let _ = v.is_null();
            let _ = v.as_bool();
            let _ = v.as_int();
            let _ = v.as_float();
            let _ = v.as_text();
            let _ = v.text_bytes();
            let _ = v.is_empty();
            let _ = v.encoded_len();
            let _ = v.as_bytes();
            let _ = v.get(b"nil");
            let _ = v.get_id(3);
            let _ = v.path("$.a.b[0]");
            let _ = format!("{v:?}");
            for i in 0..v.len().min(16) {
                let _ = v.key_at(i);
                let _ = v.key_id_at(i);
                if let Some(child) = v.at(i) {
                    go(child, depth + 1);
                }
            }
        }
        if let Some(v) = Value::new(bytes) {
            let _ = v.validate();
            go(v, 0);
        }
    }

    #[test]
    fn a_document_cut_short_anywhere_is_refused_and_never_panics() {
        let bytes = sample();
        for n in 0..bytes.len() {
            let cut = &bytes[..n];
            walk(cut);
            if let Some(v) = Value::new(cut) {
                assert!(!v.validate(), "a document missing its tail is not sound");
            }
        }
        assert!(Value::new(&bytes).expect("readable").validate());
    }

    #[test]
    fn a_document_with_a_byte_changed_is_never_worse_than_wrong() {
        let bytes = sample();
        let mut rng = Rng::new(0x5eed_0d0c);
        for _ in 0..20_000 {
            let mut damaged = bytes.clone();
            let at = rng.below(damaged.len());
            damaged[at] ^= 1 << rng.below(8);
            walk(&damaged);
        }
    }

    #[test]
    fn a_child_that_points_at_its_own_parent_is_refused() {
        let mut bytes = sample();
        let v = Value::new(&bytes).expect("readable");
        assert!(v.validate());
        // The first entry's offset lives right after the key entries and the
        // header copy. Point it at the container itself.
        let n = v.len();
        let entry = 4 + n * 4 + 4;
        bytes[entry..entry + 4].copy_from_slice(&0u32.to_le_bytes());
        let v = Value::new(&bytes).expect("the header is still fine");
        assert!(v.at(0).is_none(), "the child is not readable");
        assert!(!v.validate());
        walk(&bytes);
    }

    #[test]
    fn a_container_that_claims_more_elements_than_it_has_is_refused() {
        let bytes = sample();
        let mut damaged = bytes.clone();
        let h = u32::from_le_bytes(damaged[..4].try_into().expect("four bytes"));
        let bigger = (h & 0xff) | ((1u32 << 20) << 8);
        damaged[..4].copy_from_slice(&bigger.to_le_bytes());
        assert!(
            Value::new(&damaged).is_none(),
            "the entry table would not fit, so the value is not readable at all"
        );
        walk(&damaged);
    }

    #[test]
    fn keys_sort_by_length_and_then_by_bytes() {
        let mut keys: Vec<&[u8]> = vec![b"bb", b"a", b"", b"ab", b"z", b"aaa"];
        keys.sort_by(|a, b| key_order(a, b));
        assert_eq!(keys, [&b""[..], b"a", b"z", b"ab", b"bb", b"aaa"]);
    }

    #[test]
    fn a_document_prints_as_itself() {
        let bytes = sample();
        let v = Value::new(&bytes).expect("readable");
        let text = format!("{v:?}");
        assert!(text.contains("\"n\": -3"), "{text}");
        assert!(text.contains("\"arr\": [1, \"two\"]"), "{text}");
        assert!(text.contains("\"nil\": null"), "{text}");
    }

    #[test]
    fn an_unsorted_object_is_still_readable() {
        // Nothing this crate writes clears the sorted flag, but a later version
        // might, so a reader that finds it clear falls back to a scan rather
        // than refusing the document.
        let mut bytes = sample();
        let h = u32::from_le_bytes(bytes[..4].try_into().expect("four bytes"));
        bytes[..4].copy_from_slice(&(h & !SORTED).to_le_bytes());
        let v = Value::new(&bytes).expect("readable");
        assert!(v.validate(), "clearing the claim does not make it unsound");
        assert_eq!(v.get(b"n").expect("found by scan").as_int(), Some(-3));
        assert!(v.get(b"missing").is_none());
    }
}
