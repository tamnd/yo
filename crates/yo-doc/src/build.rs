//! Writing a YOJB value.
//!
//! The builder is a stream of pushes rather than a tree, because the thing that
//! feeds it most is a serializer walking a struct field by field and the thing
//! that feeds it second most is a parser walking JSON text. Neither has a tree
//! to hand and neither should have to build one.
//!
//! ```
//! use yo_doc::{Builder, Value};
//!
//! let mut b = Builder::new();
//! b.begin_object().unwrap();
//! b.key(b"id").unwrap();
//! b.int(7).unwrap();
//! b.key(b"tags").unwrap();
//! b.begin_array().unwrap();
//! b.text("red").unwrap();
//! b.text("blue").unwrap();
//! b.end_array().unwrap();
//! b.end_object().unwrap();
//! let bytes = b.finish().unwrap();
//!
//! let v = Value::new(&bytes).unwrap();
//! assert_eq!(v.get(b"id").unwrap().as_int(), Some(7));
//! assert_eq!(v.get(b"tags").unwrap().at(1).unwrap().as_text(), Some("blue"));
//! ```

use yo_common::{Code, Error, Result};

use crate::head::{self, ARRAY, COUNT_MAX, DEPTH_MAX, INTERNED, OFFSETS, SORTED, Tag};
use crate::layout;
use crate::read::{Value, key_order};

/// A value under construction.
///
/// Reusable: [`Builder::finish`] hands back the bytes and [`Builder::clear`]
/// puts it back to empty with its buffers intact, so a loop over a million
/// documents allocates a handful of times rather than a million.
#[derive(Debug, Default)]
pub struct Builder {
    /// Everything written so far. A container's children land here as they
    /// arrive and are moved into place once, when the container closes.
    out: Vec<u8>,
    /// One entry per container that has been begun and not yet ended.
    open: Vec<Open>,
    /// Pending members, for every open container at once. A container owns the
    /// tail of this from its own `first`.
    members: Vec<Member>,
    /// Pending key bytes, same arrangement.
    keys: Vec<u8>,
    /// Where a closing container parks its children while it writes its entry
    /// table in front of them.
    scratch: Vec<u8>,
    /// The key the next value will be stored under.
    pending: Option<Member>,
    /// Ticks once per member, so that a sort can be made stable by hand and two
    /// members with the same key can be told apart.
    seq: u32,
}

/// A container that has been begun and not yet ended.
#[derive(Debug)]
struct Open {
    /// Where its header goes. Its children start four bytes later.
    at: usize,
    /// `ARRAY` and `INTERNED`, decided when it was begun.
    flags: u32,
    /// The first of its members in [`Builder::members`].
    first: usize,
    /// Where its members' keys start in [`Builder::keys`]. Its own key, if it
    /// has one, is below this.
    keys_at: usize,
    /// The key it will be stored under in its own parent.
    key: Member,
}

/// One element of a container, while the container is still open.
#[derive(Debug, Default, Clone, Copy)]
struct Member {
    /// The element's own header, copied into the entry table at close.
    head: u32,
    /// Where the element's bytes are in [`Builder::out`] right now.
    at: u32,
    /// How many bytes they are.
    len: u32,
    /// Where its key is in [`Builder::keys`], and how long.
    key_at: u32,
    key_len: u32,
    /// Its intern table id, when the container has interned keys.
    id: u16,
    /// Insertion order.
    seq: u32,
}

impl Builder {
    /// An empty builder.
    #[must_use]
    pub fn new() -> Builder {
        Builder::default()
    }

    /// Empty, with room for `bytes` of value already reserved.
    #[must_use]
    pub fn with_capacity(bytes: usize) -> Builder {
        Builder {
            out: Vec::with_capacity(bytes),
            ..Builder::default()
        }
    }

    /// Throw away everything written so far and keep the buffers.
    pub fn clear(&mut self) {
        self.out.clear();
        self.open.clear();
        self.members.clear();
        self.keys.clear();
        self.pending = None;
        self.seq = 0;
    }

    /// The finished value.
    ///
    /// An error here means the value is not finished: a container was begun and
    /// not ended, a key was written with no value after it, or nothing was
    /// written at all.
    pub fn finish(&mut self) -> Result<&[u8]> {
        if let Some(open) = self.open.last() {
            let what = if open.flags & ARRAY != 0 {
                "array"
            } else {
                "object"
            };
            return Err(Error::fmt(
                Code::Invalid,
                format_args!("the document ends inside an unclosed {what}"),
            ));
        }
        if self.pending.is_some() {
            return Err(Error::new(Code::Invalid, "a key with no value after it"));
        }
        if self.out.is_empty() {
            return Err(Error::new(Code::Invalid, "the document holds no value"));
        }
        Ok(&self.out)
    }

    /// Write `null`.
    pub fn null(&mut self) -> Result<()> {
        self.scalar(Tag::Null, &[])
    }

    /// Write a boolean.
    pub fn bool(&mut self, v: bool) -> Result<()> {
        self.scalar(if v { Tag::True } else { Tag::False }, &[])
    }

    /// Write an integer, in as few bytes as it fits in.
    pub fn int(&mut self, v: i64) -> Result<()> {
        let raw = v.to_le_bytes();
        self.scalar(Tag::Int, &raw[..int_width(v)])
    }

    /// Write a float.
    pub fn float(&mut self, v: f64) -> Result<()> {
        self.scalar(Tag::Float, &v.to_le_bytes())
    }

    /// Write a string.
    pub fn text(&mut self, v: &str) -> Result<()> {
        self.scalar(Tag::Text, v.as_bytes())
    }

    /// Write a string that is already bytes.
    ///
    /// The bytes are stored as they are and are not checked, so a caller that
    /// hands over something that is not UTF-8 gets a document whose
    /// [`Value::as_text`] answers `None` where it should have answered a
    /// string. It exists because RESP carries strings as bytes and re-checking
    /// what a client already sent is a copy nobody asked for.
    pub fn text_bytes(&mut self, v: &[u8]) -> Result<()> {
        self.scalar(Tag::Text, v)
    }

    /// Copy a value that is already encoded.
    ///
    /// This is how a path update writes the parts of a document it is not
    /// changing: they are already in the right form, so they are memcpy and not
    /// a re-encode.
    pub fn embed(&mut self, v: &Value<'_>) -> Result<()> {
        let bytes = v
            .as_bytes()
            .ok_or_else(|| Error::new(Code::Corrupt, "the value being copied is not readable"))?;
        self.start()?;
        let at = self.out.len();
        self.out.extend_from_slice(bytes);
        self.record(at)
    }

    /// Begin an object. Every value inside it needs a [`Builder::key`] first.
    pub fn begin_object(&mut self) -> Result<()> {
        self.begin(0)
    }

    /// Begin an object whose keys are ids from a collection's intern table.
    ///
    /// Every value inside it needs a [`Builder::key_id`] first. This is what a
    /// typed collection writes, and it is where the size of a document
    /// collection mostly goes: the same twenty field names on every document
    /// cost two bytes each here instead of their bytes.
    pub fn begin_object_interned(&mut self) -> Result<()> {
        self.begin(INTERNED)
    }

    /// Begin an array.
    pub fn begin_array(&mut self) -> Result<()> {
        self.begin(ARRAY)
    }

    /// End the object begun by the matching [`Builder::begin_object`].
    pub fn end_object(&mut self) -> Result<()> {
        self.end(false)
    }

    /// End the array begun by the matching [`Builder::begin_array`].
    pub fn end_array(&mut self) -> Result<()> {
        self.end(true)
    }

    /// The key the next value goes under.
    ///
    /// Members may be written in any order, since the container sorts them when
    /// it closes. Writing the same key twice keeps the last one, which is what
    /// every JSON parser does and what `JSON.SET` has to do.
    pub fn key(&mut self, key: &[u8]) -> Result<()> {
        let open = self.expect_object()?;
        if open.flags & INTERNED != 0 {
            return Err(Error::new(
                Code::Invalid,
                "this object takes key ids, not key bytes",
            ));
        }
        if key.len() > COUNT_MAX {
            return Err(Error::new(Code::Full, "the key is longer than 16 MiB"));
        }
        self.stash(Member {
            key_at: u32::try_from(self.keys.len()).map_err(|_| too_big())?,
            key_len: key.len() as u32,
            ..Member::default()
        })?;
        self.keys.extend_from_slice(key);
        Ok(())
    }

    /// The intern table id the next value goes under.
    pub fn key_id(&mut self, id: u16) -> Result<()> {
        let open = self.expect_object()?;
        if open.flags & INTERNED == 0 {
            return Err(Error::new(
                Code::Invalid,
                "this object takes key bytes, not key ids",
            ));
        }
        self.stash(Member {
            id,
            ..Member::default()
        })
    }

    /// The innermost open container, if it is an object.
    fn expect_object(&self) -> Result<&Open> {
        match self.open.last() {
            Some(open) if open.flags & ARRAY == 0 => Ok(open),
            Some(_) => Err(Error::new(Code::Invalid, "an array element has no key")),
            None => Err(Error::new(Code::Invalid, "there is no object open")),
        }
    }

    /// Park a key until the value that goes under it arrives.
    fn stash(&mut self, key: Member) -> Result<()> {
        if self.pending.is_some() {
            return Err(Error::new(Code::Invalid, "two keys in a row"));
        }
        self.pending = Some(key);
        Ok(())
    }

    /// Write a scalar's header and payload.
    fn scalar(&mut self, tag: Tag, payload: &[u8]) -> Result<()> {
        if payload.len() > COUNT_MAX {
            return Err(Error::new(Code::Full, "the value is longer than 16 MiB"));
        }
        self.start()?;
        let at = self.out.len();
        let h = head::head(tag, 0, payload.len());
        self.out.extend_from_slice(&h.to_le_bytes());
        self.out.extend_from_slice(payload);
        self.record(at)
    }

    /// Check that a value may be written here, and that it has a key if it
    /// needs one.
    fn start(&mut self) -> Result<()> {
        match self.open.last() {
            Some(open) if open.flags & ARRAY == 0 && self.pending.is_none() => Err(Error::new(
                Code::Invalid,
                "an object member needs a key before its value",
            )),
            Some(_) => Ok(()),
            None if self.out.is_empty() => Ok(()),
            None => Err(Error::new(
                Code::Invalid,
                "a document holds one value, and it is already written",
            )),
        }
    }

    /// Note the value that was just written at `at` as a member of whatever is
    /// open around it.
    fn record(&mut self, at: usize) -> Result<()> {
        if self.open.is_empty() {
            return Ok(());
        }
        let mut m = self.pending.take().unwrap_or_default();
        m.head = head::read(&self.out, at).expect("the header was just written");
        m.at = u32::try_from(at).map_err(|_| too_big())?;
        m.len = u32::try_from(self.out.len() - at).map_err(|_| too_big())?;
        m.seq = self.seq;
        self.seq += 1;
        self.members.push(m);
        Ok(())
    }

    /// Open a container and reserve the four bytes its header will go in.
    fn begin(&mut self, flags: u32) -> Result<()> {
        if self.open.len() >= DEPTH_MAX {
            return Err(Error::fmt(
                Code::Full,
                format_args!("a document nests at most {DEPTH_MAX} deep"),
            ));
        }
        self.start()?;
        let at = self.out.len();
        self.out.extend_from_slice(&[0; 4]);
        self.open.push(Open {
            at,
            flags,
            first: self.members.len(),
            keys_at: self.keys.len(),
            key: self.pending.take().unwrap_or_default(),
        });
        Ok(())
    }

    /// Close a container: sort its members, then write its header, its entry
    /// table and its key region in front of the children that are already
    /// there.
    ///
    /// The children move once, through [`Builder::scratch`], because the entry
    /// table's size is not known until the count is and the count is not known
    /// until here. Each byte of a document is therefore copied once per level
    /// it is nested under, which is why [`DEPTH_MAX`] is a number and not a
    /// suggestion.
    fn end(&mut self, array: bool) -> Result<()> {
        let Some(open) = self.open.pop() else {
            return Err(Error::new(Code::Invalid, "nothing is open"));
        };
        if array != (open.flags & ARRAY != 0) {
            return Err(Error::new(
                Code::Invalid,
                "an object is not ended by ending an array, or the other way round",
            ));
        }
        if self.pending.is_some() {
            return Err(Error::new(Code::Invalid, "a key with no value after it"));
        }
        if !array {
            self.sort_members(&open);
        }
        let n = self.members.len() - open.first;
        if n > COUNT_MAX {
            return Err(Error::fmt(
                Code::Full,
                format_args!("a container holds at most {COUNT_MAX} elements"),
            ));
        }

        let sorted = if array { 0 } else { SORTED };
        let h = head::head(Tag::Container, open.flags | OFFSETS | sorted, n);
        let entries_end = 4 + layout::keys_area(h, n) + n * 8;
        let key_bytes: usize = self.members[open.first..]
            .iter()
            .map(|m| m.key_len as usize)
            .sum();

        let children_at = open.at + 4;
        self.scratch.clear();
        self.scratch.extend_from_slice(&self.out[children_at..]);
        self.out.truncate(children_at);
        self.out[open.at..children_at].copy_from_slice(&h.to_le_bytes());

        if !array {
            if open.flags & INTERNED != 0 {
                for i in open.first..self.members.len() {
                    self.out
                        .extend_from_slice(&self.members[i].id.to_le_bytes());
                }
                // Two byte ids leave the entry table off a four byte stride
                // half the time, so the area is padded up rather than the
                // reader being made to cope with both.
                if n % 2 == 1 {
                    self.out.extend_from_slice(&[0; 2]);
                }
            } else {
                let mut key_at = entries_end;
                for i in open.first..self.members.len() {
                    let off = u32::try_from(key_at).map_err(|_| too_big())?;
                    self.out.extend_from_slice(&off.to_le_bytes());
                    key_at += self.members[i].key_len as usize;
                }
            }
        }

        let mut val_at = entries_end + key_bytes;
        for i in open.first..self.members.len() {
            let m = self.members[i];
            self.out.extend_from_slice(&m.head.to_le_bytes());
            let off = u32::try_from(val_at).map_err(|_| too_big())?;
            self.out.extend_from_slice(&off.to_le_bytes());
            val_at += m.len as usize;
        }

        if !array && open.flags & INTERNED == 0 {
            for i in open.first..self.members.len() {
                let m = self.members[i];
                let at = m.key_at as usize;
                self.out
                    .extend_from_slice(&self.keys[at..at + m.key_len as usize]);
            }
        }

        // The children come back in entry order, so the value region ends up
        // sorted the way the entry table is. That is what lets a reader work
        // out one element's length from the next element's offset, and the
        // whole container's from its last.
        for i in open.first..self.members.len() {
            let m = self.members[i];
            let from = m.at as usize - children_at;
            self.out
                .extend_from_slice(&self.scratch[from..from + m.len as usize]);
        }

        self.members.truncate(open.first);
        self.keys.truncate(open.keys_at);
        if !self.open.is_empty() {
            self.pending = Some(open.key);
        }
        self.record(open.at)
    }

    /// Put an object's members in key order, and drop all but the last of any
    /// key written more than once.
    fn sort_members(&mut self, open: &Open) {
        let interned = open.flags & INTERNED != 0;
        let keys = &self.keys;
        let key_of = |m: &Member| {
            let at = m.key_at as usize;
            &keys[at..at + m.key_len as usize]
        };
        self.members[open.first..].sort_by(|a, b| {
            if interned {
                a.id.cmp(&b.id).then(a.seq.cmp(&b.seq))
            } else {
                key_order(key_of(a), key_of(b)).then(a.seq.cmp(&b.seq))
            }
        });

        let same = |a: &Member, b: &Member| {
            if interned {
                a.id == b.id
            } else {
                key_of(a) == key_of(b)
            }
        };
        let mut write = open.first;
        let mut read = open.first;
        while read < self.members.len() {
            let mut run = read + 1;
            while run < self.members.len() && same(&self.members[read], &self.members[run]) {
                run += 1;
            }
            // Equal keys are adjacent and in insertion order, so the last of a
            // run is the one that wins. The ones that lose stay in `out` as
            // bytes nothing points at, which costs space in a document that
            // repeats a key and nothing at all in one that does not.
            self.members[write] = self.members[run - 1];
            write += 1;
            read = run;
        }
        self.members.truncate(write);
    }
}

/// The fewest bytes `v` fits in, two's complement.
fn int_width(v: i64) -> usize {
    if i64::from(v as i8) == v {
        1
    } else if i64::from(v as i16) == v {
        2
    } else if i64::from(v as i32) == v {
        4
    } else {
        8
    }
}

fn too_big() -> Error {
    Error::new(Code::Full, "a document is at most four gigabytes")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::head::Kind;

    /// Build a value and check that it is sound in every way the reader can
    /// check, then hand the bytes back.
    fn built(f: impl FnOnce(&mut Builder) -> Result<()>) -> Vec<u8> {
        let mut b = Builder::new();
        f(&mut b).expect("the builder accepted every call");
        let bytes = b.finish().expect("the value is finished").to_vec();
        let v = Value::new(&bytes).expect("the reader accepts it");
        assert!(v.validate(), "the value is self consistent");
        assert_eq!(
            v.encoded_len(),
            Some(bytes.len()),
            "the value is exactly as long as the buffer"
        );
        bytes
    }

    #[test]
    fn every_scalar_comes_back_as_itself() {
        let cases: Vec<(Vec<u8>, Kind)> = vec![
            (built(|b| b.null()), Kind::Null),
            (built(|b| b.bool(true)), Kind::Bool),
            (built(|b| b.bool(false)), Kind::Bool),
            (built(|b| b.int(-9)), Kind::Int),
            (built(|b| b.float(1.5)), Kind::Float),
            (built(|b| b.text("hello")), Kind::Text),
        ];
        for (bytes, kind) in &cases {
            assert_eq!(Value::new(bytes).expect("readable").kind(), *kind);
        }
        assert!(Value::new(&cases[0].0).expect("readable").is_null());
        assert_eq!(
            Value::new(&cases[1].0).expect("readable").as_bool(),
            Some(true)
        );
        assert_eq!(
            Value::new(&cases[2].0).expect("readable").as_bool(),
            Some(false)
        );
        assert_eq!(
            Value::new(&cases[3].0).expect("readable").as_int(),
            Some(-9)
        );
        assert_eq!(
            Value::new(&cases[4].0).expect("readable").as_float(),
            Some(1.5)
        );
        assert_eq!(
            Value::new(&cases[5].0).expect("readable").as_text(),
            Some("hello")
        );
    }

    #[test]
    fn an_integer_takes_as_few_bytes_as_it_fits_in() {
        // The width changes where two's complement says it should, and both
        // sides of every boundary read back as themselves.
        let cases = [
            (0i64, 1usize),
            (127, 1),
            (-128, 1),
            (128, 2),
            (-129, 2),
            (32_767, 2),
            (-32_768, 2),
            (32_768, 4),
            (2_147_483_647, 4),
            (-2_147_483_648, 4),
            (2_147_483_648, 8),
            (i64::MIN, 8),
            (i64::MAX, 8),
        ];
        for (v, width) in cases {
            let bytes = built(|b| b.int(v));
            assert_eq!(bytes.len(), 4 + width, "{v} takes {width} bytes");
            assert_eq!(Value::new(&bytes).expect("readable").as_int(), Some(v));
        }
    }

    #[test]
    fn an_object_comes_back_in_key_order_whatever_order_it_went_in() {
        let bytes = built(|b| {
            b.begin_object()?;
            for k in ["zebra", "b", "aa", "a", "yak"] {
                b.key(k.as_bytes())?;
                b.text(k)?;
            }
            b.end_object()
        });
        let v = Value::new(&bytes).expect("readable");
        let keys: Vec<&[u8]> = v.members().map(|(k, _)| k).collect();
        // Shorter first, then by bytes.
        assert_eq!(keys, [&b"a"[..], b"b", b"aa", b"yak", b"zebra"]);
        for k in ["zebra", "b", "aa", "a", "yak"] {
            assert_eq!(v.get(k.as_bytes()).expect("found").as_text(), Some(k));
        }
        assert!(v.get(b"nope").is_none());
        assert!(v.get(b"").is_none());
    }

    #[test]
    fn writing_a_key_twice_keeps_the_last_one_and_leaves_no_dead_bytes() {
        let bytes = built(|b| {
            b.begin_object()?;
            b.key(b"a")?;
            b.int(1)?;
            b.key(b"b")?;
            b.int(2)?;
            b.key(b"a")?;
            b.text("the winner")?;
            b.key(b"a")?;
            b.int(3)?;
            b.end_object()
        });
        let v = Value::new(&bytes).expect("readable");
        assert_eq!(v.len(), 2, "two keys, however many times they were written");
        assert_eq!(v.get(b"a").expect("found").as_int(), Some(3));
        assert_eq!(v.get(b"b").expect("found").as_int(), Some(2));
        // `built` already checked that the encoded length is the buffer length,
        // which is the check that the losing values were not left behind.
        assert_eq!(bytes.len(), 4 + 2 * 4 + 2 * 8 + 2 + 5 + 5);
    }

    #[test]
    fn a_nested_document_reads_at_every_level() {
        let bytes = built(|b| {
            b.begin_object()?;
            b.key(b"id")?;
            b.int(7)?;
            b.key(b"lines")?;
            b.begin_array()?;
            for i in 0..3i64 {
                b.begin_object()?;
                b.key(b"sku")?;
                b.int(i)?;
                b.key(b"note")?;
                b.text("a line of some length so the offsets move")?;
                b.end_object()?;
            }
            b.end_array()?;
            b.key(b"open")?;
            b.bool(true)?;
            b.end_object()
        });
        let v = Value::new(&bytes).expect("readable");
        assert_eq!(v.get(b"id").expect("found").as_int(), Some(7));
        assert_eq!(v.get(b"open").expect("found").as_bool(), Some(true));
        let lines = v.get(b"lines").expect("found");
        assert_eq!(lines.kind(), Kind::Array);
        assert_eq!(lines.len(), 3);
        for i in 0..3i64 {
            let line = lines.at(i as usize).expect("an element");
            assert_eq!(line.get(b"sku").expect("found").as_int(), Some(i));
            assert!(line.get(b"note").expect("found").as_text().is_some());
            // A child is a whole value on its own, which is what makes a copy
            // out of a document a memcpy and not a re-encode.
            let alone = line.as_bytes().expect("a length");
            let again = Value::new(alone).expect("readable on its own");
            assert!(again.validate());
            assert_eq!(again.get(b"sku").expect("found").as_int(), Some(i));
        }
    }

    #[test]
    fn an_empty_container_is_four_bytes() {
        let obj = built(|b| {
            b.begin_object()?;
            b.end_object()
        });
        assert_eq!(obj.len(), 4);
        let v = Value::new(&obj).expect("readable");
        assert_eq!(v.kind(), Kind::Object);
        assert!(v.is_empty());
        assert!(v.get(b"a").is_none());

        let arr = built(|b| {
            b.begin_array()?;
            b.end_array()
        });
        assert_eq!(arr.len(), 4);
        let v = Value::new(&arr).expect("readable");
        assert_eq!(v.kind(), Kind::Array);
        assert!(v.is_empty());
        assert!(v.at(0).is_none());
    }

    #[test]
    fn an_interned_object_looks_up_by_id() {
        // Odd and even counts both, since an odd number of two byte ids leaves
        // the entry table off a four byte stride without the padding.
        for n in [1u16, 2, 3, 8, 9] {
            let bytes = built(|b| {
                b.begin_object_interned()?;
                for id in (0..n).rev() {
                    b.key_id(id * 3)?;
                    b.int(i64::from(id))?;
                }
                b.end_object()
            });
            let v = Value::new(&bytes).expect("readable");
            assert!(v.is_interned());
            assert_eq!(v.len(), usize::from(n));
            for id in 0..n {
                assert_eq!(
                    v.get_id(id * 3).expect("found").as_int(),
                    Some(i64::from(id))
                );
            }
            assert!(v.get_id(1).is_none(), "1 is not a multiple of 3");
            assert!(v.key_at(0).is_none(), "the names are not in the document");
            assert_eq!(v.key_id_at(0), Some(0));
        }
    }

    #[test]
    fn a_thousand_keys_are_all_findable() {
        let names: Vec<String> = (0..1_000).map(|i| format!("field{i}")).collect();
        let bytes = built(|b| {
            b.begin_object()?;
            for (i, name) in names.iter().enumerate() {
                b.key(name.as_bytes())?;
                b.int(i as i64)?;
            }
            b.end_object()
        });
        let v = Value::new(&bytes).expect("readable");
        assert_eq!(v.len(), 1_000);
        for (i, name) in names.iter().enumerate() {
            assert_eq!(
                v.get(name.as_bytes()).expect("found").as_int(),
                Some(i as i64)
            );
        }
        assert!(v.get(b"field1000").is_none());
    }

    #[test]
    fn a_value_that_is_already_encoded_can_be_copied_in() {
        let inner = built(|b| {
            b.begin_object()?;
            b.key(b"x")?;
            b.int(3)?;
            b.end_object()
        });
        let bytes = built(|b| {
            b.begin_array()?;
            b.int(1)?;
            b.embed(&Value::new(&inner).expect("readable"))?;
            b.int(2)?;
            b.end_array()
        });
        let v = Value::new(&bytes).expect("readable");
        assert_eq!(v.len(), 3);
        assert_eq!(
            v.at(1)
                .expect("an element")
                .get(b"x")
                .expect("found")
                .as_int(),
            Some(3)
        );
    }

    #[test]
    fn a_builder_can_be_used_again() {
        let mut b = Builder::new();
        b.int(1).expect("a value");
        assert_eq!(b.finish().expect("finished").len(), 5);
        b.clear();
        b.text("hello").expect("a value");
        let bytes = b.finish().expect("finished");
        assert_eq!(
            Value::new(bytes).expect("readable").as_text(),
            Some("hello")
        );
    }

    #[test]
    fn the_builder_says_no_to_every_way_of_getting_it_wrong() {
        let bad = |f: fn(&mut Builder) -> Result<()>| {
            let mut b = Builder::new();
            f(&mut b).unwrap_err()
        };

        // A key with nothing after it.
        assert!(
            bad(|b| {
                b.begin_object()?;
                b.key(b"a")?;
                b.end_object()
            })
            .message()
            .contains("no value")
        );
        // Two keys in a row.
        assert!(
            bad(|b| {
                b.begin_object()?;
                b.key(b"a")?;
                b.key(b"b")
            })
            .message()
            .contains("two keys")
        );
        // A value in an object with no key.
        assert!(
            bad(|b| {
                b.begin_object()?;
                b.int(1)
            })
            .message()
            .contains("needs a key")
        );
        // A key in an array.
        assert!(
            bad(|b| {
                b.begin_array()?;
                b.key(b"a")
            })
            .message()
            .contains("no key")
        );
        // A key with nothing open.
        assert!(bad(|b| b.key(b"a")).message().contains("no object open"));
        // Ending the wrong thing.
        assert!(
            bad(|b| {
                b.begin_object()?;
                b.end_array()
            })
            .message()
            .contains("not ended by")
        );
        // Ending nothing.
        assert!(
            bad(|b| b.end_object())
                .message()
                .contains("nothing is open")
        );
        // Two values at the top level.
        assert!(
            bad(|b| {
                b.int(1)?;
                b.int(2)
            })
            .message()
            .contains("already written")
        );
        // Key bytes into an interned object and the other way round.
        assert!(
            bad(|b| {
                b.begin_object_interned()?;
                b.key(b"a")
            })
            .message()
            .contains("key ids")
        );
        assert!(
            bad(|b| {
                b.begin_object()?;
                b.key_id(1)
            })
            .message()
            .contains("key bytes")
        );
    }

    #[test]
    fn finishing_early_is_an_error_and_not_a_short_document() {
        let mut b = Builder::new();
        assert!(b.finish().unwrap_err().message().contains("no value"));
        b.begin_array().expect("open");
        assert!(b.finish().unwrap_err().message().contains("unclosed array"));
        b.end_array().expect("close");
        b.finish().expect("finished now");

        let mut b = Builder::new();
        b.begin_object().expect("open");
        assert!(
            b.finish()
                .unwrap_err()
                .message()
                .contains("unclosed object")
        );
    }

    #[test]
    fn a_document_nests_as_deep_as_the_reader_will_walk_and_no_deeper() {
        let mut b = Builder::new();
        for _ in 0..DEPTH_MAX {
            b.begin_array().expect("within the limit");
        }
        assert!(
            b.begin_array()
                .unwrap_err()
                .message()
                .contains("nests at most"),
            "one past the limit is refused"
        );
        for _ in 0..DEPTH_MAX {
            b.end_array().expect("close");
        }
        let bytes = b.finish().expect("finished").to_vec();
        let v = Value::new(&bytes).expect("readable");
        assert!(v.validate(), "the reader walks all of it");
        assert_eq!(v.encoded_len(), Some(bytes.len()));
    }
}
