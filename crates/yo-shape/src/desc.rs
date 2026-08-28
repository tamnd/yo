//! The canonical description and the tag over it (`15` section 3.1).

use core::fmt;

use yo_common::blake3;

/// The primitives. Every one of them says its width, because "int" means a
/// different number of bytes in every language that will open the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prim {
    /// An unsigned byte.
    U8,
    /// An unsigned 16 bit integer.
    U16,
    /// An unsigned 32 bit integer.
    U32,
    /// An unsigned 64 bit integer.
    U64,
    /// A signed byte.
    I8,
    /// A signed 16 bit integer.
    I16,
    /// A signed 32 bit integer.
    I32,
    /// A signed 64 bit integer.
    I64,
    /// A 32 bit float.
    F32,
    /// A 64 bit float.
    F64,
    /// A boolean.
    Bool,
    /// UTF-8 text.
    Str,
    /// Bytes with no encoding attached.
    Bytes,
}

impl Prim {
    /// The token this primitive is written as.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Prim::U8 => "u8",
            Prim::U16 => "u16",
            Prim::U32 => "u32",
            Prim::U64 => "u64",
            Prim::I8 => "i8",
            Prim::I16 => "i16",
            Prim::I32 => "i32",
            Prim::I64 => "i64",
            Prim::F32 => "f32",
            Prim::F64 => "f64",
            Prim::Bool => "bool",
            Prim::Str => "str",
            Prim::Bytes => "bytes",
        }
    }

    /// Every token. No token is a prefix of another, so a parser can try them
    /// in any order and still read a description exactly one way.
    pub(crate) const ALL: &'static [Prim] = &[
        Prim::Bytes,
        Prim::Bool,
        Prim::Str,
        Prim::U8,
        Prim::U16,
        Prim::U32,
        Prim::U64,
        Prim::I8,
        Prim::I16,
        Prim::I32,
        Prim::I64,
        Prim::F32,
        Prim::F64,
    ];
}

impl fmt::Display for Prim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.token())
    }
}

/// How a vector is compared. Part of the shape because a collection built for
/// cosine and searched as if it were L2 gives wrong answers quietly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Euclidean distance.
    L2,
    /// Cosine similarity.
    Cosine,
    /// Inner product.
    Ip,
    /// Hamming distance, for binary vectors.
    Hamming,
}

impl Metric {
    /// The name this metric is written as.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Metric::L2 => "l2",
            Metric::Cosine => "cosine",
            Metric::Ip => "ip",
            Metric::Hamming => "hamming",
        }
    }
}

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.token())
    }
}

/// What a type contributes to a description.
///
/// A plain function pointer rather than a closure, so that a struct's field
/// list is a constant slice and the field count can never disagree with the
/// number of fields written.
pub type Describe = fn(&mut Desc);

/// A type that can describe itself.
///
/// The description is the type's identity in the file, so two types that
/// describe themselves the same way are the same type as far as `yo` is
/// concerned, whatever they are called in the host language.
pub trait Shape {
    /// Write this type into the description being built.
    fn describe(d: &mut Desc);
}

/// A canonical description, built by writing and read as bytes.
///
/// The bytes are the whole point: they are what gets stored, what gets hashed
/// into the [`Tag`], and what a binding in another language has to produce
/// exactly. Nothing here depends on Rust.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Desc {
    bytes: Vec<u8>,
    /// Ranges into `bytes` naming the structs currently being written, so that
    /// a type that contains itself writes a reference instead of recursing
    /// until the stack runs out.
    open: Vec<(usize, usize)>,
}

impl Desc {
    /// An empty description.
    #[must_use]
    pub fn new() -> Desc {
        Desc {
            bytes: Vec::new(),
            open: Vec::new(),
        }
    }

    /// The description of `T`.
    #[must_use]
    pub fn of<T: Shape + ?Sized>() -> Desc {
        let mut d = Desc::new();
        T::describe(&mut d);
        d
    }

    /// A description someone else produced, from the catalogue or the wire.
    #[must_use]
    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Desc {
        Desc {
            bytes: bytes.into(),
            open: Vec::new(),
        }
    }

    /// The bytes, which are what gets stored.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The description as text, for a message or a log line. It is all ASCII
    /// apart from names, which are UTF-8, so this never fails in practice.
    #[must_use]
    pub fn as_text(&self) -> String {
        String::from_utf8_lossy(&self.bytes).into_owned()
    }

    /// The tag: the first 128 bits of the BLAKE3 hash of these bytes.
    #[must_use]
    pub fn tag(&self) -> Tag {
        Tag::of(&self.bytes)
    }

    /// Whether anything has been written.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// A primitive.
    pub fn prim(&mut self, p: Prim) {
        self.bytes.extend_from_slice(p.token().as_bytes());
    }

    /// An optional value.
    pub fn optional(&mut self, inner: Describe) {
        self.bytes.push(b'O');
        inner(self);
    }

    /// A sequence.
    pub fn list(&mut self, inner: Describe) {
        self.bytes.push(b'L');
        inner(self);
    }

    /// A mapping, key type then value type.
    pub fn map(&mut self, key: Describe, value: Describe) {
        self.bytes.push(b'M');
        key(self);
        value(self);
    }

    /// A vector of `dim` dimensions compared with `metric`.
    pub fn vector(&mut self, dim: u32, metric: Metric) {
        self.bytes.push(b'V');
        self.varint(dim);
        self.name(metric.token());
    }

    /// A named reference, which is what a recursive type writes on the way
    /// down. Written for you by [`Desc::strukt`]; call it directly only when
    /// building a description by hand.
    pub fn reference(&mut self, name: &str) {
        self.bytes.push(b'R');
        self.name(name);
    }

    /// A struct, with its fields in declaration order.
    ///
    /// Order is layout, so it is part of the shape and reordering fields is a
    /// breaking change (`15` section 5). That is why this takes a slice rather
    /// than a map.
    ///
    /// If `name` is already being written further up, this writes a reference
    /// instead and does not descend, which is what makes a linked list or a
    /// tree describable at all.
    pub fn strukt(&mut self, name: &str, fields: &[(&str, Describe)]) {
        if self.is_open(name) {
            self.reference(name);
            return;
        }

        self.bytes.push(b'S');
        let at = self.name(name);
        self.open.push(at);
        self.varint(len_as_u32(fields.len()));
        for (field, describe) in fields {
            self.name(field);
            describe(self);
        }
        self.open.pop();
    }

    /// An enumeration, with its variants in declaration order.
    ///
    /// Variants carry no payload here. A variant that carries data is a struct
    /// in a field of its own, which is how it has to be written until the
    /// grammar grows a form for it.
    pub fn enumeration(&mut self, name: &str, variants: &[&str]) {
        self.bytes.push(b'E');
        self.name(name);
        self.varint(len_as_u32(variants.len()));
        for variant in variants {
            self.name(variant);
        }
    }

    /// Length prefixed UTF-8, returning where it landed.
    fn name(&mut self, s: &str) -> (usize, usize) {
        self.varint(len_as_u32(s.len()));
        let at = self.bytes.len();
        self.bytes.extend_from_slice(s.as_bytes());
        (at, s.len())
    }

    /// LEB128, because a length has to be one number in every language and a
    /// fixed width would either waste bytes or cap a name.
    fn varint(&mut self, mut n: u32) {
        loop {
            let byte = (n & 0x7f) as u8;
            n >>= 7;
            if n == 0 {
                self.bytes.push(byte);
                return;
            }
            self.bytes.push(byte | 0x80);
        }
    }

    fn is_open(&self, name: &str) -> bool {
        self.open
            .iter()
            .any(|&(at, len)| &self.bytes[at..at + len] == name.as_bytes())
    }
}

/// A name or a count that does not fit in 32 bits is a bug in the caller, not
/// a case to handle. Saturating keeps the description well formed either way.
fn len_as_u32(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// The 128 bit shape tag.
///
/// Half of a BLAKE3 hash, which is enough: the tag is compared, never searched
/// for, and a collision would need two descriptions someone deliberately built
/// to collide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Tag([u8; 16]);

impl Tag {
    /// The tag of a collection created over the wire, which has no type.
    pub const UNTYPED: Tag = Tag([0; 16]);

    /// The tag of these description bytes.
    #[must_use]
    pub fn of(description: &[u8]) -> Tag {
        let full = blake3::hash(description);
        let mut tag = [0u8; 16];
        tag.copy_from_slice(&full[..16]);
        Tag(tag)
    }

    /// The tag of `T`, which is the call a typed handle makes when it opens.
    #[must_use]
    pub fn for_type<T: Shape + ?Sized>() -> Tag {
        Desc::of::<T>().tag()
    }

    /// The bytes as stored in the catalogue.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// A tag read back out of the catalogue.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 16]) -> Tag {
        Tag(bytes)
    }

    /// Whether this is the untyped tag, meaning the collection was created
    /// over RESP3 and is checked per element instead (`15` section 3.3).
    #[must_use]
    pub fn is_untyped(&self) -> bool {
        self.0 == [0; 16]
    }
}

impl fmt::Display for Tag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&blake3::to_hex(&self.0))
    }
}

macro_rules! prim_shape {
    ($($t:ty => $p:expr),* $(,)?) => {
        $(impl Shape for $t {
            fn describe(d: &mut Desc) {
                d.prim($p);
            }
        })*
    };
}

prim_shape! {
    u8 => Prim::U8,
    u16 => Prim::U16,
    u32 => Prim::U32,
    u64 => Prim::U64,
    i8 => Prim::I8,
    i16 => Prim::I16,
    i32 => Prim::I32,
    i64 => Prim::I64,
    f32 => Prim::F32,
    f64 => Prim::F64,
    bool => Prim::Bool,
    str => Prim::Str,
    String => Prim::Str,
}

/// Raw bytes, for a field that holds a blob rather than text or a list.
///
/// `Vec<u8>` describes as a list of `u8`, which is a different shape and a
/// different layout, so a field that means "bytes" says so with this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Bytes;

impl Shape for Bytes {
    fn describe(d: &mut Desc) {
        d.prim(Prim::Bytes);
    }
}

impl<T: Shape> Shape for Option<T> {
    fn describe(d: &mut Desc) {
        d.optional(T::describe);
    }
}

impl<T: Shape> Shape for Vec<T> {
    fn describe(d: &mut Desc) {
        d.list(T::describe);
    }
}

impl<T: Shape> Shape for [T] {
    fn describe(d: &mut Desc) {
        d.list(T::describe);
    }
}

impl<T: Shape, const N: usize> Shape for [T; N] {
    fn describe(d: &mut Desc) {
        d.list(T::describe);
    }
}

impl<T: Shape + ?Sized> Shape for &T {
    fn describe(d: &mut Desc) {
        T::describe(d);
    }
}

impl<T: Shape + ?Sized> Shape for Box<T> {
    fn describe(d: &mut Desc) {
        T::describe(d);
    }
}

impl<K: Shape, V: Shape> Shape for std::collections::BTreeMap<K, V> {
    fn describe(d: &mut Desc) {
        d.map(K::describe, V::describe);
    }
}

impl<K: Shape, V: Shape, S> Shape for std::collections::HashMap<K, V, S> {
    fn describe(d: &mut Desc) {
        d.map(K::describe, V::describe);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitives_write_their_own_token() {
        assert_eq!(Desc::of::<u64>().as_text(), "u64");
        assert_eq!(Desc::of::<i8>().as_text(), "i8");
        assert_eq!(Desc::of::<f32>().as_text(), "f32");
        assert_eq!(Desc::of::<bool>().as_text(), "bool");
        assert_eq!(Desc::of::<String>().as_text(), "str");
        assert_eq!(Desc::of::<Bytes>().as_text(), "bytes");
    }

    #[test]
    fn containers_nest_left_to_right() {
        assert_eq!(Desc::of::<Option<u32>>().as_text(), "Ou32");
        assert_eq!(Desc::of::<Vec<Option<i64>>>().as_text(), "LOi64");
        assert_eq!(
            Desc::of::<std::collections::BTreeMap<String, Vec<u8>>>().as_text(),
            "MstrLu8"
        );
    }

    /// A borrow and a box describe as the thing they hold, because the file
    /// does not know what a pointer is.
    #[test]
    fn indirection_is_not_part_of_the_shape() {
        assert_eq!(Desc::of::<Box<u64>>(), Desc::of::<u64>());
        assert_eq!(Desc::of::<&str>(), Desc::of::<String>());
        assert_eq!(Desc::of::<[u16; 4]>(), Desc::of::<Vec<u16>>());
    }

    fn order(d: &mut Desc) {
        d.strukt("Order", &[("id", u64::describe), ("total", f64::describe)]);
    }

    #[test]
    fn a_struct_writes_its_name_then_its_fields_in_order() {
        let mut d = Desc::new();
        order(&mut d);
        assert_eq!(d.as_text(), "S\u{5}Order\u{2}\u{2}idu64\u{5}totalf64");
    }

    /// The one that matters most, because it is the mistake the tag exists to
    /// catch and the one a schema tool that sorts field names would miss.
    #[test]
    fn reordering_fields_changes_the_tag() {
        let mut a = Desc::new();
        a.strukt("P", &[("x", u64::describe), ("y", u64::describe)]);
        let mut b = Desc::new();
        b.strukt("P", &[("y", u64::describe), ("x", u64::describe)]);
        assert_ne!(a.tag(), b.tag());
    }

    #[test]
    fn a_widening_changes_the_tag() {
        let mut a = Desc::new();
        a.strukt("P", &[("x", u32::describe)]);
        let mut b = Desc::new();
        b.strukt("P", &[("x", u64::describe)]);
        assert_ne!(a.tag(), b.tag());
        assert_ne!(a.as_bytes(), b.as_bytes());
    }

    #[test]
    fn the_same_shape_written_twice_gets_the_same_tag() {
        let mut a = Desc::new();
        order(&mut a);
        let mut b = Desc::new();
        order(&mut b);
        assert_eq!(a.tag(), b.tag());
        assert_eq!(a.tag().to_string().len(), 32);
    }

    /// A type that contains itself stops at the second mention. Without this
    /// the describer runs until the stack ends, which is a poor way to learn
    /// that a tree is a tree.
    #[test]
    fn recursion_writes_a_reference() {
        fn node(d: &mut Desc) {
            d.strukt("Node", &[("value", u64::describe), ("kids", kids)]);
        }
        fn kids(d: &mut Desc) {
            d.list(node);
        }

        let mut d = Desc::new();
        node(&mut d);
        assert_eq!(
            d.as_text(),
            "S\u{4}Node\u{2}\u{5}valueu64\u{4}kidsLR\u{4}Node"
        );
    }

    /// Two fields of the same struct type are not recursion and both expand.
    #[test]
    fn siblings_of_the_same_type_both_expand() {
        fn point(d: &mut Desc) {
            d.strukt("Point", &[("x", f64::describe)]);
        }
        let mut d = Desc::new();
        d.strukt("Line", &[("a", point), ("b", point)]);
        let text = d.as_text();
        assert_eq!(text.matches("Point").count(), 2);
        assert!(!text.contains('R'));
    }

    #[test]
    fn an_enum_writes_its_variants_in_order() {
        let mut d = Desc::new();
        d.enumeration("Status", &["Open", "Paid"]);
        assert_eq!(d.as_text(), "E\u{6}Status\u{2}\u{4}Open\u{4}Paid");
    }

    /// Asserted as bytes rather than as text, because 768 in LEB128 is `80 06`
    /// and `0x80` is not a character.
    #[test]
    fn a_vector_carries_its_dimension_and_metric() {
        let mut d = Desc::new();
        d.vector(768, Metric::Cosine);
        assert_eq!(d.as_bytes(), b"V\x80\x06\x06cosine");
    }

    /// A name longer than 127 bytes needs two length bytes, which is the only
    /// interesting thing about LEB128 and the thing a binding gets wrong.
    #[test]
    fn a_long_name_gets_a_two_byte_length() {
        let long = "a".repeat(200);
        let mut d = Desc::new();
        d.enumeration(&long, &[]);
        assert_eq!(d.as_bytes()[1], 0xc8);
        assert_eq!(d.as_bytes()[2], 0x01);
        assert_eq!(d.as_bytes().len(), 1 + 2 + 200 + 1);
    }

    #[test]
    fn the_untyped_tag_is_zero_and_says_so() {
        assert!(Tag::UNTYPED.is_untyped());
        assert_eq!(Tag::UNTYPED.to_string(), "0".repeat(32));
        assert!(!Tag::for_type::<u64>().is_untyped());
    }

    /// The tag is the first half of the BLAKE3 hash, and nothing else.
    #[test]
    fn the_tag_is_the_first_half_of_the_hash() {
        let d = Desc::of::<u64>();
        let full = blake3::hash(d.as_bytes());
        assert_eq!(d.tag().as_bytes(), &full[..16]);
        assert_eq!(d.tag().to_string(), blake3::to_hex(&full[..16]));
        assert_eq!(Tag::from_bytes(*d.tag().as_bytes()), d.tag());
    }
}
