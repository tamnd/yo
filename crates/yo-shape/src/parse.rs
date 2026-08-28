//! Reading a description back into a tree.
//!
//! Writing a description is the hot side and it is a `Vec<u8>` push. Reading
//! one happens when a tag comparison already failed, or when a tool wants to
//! print what is in a file, so this side is written for clarity and for saying
//! exactly where a description went wrong.

use core::fmt;

use yo_common::{Code, Error, Result};

use crate::desc::{Desc, Metric, Prim};

/// A description, parsed.
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    /// A primitive.
    Prim(Prim),
    /// An optional value.
    Optional(Box<Type>),
    /// A sequence.
    List(Box<Type>),
    /// A mapping.
    Map(Box<Type>, Box<Type>),
    /// A struct with its fields in declaration order.
    Struct {
        /// The type's name.
        name: String,
        /// Name and type, in declaration order.
        fields: Vec<(String, Type)>,
    },
    /// An enumeration with its variants in declaration order.
    Enum {
        /// The type's name.
        name: String,
        /// Variant names, in declaration order.
        variants: Vec<String>,
    },
    /// A vector of a fixed width, compared one way.
    Vector {
        /// How many dimensions.
        dim: u32,
        /// How it is compared. Held as written, because a file may name a
        /// metric this build has never heard of.
        metric: String,
    },
    /// A reference back to a struct that encloses this one.
    Ref(String),
}

impl Type {
    /// The name, for the types that have one.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        match self {
            Type::Struct { name, .. } | Type::Enum { name, .. } | Type::Ref(name) => Some(name),
            _ => None,
        }
    }

    /// What to call this kind of type in a message.
    #[must_use]
    pub fn kind(&self) -> &'static str {
        match self {
            Type::Prim(_) => "primitive",
            Type::Optional(_) => "optional",
            Type::List(_) => "list",
            Type::Map(_, _) => "map",
            Type::Struct { .. } => "struct",
            Type::Enum { .. } => "enum",
            Type::Vector { .. } => "vector",
            Type::Ref(_) => "reference",
        }
    }
}

/// Parse a whole description. Trailing bytes are an error, because a
/// description that is a valid type followed by rubbish is not a description.
///
/// # Errors
///
/// [`Code::Corrupt`] with the byte offset, for anything that is not a
/// description this build can read.
pub fn parse(desc: &Desc) -> Result<Type> {
    let bytes = desc.as_bytes();
    let mut p = Parser { bytes, at: 0 };
    let ty = p.ty()?;
    if p.at != bytes.len() {
        return Err(p.bad("trailing bytes after the type"));
    }
    Ok(ty)
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Parser<'_> {
    fn bad(&self, what: &str) -> Error {
        Error::fmt(
            Code::Corrupt,
            format_args!("shape description is malformed at byte {}: {what}", self.at),
        )
        .at(u32::try_from(self.at).unwrap_or(u32::MAX))
    }

    fn peek(&self) -> Result<u8> {
        self.bytes
            .get(self.at)
            .copied()
            .ok_or_else(|| self.bad("the description ends here"))
    }

    fn varint(&mut self) -> Result<u32> {
        let mut value: u32 = 0;
        let mut shift = 0;
        loop {
            let byte = self.peek()?;
            self.at += 1;
            let part = u32::from(byte & 0x7f);
            value |= part
                .checked_shl(shift)
                .ok_or_else(|| self.bad("a length does not fit in 32 bits"))?;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
            if shift >= 32 {
                return Err(self.bad("a length does not fit in 32 bits"));
            }
        }
    }

    fn name(&mut self) -> Result<String> {
        let len = self.varint()? as usize;
        let end = self
            .at
            .checked_add(len)
            .filter(|&end| end <= self.bytes.len())
            .ok_or_else(|| self.bad("a name runs past the end"))?;
        let text = core::str::from_utf8(&self.bytes[self.at..end])
            .map_err(|_| self.bad("a name is not UTF-8"))?
            .to_owned();
        self.at = end;
        Ok(text)
    }

    fn ty(&mut self) -> Result<Type> {
        match self.peek()? {
            b'O' => {
                self.at += 1;
                Ok(Type::Optional(Box::new(self.ty()?)))
            }
            b'L' => {
                self.at += 1;
                Ok(Type::List(Box::new(self.ty()?)))
            }
            b'M' => {
                self.at += 1;
                let key = self.ty()?;
                let value = self.ty()?;
                Ok(Type::Map(Box::new(key), Box::new(value)))
            }
            b'R' => {
                self.at += 1;
                Ok(Type::Ref(self.name()?))
            }
            b'V' => {
                self.at += 1;
                let dim = self.varint()?;
                let metric = self.name()?;
                Ok(Type::Vector { dim, metric })
            }
            b'S' => {
                self.at += 1;
                let name = self.name()?;
                let count = self.varint()?;
                let mut fields = Vec::with_capacity(count.min(64) as usize);
                for _ in 0..count {
                    let field = self.name()?;
                    fields.push((field, self.ty()?));
                }
                Ok(Type::Struct { name, fields })
            }
            b'E' => {
                self.at += 1;
                let name = self.name()?;
                let count = self.varint()?;
                let mut variants = Vec::with_capacity(count.min(64) as usize);
                for _ in 0..count {
                    variants.push(self.name()?);
                }
                Ok(Type::Enum { name, variants })
            }
            _ => self.prim(),
        }
    }

    fn prim(&mut self) -> Result<Type> {
        let rest = &self.bytes[self.at..];
        for &p in Prim::ALL {
            if rest.starts_with(p.token().as_bytes()) {
                self.at += p.token().len();
                return Ok(Type::Prim(p));
            }
        }
        Err(self.bad("not a type"))
    }
}

/// The metric a parsed vector names, when this build knows it.
#[must_use]
pub fn metric_of(name: &str) -> Option<Metric> {
    [Metric::L2, Metric::Cosine, Metric::Ip, Metric::Hamming]
        .into_iter()
        .find(|m| m.token() == name)
}

impl fmt::Display for Type {
    /// The rendering used in messages: a struct at the top shows its fields,
    /// a struct anywhere else shows its name. A shape mismatch message has to
    /// fit on a terminal and the field that changed is usually near the top.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Struct { name, fields } => {
                write!(f, "{name} {{ ")?;
                for (i, (field, ty)) in fields.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{field}: {}", Inner(ty))?;
                }
                f.write_str(" }")
            }
            other => write!(f, "{}", Inner(other)),
        }
    }
}

/// A type in field position, where a struct is only its name.
struct Inner<'a>(&'a Type);

impl fmt::Display for Inner<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Type::Prim(p) => write!(f, "{p}"),
            Type::Optional(t) => write!(f, "O {}", Inner(t)),
            Type::List(t) => write!(f, "L {}", Inner(t)),
            Type::Map(k, v) => write!(f, "M {} {}", Inner(k), Inner(v)),
            Type::Struct { name, .. } => f.write_str(name),
            Type::Enum { name, variants } => {
                write!(f, "E {name}[{}]", variants.join(","))
            }
            Type::Vector { dim, metric } => write!(f, "V {dim} {metric}"),
            Type::Ref(name) => write!(f, "R {name}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desc::{Describe, Shape};

    fn parsed(build: impl FnOnce(&mut Desc)) -> Type {
        let mut d = Desc::new();
        build(&mut d);
        parse(&d).expect("this description was just written")
    }

    #[test]
    fn every_primitive_survives_a_round_trip() {
        for &p in Prim::ALL {
            let ty = parsed(|d| d.prim(p));
            assert_eq!(ty, Type::Prim(p), "{p}");
        }
    }

    #[test]
    fn containers_survive_a_round_trip() {
        assert_eq!(
            parsed(|d| d.optional(u64::describe)),
            Type::Optional(Box::new(Type::Prim(Prim::U64)))
        );
        assert_eq!(
            parsed(|d| d.map(String::describe, <Vec<u8> as Shape>::describe)),
            Type::Map(
                Box::new(Type::Prim(Prim::Str)),
                Box::new(Type::List(Box::new(Type::Prim(Prim::U8))))
            )
        );
    }

    #[test]
    fn a_struct_keeps_its_field_order() {
        let ty = parsed(|d| {
            d.strukt(
                "Order",
                &[
                    ("id", u64::describe),
                    ("note", <Option<String> as Shape>::describe),
                ],
            );
        });
        let Type::Struct { name, fields } = &ty else {
            panic!("expected a struct, got {ty:?}");
        };
        assert_eq!(name, "Order");
        assert_eq!(fields[0].0, "id");
        assert_eq!(fields[1].0, "note");
        assert_eq!(ty.to_string(), "Order { id: u64, note: O str }");
    }

    #[test]
    fn an_enum_keeps_its_variant_order() {
        let ty = parsed(|d| d.enumeration("Status", &["Open", "Paid", "Shipped"]));
        assert_eq!(ty.to_string(), "E Status[Open,Paid,Shipped]");
    }

    #[test]
    fn a_vector_keeps_its_dimension() {
        let ty = parsed(|d| d.vector(1536, Metric::Ip));
        assert_eq!(
            ty,
            Type::Vector {
                dim: 1536,
                metric: "ip".into()
            }
        );
        assert_eq!(ty.to_string(), "V 1536 ip");
        assert_eq!(metric_of("ip"), Some(Metric::Ip));
        assert_eq!(metric_of("euclidean"), None);
    }

    #[test]
    fn a_recursive_type_parses_to_a_reference() {
        fn node(d: &mut Desc) {
            d.strukt("Node", &[("kids", kids as Describe)]);
        }
        fn kids(d: &mut Desc) {
            d.list(node);
        }
        let ty = parsed(node);
        assert_eq!(ty.to_string(), "Node { kids: L R Node }");
    }

    /// A nested struct shows as its name, which is what `15` section 3.2's
    /// example message does.
    #[test]
    fn a_nested_struct_renders_as_its_name() {
        fn line(d: &mut Desc) {
            d.strukt("Line", &[("sku", String::describe)]);
        }
        let ty = parsed(|d| d.strukt("Order", &[("lines", |d: &mut Desc| d.list(line))]));
        assert_eq!(ty.to_string(), "Order { lines: L Line }");
    }

    #[test]
    fn rubbish_is_rejected_with_the_offset() {
        let bad = Desc::from_bytes(b"u64u64".to_vec());
        let e = parse(&bad).expect_err("two types in a row is not one type");
        assert_eq!(e.code(), Code::Corrupt);
        assert_eq!(e.position(), Some(3));

        for cut in ["S", "S\u{5}Ord", "L", "MstrL", "V\u{80}"] {
            let e = parse(&Desc::from_bytes(cut.as_bytes().to_vec()))
                .expect_err("a truncated description is not a description");
            assert_eq!(e.code(), Code::Corrupt, "{cut:?}");
        }

        let e = parse(&Desc::from_bytes(b"q".to_vec())).expect_err("q is not a type");
        assert!(e.message().contains("not a type"), "{e}");
    }

    /// A count that says more fields than are there fails rather than
    /// returning half a struct, because half a struct compares as a shape
    /// change and would send the caller after the wrong bug.
    #[test]
    fn a_lying_field_count_is_rejected() {
        let bad = Desc::from_bytes(b"S\x01P\x02\x01xu64".to_vec());
        assert_eq!(
            parse(&bad).expect_err("one field, not two").code(),
            Code::Corrupt
        );
    }
}
