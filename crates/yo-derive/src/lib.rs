//! `#[derive(Yo)]`, which writes a type's shape, its document encoding and the
//! indexes it declares (`15` sections 3 and 4).
//!
//! ```ignore
//! #[derive(Yo)]
//! struct Order {
//!     #[yo(id)]
//!     id: u64,
//!     #[yo(index)]
//!     status: String,
//!     #[yo(ordered)]
//!     total: f64,
//!     #[yo(array)]
//!     tags: Vec<String>,
//!     #[yo(text)]
//!     note: String,
//!     #[yo(vector = 3)]
//!     embedding: Vec<f32>,
//! }
//! ```
//!
//! That gives `Order` three things. A [shape], which is the canonical
//! description the collection is created with and every later open is checked
//! against. A document encoding, so a value goes into the store as YOJB without
//! passing through JSON text. And a list of indexes, which the collection
//! declares the first time it is opened, plus a `Path` constant per indexed
//! field so a query is written `Order::STATUS` rather than `"$.status"`.
//!
//! # The words
//!
//! `#[yo(id)]` names the field that is the document's id, and a type needs one
//! to be a document at all. `#[yo(index)]` asks equality, `#[yo(ordered)]` asks
//! equality and ranges, `#[yo(array)]` files the document under every element
//! of a list, and `#[yo(text)]` files it under every word of a string. They are
//! the four kinds a path index comes in and nothing is invented here.
//!
//! `#[yo(vector = 384)]` on a `Vec<f32>` asks for a vector index over the
//! embedding that field holds, which is the one mark that takes a number,
//! because how wide a collection's vectors are is decided when the type is
//! written and there is no reason to find it out from the first document
//! instead. It gives the type a `Vector` constant, so a nearest neighbour
//! search names the field the same way an equality lookup does.
//!
//! # Why there are no dependencies
//!
//! A derive is the one place a library gets to put three crates in everybody's
//! build graph without being asked, and the usual three are most of what a
//! cold build of a small program costs. What this reads is a struct with named
//! fields, and a field is an attribute, a visibility, a name, a colon and some
//! tokens. The compiler hands that over already split into tokens, so the
//! parser in `parse` is a few hundred lines and the types themselves are
//! carried straight back out without ever being understood.
//!
//! The cost of that choice is that the errors here are sentences rather than
//! spans, so a mistake points at the struct rather than at the word. That is
//! the trade, and it is written down rather than discovered.
//!
//! [shape]: https://docs.rs/yodb/latest/yo/trait.Shape.html

#![deny(missing_docs)]

mod parse;

use proc_macro::TokenStream;

use parse::{Field, Struct};

/// Write a type's shape, its document encoding and its indexes.
///
/// See the module docs for what the attributes mean.
#[proc_macro_derive(Yo, attributes(yo))]
pub fn yo(input: TokenStream) -> TokenStream {
    match parse::parse(input).and_then(emit) {
        Ok(out) => out,
        Err(why) => complain(&why),
    }
}

/// Hand a sentence back to the compiler instead of code.
fn complain(why: &str) -> TokenStream {
    format!("::core::compile_error!{{ {why:?} }}")
        .parse()
        .expect("a compile_error with a string in it")
}

fn emit(s: Struct) -> Result<TokenStream, String> {
    let name = &s.name;
    let mut out = String::new();

    out.push_str(&shape(&s));
    out.push_str(&field(&s));
    out.push_str(&indexed(&s));
    if let Some(id) = the_id(&s)? {
        out.push_str(&document(&s, id));
    }
    out.push_str(&paths(&s));

    out.parse().map_err(|e| {
        format!("Yo wrote something the compiler would not take for {name}, which is a bug in the derive: {e}")
    })
}

/// The one field marked `#[yo(id)]`, if there is one.
fn the_id(s: &Struct) -> Result<Option<&Field>, String> {
    let mut marked = s.fields.iter().filter(|f| f.id);
    let Some(first) = marked.next() else {
        return Ok(None);
    };
    if let Some(second) = marked.next() {
        return Err(format!(
            "{} marks both {} and {} as its id, and a document is stored under one",
            s.name, first.label, second.label
        ));
    }
    Ok(Some(first))
}

fn shape(s: &Struct) -> String {
    let mut fields = String::new();
    for f in &s.fields {
        let (label, ty) = (&f.label, &f.ty);
        fields.push_str(&format!("({label:?}, <{ty} as ::yo::Shape>::describe),"));
    }
    let (name, label) = (&s.name, &s.name);
    format!(
        "#[automatically_derived]
impl ::yo::Shape for {name} {{
    fn describe(d: &mut ::yo::Desc) {{
        d.strukt({label:?}, &[{fields}]);
    }}
}}
"
    )
}

fn field(s: &Struct) -> String {
    let mut write = String::new();
    let mut read = String::new();
    for f in &s.fields {
        let (name, label) = (&f.name, &f.label);
        write.push_str(&format!(
            "b.key({label:?}.as_bytes())?; ::yo::doc::Field::write(&self.{name}, b)?;"
        ));
        read.push_str(&format!("{name}: ::yo::doc::at(d, {label:?})?,"));
    }
    let name = &s.name;
    format!(
        "#[automatically_derived]
impl ::yo::doc::Field for {name} {{
    fn write(&self, b: &mut ::yo::doc::Builder) -> ::yo::Result<()> {{
        b.begin_object()?;
        {write}
        b.end_object()
    }}

    fn read(d: ::yo::doc::Doc<'_>) -> ::yo::Result<{name}> {{
        ::yo::doc::expect_object(d, {name:?})?;
        Ok({name} {{ {read} }})
    }}
}}
"
    )
}

/// The indexes a type declares, which every derived type has whether or not it
/// has an id. An edge type has no id and still declares indexes, so this is a
/// trait of its own rather than a constant on `Document`.
fn indexed(s: &Struct) -> String {
    let mut indexes = String::new();
    for f in &s.fields {
        if let Some(kind) = f.kind {
            let path = format!("$.{}", f.label);
            indexes.push_str(&format!("({path:?}, ::yo::doc::IndexKind::{kind}),"));
        }
    }
    let mut vectors = String::new();
    for f in &s.fields {
        if let Some(dim) = f.vector {
            let path = format!("$.{}", f.label);
            vectors.push_str(&format!("({path:?}, {dim}),"));
        }
    }
    let name = &s.name;
    format!(
        "#[automatically_derived]
impl ::yo::doc::Indexed for {name} {{
    const INDEXES: &'static [(&'static str, ::yo::doc::IndexKind)] = &[{indexes}];
    const VECTORS: &'static [(&'static str, usize)] = &[{vectors}];
}}
"
    )
}

fn document(s: &Struct, id: &Field) -> String {
    let (name, ty, at) = (&s.name, &id.ty, &id.name);
    format!(
        "#[automatically_derived]
impl ::yo::doc::Document for {name} {{
    type Id = {ty};

    fn id(&self) -> &{ty} {{
        &self.{at}
    }}
}}
"
    )
}

/// A `Path` constant per indexed field, so a query names the field rather than
/// a string that the compiler cannot check.
fn paths(s: &Struct) -> String {
    let mut consts = String::new();
    for f in &s.fields {
        let Some(kind) = f.kind else { continue };
        // The key of an array index is one element, so that is what a query
        // against it compares. Everything else queries its own type.
        let asked = match (kind, &f.elem) {
            ("Array", Some(elem)) => elem,
            _ => &f.ty,
        };
        let (upper, label, path) = (f.label.to_uppercase(), &f.label, format!("$.{}", f.label));
        let name = &s.name;
        // An ordered index answers ranges as well, and that is a different type
        // rather than a flag, so that a range over an equality index does not
        // compile.
        let held = if kind == "Ordered" {
            format!("::yo::doc::Ordered<{name}, {asked}>")
        } else {
            format!("::yo::doc::Path<{name}, {asked}>")
        };
        let built = if kind == "Ordered" {
            format!("::yo::doc::Ordered::new({path:?})")
        } else {
            format!("::yo::doc::Path::new({path:?}, ::yo::doc::IndexKind::{kind})")
        };
        consts.push_str(&format!(
            "    /// The `{path}` path, which is indexed for {kind} and named
    /// `{label}` on this type.
    pub const {upper}: {held} = {built};
"
        ));
    }
    for f in &s.fields {
        let Some(dim) = f.vector else { continue };
        let (upper, label, path) = (f.label.to_uppercase(), &f.label, format!("$.{}", f.label));
        let name = &s.name;
        consts.push_str(&format!(
            "    /// The `{path}` path, which holds a {dim} wide embedding and is
    /// named `{label}` on this type.
    pub const {upper}: ::yo::doc::Vector<{name}> = ::yo::doc::Vector::new({path:?}, {dim});
"
        ));
    }
    if consts.is_empty() {
        return String::new();
    }
    let name = &s.name;
    format!("impl {name} {{\n{consts}}}\n")
}
