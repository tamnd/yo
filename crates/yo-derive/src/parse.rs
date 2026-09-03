//! Reading a struct back out of the tokens the compiler hands a derive.
//!
//! The grammar is small on purpose: outer attributes, a visibility, the word
//! `struct`, a name, and a brace with `name: Type` in it. Everything a field's
//! type could be is carried through as tokens and handed back to the compiler
//! untouched, so this never has to understand a type to write code about one.

use proc_macro::{Delimiter, TokenStream, TokenTree};

/// One field of a struct.
pub struct Field {
    /// The name as it is written, which keeps a `r#` on a raw identifier
    /// because that is what the expression `self.r#type` needs.
    pub name: String,
    /// The name without the `r#`, which is what goes in the shape, in the
    /// document and in the index path.
    pub label: String,
    /// The type as tokens, handed back to the compiler as it arrived.
    pub ty: String,
    /// The element type of a `Vec<T>`, which is what one key of an array index
    /// is, and `None` for anything else.
    pub elem: Option<String>,
    /// Whether this field is the document's id.
    pub id: bool,
    /// The `IndexKind` variant this field asked for, if it asked for one.
    pub kind: Option<&'static str>,
    /// How wide the embedding at this field is, if it asked for a vector index.
    pub vector: Option<usize>,
}

/// A struct with named fields.
pub struct Struct {
    /// The type's name.
    pub name: String,
    /// Its fields, in declaration order, because order is layout.
    pub fields: Vec<Field>,
}

/// Read the item a derive was put on.
///
/// # Errors
///
/// A sentence to hand back as a `compile_error!`, for anything that is not a
/// struct with named fields or that carries an attribute `yo` does not know.
pub fn parse(input: TokenStream) -> Result<Struct, String> {
    let mut c = Cursor::new(input);
    // Whatever else is on the item is not ours. The `#[derive]` itself is gone
    // by the time we see this, and a doc comment is an attribute too.
    take_marks(&mut c)?;
    skip_vis(&mut c);

    match c.bump() {
        Some(TokenTree::Ident(word)) if word.to_string() == "struct" => {}
        Some(TokenTree::Ident(word)) => {
            return Err(format!(
                "Yo writes the shape of a struct with named fields, and this is a {word}"
            ));
        }
        _ => return Err("Yo writes the shape of a struct with named fields".to_owned()),
    }

    let name = match c.bump() {
        Some(TokenTree::Ident(name)) => name.to_string(),
        _ => return Err("this struct has no name".to_owned()),
    };

    let body = match c.bump() {
        Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Brace => g.stream(),
        Some(TokenTree::Punct(p)) if p.as_char() == '<' => {
            return Err(format!(
                "{name} has type parameters, and a shape is one description that six languages compute identically rather than one per instantiation, so Yo cannot write it"
            ));
        }
        _ => {
            return Err(format!(
                "{name} has no named fields, and a shape is its fields and their order"
            ));
        }
    };

    Ok(Struct {
        name,
        fields: fields(body)?,
    })
}

fn fields(body: TokenStream) -> Result<Vec<Field>, String> {
    let mut c = Cursor::new(body);
    let mut out: Vec<Field> = Vec::new();

    while c.peek().is_some() {
        let marks = take_marks(&mut c)?;
        skip_vis(&mut c);

        let name = match c.bump() {
            Some(TokenTree::Ident(name)) => name.to_string(),
            Some(other) => return Err(format!("{other} is where a field name should be")),
            None => break,
        };
        match c.bump() {
            Some(TokenTree::Punct(p)) if p.as_char() == ':' => {}
            _ => {
                return Err(format!(
                    "the field {name} has no type, and a shape is its types"
                ));
            }
        }
        let (ty, elem) = take_type(&mut c);

        let label = name.strip_prefix("r#").unwrap_or(&name).to_owned();
        let mut field = Field {
            name,
            label,
            ty,
            elem,
            id: false,
            kind: None,
            vector: None,
        };
        for mark in marks {
            apply(&mut field, &mark)?;
        }
        out.push(field);
    }

    Ok(out)
}

fn apply(field: &mut Field, mark: &str) -> Result<(), String> {
    if let Some(width) = mark.strip_prefix("vector=") {
        let dim = width.parse::<usize>().map_err(|_| {
            format!(
                "the field {} asks for a vector index {width} wide, and a width is a whole number",
                field.label
            )
        })?;
        if dim == 0 {
            return Err(format!(
                "the field {} asks for a vector index of no width, and there is nothing to compare",
                field.label
            ));
        }
        if let Some(already) = field.kind {
            return Err(format!(
                "the field {} asks for a vector index and a {already} one at once, and an embedding is not a key",
                field.label
            ));
        }
        field.vector = Some(dim);
        return Ok(());
    }
    let kind = match mark {
        "id" => {
            field.id = true;
            return Ok(());
        }
        "index" => "Equality",
        "ordered" => "Ordered",
        "array" => "Array",
        "text" => "Text",
        "vector" => {
            return Err(format!(
                "the field {} asks for a vector index without saying how wide, as in #[yo(vector = 384)]",
                field.label
            ));
        }
        other => {
            return Err(format!(
                "{other} is not something yo understands on the field {}. It knows id, index, ordered, array, text and vector",
                field.label
            ));
        }
    };
    if let Some(already) = field.kind {
        return Err(format!(
            "the field {} asks for two indexes at once, {already} and {kind}, and a path answers one question",
            field.label
        ));
    }
    if field.vector.is_some() {
        return Err(format!(
            "the field {} asks for a vector index and a {kind} one at once, and an embedding is not a key",
            field.label
        ));
    }
    field.kind = Some(kind);
    Ok(())
}

/// Take the `#[yo(...)]` words off whatever comes next, dropping every other
/// attribute on the way past.
fn take_marks(c: &mut Cursor) -> Result<Vec<String>, String> {
    let mut words = Vec::new();
    while matches!(c.peek(), Some(TokenTree::Punct(p)) if p.as_char() == '#') {
        c.bump();
        let body = match c.bump() {
            Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Bracket => g.stream(),
            _ => return Err("an attribute with nothing in it".to_owned()),
        };
        let mut inner = Cursor::new(body);
        let Some(TokenTree::Ident(name)) = inner.bump() else {
            continue;
        };
        if name.to_string() != "yo" {
            continue;
        }
        let list = match inner.bump() {
            Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Parenthesis => g.stream(),
            _ => return Err("yo takes a list of words, as in #[yo(index)]".to_owned()),
        };
        // A word, or a word with a number after it, which is what a vector
        // index needs and is the only place a value appears in one of these.
        let mut items = Cursor::new(list);
        while let Some(tt) = items.bump() {
            match tt {
                TokenTree::Ident(word) => {
                    let mut mark = word.to_string();
                    if matches!(items.peek(), Some(TokenTree::Punct(p)) if p.as_char() == '=') {
                        items.bump();
                        match items.bump() {
                            Some(TokenTree::Literal(n)) => {
                                mark.push('=');
                                mark.push_str(&n.to_string());
                            }
                            _ => {
                                return Err(format!(
                                    "{mark} in a yo attribute is followed by an equals sign and nothing it can use, as in #[yo(vector = 384)]"
                                ));
                            }
                        }
                    }
                    words.push(mark);
                }
                TokenTree::Punct(p) if p.as_char() == ',' => {}
                other => return Err(format!("{other} is not a word yo can read in an attribute")),
            }
        }
    }
    Ok(words)
}

/// Step over `pub`, `pub(crate)` and the rest of that family.
fn skip_vis(c: &mut Cursor) {
    if matches!(c.peek(), Some(TokenTree::Ident(word)) if word.to_string() == "pub") {
        c.bump();
        if matches!(c.peek(), Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Parenthesis)
        {
            c.bump();
        }
    }
}

/// Everything up to the comma that ends this field, plus the element type if
/// what we walked over was a `Vec<T>`.
///
/// The angle bracket count is what keeps the comma in `BTreeMap<K, V>` from
/// looking like the end of a field.
fn take_type(c: &mut Cursor) -> (String, Option<String>) {
    let mut depth = 0i32;
    let mut ty: Vec<TokenTree> = Vec::new();
    while let Some(tt) = c.peek() {
        match tt {
            TokenTree::Punct(p) if p.as_char() == ',' && depth == 0 => {
                c.bump();
                break;
            }
            TokenTree::Punct(p) if p.as_char() == '<' => depth += 1,
            TokenTree::Punct(p) if p.as_char() == '>' && depth > 0 => depth -= 1,
            _ => {}
        }
        if let Some(tt) = c.bump() {
            ty.push(tt);
        }
    }

    let elem = element(&ty);
    (render(&ty), elem)
}

/// The `T` of a `Vec<T>`, read off the tokens rather than off the text so that
/// a type called `MyVec` or a path ending in `Vec` is not mistaken for one.
fn element(ty: &[TokenTree]) -> Option<String> {
    let (first, rest) = ty.split_first()?;
    match first {
        TokenTree::Ident(name) if name.to_string() == "Vec" => {}
        _ => return None,
    }
    let (open, rest) = rest.split_first()?;
    match open {
        TokenTree::Punct(p) if p.as_char() == '<' => {}
        _ => return None,
    }
    let (close, inner) = rest.split_last()?;
    match close {
        TokenTree::Punct(p) if p.as_char() == '>' => {}
        _ => return None,
    }
    if inner.is_empty() {
        return None;
    }
    Some(render(inner))
}

fn render(tokens: &[TokenTree]) -> String {
    tokens.iter().cloned().collect::<TokenStream>().to_string()
}

/// A token stream read one item at a time, with one item of lookahead.
struct Cursor {
    tokens: Vec<TokenTree>,
    at: usize,
}

impl Cursor {
    fn new(stream: TokenStream) -> Cursor {
        Cursor {
            tokens: stream.into_iter().collect(),
            at: 0,
        }
    }

    fn peek(&self) -> Option<&TokenTree> {
        self.tokens.get(self.at)
    }

    fn bump(&mut self) -> Option<TokenTree> {
        let tt = self.tokens.get(self.at).cloned();
        if tt.is_some() {
            self.at += 1;
        }
        tt
    }
}
