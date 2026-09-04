//! Printing a parsed query back out, which is what `FT.EXPLAIN` answers with.
//!
//! The printout is the only view a client has of what its query became, so it is
//! the thing people paste into a bug report when a search returns the wrong
//! rows. That makes it an interface rather than a debugging aid, and it is
//! matched to a real server byte for byte, down to the two spaces of indent, the
//! space before `UNION {` that `NOT{` does not get, and the six decimal places
//! on a number the client wrote as `1`.

use crate::index::Index;
use crate::query::{Circle, Mask, Node, Range, Vector, What, Word};
use crate::query::{EVERY, parse};

/// The name a vector query gives its distance when the client did not.
const SCORE: &str = "_score";

/// Prints a tree the way `FT.EXPLAIN` prints it, ending in a newline.
///
/// The index is needed because a node carries the fields it asks as a set of
/// bits and the printout names them, so there is no way to print a tree without
/// the schema it was parsed against.
#[must_use]
pub fn explain(node: &Node, index: &Index) -> Vec<u8> {
    let mut out = Vec::with_capacity(64);
    write(&mut out, node, index, 0);
    out
}

/// The text fields of a schema in the order their bits are numbered.
fn text(index: &Index) -> impl Iterator<Item = &[u8]> {
    index
        .schema
        .iter()
        .filter(|f| matches!(f.kind, crate::field::Kind::Text(_)))
        .map(|f| &*f.attribute)
}

/// The bit a text field is numbered with, or nothing if it is not text.
///
/// Numbered by position among the text fields rather than by position in the
/// schema, so adding a numeric field in the middle does not move a text field's
/// bit and invalidate a mask that was worked out before it.
#[must_use]
pub fn bit(index: &Index, attribute: &[u8]) -> Option<Mask> {
    let at = text(index).position(|name| name == attribute)?;
    Some(1 << (at % Mask::BITS as usize))
}

/// The `@a|b:` a node is printed behind, or nothing when it asks every field.
fn scope(mask: Mask, index: &Index) -> Option<Vec<u8>> {
    if mask == EVERY {
        return None;
    }
    let mut out = vec![b'@'];
    if mask == 0 {
        // A modifier inside a modifier that allowed no field in common. There
        // is no name for the empty set of fields, so a real server prints the
        // word and answers nothing, and a client that sees it has written a
        // query that cannot match.
        out.extend_from_slice(b"NULL:");
        return Some(out);
    }
    let mut first = true;
    for (at, name) in text(index).enumerate() {
        if mask & (1 << (at % Mask::BITS as usize)) == 0 {
            continue;
        }
        if !first {
            out.push(b'|');
        }
        first = false;
        out.extend_from_slice(name);
    }
    out.push(b':');
    Some(out)
}

/// Writes one node and everything under it, indented by `depth` levels.
fn write(out: &mut Vec<u8>, node: &Node, index: &Index, depth: usize) {
    match &node.what {
        // A query that came to nothing is printed on its own, without the
        // fields it would have asked, because there is nothing left to ask
        // them of: `@a:the` is `<empty>` and not `@a:<empty>`.
        What::Empty => line(out, &Node::empty(), index, depth, b"<empty>"),
        What::Wildcard => line(out, node, index, depth, b"<WILDCARD>"),
        What::Term(word) => term(out, node, index, depth, word),
        What::Union(list) => group(out, node, index, depth, b"UNION {", list),
        What::Intersect(list) => group(out, node, index, depth, b"INTERSECT {", list),
        What::Exact(list) => group(out, node, index, depth, b"EXACT {", list),
        What::Not(child) => group(
            out,
            node,
            index,
            depth,
            b"NOT{",
            std::slice::from_ref(child),
        ),
        What::Optional(child) => {
            group(
                out,
                node,
                index,
                depth,
                b"OPTIONAL{",
                std::slice::from_ref(child),
            );
        }
        What::Prefix(word) => wrapped(out, node, index, depth, b"PREFIX{", word, b"*}"),
        What::Suffix(word) => wrapped(out, node, index, depth, b"SUFFIX{*", word, b"}"),
        What::Infix(word) => wrapped(out, node, index, depth, b"INFIX{*", word, b"*}"),
        What::Fuzzy(word, _) => wrapped(out, node, index, depth, b"FUZZY{", word, b"}"),
        What::Pattern(word) => wrapped(out, node, index, depth, b"WILDCARD{", word, b"}"),
        What::Numeric(range) => numeric(out, node, depth, range),
        What::Tag(field, list) => {
            let mut head = b"TAG:@".to_vec();
            head.extend_from_slice(field);
            head.extend_from_slice(b" {");
            group(out, node, index, depth, &head, list);
        }
        What::Geo(circle) => geo(out, node, depth, circle),
        What::Vector(vector) => vec_node(out, node, index, depth, vector),
    }
}

/// Two spaces per level, which is the indent a real server uses.
fn pad(out: &mut Vec<u8>, depth: usize) {
    out.extend_from_slice(&b"  ".repeat(depth));
}

/// The `@a:` in front of a node, for the nodes that carry one.
///
/// A numeric and a geo node print a mask only when it is the empty one, because
/// they name their own field in the body and a real server leaves a real mask
/// off them even when a modifier narrowed them.
fn head(out: &mut Vec<u8>, node: &Node, index: &Index, depth: usize) {
    pad(out, depth);
    if let Some(scope) = scope(node.mask, index) {
        out.extend_from_slice(&scope);
    }
}

/// The `@NULL:` a numeric or a geo node carries when a modifier left it asking
/// no field at all, which is the one mask a real server prints on these two.
fn null_head(out: &mut Vec<u8>, node: &Node, depth: usize) {
    pad(out, depth);
    if node.mask == 0 {
        out.extend_from_slice(b"@NULL:");
    }
}

/// A node that is one line and holds nothing.
fn line(out: &mut Vec<u8>, node: &Node, index: &Index, depth: usize, body: &[u8]) {
    head(out, node, index, depth);
    out.extend_from_slice(body);
    attrs(out, node);
    out.push(b'\n');
}

/// A leaf whose body is a word inside a shape, such as `PREFIX{hel*}`.
fn wrapped(
    out: &mut Vec<u8>,
    n: &Node,
    i: &Index,
    depth: usize,
    open: &[u8],
    w: &[u8],
    shut: &[u8],
) {
    let mut body = open.to_vec();
    body.extend_from_slice(w);
    body.extend_from_slice(shut);
    line(out, n, i, depth, &body);
}

/// One word, marked if the stemmer rather than the client produced it.
fn term(out: &mut Vec<u8>, node: &Node, index: &Index, depth: usize, word: &Word) {
    // A parameter can hold an empty string, and a word with nothing in it
    // prints as a pair of quotes rather than as a blank line.
    if word.word.is_empty() {
        line(out, node, index, depth, b"\"\"");
        return;
    }
    let mut body = Vec::with_capacity(word.word.len() + 12);
    if word.stem {
        body.push(b'+');
    }
    body.extend_from_slice(&word.word);
    if word.expanded {
        body.extend_from_slice(b"(expanded)");
    }
    line(out, node, index, depth, &body);
}

/// What a client hung off a node, in the order a real server prints it.
///
/// A word prints its weight flush against itself and prints nothing else, and
/// everything else prints all three with spaces inside the braces. There is no
/// reason for the two shapes to differ and a real server has them both, so both
/// are here. A slop of minus one is no limit at all and is left off, and an
/// order is only worth saying next to a slop unless it was asked for.
fn attrs(out: &mut Vec<u8>, node: &Node) {
    let weight = node.weight.filter(|w| *w != 1.0);
    if matches!(node.what, What::Term(_)) {
        let Some(weight) = weight else { return };
        out.extend_from_slice(b" => {$weight: ");
        out.extend_from_slice(short(weight).as_bytes());
        out.extend_from_slice(b";}");
        return;
    }
    let slop = node.slop.filter(|s| *s >= 0);
    if weight.is_none() && slop.is_none() && !node.inorder {
        return;
    }
    out.extend_from_slice(b" => {");
    if let Some(weight) = weight {
        out.extend_from_slice(b" $weight: ");
        out.extend_from_slice(short(weight).as_bytes());
        out.push(b';');
    }
    if let Some(slop) = slop {
        out.extend_from_slice(b" $slop: ");
        out.extend_from_slice(slop.to_string().as_bytes());
        out.push(b';');
    }
    if slop.is_some() || node.inorder {
        out.extend_from_slice(b" $inorder: ");
        out.extend_from_slice(if node.inorder { b"true" } else { b"false" });
        out.push(b';');
    }
    out.extend_from_slice(b" }");
}

/// A node that holds others, opened on its own line and shut on its own line.
fn group(out: &mut Vec<u8>, node: &Node, index: &Index, depth: usize, open: &[u8], list: &[Node]) {
    head(out, node, index, depth);
    out.extend_from_slice(open);
    out.push(b'\n');
    for child in list {
        write(out, child, index, depth + 1);
    }
    pad(out, depth);
    out.push(b'}');
    attrs(out, node);
    out.push(b'\n');
}

/// `NUMERIC {1.000000 <= @n <= 10.000000}`, with the ends open or shut.
fn numeric(out: &mut Vec<u8>, node: &Node, depth: usize, range: &Range) {
    null_head(out, node, depth);
    out.extend_from_slice(b"NUMERIC {");
    out.extend_from_slice(fixed(range.min).as_bytes());
    out.extend_from_slice(if range.min_open { b" < @" } else { b" <= @" });
    out.extend_from_slice(&range.field);
    out.extend_from_slice(if range.max_open { b" < " } else { b" <= " });
    out.extend_from_slice(fixed(range.max).as_bytes());
    out.push(b'}');
    attrs(out, node);
    out.push(b'\n');
}

/// `GEO loc:{1.000000,2.000000 --> 3.000000 km}`, with the field bare.
fn geo(out: &mut Vec<u8>, node: &Node, depth: usize, circle: &Circle) {
    null_head(out, node, depth);
    out.extend_from_slice(b"GEO ");
    out.extend_from_slice(&circle.field);
    out.extend_from_slice(b":{");
    out.extend_from_slice(fixed(circle.lon).as_bytes());
    out.push(b',');
    out.extend_from_slice(fixed(circle.lat).as_bytes());
    out.extend_from_slice(b" --> ");
    out.extend_from_slice(fixed(circle.radius).as_bytes());
    out.push(b' ');
    out.extend_from_slice(&circle.unit);
    out.push(b'}');
    attrs(out, node);
    out.push(b'\n');
}

/// A vector query, which is a sentence rather than a shape.
///
/// When it has nothing under it the sentence is the body, and when it has
/// something under it the body is that and the sentence moves out to an
/// attribute clause after the closing brace. That is two layouts for one node
/// and it is what a real server prints.
fn vec_node(out: &mut Vec<u8>, node: &Node, index: &Index, depth: usize, vector: &Vector) {
    let sentence = sentence(vector);
    let Some(over) = &vector.over else {
        let mut body = b"VECTOR {".to_vec();
        body.extend_from_slice(&sentence);
        body.push(b'}');
        line(out, node, index, depth, &body);
        return;
    };
    head(out, node, index, depth);
    out.extend_from_slice(b"VECTOR {\n");
    write(out, over, index, depth + 1);
    pad(out, depth);
    out.extend_from_slice(b"} => {");
    out.extend_from_slice(&sentence);
    out.extend_from_slice(b"}\n");
}

/// What a vector node says about itself, without the braces around it.
fn sentence(vector: &Vector) -> Vec<u8> {
    let mut out = Vec::with_capacity(96);
    if let Some(radius) = vector.radius {
        out.extend_from_slice(b"Vectors that are within ");
        out.extend_from_slice(short(radius).as_bytes());
        out.extend_from_slice(b" distance radius from `$");
        out.extend_from_slice(&vector.param);
        out.extend_from_slice(b"` in vector index associated with field @");
        out.extend_from_slice(&vector.field);
        if let Some(alias) = &vector.alias {
            out.extend_from_slice(b", yields distance as `");
            out.extend_from_slice(alias);
            out.push(b'`');
        }
        return out;
    }
    out.extend_from_slice(b"K=");
    out.extend_from_slice(vector.k.unwrap_or(0).to_string().as_bytes());
    out.extend_from_slice(b" nearest vectors to `$");
    out.extend_from_slice(&vector.param);
    out.extend_from_slice(b"` in vector index associated with field @");
    out.extend_from_slice(&vector.field);
    for (name, value) in &vector.options {
        out.extend_from_slice(b", ");
        out.extend_from_slice(name);
        out.extend_from_slice(b" = ");
        out.extend_from_slice(value);
    }
    out.extend_from_slice(b", yields distance as `");
    match &vector.alias {
        Some(alias) => out.extend_from_slice(alias),
        None => {
            out.extend_from_slice(b"__");
            out.extend_from_slice(&vector.field);
            out.extend_from_slice(SCORE.as_bytes());
        }
    }
    out.push(b'`');
    out
}

/// Six decimal places, which is what C's `%f` gives and what the printout has.
fn fixed(value: f64) -> String {
    if value.is_infinite() {
        return if value > 0.0 {
            "inf".to_owned()
        } else {
            "-inf".to_owned()
        };
    }
    format!("{value:.6}")
}

/// As few digits as say the number exactly, which is what a weight and a vector
/// radius are printed with.
fn short(value: f64) -> String {
    // A weight may be a nothing at all, and it is spelled the way C spells it
    // rather than the way Rust does.
    if value.is_nan() {
        return "nan".to_owned();
    }
    format!("{value}")
}

/// Parses a query and prints it in one go, which is the whole of `FT.EXPLAIN`.
///
/// # Errors
///
/// Whatever the parser refused the query for.
pub fn describe(query: &[u8], index: &Index, ask: &parse::Ask) -> Result<Vec<u8>, parse::Bad> {
    Ok(explain(&parse(query, index, ask)?, index))
}

/// Splits a printout into the lines `FT.EXPLAINCLI` answers with.
///
/// The printout ends in a newline and the split leaves an empty piece after it,
/// which a real server sends as a final empty string rather than dropping. It
/// looks like an off by one and it is load bearing, because a client that joins
/// the array back with newlines gets the bulk string `FT.EXPLAIN` would have
/// sent.
#[must_use]
pub fn lines(printed: &[u8]) -> Vec<&[u8]> {
    printed.split(|b| *b == b'\n').collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{Field, Kind, Text};
    use crate::index::{Definition, Index};

    fn index() -> Index {
        let schema = vec![
            Field::new(b"a", Kind::Text(Text::default())),
            Field::new(b"b", Kind::Text(Text::default())),
            Field::new(b"n", Kind::Numeric),
        ];
        Index::new(b"i", Definition::default(), schema)
    }

    #[test]
    fn a_word_asking_every_field_is_printed_without_a_scope() {
        let out = explain(&Node::term(b"hello"), &index());
        assert_eq!(out, b"hello\n");
    }

    #[test]
    fn a_word_narrowed_to_one_field_names_it() {
        let mut node = Node::term(b"hello");
        node.narrow(bit(&index(), b"a").expect("a is a text field"));
        assert_eq!(explain(&node, &index()), b"@a:hello\n");
    }

    #[test]
    fn a_word_narrowed_to_two_fields_names_them_in_schema_order() {
        let index = index();
        let both = bit(&index, b"b").unwrap() | bit(&index, b"a").unwrap();
        let mut node = Node::term(b"hello");
        node.narrow(both);
        assert_eq!(explain(&node, &index), b"@a|b:hello\n");
    }

    /// Two modifiers that allow no field in common leave a node that can never
    /// match, and it is printed rather than dropped.
    #[test]
    fn a_word_narrowed_to_nothing_is_printed_as_null() {
        let index = index();
        let mut node = Node::term(b"hello");
        node.narrow(bit(&index, b"a").unwrap());
        node.narrow(bit(&index, b"b").unwrap());
        assert_eq!(explain(&node, &index), b"@NULL:hello\n");
    }

    /// The one every reading of the grammar gets wrong: a space before the brace
    /// on some of them and none on the others.
    #[test]
    fn a_negation_has_no_space_before_its_brace_and_a_union_does() {
        let union = Node::new(What::Union(vec![Node::term(b"a"), Node::term(b"b")]));
        assert_eq!(explain(&union, &index()), b"UNION {\n  a\n  b\n}\n");
        let not = Node::new(What::Not(Box::new(Node::term(b"a"))));
        assert_eq!(explain(&not, &index()), b"NOT{\n  a\n}\n");
    }

    #[test]
    fn a_number_the_client_wrote_as_one_is_printed_with_six_places() {
        let range = Range {
            field: b"n".to_vec().into(),
            min: 1.0,
            max: 10.0,
            min_open: false,
            max_open: false,
        };
        let node = Node::new(What::Numeric(range));
        assert_eq!(
            explain(&node, &index()),
            b"NUMERIC {1.000000 <= @n <= 10.000000}\n"
        );
    }

    #[test]
    fn an_open_end_and_an_infinite_end_are_printed_differently() {
        let range = Range {
            field: b"n".to_vec().into(),
            min: 1.0,
            max: f64::INFINITY,
            min_open: true,
            max_open: false,
        };
        let node = Node::new(What::Numeric(range));
        assert_eq!(
            explain(&node, &index()),
            b"NUMERIC {1.000000 < @n <= inf}\n"
        );
    }

    /// The trailing newline leaves an empty last line and a real server sends
    /// it, so a client can join the array back into the bulk string.
    #[test]
    fn the_line_split_keeps_the_empty_piece_after_the_last_newline() {
        let out = explain(&Node::term(b"hello"), &index());
        assert_eq!(lines(&out), vec![&b"hello"[..], &b""[..]]);
    }
}
