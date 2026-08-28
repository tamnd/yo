//! A TOML reader, about as much of one as `commands.toml` needs.
//!
//! The repository has two registry files and neither of them uses more of TOML
//! than array of tables headers, comments, and keys whose values are a quoted
//! string, an integer or a boolean. Reading that is sixty lines. Depending on a
//! parser for it would put a crate and its tree into a build that has none, and
//! `xtask` is the one place in the repository where a dependency buys nothing:
//! it runs on a developer's machine and in CI, and it exists to make those two
//! agree.
//!
//! What this does not do is as important as what it does, because a reader that
//! silently ignores what it cannot handle is a reader that lets a typo through.
//! Anything outside the subset is an error naming the line, so a file that uses
//! more TOML than this understands fails loudly and gets the reader extended.

use std::collections::BTreeMap;
use std::fmt;

/// One value, in the three shapes the registry files use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Value {
    /// A quoted string, with `\"` and `\\` unescaped.
    Str(String),
    /// A signed integer, which is what `arity` and `expected` are.
    Int(i64),
    /// `true` or `false`.
    Bool(bool),
}

impl Value {
    /// The string, if it is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// The integer, if it is one.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(n) => Some(*n),
            _ => None,
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Str(s) => write!(f, "{s}"),
            Value::Int(n) => write!(f, "{n}"),
            Value::Bool(b) => write!(f, "{b}"),
        }
    }
}

/// One `[[name]]` block, with the line it started on.
#[derive(Debug, Clone)]
pub struct Table {
    /// The header name, without the brackets.
    pub name: String,
    /// The line the header is on, for error messages worth reading.
    pub line: usize,
    /// The keys, sorted, because a registry check should report in a stable
    /// order whatever order the file happens to be in.
    pub keys: BTreeMap<String, Value>,
}

impl Table {
    /// The value under `key`, if the table has one.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.keys.get(key)
    }

    /// The string under `key`, or an error naming the table and the line.
    pub fn str(&self, key: &str) -> Result<&str, String> {
        match self.get(key) {
            Some(Value::Str(s)) => Ok(s),
            Some(other) => Err(format!(
                "line {}: [[{}]] has {key} = {other}, which is not a string",
                self.line, self.name
            )),
            None => Err(format!(
                "line {}: [[{}]] has no {key}",
                self.line, self.name
            )),
        }
    }
}

/// Every array of tables block in the document, in file order.
pub fn parse(text: &str) -> Result<Vec<Table>, String> {
    let mut tables: Vec<Table> = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = i + 1;
        let s = strip_comment(raw).trim();
        if s.is_empty() {
            continue;
        }
        if let Some(rest) = s.strip_prefix("[[") {
            let name = rest
                .strip_suffix("]]")
                .ok_or_else(|| format!("line {line}: {s} does not close its header"))?;
            tables.push(Table {
                name: name.trim().to_string(),
                line,
                keys: BTreeMap::new(),
            });
            continue;
        }
        if s.starts_with('[') {
            return Err(format!(
                "line {line}: {s} is a plain table, and this reader only knows array of tables"
            ));
        }
        let (key, value) = s
            .split_once('=')
            .ok_or_else(|| format!("line {line}: {s} is neither a header nor a key"))?;
        let key = key.trim().to_string();
        let value = parse_value(value.trim(), line)?;
        let table = tables
            .last_mut()
            .ok_or_else(|| format!("line {line}: {key} comes before any header"))?;
        if table.keys.insert(key.clone(), value).is_some() {
            return Err(format!("line {line}: [[{}]] sets {key} twice", table.name));
        }
    }
    Ok(tables)
}

/// Everything before an unquoted `#`.
///
/// Quoted, because `notes` values contain sentences and a sentence can contain
/// a hash. Getting this wrong truncates a note at a `#` and the truncation is
/// invisible until somebody reads the file.
fn strip_comment(line: &str) -> &str {
    let mut in_quotes = false;
    let mut escaped = false;
    for (i, c) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match c {
            '\\' if in_quotes => escaped = true,
            '"' => in_quotes = !in_quotes,
            '#' if !in_quotes => return &line[..i],
            _ => {}
        }
    }
    line
}

fn parse_value(s: &str, line: usize) -> Result<Value, String> {
    if s == "true" {
        return Ok(Value::Bool(true));
    }
    if s == "false" {
        return Ok(Value::Bool(false));
    }
    if let Some(body) = s.strip_prefix('"') {
        let body = body
            .strip_suffix('"')
            .ok_or_else(|| format!("line {line}: {s} does not close its quote"))?;
        let mut out = String::with_capacity(body.len());
        let mut chars = body.chars();
        while let Some(c) = chars.next() {
            if c != '\\' {
                out.push(c);
                continue;
            }
            match chars.next() {
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some(other) => {
                    return Err(format!(
                        "line {line}: \\{other} is not an escape this reader knows"
                    ));
                }
                None => return Err(format!("line {line}: {s} ends in a backslash")),
            }
        }
        return Ok(Value::Str(out));
    }
    s.parse::<i64>()
        .map(Value::Int)
        .map_err(|_| format!("line {line}: {s} is not a string, an integer or a boolean"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_document_comes_apart_into_its_tables() {
        let doc = r#"
# a comment on its own line
[[command]]
name = "SET"
arity = -3
divergent = true

[[command]]
name = "GET"   # a trailing comment
arity = 2
"#;
        let t = parse(doc).unwrap();
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].name, "command");
        assert_eq!(t[0].str("name").unwrap(), "SET");
        assert_eq!(t[0].get("arity").unwrap().as_int(), Some(-3));
        assert_eq!(t[0].get("divergent"), Some(&Value::Bool(true)));
        assert_eq!(t[1].str("name").unwrap(), "GET");
        assert_eq!(t[1].line, 8);
    }

    #[test]
    fn a_hash_inside_a_string_is_not_a_comment() {
        let t = parse("[[x]]\nnotes = \"tagged #1 and #2\"\n").unwrap();
        assert_eq!(t[0].str("notes").unwrap(), "tagged #1 and #2");
    }

    #[test]
    fn the_escapes_this_reader_knows_round_trip() {
        let t = parse("[[x]]\na = \"say \\\"hi\\\"\"\nb = \"back\\\\slash\"\n").unwrap();
        assert_eq!(t[0].str("a").unwrap(), "say \"hi\"");
        assert_eq!(t[0].str("b").unwrap(), "back\\slash");
    }

    #[test]
    fn anything_outside_the_subset_is_an_error_and_not_a_shrug() {
        // A plain table, which the registry files do not use and which this
        // reader would otherwise fold into whatever came before it.
        assert!(parse("[package]\nname = \"x\"\n").is_err());
        // A list, which would be silently dropped if it were tolerated.
        assert!(parse("[[x]]\na = [1, 2]\n").is_err());
        // A key before any header has nowhere to go.
        assert!(parse("a = 1\n").is_err());
        // The same key twice means one of the two is being ignored.
        assert!(parse("[[x]]\na = 1\na = 2\n").is_err());
        // An unterminated quote.
        assert!(parse("[[x]]\na = \"oops\n").is_err());
    }

    #[test]
    fn a_missing_key_says_which_table_and_which_line() {
        let t = parse("[[command]]\nname = \"SET\"\n").unwrap();
        let e = t[0].str("plan").unwrap_err();
        assert!(e.contains("line 1"), "{e}");
        assert!(e.contains("command"), "{e}");
        assert!(e.contains("plan"), "{e}");
    }
}
