//! Reading `errors.toml`.
//!
//! yo-common's build script already parses this file to make the Rust enum. This
//! is a second reader for the same file rather than a shared crate, because a
//! build script cannot export anything and the alternative is a crate that
//! exists only so two callers can share forty lines.
//!
//! Duplicated parsing usually drifts. Here it cannot: the test at the bottom
//! checks this reader against `yo_common::Code`, field by field, for every
//! variant. If the two ever disagree the test says so by name.

use std::fs;
use std::path::Path;

/// One error, exactly as the file spells it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// The Rust variant name.
    pub name: String,
    /// The wire value. Frozen once released.
    pub code: u32,
    /// The C enumerator.
    pub c_name: String,
    /// Whether the identical call could succeed later.
    pub retryable: bool,
    /// One paragraph of prose.
    pub doc: String,
    /// The documentation page, if the condition has one.
    pub url: Option<String>,
}

/// Reads and validates the error table.
///
/// Panics with the offending line on anything the format does not allow, which
/// is the right behaviour for a build tool that runs in CI.
pub fn load(root: &Path) -> Vec<Entry> {
    let path = root.join("crates/yo-common/errors.toml");
    let src = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("could not read {}: {e}", path.display()));

    let mut out: Vec<Entry> = Vec::new();
    let mut cur: Option<Entry> = None;

    for line in src.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "[[error]]" {
            if let Some(e) = cur.take() {
                out.push(e);
            }
            cur = Some(Entry {
                name: String::new(),
                code: u32::MAX,
                c_name: String::new(),
                retryable: false,
                doc: String::new(),
                url: None,
            });
            continue;
        }
        let e = cur
            .as_mut()
            .expect("a key appeared before any [[error]] header");
        let (key, value) = line
            .split_once('=')
            .unwrap_or_else(|| panic!("not a key value pair: {line}"));
        match key.trim() {
            "name" => e.name = unquote(value),
            "code" => e.code = value.trim().parse().expect("code is not a number"),
            "c_name" => e.c_name = unquote(value),
            "retryable" => e.retryable = value.trim() == "true",
            "doc" => e.doc = unquote(value),
            "url" => e.url = Some(unquote(value)),
            other => panic!("unknown key in errors.toml: {other}"),
        }
    }
    if let Some(e) = cur.take() {
        out.push(e);
    }

    assert!(!out.is_empty(), "errors.toml defines no errors");
    for (i, e) in out.iter().enumerate() {
        assert_eq!(
            e.code as usize, i,
            "codes must be dense and in order from zero, found {} at position {i}",
            e.code
        );
        assert!(!e.name.is_empty(), "error at position {i} has no name");
        assert!(!e.c_name.is_empty(), "error at position {i} has no c_name");
        assert!(!e.doc.is_empty(), "error at position {i} has no doc");
    }
    out
}

fn unquote(v: &str) -> String {
    let v = v.trim();
    let b = v.as_bytes();
    assert!(
        b.len() >= 2 && b[0] == b'"' && b[b.len() - 1] == b'"',
        "value is not a quoted string: {v}"
    );
    v[1..v.len() - 1].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::root;

    #[test]
    fn agrees_with_the_generated_rust_enum() {
        let table = load(&root());
        let all = yo_common::Code::ALL;
        assert_eq!(
            table.len(),
            all.len(),
            "the two readers see a different number of errors"
        );
        for (e, code) in table.iter().zip(all) {
            assert_eq!(e.code, code.as_u32(), "{} has a different value", e.name);
            assert_eq!(e.c_name, code.c_name(), "{} has a different C name", e.name);
            assert_eq!(
                e.retryable,
                code.is_retryable(),
                "{} disagrees on retryability",
                e.name
            );
            assert_eq!(
                e.url.as_deref(),
                code.url(),
                "{} has a different url",
                e.name
            );
        }
    }
}
