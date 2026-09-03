//! Reaching one field of a document without decoding the rest of it.
//!
//! `$.a.b[3].c` is four steps, and each step is a header read, a binary search
//! or an index, and a seek. A ten kilobyte document with a four level path is
//! four cache lines and no allocation, which is the whole reason the encoding
//! is shaped the way it is.
//!
//! ```
//! use yo_doc::{Builder, Value};
//!
//! let mut b = Builder::new();
//! b.begin_object().unwrap();
//! b.key(b"user").unwrap();
//! b.begin_object().unwrap();
//! b.key(b"tags").unwrap();
//! b.begin_array().unwrap();
//! b.text("a").unwrap();
//! b.text("b").unwrap();
//! b.end_array().unwrap();
//! b.end_object().unwrap();
//! b.end_object().unwrap();
//! let bytes = b.finish().unwrap();
//!
//! let v = Value::new(&bytes).unwrap();
//! assert_eq!(v.path("$.user.tags[1]").unwrap().unwrap().as_text(), Some("b"));
//! assert_eq!(v.path("$.user.tags[-1]").unwrap().unwrap().as_text(), Some("b"));
//! assert!(v.path("$.user.missing").unwrap().is_none());
//! ```
//!
//! # What this grammar is
//!
//! The part of JSONPath that names exactly one place: a root, member access by
//! name, and element access by index counting from either end. The parts that
//! name a set of places, `[*]` and `..` and a slice and a union, are in
//! [`query`](crate::query), because they answer a list and everything here
//! answers at most one value.
//!
//! Both grammars read the same path the same way where they overlap, so a
//! caller that knows it wants one field uses this and pays no allocation, and
//! one that has taken a path from a client parses it with
//! [`Path`](crate::Path) and finds out from
//! [`is_definite`](crate::Path::is_definite) whether it named one place.

use yo_common::{Code, Error, Result};

use crate::head::Kind;
use crate::read::Value;

/// One step of a path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step<'a> {
    /// A member of an object, by name.
    Key(&'a [u8]),
    /// An element of an array. Negative counts back from the end, so -1 is the
    /// last element.
    Index(i64),
}

/// The steps of a path, parsed as they are walked.
///
/// Nothing is collected, so a path costs no allocation and a path that turns
/// out to be nonsense costs only as much of itself as was read before the
/// nonsense.
#[derive(Debug, Clone)]
pub struct Steps<'a> {
    rest: &'a [u8],
}

impl<'a> Steps<'a> {
    /// The steps of `path`.
    ///
    /// A leading `$` is the root and is optional, so both `$.a.b` and `a.b`
    /// parse to the same two steps.
    #[must_use]
    pub fn new(path: &'a [u8]) -> Steps<'a> {
        let rest = path.strip_prefix(b"$").unwrap_or(path);
        Steps { rest }
    }
}

impl<'a> Iterator for Steps<'a> {
    type Item = Result<Step<'a>>;

    fn next(&mut self) -> Option<Result<Step<'a>>> {
        if self.rest.is_empty() {
            return None;
        }
        Some(self.step())
    }
}

impl<'a> Steps<'a> {
    fn step(&mut self) -> Result<Step<'a>> {
        match self.rest[0] {
            b'.' => {
                self.rest = &self.rest[1..];
                if self.rest.first() == Some(&b'.') {
                    return Err(bad(
                        "a descent, `..`, names more than one place, so it is read by `Path` and not here",
                    ));
                }
                let end = self
                    .rest
                    .iter()
                    .position(|&c| c == b'.' || c == b'[')
                    .unwrap_or(self.rest.len());
                if end == 0 {
                    return Err(bad("a `.` with no name after it"));
                }
                let (name, rest) = self.rest.split_at(end);
                self.rest = rest;
                Ok(Step::Key(name))
            }
            b'[' => self.bracket(),
            _ => {
                // A path may start with a bare name, so that `a.b` and `$.a.b`
                // both work. Anywhere else this is a missing separator.
                let end = self
                    .rest
                    .iter()
                    .position(|&c| c == b'.' || c == b'[')
                    .unwrap_or(self.rest.len());
                let (name, rest) = self.rest.split_at(end);
                self.rest = rest;
                Ok(Step::Key(name))
            }
        }
    }

    fn bracket(&mut self) -> Result<Step<'a>> {
        let body = &self.rest[1..];
        let Some(close) = body.iter().position(|&c| c == b']') else {
            return Err(bad("a `[` with no `]` after it"));
        };
        let inner = &body[..close];
        self.rest = &body[close + 1..];
        if inner == b"*" {
            return Err(bad(
                "a wildcard, `[*]`, names more than one place, so it is read by `Path` and not here",
            ));
        }
        if let Some(quoted) = quoted(inner) {
            return Ok(Step::Key(quoted));
        }
        let text = core::str::from_utf8(inner).map_err(|_| bad("an index that is not a number"))?;
        let i: i64 = text
            .parse()
            .map_err(|_| bad("an index that is not a number"))?;
        Ok(Step::Index(i))
    }
}

/// The bytes inside `"..."` or `'...'`, if that is what this is.
fn quoted(inner: &[u8]) -> Option<&[u8]> {
    if inner.len() >= 2 {
        let (first, last) = (inner[0], inner[inner.len() - 1]);
        if (first == b'"' || first == b'\'') && last == first {
            return Some(&inner[1..inner.len() - 1]);
        }
    }
    None
}

impl<'a> Value<'a> {
    /// The value at `path`, if there is one there.
    ///
    /// `Ok(None)` is a path that is well formed and names nothing, which is a
    /// normal answer and not a failure. `Err` is a path that does not parse.
    pub fn path(&self, path: &str) -> Result<Option<Value<'a>>> {
        self.path_bytes(path.as_bytes())
    }

    /// [`Value::path`] over bytes, for the RESP side where a path arrives as a
    /// bulk string.
    pub fn path_bytes(&self, path: &[u8]) -> Result<Option<Value<'a>>> {
        let mut at = *self;
        for step in Steps::new(path) {
            let Some(next) = at.step(step?) else {
                return Ok(None);
            };
            at = next;
        }
        Ok(Some(at))
    }

    /// One step down from here.
    #[must_use]
    pub fn step(&self, step: Step<'_>) -> Option<Value<'a>> {
        match step {
            Step::Key(k) => self.get(k),
            Step::Index(i) => {
                if self.kind() != Kind::Array {
                    return None;
                }
                let n = self.len();
                let at = if i < 0 {
                    n.checked_sub(i.unsigned_abs() as usize)?
                } else {
                    i as usize
                };
                self.at(at)
            }
        }
    }
}

fn bad(what: &str) -> Error {
    Error::new(Code::Invalid, what)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Builder;

    /// `{"a": {"b": [10, 20, {"c": "deep"}]}, "empty": {}}`
    fn doc() -> Vec<u8> {
        let mut b = Builder::new();
        b.begin_object().expect("open");
        b.key(b"a").expect("key");
        b.begin_object().expect("open");
        b.key(b"b").expect("key");
        b.begin_array().expect("open");
        b.int(10).expect("value");
        b.int(20).expect("value");
        b.begin_object().expect("open");
        b.key(b"c").expect("key");
        b.text("deep").expect("value");
        b.end_object().expect("close");
        b.end_array().expect("close");
        b.end_object().expect("close");
        b.key(b"empty").expect("key");
        b.begin_object().expect("open");
        b.end_object().expect("close");
        b.end_object().expect("close");
        b.finish().expect("finished").to_vec()
    }

    #[test]
    fn a_path_reaches_what_it_names() {
        let bytes = doc();
        let v = Value::new(&bytes).expect("readable");
        let at = |p: &str| v.path(p).expect("the path parses");
        assert_eq!(at("$.a.b[0]").expect("there").as_int(), Some(10));
        assert_eq!(at("$.a.b[2].c").expect("there").as_text(), Some("deep"));
        assert_eq!(at("$.a.b[-1].c").expect("there").as_text(), Some("deep"));
        assert_eq!(at("$.a.b[-3]").expect("there").as_int(), Some(10));
        assert_eq!(at("$").expect("there").len(), 2);
        assert_eq!(at("").expect("there").len(), 2);
        // The three ways of writing the same step.
        assert_eq!(at("$.a.b[0]").expect("there").as_int(), Some(10));
        assert_eq!(at("$[\"a\"][\"b\"][0]").expect("there").as_int(), Some(10));
        assert_eq!(at("a.b[0]").expect("there").as_int(), Some(10));
        assert_eq!(at("$['a'].b[0]").expect("there").as_int(), Some(10));
    }

    #[test]
    fn a_path_that_names_nothing_is_an_answer_and_not_a_failure() {
        let bytes = doc();
        let v = Value::new(&bytes).expect("readable");
        let at = |p: &str| v.path(p).expect("the path parses");
        assert!(at("$.nope").is_none());
        assert!(at("$.a.nope.deeper").is_none());
        assert!(at("$.a.b[3]").is_none(), "past the end");
        assert!(at("$.a.b[-4]").is_none(), "past the start");
        assert!(at("$.empty.anything").is_none());
        assert!(at("$.a.b.c").is_none(), "a name into an array");
        assert!(at("$.a[0]").is_none(), "an index into an object");
        assert!(at("$.a.b[0][0]").is_none(), "an index into a number");
    }

    #[test]
    fn a_path_that_does_not_parse_says_so() {
        let bytes = doc();
        let v = Value::new(&bytes).expect("readable");
        let why = |p: &str| v.path(p).unwrap_err().message().to_string();
        assert!(why("$.a[").contains("no `]`"));
        assert!(why("$.a[x]").contains("not a number"));
        assert!(why("$..a").contains("more than one place"));
        assert!(why("$.a[*]").contains("more than one place"));
        assert!(why("$.a.").contains("no name after it"));
    }

    #[test]
    fn the_steps_of_a_path_are_what_they_look_like() {
        let steps: Vec<Step<'_>> = Steps::new(b"$.a[3].bb[-1][\"c c\"]")
            .map(|s| s.expect("parses"))
            .collect();
        assert_eq!(
            steps,
            [
                Step::Key(b"a"),
                Step::Index(3),
                Step::Key(b"bb"),
                Step::Index(-1),
                Step::Key(b"c c"),
            ]
        );
    }

    #[test]
    fn a_path_over_bytes_reads_the_same_as_a_path_over_text() {
        let bytes = doc();
        let v = Value::new(&bytes).expect("readable");
        let one = v.path("$.a.b[1]").expect("parses").expect("there");
        let two = v.path_bytes(b"$.a.b[1]").expect("parses").expect("there");
        assert_eq!(one.as_int(), two.as_int());
    }
}
