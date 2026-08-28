//! A JSON writer, about eighty lines of it.
//!
//! The generated files are diffed in CI, so the output has to be byte stable:
//! same key order every run, two space indent, a newline at the end, no trailing
//! commas. A serialiser that sorts keys or prints floats in its own way would
//! make the diff check noise. Writing it here is cheaper than depending on one
//! and configuring it into behaving.

use std::fmt::Write as _;

/// Accumulates JSON with a fixed layout.
pub struct Out {
    buf: String,
    depth: usize,
    /// Whether the current container already holds an element.
    started: Vec<bool>,
}

impl Out {
    /// A new writer, ready for an opening brace.
    pub fn new() -> Out {
        Out {
            buf: String::new(),
            depth: 0,
            started: Vec::new(),
        }
    }

    /// The finished document, with a trailing newline.
    pub fn finish(mut self) -> String {
        self.buf.push('\n');
        self.buf
    }

    fn comma(&mut self) {
        if let Some(started) = self.started.last_mut() {
            if *started {
                self.buf.push(',');
            }
            *started = true;
        }
        if !self.started.is_empty() {
            self.buf.push('\n');
            for _ in 0..self.depth {
                self.buf.push_str("  ");
            }
        }
    }

    fn key(&mut self, k: &str) {
        self.comma();
        write!(self.buf, "{}: ", quote(k)).unwrap();
    }

    /// Opens an object, as a value inside the current container.
    pub fn obj(&mut self) {
        self.comma();
        self.buf.push('{');
        self.depth += 1;
        self.started.push(false);
    }

    /// Opens an object under a key.
    pub fn obj_at(&mut self, k: &str) {
        self.key(k);
        self.buf.push('{');
        self.depth += 1;
        self.started.push(false);
    }

    /// Opens an array under a key.
    pub fn arr_at(&mut self, k: &str) {
        self.key(k);
        self.buf.push('[');
        self.depth += 1;
        self.started.push(false);
    }

    /// Closes the innermost object.
    pub fn end_obj(&mut self) {
        self.close('}');
    }

    /// Closes the innermost array.
    pub fn end_arr(&mut self) {
        self.close(']');
    }

    fn close(&mut self, c: char) {
        let had = self.started.pop().unwrap_or(false);
        self.depth -= 1;
        if had {
            self.buf.push('\n');
            for _ in 0..self.depth {
                self.buf.push_str("  ");
            }
        }
        self.buf.push(c);
    }

    /// A string field.
    pub fn str(&mut self, k: &str, v: &str) {
        self.key(k);
        self.buf.push_str(&quote(v));
    }

    /// A string field that is omitted when the value is absent, so that an
    /// optional field never shows up as an explicit null nobody handles.
    pub fn opt_str(&mut self, k: &str, v: Option<&str>) {
        if let Some(v) = v {
            self.str(k, v);
        }
    }

    /// An unsigned field.
    pub fn num(&mut self, k: &str, v: u64) {
        self.key(k);
        write!(self.buf, "{v}").unwrap();
    }

    /// A boolean field.
    pub fn bool(&mut self, k: &str, v: bool) {
        self.key(k);
        self.buf.push_str(if v { "true" } else { "false" });
    }

    /// An array of strings, on one line per element.
    pub fn strs(&mut self, k: &str, vs: &[&str]) {
        self.arr_at(k);
        for v in vs {
            self.comma();
            self.buf.push_str(&quote(v));
        }
        self.end_arr();
    }
}

/// A JSON string literal, escaped the way the spec requires and no further.
pub fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => write!(out, "\\u{:04x}", c as u32).unwrap(),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nesting_and_commas_come_out_right() {
        let mut o = Out::new();
        o.obj();
        o.str("name", "yo");
        o.arr_at("items");
        o.obj();
        o.num("n", 1);
        o.end_obj();
        o.obj();
        o.num("n", 2);
        o.end_obj();
        o.end_arr();
        o.end_obj();
        assert_eq!(
            o.finish(),
            "{\n  \"name\": \"yo\",\n  \"items\": [\n    {\n      \"n\": 1\n    },\n    {\n      \"n\": 2\n    }\n  ]\n}\n"
        );
    }

    #[test]
    fn empty_containers_stay_on_one_line() {
        let mut o = Out::new();
        o.obj();
        o.arr_at("none");
        o.end_arr();
        o.end_obj();
        assert_eq!(o.finish(), "{\n  \"none\": []\n}\n");
    }

    #[test]
    fn escaping_covers_what_the_spec_requires() {
        assert_eq!(quote("a\"b\\c\nd\te"), "\"a\\\"b\\\\c\\nd\\te\"");
        assert_eq!(quote("\u{1}"), "\"\\u0001\"");
        // Not escaped, because JSON does not require it and escaping it would
        // make the diff of a doc line unreadable.
        assert_eq!(quote("a/b"), "\"a/b\"");
    }
}
