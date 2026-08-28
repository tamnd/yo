//! Generates the Rust error model from `errors.toml`.
//!
//! This is deliberately a hand written parser over a format we control rather
//! than a `toml` build dependency. The file has one shape, this reads that
//! shape, and it fails loudly on anything else. `16` section 0 lists the
//! dependencies this project takes and a build time TOML parser is not one of
//! them.

use std::fmt::Write as _;
use std::{env, fs, path::PathBuf};

struct Error {
    name: String,
    code: u32,
    c_name: String,
    retryable: bool,
    doc: String,
    url: Option<String>,
}

fn main() {
    println!("cargo:rerun-if-changed=errors.toml");
    let src = fs::read_to_string("errors.toml").expect("errors.toml is missing");
    let errors = parse(&src);

    assert!(!errors.is_empty(), "errors.toml defines no errors");
    for (i, e) in errors.iter().enumerate() {
        assert_eq!(
            e.code as usize, i,
            "error codes must be dense and in order starting at 0, found {} at position {}",
            e.code, i
        );
    }

    let mut out = String::new();
    out.push_str("// Generated from errors.toml by build.rs. Do not edit.\n\n");

    out.push_str("/// A stable, wire visible condition code.\n");
    out.push_str("///\n");
    out.push_str("/// These numbers are frozen once released. They are the same integers the C\n");
    out.push_str("/// ABI exposes as `yo_code` and the same ones every binding reports.\n");
    out.push_str("#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]\n");
    out.push_str("#[repr(u32)]\n");
    out.push_str("#[non_exhaustive]\n");
    out.push_str("pub enum Code {\n");
    for e in &errors {
        writeln!(out, "    /// {}", e.doc).unwrap();
        writeln!(out, "    ///\n    /// C name: `{}`.", e.c_name).unwrap();
        writeln!(out, "    {} = {},", e.name, e.code).unwrap();
    }
    out.push_str("}\n\n");

    out.push_str("impl Code {\n");

    out.push_str("    /// Every code, in wire order. Index equals the numeric value.\n");
    out.push_str("    pub const ALL: &'static [Code] = &[\n");
    for e in &errors {
        writeln!(out, "        Code::{},", e.name).unwrap();
    }
    out.push_str("    ];\n\n");

    out.push_str("    /// The wire value.\n");
    out.push_str("    #[inline]\n");
    out.push_str("    pub const fn as_u32(self) -> u32 {\n        self as u32\n    }\n\n");

    out.push_str("    /// The code for a wire value, or `None` if this build does not know it.\n");
    out.push_str("    ///\n");
    out.push_str("    /// An unknown code from a newer peer is a value to report, not a panic.\n");
    out.push_str("    #[inline]\n");
    out.push_str("    pub const fn from_u32(v: u32) -> Option<Code> {\n");
    out.push_str("        match v {\n");
    for e in &errors {
        writeln!(out, "            {} => Some(Code::{}),", e.code, e.name).unwrap();
    }
    out.push_str("            _ => None,\n        }\n    }\n\n");

    out.push_str("    /// The C ABI spelling, which is also what appears in log lines.\n");
    out.push_str("    #[inline]\n");
    out.push_str("    pub const fn c_name(self) -> &'static str {\n");
    out.push_str("        match self {\n");
    for e in &errors {
        writeln!(out, "            Code::{} => {:?},", e.name, e.c_name).unwrap();
    }
    out.push_str("        }\n    }\n\n");

    out.push_str("    /// Whether the identical call could succeed later.\n");
    out.push_str("    ///\n");
    out.push_str(
        "    /// This is a property of the condition and not of the caller, which is why\n",
    );
    out.push_str("    /// it is generated rather than decided at each call site.\n");
    out.push_str("    #[inline]\n");
    out.push_str("    pub const fn is_retryable(self) -> bool {\n");
    out.push_str("        match self {\n");
    for e in &errors {
        writeln!(out, "            Code::{} => {},", e.name, e.retryable).unwrap();
    }
    out.push_str("        }\n    }\n\n");

    out.push_str("    /// The C ABI spelling with a trailing NUL, ready to cross the boundary.\n");
    out.push_str("    ///\n");
    out.push_str("    /// The C ABI needs NUL terminated strings and Rust literals are not,\n");
    out.push_str("    /// so the terminator is put here where it is generated rather than in a\n");
    out.push_str("    /// second table somebody has to keep in step by hand.\n");
    out.push_str("    #[inline]\n");
    out.push_str("    pub const fn c_name_z(self) -> &'static str {\n");
    out.push_str("        match self {\n");
    for e in &errors {
        writeln!(out, "            Code::{} => \"{}\\0\",", e.name, e.c_name).unwrap();
    }
    out.push_str("        }\n    }\n\n");

    out.push_str("    /// The documentation page with a trailing NUL, if there is one.\n");
    out.push_str("    #[inline]\n");
    out.push_str("    pub const fn url_z(self) -> Option<&'static str> {\n");
    out.push_str("        match self {\n");
    for e in &errors {
        match &e.url {
            Some(u) => {
                writeln!(out, "            Code::{} => Some(\"{}\\0\"),", e.name, u).unwrap()
            }
            None => writeln!(out, "            Code::{} => None,", e.name).unwrap(),
        }
    }
    out.push_str("        }\n    }\n\n");

    out.push_str("    /// The documentation page for this condition, if it has one.\n");
    out.push_str("    #[inline]\n");
    out.push_str("    pub const fn url(self) -> Option<&'static str> {\n");
    out.push_str("        match self {\n");
    for e in &errors {
        match &e.url {
            Some(u) => writeln!(out, "            Code::{} => Some({:?}),", e.name, u).unwrap(),
            None => writeln!(out, "            Code::{} => None,", e.name).unwrap(),
        }
    }
    out.push_str("        }\n    }\n");
    out.push_str("}\n");

    let dest = PathBuf::from(env::var("OUT_DIR").unwrap()).join("code.rs");
    fs::write(&dest, out).expect("could not write the generated error model");
}

fn parse(src: &str) -> Vec<Error> {
    let mut out: Vec<Error> = Vec::new();
    let mut name = None;
    let mut code = None;
    let mut c_name = None;
    let mut retryable = None;
    let mut doc = None;
    let mut url = None;

    let flush = |out: &mut Vec<Error>,
                 name: &mut Option<String>,
                 code: &mut Option<u32>,
                 c_name: &mut Option<String>,
                 retryable: &mut Option<bool>,
                 doc: &mut Option<String>,
                 url: &mut Option<String>| {
        if let Some(n) = name.take() {
            out.push(Error {
                name: n,
                code: code.take().expect("error entry has no code"),
                c_name: c_name.take().expect("error entry has no c_name"),
                retryable: retryable.take().expect("error entry has no retryable"),
                doc: doc.take().expect("error entry has no doc"),
                url: url.take(),
            });
        }
    };

    for line in src.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "[[error]]" {
            flush(
                &mut out,
                &mut name,
                &mut code,
                &mut c_name,
                &mut retryable,
                &mut doc,
                &mut url,
            );
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .unwrap_or_else(|| panic!("errors.toml line is not a key value pair: {line}"));
        let key = key.trim();
        let value = value.trim();
        match key {
            "name" => name = Some(unquote(value)),
            "code" => code = Some(value.parse().expect("code is not a number")),
            "c_name" => c_name = Some(unquote(value)),
            "retryable" => retryable = Some(value == "true"),
            "doc" => doc = Some(unquote(value)),
            "url" => url = Some(unquote(value)),
            other => panic!("errors.toml has an unknown key: {other}"),
        }
    }
    flush(
        &mut out,
        &mut name,
        &mut code,
        &mut c_name,
        &mut retryable,
        &mut doc,
        &mut url,
    );
    out
}

fn unquote(v: &str) -> String {
    let v = v.trim();
    let bytes = v.as_bytes();
    assert!(
        bytes.len() >= 2 && bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"',
        "errors.toml value is not a quoted string: {v}"
    );
    v[1..v.len() - 1].to_string()
}
