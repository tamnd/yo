//! Repository chores that need to be the same on every machine.
//!
//! Two commands and no configuration:
//!
//! ```text
//! cargo xtask generate   write api.model.json and include/yo.h
//! cargo xtask check      regenerate in memory and fail on a diff
//! ```
//!
//! `check` is what CI runs. Generated files are checked in so that a binding
//! generator does not need a Rust toolchain to read the model, and the diff check
//! is what stops checked in from turning into stale.

mod emit_header;
mod emit_model;
mod errors;
mod json;
mod model;

use std::path::{Path, PathBuf};
use std::{fs, process};

/// The repository root, found from this file rather than from the working
/// directory, so `cargo xtask` works from anywhere in the tree.
pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask should live one level under the root")
        .to_path_buf()
}

/// One generated file: where it goes and how it is made.
struct Artifact {
    path: &'static str,
    render: fn() -> String,
}

const ARTIFACTS: &[Artifact] = &[
    Artifact {
        path: "api.model.json",
        render: emit_model::render,
    },
    Artifact {
        path: "include/yo.h",
        render: emit_header::render,
    },
];

fn main() {
    let cmd = std::env::args().nth(1);
    match cmd.as_deref() {
        Some("generate") => generate(),
        Some("check") => check(),
        Some(other) => {
            eprintln!("unknown command: {other}");
            eprintln!("usage: cargo xtask [generate|check]");
            process::exit(2);
        }
        None => {
            eprintln!("usage: cargo xtask [generate|check]");
            process::exit(2);
        }
    }
}

fn generate() {
    let root = root();
    for a in ARTIFACTS {
        let dest = root.join(a.path);
        let text = (a.render)();
        if let Some(dir) = dest.parent() {
            fs::create_dir_all(dir).expect("could not make the output directory");
        }
        let same = fs::read_to_string(&dest)
            .map(|old| lf(&old) == text)
            .unwrap_or(false);
        fs::write(&dest, text)
            .unwrap_or_else(|e| panic!("could not write {}: {e}", dest.display()));
        println!(
            "{} {}",
            if same { "unchanged" } else { "wrote    " },
            a.path
        );
    }
}

fn check() {
    let root = root();
    let mut bad = Vec::new();
    for a in ARTIFACTS {
        let dest = root.join(a.path);
        let want = (a.render)();
        match fs::read_to_string(&dest).map(|s| lf(&s)) {
            Ok(got) if got == want => println!("ok       {}", a.path),
            Ok(got) => {
                println!("STALE    {}", a.path);
                show_first_difference(&got, &want);
                bad.push(a.path);
            }
            Err(_) => {
                println!("MISSING  {}", a.path);
                bad.push(a.path);
            }
        }
    }
    if !bad.is_empty() {
        eprintln!();
        eprintln!("{} generated file(s) do not match the model.", bad.len());
        eprintln!("Run `cargo xtask generate` and commit the result.");
        process::exit(1);
    }
}

/// Line endings, normalised to what the generator emits.
///
/// .gitattributes asks for an LF checkout everywhere, so on a correctly
/// configured clone this changes nothing. It exists because a Windows clone
/// with core.autocrlf set the other way turns every line into CRLF, and then
/// this check fails on a difference that is not in the model and that the
/// person reading the failure cannot see.
fn lf(s: &str) -> String {
    s.replace("\r\n", "\n")
}

/// Prints the first line that differs, because a whole diff of a generated file
/// is noise and the first difference is almost always the whole story.
fn show_first_difference(got: &str, want: &str) {
    for (i, (a, b)) in got.lines().zip(want.lines()).enumerate() {
        if a != b {
            eprintln!("  line {}:", i + 1);
            eprintln!("    on disk:   {a}");
            eprintln!("    from model: {b}");
            return;
        }
    }
    let (g, w) = (got.lines().count(), want.lines().count());
    if g != w {
        eprintln!("  the file has {g} lines and the model produces {w}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_deterministic() {
        // If this ever fails, the diff check in CI becomes a coin toss and
        // everybody learns to ignore it.
        for a in ARTIFACTS {
            assert_eq!(
                (a.render)(),
                (a.render)(),
                "{} is not stable across runs",
                a.path
            );
        }
    }

    #[test]
    fn the_checked_in_files_match_the_model() {
        // This is the same assertion CI makes. It is here as well so that a
        // local `cargo test` catches a forgotten regenerate before the push.
        let root = root();
        for a in ARTIFACTS {
            let dest = root.join(a.path);
            let got = fs::read_to_string(&dest)
                .map(|s| lf(&s))
                .unwrap_or_else(|_| panic!("{} is missing, run `cargo xtask generate`", a.path));
            assert_eq!(
                got,
                (a.render)(),
                "{} is stale, run `cargo xtask generate`",
                a.path
            );
        }
    }
}
