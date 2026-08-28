//! Repository chores that need to be the same on every machine.
//!
//! Two commands and no configuration:
//!
//! ```text
//! cargo xtask generate   write api.model.json and include/yo.h
//! cargo xtask check      regenerate in memory and fail on a diff
//! cargo xtask reserve    the registry name audit, in xtask/reserve.py
//! ```
//!
//! `check` is what CI runs. Generated files are checked in so that a binding
//! generator does not need a Rust toolchain to read the model, and the diff check
//! is what stops checked in from turning into stale.
//!
//! `check` also reads the two registry files. `commands.toml` is the command
//! audit and it gates: a command with no storage plan does not ship.
//! `divergences.toml` is the register of every place yo is knowingly not Redis,
//! and a command that claims a divergence has to name a row that is in it. Both
//! rules are in `12` sections 3 and 10, and neither is worth writing down
//! unless something enforces it.
//!
//! `reserve` is the odd one out: it is a Python program that this command only
//! launches. `dx/16` §10 has the argument, and the short version is that doing
//! it in Rust means an HTTP client, a TLS stack and a JSON parser in a
//! workspace whose dependency list is short enough that a user can read it, for
//! the sake of a tool that runs once a week.

mod emit_header;
mod emit_model;
mod errors;
mod json;
mod model;
mod registry;
mod toml;

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
        Some("reserve") => reserve(),
        Some(other) => {
            eprintln!("unknown command: {other}");
            eprintln!("usage: cargo xtask [generate|check|reserve]");
            process::exit(2);
        }
        None => {
            eprintln!("usage: cargo xtask [generate|check|reserve]");
            process::exit(2);
        }
    }
}

/// Hands everything after `reserve` to `xtask/reserve.py` and exits with its
/// status.
///
/// The exit code is forwarded rather than collapsed to zero or one, because
/// `reserve verify` uses three of them and the difference is the point: 0 is
/// held, 1 is a name lost or transferred, and 2 is a probe that could not get
/// an answer at all. A wrapper that turned 2 into 1 would be reintroducing the
/// bug `dx/16` §10 property 3 was written about.
fn reserve() {
    let script = root().join("xtask").join("reserve.py");
    let args: Vec<String> = std::env::args().skip(2).collect();

    // `python3` and not `python`. On a machine where both exist, `python` is as
    // likely to be a 2.7 that dies on the first f-string as anything else, and
    // the error it gives says "invalid syntax" rather than "wrong interpreter".
    let status = process::Command::new("python3")
        .arg(&script)
        .args(&args)
        .status();

    match status {
        Ok(s) => process::exit(s.code().unwrap_or(1)),
        Err(e) => {
            eprintln!("could not run {}: {e}", script.display());
            eprintln!("It needs python3 on PATH and nothing else: no pip, no venv,");
            eprintln!("no third-party packages. See dx/16 section 10.");
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
    let mut bad: Vec<&str> = Vec::new();
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
    let registry = registry::problems();
    if registry.is_empty() {
        println!("ok       commands.toml and divergences.toml");
    } else {
        println!("BAD      commands.toml and divergences.toml");
        for p in &registry {
            println!("  {p}");
        }
    }

    if !bad.is_empty() {
        eprintln!();
        eprintln!("{} generated file(s) do not match the model.", bad.len());
        eprintln!("Run `cargo xtask generate` and commit the result.");
    }
    if !registry.is_empty() {
        eprintln!();
        eprintln!(
            "{} problem(s) in the command audit. A command with no storage plan does not ship, and a divergence needs a row in the register.",
            registry.len()
        );
    }
    if !bad.is_empty() || !registry.is_empty() {
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
