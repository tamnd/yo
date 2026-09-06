//! Repository chores that need to be the same on every machine.
//!
//! No configuration anywhere in here, and the commands are:
//!
//! ```text
//! cargo xtask generate   write api.model.json and include/yo.h
//! cargo xtask check      regenerate in memory and fail on a diff
//! cargo xtask reserve    the registry name audit, in xtask/reserve.py
//! cargo xtask cross      clippy for the two targets this laptop is not
//! cargo xtask alloc      what allocates on a command path, in xtask/alloc.py
//! cargo xtask ltm        the larger than memory gate, in xtask/src/ltm.rs
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
//! `reserve` and `alloc` are the odd ones out: they are Python programs that
//! this command only launches. `dx/16` §10 has the argument, and the short
//! version is that doing them in Rust means an HTTP client, a TLS stack and a
//! JSON parser in a workspace whose dependency list is short enough that a user
//! can read it, for the sake of tools that run once a week.
//!
//! `alloc` is the instrument behind the Y7 work. It builds `yodb`, starts it
//! with the allocation check in report mode, sends a few thousand commands
//! covering every type and prints one line per place that allocated on a
//! command path. It lives here rather than in a test because it needs a server
//! and a socket, and it lives in the repository rather than in somebody's `tmp`
//! because a list of violations is worth nothing if the next person cannot
//! produce the same one.
//!
//! `ltm` is the same idea for M5. It starts a server with a store file and a
//! memory limit, loads ten times that limit, reads at random and reports what
//! each point read cost in reads from the file. The gate rows in `06` are
//! ratios, and a ratio nobody can reproduce is a claim.

mod emit_header;
mod emit_model;
mod errors;
mod json;
mod ltm;
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
        Some("reserve") => python("reserve.py"),
        Some("cross") => cross(),
        Some("alloc") => python("alloc.py"),
        Some("ltm") => ltm::ltm(),
        Some(other) => {
            eprintln!("unknown command: {other}");
            eprintln!("usage: cargo xtask [generate|check|reserve|cross|alloc|ltm]");
            process::exit(2);
        }
        None => {
            eprintln!("usage: cargo xtask [generate|check|reserve|cross|alloc|ltm]");
            process::exit(2);
        }
    }
}

/// The targets `cross` lints for, which are the two the developer laptop is
/// not.
///
/// Nothing here is built or linked. Clippy stops after analysis, so a target
/// needs its standard library and nothing else, and `rustup target add` is the
/// whole setup.
const CROSS: &[Target] = &[
    Target {
        triple: "x86_64-pc-windows-msvc",
        zig: "x86_64-windows-gnu",
        archiver: "lib",
    },
    Target {
        triple: "x86_64-unknown-linux-gnu",
        zig: "x86_64-linux-gnu",
        archiver: "ar",
    },
];

/// A target `cross` lints for, and what zig has to be told to compile C for it.
///
/// `zig` is the third column because the two spellings do not match. Rust names
/// a vendor and zig does not, so `x86_64-unknown-linux-gnu` has to become
/// `x86_64-linux-gnu` on the way through, and the MSVC target has to become the
/// GNU one because a mac has no MSVC runtime headers to compile against. That
/// substitution is only sound because nothing here is linked or run: clippy
/// stops after analysis, so the archive the build script produces is never read
/// by anything, and the ABI it was built for does not matter.
struct Target {
    triple: &'static str,
    zig: &'static str,
    /// `ar` or `lib`, which is the archiver the `cc` crate will drive for this
    /// target. It picks by ABI, so the MSVC target gets `lib.exe` style
    /// arguments and everything else gets `ar` style, and the two do not
    /// understand each other's flags at all.
    archiver: &'static str,
}

/// Run clippy for the platforms this machine is not.
///
/// Three release runs in a row went red on a lint that cannot fire on a mac:
/// an `if let` over a one variant enum, a loop around it that never loops, a
/// closure that only ever answers `Some`. Every one of them was in code behind
/// `#[cfg(unix)]` or beside it, every one took a push and a wait to find, and
/// every fix traded one of them for the next. This finds all three in about a
/// minute without leaving the laptop.
///
/// `--all-targets` is deliberately not passed. It pulls in criterion, whose
/// `alloca` has a C build script that needs a real MSVC to build, and the benches
/// are not where the platform code is anyway.
///
/// The workspace stopped being pure Rust when the Lua engine landed. `mlua-sys`
/// compiles the interpreter from C in a build script, and a build script runs
/// for the target being linted, so from that point on this command needed a C
/// compiler for linux and one for windows and a mac has neither. It brings its
/// own now: zig ships a clang that can target anything, and `cross_toolchain`
/// writes the two wrapper scripts that point the `cc` crate at it.
fn cross() {
    let tools = match cross_toolchain() {
        Ok(tools) => tools,
        Err(why) => {
            eprintln!("{why}");
            process::exit(1);
        }
    };
    let mut bad = false;
    for target in CROSS {
        println!("clippy {}", target.triple);
        let under = target.triple.replace('-', "_");
        let status = process::Command::new(env!("CARGO"))
            .current_dir(root())
            .args([
                "clippy",
                "--workspace",
                "--all-features",
                "--target",
                target.triple,
            ])
            .env(
                format!("CC_{under}"),
                tools.join(format!("cc-{}", target.zig)),
            )
            .env(
                format!("AR_{under}"),
                tools.join(format!("{}-{}", target.archiver, target.zig)),
            )
            .status();
        match status {
            Ok(s) if s.success() => {}
            Ok(_) => bad = true,
            Err(e) => {
                eprintln!("could not run cargo clippy for {}: {e}", target.triple);
                eprintln!("Add the target first: rustup target add {}", target.triple);
                bad = true;
            }
        }
    }
    if bad {
        process::exit(1);
    }
    println!("ok       every target lints clean");
}

/// Write the compiler and archiver wrappers `cross` hands to the `cc` crate,
/// and answer the directory holding them.
///
/// The wrappers exist because of one argument. `cc` appends `-target <triple>`
/// with the Rust spelling, zig only understands its own, and the last `-target`
/// on the line wins, so a wrapper that merely appended the right one would be
/// overruled by the wrong one. Each wrapper therefore drops every `-target` it
/// was handed and supplies its own.
fn cross_toolchain() -> Result<PathBuf, String> {
    let zig = find_zig()?;
    let dir = std::env::temp_dir().join("yo-xtask-cross");
    fs::create_dir_all(&dir).map_err(|e| format!("could not make {}: {e}", dir.display()))?;
    for target in CROSS {
        let cc = dir.join(format!("cc-{}", target.zig));
        write_tool(
            &cc,
            &format!(
                "#!/bin/sh\n\
                 kept=\"\"\n\
                 skip=0\n\
                 for a in \"$@\"; do\n\
                 \x20 if [ $skip -eq 1 ]; then skip=0; continue; fi\n\
                 \x20 case \"$a\" in\n\
                 \x20   -target) skip=1; continue ;;\n\
                 \x20   --target=*) continue ;;\n\
                 \x20 esac\n\
                 \x20 kept=\"$kept $a\"\n\
                 done\n\
                 exec {zig} cc -target {} $kept\n",
                target.zig,
                zig = zig.display(),
            ),
        )?;
        let ar = dir.join(format!("{}-{}", target.archiver, target.zig));
        write_tool(
            &ar,
            &format!(
                "#!/bin/sh\nexec {} {} \"$@\"\n",
                zig.display(),
                target.archiver
            ),
        )?;
    }
    Ok(dir)
}

/// Write one wrapper and make it runnable.
fn write_tool(path: &Path, body: &str) -> Result<(), String> {
    fs::write(path, body).map_err(|e| format!("could not write {}: {e}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o755))
            .map_err(|e| format!("could not chmod {}: {e}", path.display()))?;
    }
    Ok(())
}

/// Find a zig, on the path or in the `ziglang` wheel that pip installs.
///
/// The wheel is worth looking in because it is how this ends up on a machine
/// that already has Python for the other xtask commands, and because
/// `cargo-zigbuild` looks there too, so a developer who set one of these up has
/// usually set up the other by accident.
fn find_zig() -> Result<PathBuf, String> {
    if let Ok(out) = process::Command::new("zig").arg("version").output()
        && out.status.success()
    {
        return Ok(PathBuf::from("zig"));
    }
    let out = process::Command::new("python3")
        .args([
            "-c",
            "import ziglang, os; print(os.path.dirname(ziglang.__file__))",
        ])
        .output();
    if let Ok(out) = out
        && out.status.success()
    {
        let dir = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
        let zig = dir.join("zig");
        if zig.exists() {
            return Ok(zig);
        }
    }
    Err(
        "cross needs a C compiler for linux and for windows, and this machine has \
         neither. The workspace builds Lua from C, so clippy for another target \
         runs a build script that has to compile it.\n\
         Install one with `pip install ziglang`, or `brew install zig`, and run \
         this again. Without it the release matrix in CI is the only cross check \
         there is."
            .to_string(),
    )
}

/// Hands everything after the subcommand to a script in `xtask/` and exits with
/// its status.
///
/// The exit code is forwarded rather than collapsed to zero or one, because
/// `reserve verify` uses three of them and the difference is the point: 0 is
/// held, 1 is a name lost or transferred, and 2 is a probe that could not get
/// an answer at all. A wrapper that turned 2 into 1 would be reintroducing the
/// bug `dx/16` §10 property 3 was written about.
fn python(name: &str) {
    let script = root().join("xtask").join(name);
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
