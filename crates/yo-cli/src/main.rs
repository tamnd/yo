//! The `yodb` command line tool.
//!
//! Two subcommands: `check`, which is the M1 deliverable, and `serve`, which
//! puts the RESP engine on a socket. The others arrive with the milestones that
//! need them.
//!
//! Argument parsing is done by hand rather than with a library. That is a
//! choice worth defending exactly once, here: `yodb check` is the tool you run
//! when a database will not start, and a tool for that moment that pulls in a
//! dependency tree is a tool that can fail to build on the machine where you
//! need it. It parses six flags across two commands. When this grows a
//! benchmark runner and a config file the calculation changes and so should the
//! code.

mod check;
mod poll;
mod serve;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;

use check::Severity;

const USAGE: &str = "\
yodb, an embedded knowledge engine

usage:
  yodb check FILE [--quick] [--quiet]
  yodb serve [--bind ADDR] [--port PORT]

  check    read a .yo file and report anything wrong with it. Never writes.
             --quick   skip the records and read only the headers
             --quiet   print findings and the summary, nothing else

  serve    speak RESP on a socket, so a Redis client can talk to it.
             --bind    address to listen on, 127.0.0.1 by default
             --port    port to listen on, 6379 by default

exit codes:
  0  nothing wrong
  1  something wrong
  2  the arguments did not make sense, or the file could not be read at all
";

/// What a Redis client tries first, so it is what we listen on.
const DEFAULT_PORT: u16 = 6379;

/// Loopback, not every interface.
///
/// Redis shipped bound to every interface for years, and the result was tens of
/// thousands of open databases on the internet. Reaching this server from
/// another machine should be a thing somebody typed on purpose.
const DEFAULT_BIND: &str = "127.0.0.1";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut rest: Vec<&str> = args.iter().map(String::as_str).collect();

    match rest.first().copied() {
        Some("check") => {
            rest.remove(0);
            check_command(&rest)
        }
        Some("serve") => {
            rest.remove(0);
            serve_command(&rest)
        }
        Some("-h" | "--help") | None => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Some("-V" | "--version") => {
            println!("yo {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("yo: no such command: {other}\n");
            eprint!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn check_command(args: &[&str]) -> ExitCode {
    let mut path: Option<PathBuf> = None;
    let mut quick = false;
    let mut quiet = false;

    for a in args {
        match *a {
            "--quick" => quick = true,
            "--quiet" => quiet = true,
            "-h" | "--help" => {
                print!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            other if other.starts_with('-') => {
                eprintln!("yodb check: no such option: {other}");
                return ExitCode::from(2);
            }
            other if path.is_none() => path = Some(PathBuf::from(other)),
            other => {
                eprintln!("yodb check: takes one file, and was also given {other}");
                return ExitCode::from(2);
            }
        }
    }

    let Some(path) = path else {
        eprintln!("yodb check: which file?\n");
        eprint!("{USAGE}");
        return ExitCode::from(2);
    };

    let report = match check::check(&path, !quick) {
        Ok(r) => r,
        Err(e) => {
            // Getting here means the file could not be opened far enough to say
            // anything at all, which is a different thing from a file with
            // problems in it and gets a different exit code.
            eprintln!("yodb check: {}: {e}", path.display());
            return ExitCode::from(2);
        }
    };

    if !quiet {
        println!("{}", path.display());
    }
    for f in &report.findings {
        println!("{f}");
    }

    let c = report.counts;
    if !quiet {
        // Off `quick` rather than off the counts. A segment that stops parsing
        // leaves the count at zero, and printing "records not walked" there
        // would say the walk did not happen when what happened is that it ran
        // and hit something.
        if quick {
            println!("{} segments, records not walked", c.regions);
        } else {
            println!(
                "{} segments, {} records, {} record bytes, {} dead",
                c.regions, c.records, c.record_bytes, c.dead_bytes
            );
        }
    }

    let errors = report.count(Severity::Error);
    let warns = report.count(Severity::Warn);
    if report.is_sound() {
        println!(
            "OK{}",
            if warns > 0 {
                format!(", with {warns} warning{}", plural(warns))
            } else {
                String::new()
            }
        );
        ExitCode::SUCCESS
    } else {
        println!("FAILED: {errors} problem{}", plural(errors));
        ExitCode::FAILURE
    }
}

fn serve_command(args: &[&str]) -> ExitCode {
    let mut bind = DEFAULT_BIND.to_string();
    let mut port = DEFAULT_PORT;

    let mut at = 0;
    while at < args.len() {
        let arg = args[at];
        at += 1;
        match arg {
            "-h" | "--help" => {
                print!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            "--bind" | "--port" => {
                let Some(value) = args.get(at) else {
                    eprintln!("yodb serve: {arg} needs a value");
                    return ExitCode::from(2);
                };
                at += 1;
                if arg == "--bind" {
                    bind = (*value).to_string();
                } else {
                    match value.parse() {
                        Ok(p) => port = p,
                        Err(_) => {
                            eprintln!("yodb serve: {value} is not a port");
                            return ExitCode::from(2);
                        }
                    }
                }
            }
            other => {
                eprintln!("yodb serve: no such option: {other}");
                return ExitCode::from(2);
            }
        }
    }

    let Ok(addr) = format!("{bind}:{port}").parse::<SocketAddr>() else {
        eprintln!("yodb serve: {bind} is not an address to listen on");
        return ExitCode::from(2);
    };

    let mut server = match serve::Server::bind(addr) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("yodb serve: {addr}: {e}");
            return ExitCode::from(2);
        }
    };

    // What it actually bound to, which is the only way to find out when the
    // port asked for was zero.
    let bound = server.local_addr().unwrap_or(addr);
    println!("yodb {} listening on {bound}", env!("CARGO_PKG_VERSION"));

    // Nothing sets this yet. Ctrl-C ends the process, and there is no state to
    // flush until the file arrives in M5, which is when a shutdown that waits
    // for something starts to mean anything.
    static STOP: AtomicBool = AtomicBool::new(false);
    match server.run(&STOP) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("yodb serve: {e}");
            ExitCode::FAILURE
        }
    }
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}
