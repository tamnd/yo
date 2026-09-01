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
mod signal;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;

use check::Severity;

const USAGE: &str = "\
yodb, an embedded knowledge engine

usage:
  yodb check FILE [--quick] [--quiet]
  yodb serve [--bind ADDR] [--port PORT] [--unixsocket PATH] [--no-port]

  check    read a .yo file and report anything wrong with it. Never writes.
             --quick   skip the records and read only the headers
             --quiet   print findings and the summary, nothing else

  serve    speak RESP on a socket, so a Redis client can talk to it.
             --bind        address to listen on, 127.0.0.1 by default
             --port        port to listen on, 6379 by default
             --unixsocket  also listen on a socket file, which skips the
                           TCP stack and is the faster way in for a client
                           on the same machine
             --no-port     no TCP at all, socket file only

environment:
  YO_ALLOC  what to do when a command path allocates. off by default, which
            is the check turned off. report prints each place it happens once
            and carries on. abort stops the process on the first one.

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

/// The allocator that enforces Y7, no heap on a command path.
///
/// Installed here because picking a global allocator belongs to the program and
/// not to any library it links. It forwards everything to the system allocator
/// and does nothing else until `YO_ALLOC` asks it to, so a release build of
/// `yodb` behaves exactly as it did before this line existed.
#[global_allocator]
static ALLOC: yo_alloc::YoAlloc = yo_alloc::YoAlloc::new();

fn main() -> ExitCode {
    // Before anything else, because it decides what happens for the rest of the
    // process and a value nobody understands has to be an error rather than a
    // quiet off. Somebody typing YO_ALLOC=abrot believes the check is running.
    if yo_alloc::set_mode_from_env().is_none() {
        eprintln!("yodb: YO_ALLOC is off, report or abort");
        return ExitCode::from(2);
    }

    let code = run();

    // The tally, for a run that reaches the end. `serve` normally does not,
    // because Ctrl-C ends the process rather than the loop, so in practice this
    // is for `check` and for a server that was told to stop. Each site already
    // printed itself on the way past.
    if yo_alloc::mode() == yo_alloc::Mode::Report {
        let (sites, total) = yo_alloc::seen();
        eprintln!("yodb: {total} allocation(s) on a command path, at {sites} place(s)");
    }
    code
}

fn run() -> ExitCode {
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
    let mut unixsocket: Option<std::path::PathBuf> = None;
    let mut tcp = true;

    let mut at = 0;
    while at < args.len() {
        let arg = args[at];
        at += 1;
        match arg {
            "-h" | "--help" => {
                print!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            "--no-port" => tcp = false,
            "--bind" | "--port" | "--unixsocket" => {
                let Some(value) = args.get(at) else {
                    eprintln!("yodb serve: {arg} needs a value");
                    return ExitCode::from(2);
                };
                at += 1;
                if arg == "--bind" {
                    bind = (*value).to_string();
                } else if arg == "--unixsocket" {
                    unixsocket = Some(std::path::PathBuf::from(*value));
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
    if !tcp && unixsocket.is_none() {
        eprintln!("yodb serve: --no-port with no --unixsocket leaves nothing to connect to");
        return ExitCode::from(2);
    }

    let want = if tcp { Some(addr) } else { None };
    let mut server = match serve::Server::open(want, unixsocket.clone()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("yodb serve: {e}");
            return ExitCode::from(2);
        }
    };

    // What it actually bound to, which is the only way to find out when the
    // port asked for was zero.
    let version = env!("CARGO_PKG_VERSION");
    match (tcp, &unixsocket) {
        (true, Some(path)) => {
            let bound = server.local_addr().unwrap_or(addr);
            println!(
                "yodb {version} listening on {bound} and on {}",
                path.display()
            );
        }
        (true, None) => {
            let bound = server.local_addr().unwrap_or(addr);
            println!("yodb {version} listening on {bound}");
        }
        (false, Some(path)) => {
            println!("yodb {version} listening on {}", path.display());
        }
        (false, None) => unreachable!("refused above"),
    }

    // After the listening line and not before it, so a Ctrl-C that arrives in
    // the moment between the two is a process that was never told to serve
    // rather than one that says it is serving and then stops.
    signal::listen();
    let outcome = match server.run(signal::stop()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("yodb serve: {e}");
            ExitCode::FAILURE
        }
    };
    // Explicitly, and before the line that says so, because dropping the server
    // is what unlinks the socket file and closes the doors. Leaving it to the
    // end of the function would print that it had shut down while the path it
    // was listening on was still there for somebody to connect to.
    drop(server);
    if signal::stopped() {
        println!("yodb {version} shutting down");
    }
    outcome
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}
