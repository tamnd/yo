//! The `yodb` command line tool.
//!
//! Three subcommands: `check`, which is the M1 deliverable, `serve`, which puts
//! the RESP engine on a socket, and `restore`, which reads a Redis RDB file and
//! says what is in it. The others arrive with the milestones that need them.
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
mod restore;
mod serve;
mod signal;
mod store;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;

use check::Severity;

const USAGE: &str = "\
yodb, an embedded knowledge engine

usage:
  yodb check FILE [--quick] [--quiet]
  yodb restore FILE [--quiet]
  yodb serve [--bind ADDR] [--port PORT] [--unixsocket PATH] [--no-port]
             [--threads N] [--dir PATH] [--store PATH --maxmemory BYTES]
             [--requirepass PASSWORD] [--aclfile PATH] [--restore FILE]
             [--replicaof HOST PORT] [--masterauth PASSWORD]
             [--masteruser USER] [--replica-read-only yes|no]

  check    read a .yo file and report anything wrong with it. Never writes.
             --quick   skip the records and read only the headers
             --quiet   print findings and the summary, nothing else

  restore  read a Redis RDB file, build every value in it and report what
           it holds. Nothing is written and no server is started, so this
           is the safe way to find out whether a dump from somewhere else
           will load here. Use serve --restore to start a server on it.
             --quiet   print the summary, nothing else

  serve    speak RESP on a socket, so a Redis client can talk to it.
             --bind        address to listen on, 127.0.0.1 by default
             --port        port to listen on, 6379 by default
             --unixsocket  also listen on a socket file, which skips the
                           TCP stack and is the faster way in for a client
                           on the same machine
             --no-port     no TCP at all, socket file only
             --threads     how many threads serve connections. One by
                           default. 0 means one per core. A connection
                           belongs to the thread that accepted it and the
                           keyspace is shared, so the count also decides
                           how finely each database is striped
             --requirepass the password every connection has to send AUTH
                           with before it can send anything else. No
                           password by default, which is a server anybody
                           who can reach the port can read and write.
             --aclfile     a file to read the users out of at startup, and
                           the one ACL SAVE writes back to. Without one the
                           server starts with a single default user that can
                           do anything, and ACL LOAD and ACL SAVE both say so.
                           A file that will not parse stops the server from
                           starting
             --dir         where the server writes, which is where BACKUP
                           puts its files and what CONFIG GET dir answers.
                           The directory the command was run from by default
             --maxmemory   how much memory to use before something has to
                           go, in the units CONFIG SET takes, so 100mb is
                           a hundred mebibytes and 100m is a hundred
                           million. No limit by default
             --restore     a Redis RDB file to build the dataset out of
                           before the port opens, so the first client to
                           connect finds the data already there. The file
                           is read and not held, and nothing writes back
                           to it. A file that will not load stops the
                           server from starting rather than leaving it
                           serving half a dataset
             --replicaof   a master to follow, given as a host and a port the
                           same way redis-server takes them. The server comes
                           up as a replica: it takes a snapshot of the master,
                           applies everything the master does from then on, and
                           refuses writes from ordinary clients. A master that
                           is not up yet is a replica that keeps trying rather
                           than a server that will not start, and REPLICAOF NO
                           ONE at any point makes it a master again
             --masterauth  the password the link to the master sends AUTH with.
                           Nothing by default, which is a master that asks for
                           nothing
             --masteruser  the user to send with that password, for a master
                           with an access control list rather than a single
                           password. Ignored without --masterauth, since a user
                           with no password is not a thing to send
             --replica-read-only
                           whether a write from an ordinary client is refused
                           while this server follows a master. yes by default,
                           which is the only safe setting: a write that lands on
                           a replica is one the master never hears about and
                           that the next full resync throws away
             --store       a file to put cold values in when memory fills
                           up, instead of throwing keys away. The path has
                           to be a new one, because what a previous run
                           left in a store is reachable only through an
                           index that died with it. Needs --maxmemory,
                           since a server with no limit never fills up

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
        Some("restore") => {
            rest.remove(0);
            restore_command(&rest)
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

/// `yodb restore FILE [--quiet]`.
///
/// Shaped like `check` on purpose, down to the exit codes, because it is the
/// same job on the other format: read a file somebody handed you and say whether
/// it is any good. The difference is that a file with something wrong in it does
/// not produce a list of findings, since an RDB is a chain where each item says
/// how long it is and one that does not parse takes the rest of the file with it.
/// So there is one sentence and it is the first thing that went wrong.
fn restore_command(args: &[&str]) -> ExitCode {
    let mut path: Option<PathBuf> = None;
    let mut quiet = false;

    for a in args {
        match *a {
            "--quiet" => quiet = true,
            "-h" | "--help" => {
                print!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            other if other.starts_with('-') => {
                eprintln!("yodb restore: no such option: {other}");
                return ExitCode::from(2);
            }
            other if path.is_none() => path = Some(PathBuf::from(other)),
            other => {
                eprintln!("yodb restore: takes one file, and was also given {other}");
                return ExitCode::from(2);
            }
        }
    }

    let Some(path) = path else {
        eprintln!("yodb restore: which file?\n");
        eprint!("{USAGE}");
        return ExitCode::from(2);
    };

    let done = match restore::restore(&path) {
        Ok(done) => done,
        Err(trouble) => {
            eprintln!("yodb restore: {trouble}");
            return match trouble {
                restore::Trouble::Unreadable(_) => ExitCode::from(2),
                restore::Trouble::Refused(_) => ExitCode::FAILURE,
            };
        }
    };
    if !quiet {
        println!("{}", path.display());
    }
    let mut lines = String::new();
    restore::report(&done, &mut lines);
    print!("{lines}");
    ExitCode::SUCCESS
}

fn serve_command(args: &[&str]) -> ExitCode {
    let mut bind = DEFAULT_BIND.to_string();
    let mut port = DEFAULT_PORT;
    let mut unixsocket: Option<std::path::PathBuf> = None;
    let mut tcp = true;
    let mut store: Option<std::path::PathBuf> = None;
    let mut maxmemory: Option<u64> = None;
    let mut dir: Option<std::path::PathBuf> = None;
    let mut threads = 1usize;
    let mut requirepass: Option<&str> = None;
    let mut aclfile: Option<std::path::PathBuf> = None;
    let mut from_rdb: Option<std::path::PathBuf> = None;
    let mut upstream: Option<(String, u16)> = None;
    let mut masterauth: &str = "";
    let mut masteruser: &str = "";
    let mut replica_read_only = true;

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
            // Two values rather than one, because that is how redis-server takes
            // it and the whole point of the spelling is that a line somebody
            // already has works here.
            "--replicaof" | "--slaveof" => {
                let (Some(host), Some(port)) = (args.get(at), args.get(at + 1)) else {
                    eprintln!("yodb serve: {arg} needs a host and a port");
                    return ExitCode::from(2);
                };
                at += 2;
                let Ok(port) = port.parse::<u16>() else {
                    eprintln!("yodb serve: {port} is not a port to follow");
                    return ExitCode::from(2);
                };
                upstream = Some(((*host).to_string(), port));
            }
            "--bind"
            | "--port"
            | "--unixsocket"
            | "--store"
            | "--maxmemory"
            | "--dir"
            | "--threads"
            | "--requirepass"
            | "--aclfile"
            | "--restore"
            | "--masterauth"
            | "--masteruser"
            | "--replica-read-only" => {
                let Some(value) = args.get(at) else {
                    eprintln!("yodb serve: {arg} needs a value");
                    return ExitCode::from(2);
                };
                at += 1;
                if arg == "--bind" {
                    bind = (*value).to_string();
                } else if arg == "--unixsocket" {
                    unixsocket = Some(std::path::PathBuf::from(*value));
                } else if arg == "--store" {
                    store = Some(std::path::PathBuf::from(*value));
                } else if arg == "--dir" {
                    dir = Some(std::path::PathBuf::from(*value));
                } else if arg == "--restore" {
                    from_rdb = Some(std::path::PathBuf::from(*value));
                } else if arg == "--aclfile" {
                    aclfile = Some(std::path::PathBuf::from(*value));
                } else if arg == "--masterauth" {
                    masterauth = value;
                } else if arg == "--masteruser" {
                    masteruser = value;
                } else if arg == "--replica-read-only" {
                    match *value {
                        "yes" => replica_read_only = true,
                        "no" => replica_read_only = false,
                        other => {
                            eprintln!("yodb serve: --replica-read-only is yes or no, not {other}");
                            return ExitCode::from(2);
                        }
                    }
                } else if arg == "--requirepass" {
                    // An empty one is no password, which is the same thing
                    // `CONFIG SET requirepass ""` means by it.
                    requirepass = Some(value);
                } else if arg == "--threads" {
                    match value.parse::<usize>() {
                        // Zero is one per core, which is what a benchmark
                        // client means by it and is the only reading of a
                        // server with no threads that is worth anything.
                        Ok(0) => {
                            threads = std::thread::available_parallelism()
                                .map_or(1, std::num::NonZero::get);
                        }
                        Ok(n) => threads = n,
                        Err(_) => {
                            eprintln!("yodb serve: {value} is not a number of threads");
                            return ExitCode::from(2);
                        }
                    }
                } else if arg == "--maxmemory" {
                    // The parser `CONFIG SET maxmemory` uses, so that the two
                    // ways of setting the same limit read it the same way.
                    match yo_resp::dispatch::parse_memory(value.as_bytes()) {
                        Some(n) => maxmemory = Some(n),
                        None => {
                            eprintln!("yodb serve: {value} is not an amount of memory");
                            return ExitCode::from(2);
                        }
                    }
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
    if store.is_some() && maxmemory.is_none() {
        eprintln!("yodb serve: --store with no --maxmemory is a file nothing would ever be put in");
        return ExitCode::from(2);
    }
    // Checked and made absolute here rather than when a backup is asked for,
    // because the person who mistyped the path is still watching now and will
    // not be then. Joined onto the working directory rather than canonicalised,
    // since canonicalising on Windows produces a path with a prefix on it that
    // nobody expects to read back out of `CONFIG GET dir`.
    let dir = match dir {
        Some(path) if !path.is_dir() => {
            eprintln!(
                "yodb serve: {}: not a directory to write in",
                path.display()
            );
            return ExitCode::from(2);
        }
        Some(path) if path.is_absolute() => Some(path),
        Some(path) => Some(std::env::current_dir().unwrap_or_default().join(path)),
        None => None,
    };

    // Read before the listener for the same reason the store is opened before
    // it, which is that a mistyped path should not leave a port bound behind it.
    // The bytes are held rather than the file, because the load wants the whole
    // image anyway and holding a file open across the startup would keep a handle
    // on something the server has no further use for.
    let image = match &from_rdb {
        Some(path) => match std::fs::read(path) {
            Ok(bytes) => Some(bytes),
            Err(e) => {
                eprintln!("yodb serve: {}: {e}", path.display());
                return ExitCode::from(2);
            }
        },
        None => None,
    };

    // Before the listener, because a file that cannot be made should not leave a
    // port bound behind it, and because failing here is a mistyped path and the
    // person who typed it is still watching.
    let opened = match &store {
        Some(path) => match store::Store::create(path) {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("yodb serve: {}: {e}", path.display());
                return ExitCode::from(2);
            }
        },
        None => None,
    };

    let want = if tcp { Some(addr) } else { None };
    let mut server = match serve::Server::open(want, unixsocket.clone(), threads) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("yodb serve: {e}");
            return ExitCode::from(2);
        }
    };
    if let Some(password) = requirepass {
        server.set_password(password.as_bytes());
    }
    if let Some(limit) = maxmemory {
        server.set_maxmemory(limit);
    }
    if let Some(dir) = dir {
        server.set_dir(dir);
    }
    // Before the store and before the image, because this is the only startup
    // step that decides who is allowed to see what came out of either.
    if let Some(path) = aclfile {
        server.set_aclfile(path);
        if let Err(errors) = server.load_acl() {
            eprintln!("yodb serve: {errors}");
            return ExitCode::from(2);
        }
    }
    if let Some(opened) = opened {
        server.use_store(opened);
    }
    server.set_replica_read_only(replica_read_only);
    server.set_master_auth(masteruser.as_bytes(), masterauth.as_bytes());
    // What the listener actually got and not what was asked for, because a
    // server started on port zero would otherwise tell its master to dial back
    // on a port nobody is listening on. A server with no port at all announces
    // nothing, which is what a master shows for a replica that did not say.
    if let Ok(bound) = server.local_addr() {
        server.announce_port(bound.port());
    }

    // Last of the startup steps, because it is the only one that can take a while
    // and because everything above changes how the keyspace behaves. A limit set
    // after the dataset was built would be a limit the dataset has already gone
    // past without anything noticing.
    if let Some(image) = &image {
        let path = from_rdb.as_ref().expect("an image came from a path");
        match server.restore(image) {
            Ok(done) => {
                let total = done.total();
                println!(
                    "yodb {} restored {total} key{} from {}",
                    env!("CARGO_PKG_VERSION"),
                    plural(total),
                    path.display()
                );
            }
            Err(refused) => {
                eprintln!("yodb serve: {}: {refused}", path.display());
                return ExitCode::FAILURE;
            }
        }
    }

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
    if threads > 1 {
        println!("yodb {version} serving on {threads} threads");
    }

    // Both numbers, on purpose. A server that only printed the quarter would
    // leave the next person to wonder a quarter of what, and the answer to that
    // is the thing they need when the pool comes out the wrong size. See
    // `yo_resp::cap` for why it is a quarter at all.
    let cap = yo_resp::cap::cap();
    match cap.limit() {
        Some(limit) => println!(
            "yodb {version} may use {} and will size pools from {}",
            bytes(limit),
            bytes(cap.budget())
        ),
        None => println!("yodb {version} found no memory limit to size pools from"),
    }

    // Which of the two things a memory limit means here, said out loud at
    // startup, because they are opposites and the difference is a file.
    match (&store, maxmemory) {
        (Some(path), Some(limit)) => println!(
            "yodb {version} keeps {} in memory and moves the rest into {}",
            bytes(limit),
            path.display()
        ),
        (None, Some(limit)) => println!("yodb {version} evicts keys above {}", bytes(limit)),
        (_, None) => {}
    }

    // Last of everything, because from here on a thread is applying a master's
    // writes and every setting above decides what applying means. The restore
    // is above it too, so a server given both a file and a master starts from
    // the file and then has whatever the master has instead.
    if let Some((host, port)) = &upstream {
        server.follow(host, *port);
        println!("yodb {version} following {host} port {port}");
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

/// A byte count a person can read, for the startup lines only.
///
/// Powers of two with the short names, because that is what `maxmemory` takes
/// and a server that prints one unit and accepts another is a server that gets
/// misconfigured.
fn bytes(n: u64) -> String {
    const UNITS: [(u64, &str); 3] = [(1 << 30, "gb"), (1 << 20, "mb"), (1 << 10, "kb")];
    for (size, name) in UNITS {
        if n >= size {
            let whole = n / size;
            let tenths = (n % size) * 10 / size;
            return if tenths == 0 {
                format!("{whole}{name}")
            } else {
                format!("{whole}.{tenths}{name}")
            };
        }
    }
    format!("{n} bytes")
}

#[cfg(test)]
mod tests {
    use super::bytes;

    #[test]
    fn a_byte_count_prints_in_the_units_maxmemory_takes() {
        assert_eq!(bytes(0), "0 bytes");
        assert_eq!(bytes(512), "512 bytes");
        assert_eq!(bytes(1024), "1kb");
        assert_eq!(bytes(2 * 1024 * 1024 * 1024), "2gb");
        // One tenth is enough to tell 7.5gb from 7gb and not so much that the
        // line stops being readable.
        assert_eq!(bytes(7 * (1 << 30) + (1 << 29)), "7.5gb");
        assert_eq!(bytes(1536), "1.5kb");
    }
}
