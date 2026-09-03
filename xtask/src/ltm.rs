//! The larger than memory gate harness.
//!
//! M5 has four exit gates and every one of them is a ratio measured against a
//! server that is holding more data than it has memory for. Until `yodb serve
//! --store PATH --maxmemory BYTES` existed there was nothing to point a harness
//! at, so this is the first thing that can produce one of those numbers. The
//! first row is the one the whole design stands on: at a working set ten times
//! memory, a point read should cost 1.05 reads from the store or fewer.
//!
//! It lives in xtask rather than in yo-bench because the gate is about our own
//! internals. yo-bench compares us against redis and valkey over a socket both
//! of them speak, and neither of them has a counter for how many times a read
//! went to a file. The denominator here comes from what this harness issued and
//! the numerator comes from our own INFO, so both halves are ours.
//!
//! What it does, in order: build `yodb`, start it on a port the kernel picks
//! with a store file and a limit, load a working set of the requested multiple
//! of that limit, read the whole keyspace uniformly at random one command at a
//! time, and divide the change in `yo_cold_faults` by the number of reads it
//! issued. Sequential and not pipelined, because a pipelined read has no
//! latency worth reporting and the gate wants the percentiles too.
//!
//! # What the numerator is and is not
//!
//! `yo_cold_faults` counts reads that went to the store rather than reads that
//! went to the device. A log serves a read out of a resident page without
//! touching a disk, and underneath that the operating system has its own cache,
//! so a fault is an upper bound on a device read and never a lower one. At ten
//! times memory almost every fault is a real read, which is why the gate is
//! written against this counter. On Linux the harness also reads `read_bytes`
//! out of `/proc/<pid>/io`, which is the number of bytes the process actually
//! pulled off the block device, and prints it beside the faults. A run where
//! those two disagree by a lot is a run where the page cache is doing the work
//! and the number is not the one the gate is asking for.

use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{self, Child, Command, Stdio};
use std::time::{Duration, Instant};

use crate::root;

/// The bar the first M5 exit gate sets: store reads per point read.
const BAR: f64 = 1.05;

/// The fewest bytes an element can be and still be told apart from its neighbour.
///
/// An element carries its own index in the first ten bytes, so anything shorter
/// than that loses the index and a key ends up holding sixteen copies of one
/// element, which for a set is a set of one.
const MIN_ELEMENT: usize = 16;

/// How many commands go out before the replies are read, during the load.
///
/// The load is not being measured, it just has to finish, and a round trip per
/// key on a working set of half a million of them is minutes of nothing.
const BATCH: usize = 256;

/// What the harness was told to do.
struct Options {
    store: PathBuf,
    maxmemory: u64,
    times: u64,
    kind: Kind,
    /// The command the read phase issues.
    ///
    /// Resolved after the whole command line has been read, because `--read`
    /// has to be checked against `--type` and the two can arrive in either
    /// order.
    read: Point,
    value: usize,
    elements: usize,
    reads: u64,
    seed: u64,
    yodb: Option<PathBuf>,
    keep: bool,
    bar: f64,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            // Not in the repository, because a 2.5 GB file under a source tree
            // is a file somebody commits by accident.
            store: std::env::temp_dir().join(format!("ltm-{}.yo", process::id())),
            maxmemory: 256 * 1024 * 1024,
            times: 10,
            kind: Kind::String,
            read: Point::Get,
            value: 4096,
            elements: 16,
            reads: 100_000,
            seed: 1,
            yodb: None,
            keep: false,
            bar: BAR,
        }
    }
}

const USAGE: &str = "\
usage: cargo xtask ltm [options]

  --store PATH      where to put the store file. A new path under the
                    system temp directory by default, removed at the end
  --maxmemory SIZE  the limit the server runs under, 256mb by default,
                    in the units CONFIG SET takes
  --times N         how many times the limit the working set should be,
                    10 by default, which is the multiple the gate names
  --type NAME       what a key holds: string, set, hash, list, zset,
                    array or stream. string by default
  --value LEN       bytes a key holds, 4096 by default. For a collection
                    it is split across its elements
  --elements N      elements a collection key holds, 16 by default, and
                    ignored for a string
  --read NAME       the command the read phase issues. One choice for
                    every type but the sorted set, which takes ZSCORE by
                    default and also takes ZRANK, which is the command
                    prediction P-4 is about
  --reads N         how many point reads to issue, 100000 by default
  --seed N          the seed for the read order, 1 by default
  --yodb PATH       use this binary instead of building one
  --bar RATIO       the ratio to pass or fail against, 1.05 by default
  --keep            leave the store file behind

exit codes:
  0  the run finished and came in at or under the bar
  1  the run finished and came in over it
  2  the run could not be made at all
";

/// Parse the command line, run the load and the reads, print the report.
pub fn ltm() {
    let args: Vec<String> = std::env::args().skip(2).collect();
    let opts = match parse(&args) {
        Ok(Some(opts)) => opts,
        Ok(None) => return,
        Err(message) => {
            eprintln!("cargo xtask ltm: {message}\n");
            eprint!("{USAGE}");
            process::exit(2);
        }
    };
    match measure(&opts) {
        Ok(true) => {}
        Ok(false) => process::exit(1),
        Err(message) => {
            eprintln!("cargo xtask ltm: {message}");
            process::exit(2);
        }
    }
}

fn parse(args: &[String]) -> Result<Option<Options>, String> {
    let mut opts = Options::default();
    let mut read: Option<String> = None;
    let mut at = 0;
    while at < args.len() {
        let arg = args[at].as_str();
        at += 1;
        match arg {
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(None);
            }
            "--keep" => opts.keep = true,
            _ => {
                let Some(value) = args.get(at) else {
                    return Err(format!("{arg} needs a value"));
                };
                at += 1;
                match arg {
                    "--store" => opts.store = PathBuf::from(value),
                    "--yodb" => opts.yodb = Some(PathBuf::from(value)),
                    "--maxmemory" => {
                        opts.maxmemory = yo_resp::dispatch::parse_memory(value.as_bytes())
                            .ok_or_else(|| format!("{value} is not an amount of memory"))?;
                    }
                    "--times" => opts.times = number(arg, value)?,
                    "--type" => {
                        opts.kind = Kind::parse(value)
                            .ok_or_else(|| format!("{value} is not a type this measures"))?;
                    }
                    "--elements" => {
                        opts.elements = usize::try_from(number(arg, value)?)
                            .map_err(|_| format!("{value} is not an element count"))?;
                    }
                    "--read" => read = Some(value.to_uppercase()),
                    "--reads" => opts.reads = number(arg, value)?,
                    "--seed" => opts.seed = number(arg, value)?,
                    "--value" => {
                        opts.value = usize::try_from(number(arg, value)?)
                            .map_err(|_| format!("{value} is not a value length"))?;
                    }
                    "--bar" => {
                        opts.bar = value
                            .parse()
                            .map_err(|_| format!("{value} is not a ratio"))?;
                    }
                    other => return Err(format!("no such option: {other}")),
                }
            }
        }
    }
    if opts.value == 0 {
        return Err("a value length of zero leaves nothing to demote".into());
    }
    if opts.elements == 0 {
        return Err("a collection with no elements in it is not one".into());
    }
    if opts.kind != Kind::String && opts.value / opts.elements < MIN_ELEMENT {
        return Err(format!(
            "{} bytes across {} elements leaves under {MIN_ELEMENT} in each one, which is too few for one element to differ from the next",
            opts.value, opts.elements
        ));
    }
    if opts.times == 0 || opts.reads == 0 {
        return Err("--times and --reads both have to be at least one".into());
    }
    // After the loop and not inside it, because this is the one option that
    // depends on another one and the two can be given in either order.
    opts.read = match &read {
        None => opts.kind.reads()[0],
        Some(name) => *opts
            .kind
            .reads()
            .iter()
            .find(|p| p.name() == name)
            .ok_or_else(|| {
                let names: Vec<&str> = opts.kind.reads().iter().map(|p| p.name()).collect();
                format!(
                    "a {} is not read with {name}, only with {}",
                    opts.kind.name(),
                    names.join(" or ")
                )
            })?,
    };
    Ok(Some(opts))
}

fn number(arg: &str, value: &str) -> Result<u64, String> {
    value
        .parse()
        .map_err(|_| format!("{arg} takes a number and was given {value}"))
}

/// Everything the run has to say for itself.
struct Report {
    keys: u64,
    loaded: Duration,
    read: Duration,
    faults: u64,
    demoted: u64,
    promoted: u64,
    served: u64,
    bytes_out: u64,
    bytes_in: u64,
    evicted: u64,
    device_bytes: Option<u64>,
    used_memory: u64,
    store_bytes: u64,
    regime: String,
    dbsize: u64,
    latency: Vec<u64>,
}

fn measure(opts: &Options) -> Result<bool, String> {
    let keys = opts.maxmemory * opts.times / opts.value as u64;
    if keys == 0 {
        return Err("that limit and value length leave no keys to load".into());
    }
    if opts.store.exists() {
        return Err(format!(
            "{} is already there. The store has to be a new path, because what a previous run left in one is reachable only through an index that died with it",
            opts.store.display()
        ));
    }

    let binary = match &opts.yodb {
        Some(path) => path.clone(),
        None => build()?,
    };

    println!(
        "yodb {}, {} of memory, {keys} {} keys of {} bytes, which is {} times the limit",
        binary.display(),
        human(opts.maxmemory),
        opts.kind.name(),
        opts.value,
        opts.times
    );

    let mut server = Server::start(&binary, opts)?;
    let outcome = run_against(&mut server, opts, keys);
    server.stop();
    if !opts.keep {
        let _ = fs::remove_file(&opts.store);
    }
    let report = outcome?;

    Ok(present(opts, &report))
}

fn run_against(server: &mut Server, opts: &Options, keys: u64) -> Result<Report, String> {
    let mut conn = Conn::connect(&server.addr)?;

    let fix = Fixtures::new(opts);

    let at = Instant::now();
    load(&mut conn, keys, opts, &fix)?;
    let loaded = at.elapsed();
    println!(
        "loaded  {keys} keys in {:.1}s, {:.0} writes a second",
        loaded.as_secs_f64(),
        keys as f64 / loaded.as_secs_f64()
    );

    let before = stats(&mut conn)?;
    let device_before = server.device_bytes();

    let at = Instant::now();
    let latency = reads(&mut conn, keys, opts, &fix)?;
    let read = at.elapsed();

    let after = stats(&mut conn)?;
    let device_after = server.device_bytes();
    let memory = section(&mut conn, "memory")?;
    let dbsize = dbsize(&mut conn)?;

    let delta = |name: &str| after.get(name).saturating_sub(before.get(name));
    Ok(Report {
        keys,
        loaded,
        read,
        faults: delta("yo_cold_faults"),
        demoted: after.get("yo_cold_demoted"),
        promoted: delta("yo_cold_promoted"),
        served: delta("yo_cold_served"),
        bytes_out: after.get("yo_cold_bytes_out"),
        bytes_in: delta("yo_cold_bytes_in"),
        evicted: after.get("evicted_keys"),
        device_bytes: match (device_before, device_after) {
            (Some(a), Some(b)) => Some(b.saturating_sub(a)),
            _ => None,
        },
        used_memory: field(&memory, "used_memory")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
        store_bytes: field(&memory, "yo_store_bytes")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
        regime: field(&memory, "yo_memory_regime")
            .unwrap_or("?")
            .to_string(),
        dbsize,
        latency,
    })
}

/// Print the report and answer whether it passed.
fn present(opts: &Options, r: &Report) -> bool {
    let reads = opts.reads as f64;
    let ratio = r.faults as f64 / reads;
    let mut sorted = r.latency.clone();
    sorted.sort_unstable();
    let at = |p: f64| -> u64 {
        if sorted.is_empty() {
            return 0;
        }
        let i = ((sorted.len() as f64 - 1.0) * p).round() as usize;
        sorted[i]
    };

    println!();
    println!(
        "working set   {} in {} {} keys, read with {}",
        human(r.keys * opts.value as u64),
        r.keys,
        opts.kind.name(),
        opts.read.name()
    );
    println!(
        "memory        {} used against a {} limit, regime {}",
        human(r.used_memory),
        human(opts.maxmemory),
        r.regime
    );
    println!(
        "store         {} in the file, {} written, {} read back",
        human(r.store_bytes),
        human(r.bytes_out),
        human(r.bytes_in)
    );
    println!(
        "keyspace      {} keys still there, {} evicted, {} values demoted",
        r.dbsize, r.evicted, r.demoted
    );
    println!(
        "load          {:.1}s, {:.0} writes a second",
        r.loaded.as_secs_f64(),
        r.keys as f64 / r.loaded.as_secs_f64()
    );
    println!(
        "reads         {} in {:.1}s, {:.0} a second",
        opts.reads,
        r.read.as_secs_f64(),
        reads / r.read.as_secs_f64()
    );
    println!(
        "latency       p50 {}us  p99 {}us  p999 {}us  max {}us",
        at(0.50),
        at(0.99),
        at(0.999),
        sorted.last().copied().unwrap_or(0)
    );
    println!(
        "store reads   {} faults, {} served from the file, {} promoted back",
        r.faults, r.served, r.promoted
    );
    if let Some(bytes) = r.device_bytes {
        // What the kernel says actually came off the device, which is the only
        // check on whether the page cache answered instead of the disk.
        println!(
            "device        {} read during the read phase, {} a point read",
            human(bytes),
            human(bytes / opts.reads.max(1))
        );
    }
    println!();

    if r.dbsize != r.keys {
        println!(
            "FAIL  {} of {} keys are gone, so this run threw data away rather than moving it",
            r.keys - r.dbsize,
            r.keys
        );
        false
    } else if ratio <= opts.bar {
        println!(
            "PASS  {ratio:.3} store reads a point read, against a bar of {:.2}",
            opts.bar
        );
        true
    } else {
        println!(
            "FAIL  {ratio:.3} store reads a point read, against a bar of {:.2}",
            opts.bar
        );
        false
    }
}

/// Build the binary under test.
///
/// Release and not debug, because every number here is a timing or a ratio
/// against one, and a debug build of the engine is not the thing the gate is
/// about.
fn build() -> Result<PathBuf, String> {
    println!("building yodb");
    let status = Command::new(env!("CARGO"))
        .current_dir(root())
        .args(["build", "--release", "-p", "yo-cli", "--bin", "yodb"])
        .status()
        .map_err(|e| format!("could not run cargo: {e}"))?;
    if !status.success() {
        return Err("yodb did not build".into());
    }
    let dir = match std::env::var_os("CARGO_TARGET_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => root().join("target"),
    };
    let name = if cfg!(windows) { "yodb.exe" } else { "yodb" };
    let path = dir.join("release").join(name);
    if path.exists() {
        Ok(path)
    } else {
        Err(format!("built, but {} is not there", path.display()))
    }
}

/// The server under test, and the two things the harness needs from it.
struct Server {
    child: Child,
    addr: String,
}

impl Server {
    fn start(binary: &Path, opts: &Options) -> Result<Server, String> {
        // Port zero, so two of these can run on the same machine and neither
        // has to be told which port the other took.
        let mut child = Command::new(binary)
            .args([
                "serve",
                "--port",
                "0",
                "--store",
                &opts.store.display().to_string(),
                "--maxmemory",
                &opts.maxmemory.to_string(),
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| format!("could not start {}: {e}", binary.display()))?;

        let stdout = child.stdout.take().expect("stdout was piped");
        let mut lines = BufReader::new(stdout).lines();
        let mut addr = None;
        for line in lines.by_ref() {
            let line = line.map_err(|e| format!("could not read the server's output: {e}"))?;
            println!("yodb: {line}");
            if let Some(rest) = line.split(" listening on ").nth(1) {
                addr = Some(rest.trim().to_string());
                break;
            }
        }
        let Some(addr) = addr else {
            let _ = child.kill();
            return Err("the server stopped before it said what it was listening on".into());
        };

        // The rest of its output goes to ours on a thread of its own. A server
        // whose pipe fills up stops serving, and the startup lines are not the
        // only ones it prints.
        std::thread::spawn(move || {
            for line in lines.map_while(Result::ok) {
                println!("yodb: {line}");
            }
        });

        Ok(Server { child, addr })
    }

    /// Bytes this process has pulled off the block device, on a kernel that
    /// keeps that count. None everywhere else, which is every platform but
    /// Linux.
    fn device_bytes(&self) -> Option<u64> {
        let text = fs::read_to_string(format!("/proc/{}/io", self.child.id())).ok()?;
        field(&text, "read_bytes").and_then(|v| v.trim().parse().ok())
    }

    fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A RESP connection, with the four reply shapes this harness asks for.
struct Conn {
    w: TcpStream,
    r: BufReader<TcpStream>,
    line: String,
    body: Vec<u8>,
}

/// One reply, as much of it as this harness needs.
enum Reply {
    Status,
    Error(String),
    Int(i64),
    /// A bulk string, as its length. The body goes in `Conn::body`.
    Bulk(Option<usize>),
    /// An array, as how many replies follow it. Only the stream row asks for
    /// one, and only to find out whether the entry it named came back.
    Array(Option<usize>),
}

impl Conn {
    fn connect(addr: &str) -> Result<Conn, String> {
        let s =
            TcpStream::connect(addr).map_err(|e| format!("could not connect to {addr}: {e}"))?;
        s.set_nodelay(true).map_err(|e| format!("{e}"))?;
        let r = BufReader::new(s.try_clone().map_err(|e| format!("{e}"))?);
        Ok(Conn {
            w: s,
            r,
            line: String::new(),
            body: Vec::new(),
        })
    }

    fn send(&mut self, bytes: &[u8]) -> Result<(), String> {
        self.w
            .write_all(bytes)
            .map_err(|e| format!("could not write to the server: {e}"))
    }

    fn reply(&mut self) -> Result<Reply, String> {
        self.line.clear();
        let read = self
            .r
            .read_line(&mut self.line)
            .map_err(|e| format!("could not read a reply: {e}"))?;
        if read == 0 {
            return Err("the server closed the connection".into());
        }
        let line = self.line.trim_end().to_string();
        let (tag, rest) = line.split_at(1);
        match tag {
            "+" => Ok(Reply::Status),
            "-" => Ok(Reply::Error(rest.to_string())),
            ":" => Ok(Reply::Int(
                rest.parse().map_err(|_| format!("bad integer: {rest}"))?,
            )),
            "$" => {
                let len: i64 = rest
                    .parse()
                    .map_err(|_| format!("bad bulk length: {rest}"))?;
                if len < 0 {
                    return Ok(Reply::Bulk(None));
                }
                let len = len as usize;
                self.body.resize(len + 2, 0);
                self.r
                    .read_exact(&mut self.body)
                    .map_err(|e| format!("could not read a bulk body: {e}"))?;
                self.body.truncate(len);
                Ok(Reply::Bulk(Some(len)))
            }
            "*" => {
                let len: i64 = rest
                    .parse()
                    .map_err(|_| format!("bad array length: {rest}"))?;
                if len < 0 {
                    return Ok(Reply::Array(None));
                }
                Ok(Reply::Array(Some(len as usize)))
            }
            other => Err(format!("a reply this harness does not read: {other}{rest}")),
        }
    }

    /// Read the replies nested inside an array of `n`, and throw them away.
    ///
    /// An XRANGE row is an array of entries, each an id and an array of field
    /// and value, and what this row is measuring is the fault that produced it
    /// rather than anything in it. What matters is that the whole reply comes
    /// off the socket, because a connection with half a reply left on it
    /// answers the next command with the other half.
    fn drain(&mut self, n: usize) -> Result<(), String> {
        for _ in 0..n {
            match self.reply()? {
                Reply::Array(Some(inner)) => self.drain(inner)?,
                Reply::Error(e) => return Err(e),
                _ => {}
            }
        }
        Ok(())
    }
}

/// One command, appended to a buffer that may already hold others.
fn cmd(out: &mut Vec<u8>, parts: &[&[u8]]) {
    out.extend_from_slice(format!("*{}\r\n", parts.len()).as_bytes());
    for p in parts {
        out.extend_from_slice(format!("${}\r\n", p.len()).as_bytes());
        out.extend_from_slice(p);
        out.extend_from_slice(b"\r\n");
    }
}

/// Everything a key of one kind is written from and read back with.
///
/// Built once for the whole run. What is being measured is the server, and a
/// harness that formats sixteen member names per key spends the run in
/// `format!` rather than in the thing under test.
struct Fixtures {
    /// One element's bytes, `elements` of them, all the same length.
    parts: Vec<Vec<u8>>,
    /// Hash field names, one an element.
    fields: Vec<Vec<u8>>,
    /// Element positions as decimal, for LINDEX and ARGET.
    indexes: Vec<Vec<u8>>,
    /// Zset scores, one an element.
    scores: Vec<Vec<u8>>,
    /// Stream ids, one an element. They start at one because `0-0` is not an id
    /// a stream will take.
    ids: Vec<Vec<u8>>,
}

impl Fixtures {
    fn new(opts: &Options) -> Fixtures {
        let (n, each) = match opts.kind {
            Kind::String => (1, opts.value),
            _ => (opts.elements, opts.value / opts.elements),
        };
        Fixtures {
            parts: (0..n).map(|i| element(i as u64, each)).collect(),
            fields: (0..n).map(|i| format!("f:{i}").into_bytes()).collect(),
            indexes: (0..n).map(|i| format!("{i}").into_bytes()).collect(),
            scores: (0..n).map(|i| format!("{i}").into_bytes()).collect(),
            ids: (0..n)
                .map(|i| format!("{}-0", i + 1).into_bytes())
                .collect(),
        }
    }

    /// Append the commands that write one key, and answer how many replies they
    /// will produce.
    ///
    /// One command for every kind but the stream, which has no command that
    /// appends more than one entry.
    fn write(&self, out: &mut Vec<u8>, kind: Kind, key: &[u8]) -> usize {
        let mut parts: Vec<&[u8]> = Vec::with_capacity(3 + self.parts.len() * 2);
        match kind {
            Kind::String => {
                cmd(out, &[b"SET", key, &self.parts[0]]);
                return 1;
            }
            Kind::Set | Kind::List => {
                parts.push(if kind == Kind::Set { b"SADD" } else { b"RPUSH" });
                parts.push(key);
                parts.extend(self.parts.iter().map(Vec::as_slice));
            }
            Kind::Hash => {
                parts.push(b"HSET");
                parts.push(key);
                for (field, part) in self.fields.iter().zip(&self.parts) {
                    parts.push(field);
                    parts.push(part);
                }
            }
            Kind::Zset => {
                parts.push(b"ZADD");
                parts.push(key);
                for (score, part) in self.scores.iter().zip(&self.parts) {
                    parts.push(score);
                    parts.push(part);
                }
            }
            Kind::Array => {
                // From index zero, so the array is dense and the whole of it is
                // one body rather than a slice per element.
                parts.push(b"ARSET");
                parts.push(key);
                parts.push(b"0");
                parts.extend(self.parts.iter().map(Vec::as_slice));
            }
            Kind::Stream => {
                for (id, part) in self.ids.iter().zip(&self.parts) {
                    cmd(out, &[b"XADD", key, id, b"f", part]);
                }
                return self.parts.len();
            }
        }
        cmd(out, &parts);
        1
    }

    /// Append the one command that reads element `at` of one key.
    fn read(&self, out: &mut Vec<u8>, read: Point, key: &[u8], at: usize) {
        match read {
            Point::Get => cmd(out, &[b"GET", key]),
            Point::Sismember => cmd(out, &[b"SISMEMBER", key, &self.parts[at]]),
            Point::Hget => cmd(out, &[b"HGET", key, &self.fields[at]]),
            Point::Lindex => cmd(out, &[b"LINDEX", key, &self.indexes[at]]),
            Point::Zscore => cmd(out, &[b"ZSCORE", key, &self.parts[at]]),
            Point::Zrank => cmd(out, &[b"ZRANK", key, &self.parts[at]]),
            Point::Arget => cmd(out, &[b"ARGET", key, &self.indexes[at]]),
            Point::Xrange => cmd(out, &[b"XRANGE", key, &self.ids[at], &self.ids[at]]),
        }
    }
}

fn load(conn: &mut Conn, keys: u64, opts: &Options, fix: &Fixtures) -> Result<(), String> {
    let mut out = Vec::with_capacity(BATCH * (opts.value + 64 * fix.parts.len()));
    let mut done = 0u64;
    let mut next_report = keys / 10;

    while done < keys {
        let batch = BATCH.min((keys - done) as usize);
        out.clear();
        let mut replies = 0;
        for i in 0..batch {
            let key = format!("key:{}", done + i as u64);
            replies += fix.write(&mut out, opts.kind, key.as_bytes());
        }
        conn.send(&out)?;
        let mut refused: Option<String> = None;
        for _ in 0..replies {
            match conn.reply()? {
                // Not returned here. There are up to a few thousand more replies
                // still in flight, and a connection with those left on it cannot
                // be asked anything, which is exactly when somebody wants to ask
                // it what its memory looks like.
                Reply::Error(e) => refused = refused.or(Some(e)),
                Reply::Array(Some(n)) => conn.drain(n)?,
                _ => {}
            }
        }
        if let Some(message) = refused {
            let memory = section(conn, "memory").unwrap_or_default();
            let stats = section(conn, "stats").unwrap_or_default();
            let keys_now = dbsize(conn).unwrap_or(0);
            return Err(format!(
                "the server refused a write in the batch at key {done}: {message}\nA server with a file should have made room rather than answering that. What it looked like when it did, at {keys_now} keys in:\n{}\n{}",
                memory.trim(),
                stats.trim()
            ));
        }
        done += batch as u64;
        if done >= next_report && keys >= 10 {
            println!("  loaded {done} of {keys}");
            next_report += keys / 10;
        }
    }
    Ok(())
}

fn reads(conn: &mut Conn, keys: u64, opts: &Options, fix: &Fixtures) -> Result<Vec<u64>, String> {
    let mut rng = Rng::new(opts.seed);
    let mut latency = Vec::with_capacity(opts.reads as usize);
    let mut out = Vec::with_capacity(64);
    let mut misses = 0u64;
    let elements = fix.parts.len() as u64;

    for _ in 0..opts.reads {
        let key = format!("key:{}", rng.below(keys));
        // A random element as well as a random key. Reading element zero every
        // time would be a fair measurement of the fault, since the whole body
        // comes back at once either way, but it would stop being one the day a
        // body learns to come back in pieces.
        let at = rng.below(elements) as usize;
        out.clear();
        fix.read(&mut out, opts.read, key.as_bytes(), at);
        let start = Instant::now();
        conn.send(&out)?;
        let reply = conn.reply()?;
        // Nested replies before the clock stops, because the entry is not back
        // until the last of it is off the socket.
        if let Reply::Array(Some(n)) = reply {
            conn.drain(n)?;
        }
        latency.push(start.elapsed().as_micros() as u64);
        match &reply {
            // The string row checks the length as well, because it is the one
            // kind whose whole value comes back and a short one would mean the
            // file handed back the wrong bytes.
            Reply::Bulk(Some(len)) if opts.kind == Kind::String && *len != opts.value => {
                return Err(format!(
                    "{key} came back {len} bytes long instead of {}",
                    opts.value
                ));
            }
            Reply::Error(e) => {
                return Err(format!(
                    "the server answered a read of {key} with an error: {e}"
                ));
            }
            Reply::Status => {
                return Err(format!("the read of {key} was answered with a status"));
            }
            _ => {}
        }
        if !opts.read.found(&reply) {
            misses += 1;
        }
    }
    if misses > 0 {
        return Err(format!(
            "{misses} of {} reads found nothing, so this run lost data rather than moving it",
            opts.reads
        ));
    }
    Ok(latency)
}

/// The INFO stats counters, as name and value.
struct Counters(String);

impl Counters {
    fn get(&self, name: &str) -> u64 {
        field(&self.0, name)
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    }
}

fn stats(conn: &mut Conn) -> Result<Counters, String> {
    Ok(Counters(section(conn, "stats")?))
}

fn section(conn: &mut Conn, name: &str) -> Result<String, String> {
    let mut out = Vec::new();
    cmd(&mut out, &[b"INFO", name.as_bytes()]);
    conn.send(&out)?;
    match conn.reply()? {
        Reply::Bulk(Some(_)) => Ok(String::from_utf8_lossy(&conn.body).into_owned()),
        _ => Err(format!("INFO {name} did not answer with a section")),
    }
}

fn dbsize(conn: &mut Conn) -> Result<u64, String> {
    let mut out = Vec::new();
    cmd(&mut out, &[b"DBSIZE"]);
    conn.send(&out)?;
    match conn.reply()? {
        Reply::Int(n) => Ok(n.max(0) as u64),
        _ => Err("DBSIZE did not answer with a number".into()),
    }
}

/// The value of one `name:value` line, from INFO or from /proc.
fn field<'a>(text: &'a str, name: &str) -> Option<&'a str> {
    text.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        (key.trim() == name).then(|| value.trim())
    })
}

/// What a key holds, which is the axis the first M5 gate is measured across.
///
/// The gate says 1.05 store reads a point read at ten times memory across every
/// type, and a string was the only type that could be measured until every
/// collection body could leave memory too. A collection is the harder case and
/// not the easier one: a string demotes as the bytes it already is, and a
/// collection has to be frozen to a form on the way out and rebuilt on the way
/// back, so a row here is measuring the promote path as well as the read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    String,
    Set,
    Hash,
    List,
    Zset,
    Array,
    Stream,
}

impl Kind {
    fn parse(name: &str) -> Option<Kind> {
        Some(match name {
            "string" => Kind::String,
            "set" => Kind::Set,
            "hash" => Kind::Hash,
            "list" => Kind::List,
            "zset" | "sortedset" => Kind::Zset,
            "array" => Kind::Array,
            "stream" => Kind::Stream,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Kind::String => "string",
            Kind::Set => "set",
            Kind::Hash => "hash",
            Kind::List => "list",
            Kind::Zset => "zset",
            Kind::Array => "array",
            Kind::Stream => "stream",
        }
    }

    /// The point reads this type can be measured with, the default first.
    ///
    /// One element out of one key in every case, because the gate counts store
    /// reads a point read and a command that walks a whole collection would be
    /// answering a different question with the same arithmetic.
    ///
    /// Only the sorted set has two, and it has two because the gate names both:
    /// the row across every type is a lookup, which is ZSCORE, and prediction
    /// P-4 is about ZRANK, which finds the same element and then has to know
    /// where it sits.
    fn reads(self) -> &'static [Point] {
        match self {
            Kind::String => &[Point::Get],
            Kind::Set => &[Point::Sismember],
            Kind::Hash => &[Point::Hget],
            Kind::List => &[Point::Lindex],
            Kind::Zset => &[Point::Zscore, Point::Zrank],
            Kind::Array => &[Point::Arget],
            Kind::Stream => &[Point::Xrange],
        }
    }
}

/// The command one point read is issued with.
///
/// A command and not just a name, because the reply decides what counts as a
/// hit and the two are not the same rule. SISMEMBER answers one for a hit and
/// zero for a miss, so a zero there is a key that is not holding what the
/// harness put in it. ZRANK answers the position, and a zero there is the first
/// element and the most ordinary hit there is. Reading one rule onto the other
/// would turn every ZRANK row into a run of misses or hide a broken one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Point {
    Get,
    Sismember,
    Hget,
    Lindex,
    Zscore,
    Zrank,
    Arget,
    Xrange,
}

impl Point {
    /// The command name, which is what the report prints and what `--read`
    /// takes.
    fn name(self) -> &'static str {
        match self {
            Point::Get => "GET",
            Point::Sismember => "SISMEMBER",
            Point::Hget => "HGET",
            Point::Lindex => "LINDEX",
            Point::Zscore => "ZSCORE",
            Point::Zrank => "ZRANK",
            Point::Arget => "ARGET",
            Point::Xrange => "XRANGE",
        }
    }

    /// Whether this reply means the element was there.
    fn found(self, reply: &Reply) -> bool {
        match reply {
            Reply::Bulk(Some(_)) => true,
            Reply::Bulk(None) => false,
            // The rank of the first element is zero and the answer to a
            // membership test that missed is also zero, so which one this is
            // depends on the command and not on the reply.
            Reply::Int(_) if self == Point::Zrank => true,
            Reply::Int(n) => *n != 0,
            Reply::Array(Some(n)) => *n != 0,
            Reply::Array(None) => false,
            Reply::Error(_) | Reply::Status => false,
        }
    }
}

/// One element's bytes, `len` of them, with `at` written into the front.
///
/// The front has to differ between elements of the same key, because a set with
/// the same member sixteen times is a set of one, and it has to be a fixed width
/// so that every element is the length the reader checks for.
fn element(at: u64, len: usize) -> Vec<u8> {
    let mut v = format!("{at:09}:").into_bytes();
    v.resize(len, b'e');
    v
}

/// A byte count a person can read.
fn human(n: u64) -> String {
    const UNITS: [(u64, &str); 3] = [(1 << 30, "GB"), (1 << 20, "MB"), (1 << 10, "KB")];
    for (size, name) in UNITS {
        if n >= size {
            return format!("{:.2} {name}", n as f64 / size as f64);
        }
    }
    format!("{n} bytes")
}

/// splitmix64, which is eight lines and has no dependency behind it.
///
/// The read order has to be the same on every machine that runs this, or two
/// runs of the same gate are not comparable. A seeded generator in the harness
/// is the way to get that, and this one passes the tests that matter for
/// picking a key at random.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A number under `n`, by the multiply and shift that costs one multiply
    /// and has a bias too small to matter at these key counts.
    fn below(&mut self, n: u64) -> u64 {
        ((u128::from(self.next()) * u128::from(n)) >> 64) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_field_comes_out_of_an_info_section() {
        let text = "# Stats\r\nevicted_keys:0\r\nyo_cold_faults:1234\r\n";
        assert_eq!(field(text, "yo_cold_faults"), Some("1234"));
        assert_eq!(field(text, "nothing_like_it"), None);
    }

    #[test]
    fn a_proc_io_line_reads_the_same_way() {
        // The other place a name and a value are separated by a colon, which is
        // why there is one parser and not two.
        let text = "rchar: 100\nread_bytes: 8192\nwrite_bytes: 0\n";
        assert_eq!(field(text, "read_bytes"), Some("8192"));
    }

    #[test]
    fn a_command_goes_out_as_an_array_of_bulk_strings() {
        let mut out = Vec::new();
        cmd(&mut out, &[b"GET", b"key:1"]);
        assert_eq!(out, b"*2\r\n$3\r\nGET\r\n$5\r\nkey:1\r\n");
    }

    #[test]
    fn random_keys_stay_inside_the_keyspace_and_cover_it() {
        let mut rng = Rng::new(1);
        let mut seen = [false; 64];
        for _ in 0..4096 {
            let k = rng.below(64);
            assert!(k < 64, "{k} is outside a keyspace of 64");
            seen[k as usize] = true;
        }
        assert!(seen.iter().all(|hit| *hit), "some keys are never picked");
    }

    #[test]
    fn the_same_seed_reads_in_the_same_order() {
        // Two runs of a gate that read different keys are not two runs of the
        // same gate.
        let (mut a, mut b) = (Rng::new(7), Rng::new(7));
        for _ in 0..100 {
            assert_eq!(a.below(1_000_000), b.below(1_000_000));
        }
    }

    #[test]
    fn defaults_load_ten_times_the_limit() {
        let o = Options::default();
        let keys = o.maxmemory * o.times / o.value as u64;
        assert_eq!(keys * o.value as u64, o.maxmemory * 10);
    }

    /// The options a row of a given kind runs with, without a server.
    fn opts(kind: Kind) -> Options {
        Options {
            kind,
            value: 4096,
            elements: 16,
            ..Options::default()
        }
    }

    #[test]
    fn every_type_the_gate_names_can_be_asked_for() {
        for name in ["string", "set", "hash", "list", "zset", "array", "stream"] {
            let kind = Kind::parse(name).unwrap_or_else(|| panic!("{name} is not a type"));
            assert_eq!(kind.name(), name);
        }
        assert_eq!(Kind::parse("sortedset"), Some(Kind::Zset));
        assert_eq!(Kind::parse("geo"), None);
    }

    #[test]
    fn the_elements_of_one_key_all_differ_and_are_all_the_same_length() {
        // A set of sixteen copies of one member is a set of one, and a run over
        // it would be measuring a body a sixteenth of the size the report says.
        let fix = Fixtures::new(&opts(Kind::Set));
        assert_eq!(fix.parts.len(), 16);
        for part in &fix.parts {
            assert_eq!(part.len(), 4096 / 16);
        }
        let mut sorted = fix.parts.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 16, "two elements of one key are the same");
    }

    #[test]
    fn a_string_key_holds_the_whole_value_in_one_element() {
        let fix = Fixtures::new(&opts(Kind::String));
        assert_eq!(fix.parts.len(), 1);
        assert_eq!(fix.parts[0].len(), 4096);
    }

    #[test]
    fn a_write_says_how_many_replies_it_is_going_to_get() {
        // The load reads exactly this many replies before it sends the next
        // batch, so a count that is one out leaves the connection holding a
        // reply that the next command gets answered with.
        for (kind, expected) in [
            (Kind::String, 1),
            (Kind::Set, 1),
            (Kind::Hash, 1),
            (Kind::List, 1),
            (Kind::Zset, 1),
            (Kind::Array, 1),
            (Kind::Stream, 16),
        ] {
            let o = opts(kind);
            let fix = Fixtures::new(&o);
            let mut out = Vec::new();
            let replies = fix.write(&mut out, kind, b"key:0");
            assert_eq!(replies, expected, "{} said the wrong count", kind.name());
            assert_eq!(
                out.iter().filter(|b| **b == b'*').count(),
                expected,
                "{} sent a number of commands its count does not match",
                kind.name()
            );
        }
    }

    #[test]
    fn a_read_asks_for_one_element_with_the_command_the_report_names() {
        for kind in [
            Kind::String,
            Kind::Set,
            Kind::Hash,
            Kind::List,
            Kind::Zset,
            Kind::Array,
            Kind::Stream,
        ] {
            let o = opts(kind);
            let fix = Fixtures::new(&o);
            let at = if kind == Kind::String { 0 } else { 3 };
            // Every read the type offers and not just the default one, so a
            // second one added to a type cannot go out as the first one.
            for read in kind.reads() {
                let mut out = Vec::new();
                fix.read(&mut out, *read, b"key:0", at);
                let text = String::from_utf8_lossy(&out).into_owned();
                assert!(
                    text.contains(read.name()),
                    "a {} read went out as {text:?} and the report calls it {}",
                    kind.name(),
                    read.name()
                );
                assert_eq!(
                    out.iter().filter(|b| **b == b'*').count(),
                    1,
                    "a {} read is more than one command",
                    kind.name()
                );
            }
        }
    }

    /// The trap the second read on a type walks into. SISMEMBER answers one for
    /// a hit and zero for a miss, so a zero there is a key that is not holding
    /// what the harness put in it. ZRANK answers the position, so a zero there
    /// is the first element and the most ordinary hit there is. One rule read
    /// onto the other turns every ZRANK row into a run of misses and stops the
    /// run before it reports anything.
    #[test]
    fn a_rank_of_zero_is_the_first_element_and_not_a_miss() {
        assert!(Point::Zrank.found(&Reply::Int(0)));
        assert!(Point::Zrank.found(&Reply::Int(41)));
        assert!(!Point::Sismember.found(&Reply::Int(0)));
        assert!(Point::Sismember.found(&Reply::Int(1)));
        // And a key that is not there is a miss for both of them.
        assert!(!Point::Zrank.found(&Reply::Bulk(None)));
        assert!(!Point::Zscore.found(&Reply::Bulk(None)));
    }

    /// `--read` names a command, and a command that does not read the type
    /// under test is a mistake worth stopping for rather than one to fall back
    /// from. Falling back would run the whole load, minutes of it, and then
    /// report a row the caller did not ask for.
    #[test]
    fn a_read_that_does_not_fit_the_type_is_refused() {
        let args = |kind: &str, read: &str| {
            ["--type", kind, "--read", read]
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
        };
        let opts = parse(&args("zset", "zrank"))
            .expect("zrank reads a zset")
            .expect("not the help");
        assert_eq!(opts.read, Point::Zrank);
        // The default is the first one the type lists, whichever order the
        // options arrived in.
        let opts = parse(&args("zset", "zscore")).unwrap().unwrap();
        assert_eq!(opts.read, Point::Zscore);
        let mut swapped = vec!["--read".to_string(), "zrank".to_string()];
        swapped.extend(["--type".to_string(), "zset".to_string()]);
        assert_eq!(parse(&swapped).unwrap().unwrap().read, Point::Zrank);

        let err = parse(&args("set", "zrank"))
            .err()
            .expect("a set is not read with ZRANK");
        assert!(err.contains("SISMEMBER"), "{err}");
    }

    #[test]
    fn a_stream_never_writes_the_id_a_stream_will_not_take() {
        // `0-0` is not an id XADD accepts, so the ids are one ahead of the
        // element positions and the read has to use the same ones.
        let fix = Fixtures::new(&opts(Kind::Stream));
        assert_eq!(fix.ids[0], b"1-0");
        assert_eq!(fix.ids[15], b"16-0");
    }

    #[test]
    fn elements_too_small_to_tell_apart_are_refused() {
        let args = |kind: &str, elements: &str| {
            vec![
                "--type".to_string(),
                kind.to_string(),
                "--value".to_string(),
                "512".to_string(),
                "--elements".to_string(),
                elements.to_string(),
            ]
        };
        assert!(parse(&args("set", "16")).is_ok());
        assert!(parse(&args("set", "64")).is_err());
        assert!(parse(&args("set", "0")).is_err());
        // A string is one element and the split does not apply to it.
        assert!(parse(&args("string", "64")).is_ok());
    }
}
