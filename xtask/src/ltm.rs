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
    value: usize,
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
            value: 4096,
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
  --value LEN       value length in bytes, 4096 by default
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
    if opts.times == 0 || opts.reads == 0 {
        return Err("--times and --reads both have to be at least one".into());
    }
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
        "yodb {}, {} of memory, {keys} keys of {} bytes, which is {} times the limit",
        binary.display(),
        human(opts.maxmemory),
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

    let at = Instant::now();
    load(&mut conn, keys, opts.value)?;
    let loaded = at.elapsed();
    println!(
        "loaded  {keys} keys in {:.1}s, {:.0} writes a second",
        loaded.as_secs_f64(),
        keys as f64 / loaded.as_secs_f64()
    );

    let before = stats(&mut conn)?;
    let device_before = server.device_bytes();

    let at = Instant::now();
    let latency = reads(&mut conn, keys, opts)?;
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
        "working set   {} in {} keys",
        human(r.keys * opts.value as u64),
        r.keys
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
            other => Err(format!("a reply this harness does not read: {other}{rest}")),
        }
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

fn load(conn: &mut Conn, keys: u64, value: usize) -> Result<(), String> {
    // One buffer for every value. What is being measured is what the server
    // does with them, not whether the harness can make them up quickly.
    let payload: Vec<u8> = (0..value).map(|i| b'a' + (i % 26) as u8).collect();
    let mut out = Vec::with_capacity(BATCH * (value + 64));
    let mut done = 0u64;
    let mut next_report = keys / 10;

    while done < keys {
        let batch = BATCH.min((keys - done) as usize);
        out.clear();
        for i in 0..batch {
            let key = format!("key:{}", done + i as u64);
            cmd(&mut out, &[b"SET", key.as_bytes(), &payload]);
        }
        conn.send(&out)?;
        let mut refused: Option<(u64, String)> = None;
        for i in 0..batch {
            match conn.reply()? {
                Reply::Status => {}
                // Not returned here. There are up to 255 more replies still in
                // flight, and a connection with those left on it cannot be
                // asked anything, which is exactly when somebody wants to ask
                // it what its memory looks like.
                Reply::Error(e) => refused = refused.or(Some((done + i as u64, e))),
                _ => return Err("a SET answered with something that is not a status".into()),
            }
        }
        if let Some((key, message)) = refused {
            let memory = section(conn, "memory").unwrap_or_default();
            let stats = section(conn, "stats").unwrap_or_default();
            let keys_now = dbsize(conn).unwrap_or(0);
            return Err(format!(
                "the server refused the write at key {key}: {message}\nA server with a file should have made room rather than answering that. What it looked like when it did, at {keys_now} keys in:\n{}\n{}",
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

fn reads(conn: &mut Conn, keys: u64, opts: &Options) -> Result<Vec<u64>, String> {
    let mut rng = Rng::new(opts.seed);
    let mut latency = Vec::with_capacity(opts.reads as usize);
    let mut out = Vec::with_capacity(64);
    let mut misses = 0u64;

    for _ in 0..opts.reads {
        let key = format!("key:{}", rng.below(keys));
        out.clear();
        cmd(&mut out, &[b"GET", key.as_bytes()]);
        let at = Instant::now();
        conn.send(&out)?;
        let reply = conn.reply()?;
        latency.push(at.elapsed().as_micros() as u64);
        match reply {
            Reply::Bulk(Some(len)) if len == opts.value => {}
            Reply::Bulk(Some(len)) => {
                return Err(format!(
                    "{key} came back {len} bytes long instead of {}",
                    opts.value
                ));
            }
            Reply::Bulk(None) => misses += 1,
            Reply::Error(e) => {
                return Err(format!(
                    "the server answered a read of {key} with an error: {e}"
                ));
            }
            _ => {
                return Err(format!(
                    "the read of {key} was answered with something that is not a bulk string"
                ));
            }
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
}
