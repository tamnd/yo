//! `MEMORY`, which is the server answering questions about itself rather than
//! about anything a client stored.
//!
//! # Why a store like this can answer at all
//!
//! Redis works out what it is using by asking jemalloc, because a value in
//! Redis is a graph of small allocations and there is no other way to add them
//! up. This store is not built that way: a key is a record in an arena and a
//! collection is a body in a slab, and both of those already know what they
//! cost, because a `maxmemory` server has to be able to ask them once a batch.
//! So every number here is read out of the store rather than out of an
//! allocator, and the ones an allocator would have to answer are the ones this
//! file says nothing about.
//!
//! That is the shape of the whole difference. `MEMORY USAGE` is exact here and
//! sampled there. `MEMORY STATS` reports the same thirty six fields plus one a
//! database, because a client library and a dashboard both read it by name, but
//! the four that describe an allocator's own fragmentation describe the arena's
//! instead, which is the thing that actually fragments here. And the fields
//! about a replication backlog, an append only file, a cluster link and a
//! replica's buffer are zero because there is nothing behind them yet, rather
//! than zero because the server did not look.
//!
//! # `MEMORY USAGE` and `SAMPLES`
//!
//! A real server walks a collection and multiplies, so it takes a sample count
//! and the answer for a big hash is an estimate that moves when you ask twice.
//! This one asks the collection, which has been keeping the number the whole
//! time, so the answer is exact and asking twice gives the same number. The
//! `SAMPLES` argument is still parsed and still refused when it is malformed,
//! because a client that sends it should get the same answer about whether it
//! sent it correctly. It just does not change the reply. That is D-6.
//!
//! # `MEMORY DOCTOR`
//!
//! Sam is a real server's joke and it is kept, sentence for sentence, because
//! the report is a thing people paste into issues and a support engineer reads
//! by shape. Five of the eight complaints can be decided here: the instance
//! being too small to judge, the peak being far above the current total, the
//! arena holding much more than it is storing, the clients' buffers being large
//! on average, and there being more scripts cached than anybody meant to cache.
//!
//! The other three cannot. Two of them compare the process's resident set with
//! what the allocator holds, and this server does not read its own resident set;
//! the third is about replica output buffers and there are no replicas yet. A
//! complaint that can never fire is worse than one that is missing, so those
//! three are not written down here at all.

use core::fmt::Write as _;

use yo_common::Result;

use super::args::{self, Args, is};
use super::{Server, Session};
use crate::reply::Out;

/// The text `MEMORY HELP` prints, one line an entry.
const HELP: &[&str] = &[
    "MEMORY <subcommand> [<arg> [value] [opt] ...]. Subcommands are:",
    "DOCTOR",
    "    Return memory problems reports.",
    "MALLOC-STATS",
    "    Return internal statistics report from the memory allocator.",
    "PURGE",
    "    Attempt to purge dirty pages for reclamation by the allocator.",
    "STATS",
    "    Return information about the memory usage of the server.",
    "USAGE <key> [SAMPLES <count>]",
    "    Return memory in bytes used by <key> and its value. Nested values are",
    "    sampled up to <count> times (default: 5, 0 means sample all).",
    "HELP",
    "    Print this help.",
];

/// What `MEMORY MALLOC-STATS` says.
///
/// The sentence a real server gives when it was not built against jemalloc,
/// which is exactly the position this one is in and will stay in: the store
/// keeps its own accounting and there is no allocator underneath it with a
/// report to hand over. Saying it in the reference's words rather than in ours
/// means a client that recognises the answer still recognises it.
const NO_MALLOC_STATS: &str = "Stats not supported for the current allocator";

/// What `MEMORY DOCTOR` says about a server too small to have a problem.
const TOO_SMALL: &str = "Hi Sam, this instance is empty or is using very little memory, my issues detector can't be used in these conditions. Please, leave for your mission on Earth and fill it with some data. The new Sam and I will be back to our programming as soon as I finished rebooting.\n";

/// What it says about a server it has looked at and has nothing to say about.
const NOTHING_WRONG: &str = "Hi Sam, I can't find any memory issue in your instance. I can only account for what occurs on this base.\n";

/// Below this the doctor will not form an opinion, which is five megabytes.
const TOO_SMALL_BYTES: usize = 5 * 1024 * 1024;

/// Above this ratio of peak to total the doctor mentions the peak.
const BIG_PEAK: f64 = 1.5;

/// Above this ratio of held to used the doctor mentions fragmentation, and only
/// then if there are also more than ten megabytes of it.
const BIG_FRAG: f64 = 1.1;

/// The ten megabytes that goes with [`BIG_FRAG`].
const BIG_FRAG_BYTES: usize = 10 << 20;

/// Above this many bytes a connection on average the doctor mentions buffers.
const BIG_CLIENT_BYTES: usize = 200 * 1024;

/// Above this many cached scripts the doctor mentions the script cache.
const MANY_SCRIPTS: usize = 1000;

/// How many samples `MEMORY USAGE` takes when it is not told, which is a number
/// this server does not use and does report.
const DEFAULT_SAMPLES: i64 = 5;

/// Run one `MEMORY` subcommand.
pub(super) fn execute(
    server: &Server,
    session: &Session,
    args: Args<'_>,
    out: &mut Out,
) -> Result<()> {
    let sub = args.get(1);
    if is(sub, b"usage") {
        return usage(server, session, args, out);
    }
    // The other five take the subcommand and nothing after it, and a real server
    // refuses a sixth argument by naming the subcommand rather than the
    // container, which is why the count is checked here once for all of them.
    let named = ["doctor", "malloc-stats", "purge", "stats", "help"]
        .into_iter()
        .find(|n| is(sub, n.as_bytes()));
    let Some(named) = named else {
        return Err(args::unknown_subcommand(sub, "MEMORY"));
    };
    if args.len() != 2 {
        return Err(args::wrong_arity_sub("memory", named));
    }
    match named {
        "help" => super::server::help(out, HELP),
        "malloc-stats" => out.bulk(NO_MALLOC_STATS.as_bytes()),
        // Nothing to give back. On a real server this hands dirty pages to the
        // allocator, and here the equivalent is arena compaction, which the
        // maintenance slice is already doing between batches and which a client
        // asking about memory should not be able to make happen at a moment of
        // its choosing. So the reply is honest about having succeeded at doing
        // nothing, which is also what a real server without jemalloc answers.
        "purge" => out.ok(),
        "doctor" => doctor(server, out),
        "stats" => stats(server, out),
        other => unreachable!("no body for memory {other}"),
    }
    Ok(())
}

/// `MEMORY USAGE key [SAMPLES count]`.
///
/// The count is read and thrown away, for the reason the module header gives.
/// It is still read, so `SAMPLES` with nothing after it is a syntax error and
/// `SAMPLES nine` is not an integer, both in the reference's words.
fn usage(server: &Server, session: &Session, args: Args<'_>, out: &mut Out) -> Result<()> {
    if args.len() < 3 {
        return Err(args::wrong_arity_sub("memory", "usage"));
    }
    let mut samples = DEFAULT_SAMPLES;
    let mut at = 3;
    while at < args.len() {
        if !is(args.get(at), b"samples") || at + 1 >= args.len() {
            return Err(args::syntax());
        }
        // A repeated `SAMPLES` is allowed and the last one wins, which is what
        // the reference's loop does rather than something it decided to do.
        samples = args.int(at + 1)?;
        at += 2;
    }
    if samples < 0 {
        return Err(args::syntax());
    }
    let key = args.get(2);
    let mut db = server.dbs[session.db].hold(key);
    match db.key_bytes(key) {
        Some(bytes) => out.int(i64::try_from(bytes).unwrap_or(i64::MAX)),
        // A missing key is a null and not an error, the same as `OBJECT`.
        None => out.nil(),
    }
    Ok(())
}

/// Every number `MEMORY STATS` reports, worked out once.
///
/// One struct rather than thirty six calls, because half of them are differences
/// between the other half and computing a total twice is how two fields in one
/// reply end up disagreeing about the same server.
struct Overhead {
    /// The highest reading anything has taken here.
    peak: usize,
    /// What the server is holding now.
    total: usize,
    /// What it was holding before a client wrote anything.
    startup: usize,
    /// The connections' read and reply buffers.
    clients: usize,
    /// What the script cache costs.
    lua: usize,
    /// What the loaded libraries cost.
    functions: usize,
    /// What the indexes cost, across every database.
    lut: usize,
    /// Everything that is not the data.
    overhead: usize,
    /// The data, which is the total minus that.
    dataset: usize,
    /// Keys in every database.
    keys: usize,
    /// Arena space held for records that are dead and not yet collected.
    ///
    /// The nearest thing here to an allocator's external fragmentation, and the
    /// honest one: it is space the process is holding and is not storing
    /// anything in, and compaction is what gives it back.
    dead: usize,
}

impl Overhead {
    /// Take every reading, in one pass over the databases each.
    fn read(server: &Server) -> Overhead {
        // The peak first, because taking it is what moves it, and the total
        // after, so the two are never a reading apart in the direction that
        // would put the total above the peak in the same reply.
        let peak = server.peak_bytes();
        let total = server.memory_bytes();
        let startup = server.startup_bytes();
        let clients = server.conn_bytes();
        let lua = yo_alloc::allow(|| server.scripts.lock().memory_bytes());
        let functions = yo_alloc::allow(|| server.libraries.lock().memory_bytes());
        let lut = server.index_bytes();
        let dead = server.arena_bytes().saturating_sub(server.dataset_bytes());
        let keys = server.dbs.iter().map(|db| db.len()).sum();
        // What the client's data is not: the index that finds it, the arena
        // space nothing is stored in yet, the buffers the connections are
        // holding and the two script caches. Everything else in the total is a
        // record, a body or the slot table that reaches one, and all three of
        // those are the data.
        let overhead = lut + dead + clients + lua + functions;
        Overhead {
            peak,
            total,
            startup,
            clients,
            lua,
            functions,
            lut,
            overhead,
            dataset: total.saturating_sub(overhead),
            keys,
            dead,
        }
    }

    /// The total with the baseline taken off, which is what the two percentages
    /// and the bytes a key are worked out against.
    ///
    /// Never zero, because all three of those divide by it and a server that has
    /// somehow read a total below its own baseline should answer a silly number
    /// rather than a not-a-number.
    fn grown(&self) -> f64 {
        (self.total.saturating_sub(self.startup)).max(1) as f64
    }
}

/// `MEMORY STATS`.
///
/// Thirty six fields and one more for every database that has a key in it,
/// which is the reference's shape exactly: a dashboard reads this by name and a
/// field that is missing is worse for it than a field that is honestly zero.
fn stats(server: &Server, out: &mut Out) {
    let mh = Overhead::read(server);
    // Which databases get a row of their own. A real server skips the empty
    // ones, so a fresh server answers thirty six fields and one that has been
    // written to in two databases answers thirty eight.
    let filled: Vec<usize> = (0..super::DATABASES)
        .filter(|&i| !server.dbs[i].is_empty())
        .collect();
    out.map(36 + filled.len());

    let pair = |out: &mut Out, name: &str, n: usize| {
        out.bulk(name.as_bytes());
        out.int(i64::try_from(n).unwrap_or(i64::MAX));
    };
    let ratio = |out: &mut Out, name: &str, d: f64| {
        out.bulk(name.as_bytes());
        out.double(d);
    };

    pair(out, "peak.allocated", mh.peak);
    pair(out, "total.allocated", mh.total);
    pair(out, "startup.allocated", mh.startup);
    // Nothing behind these five yet. Replication and cluster are M8's remaining
    // work and there is no append only file, so they are zero because there is
    // nothing there rather than because nobody counted.
    pair(out, "replication.backlog", 0);
    pair(out, "replica.fullsync.buffer", 0);
    pair(out, "clients.slaves", 0);
    pair(out, "clients.normal", mh.clients);
    // A real server splits the clients' memory into the reply buffer every
    // client shares and the part each one holds alone. There is one buffer a
    // connection here and nothing shared, so the split is the whole of it on the
    // unshared side, which is true rather than convenient.
    pair(out, "clients.normal.shared", 0);
    pair(out, "clients.normal.unshared", mh.clients);
    pair(out, "cluster.links", 0);
    pair(out, "aof.buffer", 0);
    pair(out, "lua.caches", mh.lua);
    pair(out, "functions.caches", mh.functions);
    // Redis counts one Lua interpreter here and this server has one per thread,
    // compiled on first use, which belongs to the thread and not to the server.
    // There is nothing here that can say how many of them exist.
    pair(out, "script.VMs", 0);
    pair(out, "hash.templates", 0);
    let mut name = String::with_capacity(8);
    for &i in &filled {
        yo_alloc::allow(|| {
            name.clear();
            let _ = write!(name, "db.{i}");
        });
        out.bulk(name.as_bytes());
        out.map(2);
        // The index that finds the keys of this database, summed over its
        // stripes. A striped database is several tables and this is what all of
        // them cost, which is the number a real server's one table reports.
        let db = &server.dbs[i];
        let main: usize = (0..db.width())
            .map(|s| db.hold_stripe(s).map().index().memory_bytes())
            .sum();
        pair(out, "overhead.hashtable.main", main);
        // Nought, and it will stay nought. A deadline lives in the record with
        // the key it belongs to rather than in a second table beside it, so
        // there is no expires table to report the size of: the cost of a key
        // having a deadline is eight bytes inside its own record, which is
        // already counted as part of the key.
        pair(out, "overhead.hashtable.expires", 0);
    }
    pair(out, "overhead.db.hashtable.lut", mh.lut);
    // An extendible hash index splits one segment at a time and finishes before
    // the write that caused it returns, so there is never a table half moved
    // into another one. That is the whole of why this and the rehashing count
    // below are always nought, and it is a property worth having rather than a
    // field going unfilled.
    pair(out, "overhead.db.hashtable.rehashing", 0);
    pair(out, "overhead.total", mh.overhead);
    pair(out, "db.dict.rehashing.count", 0);
    pair(out, "keys.count", mh.keys);
    pair(
        out,
        "keys.bytes-per-key",
        mh.total
            .saturating_sub(mh.startup)
            .checked_div(mh.keys)
            .unwrap_or(0),
    );
    pair(out, "dataset.bytes", mh.dataset);
    ratio(
        out,
        "dataset.percentage",
        mh.dataset as f64 * 100.0 / mh.grown(),
    );
    ratio(
        out,
        "peak.percentage",
        mh.total as f64 * 100.0 / mh.peak.max(1) as f64,
    );
    // The four allocator fields, read off the arena, which is the thing here
    // that plays the allocator's part. What it is storing is `allocated`, what
    // it is holding is `active`, and the pages are all real once they have been
    // written to, so `resident` is the same as `active` and there is no muzzy
    // state between the two.
    let used = mh.total.saturating_sub(mh.dead);
    pair(out, "allocator.allocated", used);
    pair(out, "allocator.active", mh.total);
    pair(out, "allocator.resident", mh.total);
    pair(out, "allocator.muzzy", 0);
    ratio(
        out,
        "allocator-fragmentation.ratio",
        mh.total as f64 / used.max(1) as f64,
    );
    pair(out, "allocator-fragmentation.bytes", mh.dead);
    // Resident and active are the same number above, so this pair is the ratio
    // of a thing to itself and the bytes between them are none.
    ratio(out, "allocator-rss.ratio", 1.0);
    pair(out, "allocator-rss.bytes", 0);
    // And this pair would need the process's own resident set, which this server
    // does not read. One is the honest answer for a ratio nobody measured, and
    // it is also what a real server answers when the two happen to agree.
    ratio(out, "rss-overhead.ratio", 1.0);
    pair(out, "rss-overhead.bytes", 0);
    ratio(out, "fragmentation", mh.total as f64 / used.max(1) as f64);
    pair(out, "fragmentation.bytes", mh.dead);
}

/// `MEMORY DOCTOR`.
///
/// A verbatim reply on RESP3 and a bulk string on RESP2, which is what the
/// reference sends and is why this goes out through `verbatim` rather than
/// `bulk`: the text has newlines in it and a client that knows to render it as
/// text should be told it is text.
fn doctor(server: &Server, out: &mut Out) {
    let mh = Overhead::read(server);
    let report = yo_alloc::allow(|| {
        if mh.total < TOO_SMALL_BYTES {
            return TOO_SMALL.to_string();
        }
        let mut found: Vec<&str> = Vec::new();
        if mh.peak as f64 / mh.total.max(1) as f64 > BIG_PEAK {
            found.push(PEAK_REPORT);
        }
        let used = mh.total.saturating_sub(mh.dead).max(1);
        if mh.total as f64 / used as f64 > BIG_FRAG && mh.dead > BIG_FRAG_BYTES {
            found.push(FRAG_REPORT);
        }
        let clients = usize::try_from(server.totals().clients)
            .unwrap_or(usize::MAX)
            .max(1);
        if mh.clients / clients > BIG_CLIENT_BYTES {
            found.push(CLIENT_REPORT);
        }
        if server.scripts.lock().count() > MANY_SCRIPTS {
            found.push(SCRIPT_REPORT);
        }
        if found.is_empty() {
            return NOTHING_WRONG.to_string();
        }
        let mut s = String::from(OPENING);
        for one in found {
            s.push_str(one);
        }
        s.push_str(CLOSING);
        s
    });
    out.verbatim(b"txt", report.as_bytes());
}

/// The line the doctor opens a report with when it has found something.
const OPENING: &str = "Sam, I detected a few issues in this Redis instance memory implants:\n\n";

/// And the line it closes with.
const CLOSING: &str = "I'm here to keep you safe, Sam. I want to help you.\n";

/// The peak complaint, word for word as the reference writes it.
const PEAK_REPORT: &str = " * Peak memory: In the past this instance used more than 150% the memory that is currently using. The allocator is normally not able to release memory after a peak, so you can expect to see a big fragmentation ratio, however this is actually harmless and is only due to the memory peak, and if the Redis instance Resident Set Size (RSS) is currently bigger than expected, the memory will be used as soon as you fill the Redis instance with more data. If the memory peak was only occasional and you want to try to reclaim memory, please try the MEMORY PURGE command, otherwise the only other option is to shutdown and restart the instance.\n\n";

/// The fragmentation complaint, which here means the arena is holding a lot of
/// dead records that compaction has not reached.
const FRAG_REPORT: &str = " * High allocator fragmentation: This instance has an allocator external fragmentation greater than 1.1. This problem is usually due either to a large peak memory (check if there is a peak memory entry above in the report) or may result from a workload that causes the allocator to fragment memory a lot. You can try enabling 'activedefrag' config option.\n\n";

/// The client buffer complaint.
const CLIENT_REPORT: &str = " * Big client buffers: The clients output buffers in this instance are greater than 200K per client (on average). This may result from different causes, like Pub/Sub clients subscribed to channels bot not receiving data fast enough, so that data piles on the Redis instance output buffer, or clients sending commands with large replies or very large sequences of commands in the same pipeline. Please use the CLIENT LIST command in order to investigate the issue if it causes problems in your instance, or to understand better why certain clients are using a big amount of memory.\n\n";

/// The script cache complaint.
const SCRIPT_REPORT: &str = " * Many scripts: There seem to be many cached scripts in this instance (more than 1000). This may be because scripts are generated and `EVAL`ed, instead of being parameterized (with KEYS and ARGV), `SCRIPT LOAD`ed and `EVALSHA`ed. Unless `SCRIPT FLUSH` is called periodically, the scripts' caches may end up consuming most of your memory.\n\n";
