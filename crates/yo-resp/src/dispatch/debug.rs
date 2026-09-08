//! `DEBUG`, the container a test suite talks to rather than a client.
//!
//! # What it is for
//!
//! Every other command here exists so that somebody can store something and get
//! it back. This one exists so that somebody can make the server do a thing that
//! would otherwise be impossible to arrange from the outside: send a reply of a
//! type no ordinary command sends, stop sweeping expired keys, stop the clock
//! work, fill a database with a hundred thousand keys without a hundred thousand
//! round trips, or answer with an error whose text the caller chose.
//!
//! Redis's own test suite leans on it heavily, which is why it is here at all:
//! most of the suite's `assert_encoding` and expiry tests do not run at all
//! against a server that has no `DEBUG`.
//!
//! # Which subcommands are here
//!
//! A real server has around sixty and most of them are about parts that do not
//! exist here: the AOF, cluster links, atomic slot migration, forking, crashing
//! on purpose. What is here is the part that is about this server, and `DEBUG
//! HELP` lists exactly that rather than listing what Redis has, for the same
//! reason `CLIENT HELP` does: somebody reading it to find out what they can send
//! should not be told about a subcommand that would come back unknown.
//!
//! The four knobs are the interesting ones, because a knob that is remembered
//! and read by nothing is worse than no knob at all. Three of them really move
//! something: `SET-ACTIVE-EXPIRE` gates the sweep that reclaims keys nobody asks
//! for again, `PAUSE-CRON` gates the whole maintenance slice the shard loop runs
//! between batches, and `SET-SKIP-CHECKSUM-VALIDATION` is read by the code that
//! opens a `RESTORE` payload. `DICT-RESIZING` gates arena compaction, which is
//! the nearest thing here to the dictionary resize it turns off on a real
//! server: both are the background reclaim of room a table no longer needs. The
//! one that is remembered and does nothing is
//! `QUICKLIST-PACKED-THRESHOLD`, which is D-128.
//!
//! `RELOAD` is the odd one out, because it moves the whole dataset rather than a
//! knob. It is here for the same reason the knobs are: the suite calls it after
//! almost every case, and a case that passes on both sides of it has proved that
//! the writer and the reader of the file agree about the value it just made.
//!
//! # How the errors work
//!
//! Every complaint in this file is the same sentence, `unknown subcommand or
//! wrong number of arguments for '<what was sent>'. Try DEBUG HELP.`, and that
//! is not a shortcut. A real server's `DEBUG` is a chain of `strcasecmp` tests
//! each of which also checks `argc`, and anything that falls off the end of the
//! chain gets that one line, so a subcommand that does not exist and a
//! subcommand handed the wrong number of arguments are the same case. The name
//! is echoed in the case it was sent in.
//!
//! The two exceptions are the two subcommands that read their argument and can
//! fail on the value rather than on the count, which are
//! `QUICKLIST-PACKED-THRESHOLD` and `POPULATE`, and each has its own sentence.

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;

use yo_common::num::parse_i64;
use yo_common::{Code, Error, Result};
use yo_kv::SetOptions;

use super::args::{self, Args, is};
use super::{Server, Session, persist};
use crate::reply::Out;

/// The knobs `DEBUG` turns, all of them on a word each.
///
/// One word rather than a lock because the readers are the shard loop's
/// maintenance slice and the payload reader, which is to say the hottest places
/// that could possibly read a debugging flag, and the writer is a human at a
/// test suite. The three gates are stored as their `true` meaning, so a default
/// `Knobs` is a server with everything running.
#[derive(Debug)]
pub(crate) struct Knobs {
    /// Whether the expiry sweep runs, which `SET-ACTIVE-EXPIRE 0` turns off.
    expiring: AtomicU64,
    /// Whether the maintenance slice runs at all, which `PAUSE-CRON 1` stops.
    cron: AtomicU64,
    /// Whether arena compaction runs, which `DICT-RESIZING 0` stops.
    resizing: AtomicU64,
    /// The packed node threshold, which nothing here reads. See D-128.
    packed: AtomicU64,
}

impl Default for Knobs {
    fn default() -> Knobs {
        Knobs {
            expiring: AtomicU64::new(1),
            cron: AtomicU64::new(1),
            resizing: AtomicU64::new(1),
            packed: AtomicU64::new(DEFAULT_PACKED),
        }
    }
}

/// What the packed threshold goes back to when it is set to nought, which is a
/// gigabyte and is Redis's default.
const DEFAULT_PACKED: u64 = 1 << 30;

/// The largest packed threshold that is taken, which is four gigabytes less a
/// megabyte.
///
/// Redis's `quicklistSetPackedThreshold` refuses anything above this, with a
/// comment saying it will not allow the threshold even slightly below four
/// gigabytes. The error text says bigger than one and smaller than 4gb, and
/// neither half of that sentence is quite what the code checks, since one is
/// taken and `4294967295` is not.
const MAX_PACKED: u64 = (1 << 32) - (1 << 20);

impl Server {
    /// Whether the expiry sweep should run.
    #[must_use]
    pub(crate) fn expiring(&self) -> bool {
        self.debug.expiring.load(Relaxed) != 0
    }

    /// Whether the maintenance slice should run at all.
    #[must_use]
    pub fn cron_running(&self) -> bool {
        self.debug.cron.load(Relaxed) != 0
    }

    /// Whether arena compaction should run.
    #[must_use]
    pub(crate) fn resizing(&self) -> bool {
        self.debug.resizing.load(Relaxed) != 0
    }
}

/// `DEBUG <subcommand> [...]`.
pub(super) fn execute(
    server: &Server,
    session: &mut Session,
    args: Args<'_>,
    out: &mut Out,
) -> Result<()> {
    let sub = args.get(1);
    if is(sub, b"HELP") && args.len() == 2 {
        super::server::help(out, HELP);
    } else if is(sub, b"PROTOCOL") && args.len() == 3 {
        return protocol(args.get(2), out);
    } else if is(sub, b"ERROR") && args.len() == 3 {
        // Straight out, with no code in front of it and no checking of what is
        // in it beyond the newlines, because the whole point is to hand a client
        // library an error line it chose. The empty prefix is there because this
        // is the one error line the server did not write any of, and the newline
        // folding that comes with it is what a real server does too and is what
        // stops this from being a way to write two replies with one command.
        out.error_line(b"", args.get(2));
    } else if is(sub, b"LOG") && args.len() == 3 {
        // The server log is stderr here, which is what the service file or the
        // shell redirection points wherever the operator wants it.
        yo_alloc::allow(|| {
            eprintln!("yodb: DEBUG LOG: {}", String::from_utf8_lossy(args.get(2)));
        });
        out.ok();
    } else if is(sub, b"SLEEP") && args.len() == 3 {
        sleep(args.get(2));
        out.ok();
    } else if is(sub, b"POPULATE") && (3..=5).contains(&args.len()) {
        return populate(server, session, args, out);
    } else if is(sub, b"SET-ACTIVE-EXPIRE") && args.len() == 3 {
        server.debug.expiring.store(flag(args.get(2)), Relaxed);
        out.ok();
    } else if is(sub, b"PAUSE-CRON") && args.len() == 3 {
        // The one gate that is stored the other way up from how it is written,
        // because the subcommand names the stopping and the field names the
        // running.
        server.debug.cron.store(1 - flag(args.get(2)), Relaxed);
        out.ok();
    } else if is(sub, b"DICT-RESIZING") && args.len() == 3 {
        server.debug.resizing.store(flag(args.get(2)), Relaxed);
        out.ok();
    } else if is(sub, b"SET-SKIP-CHECKSUM-VALIDATION") && args.len() == 3 {
        yo_kv::rdb::skip_checksums(flag(args.get(2)) != 0);
        out.ok();
    } else if is(sub, b"QUICKLIST-PACKED-THRESHOLD") && args.len() == 3 {
        return packed(server, args.get(2), out);
    } else if is(sub, b"RELOAD") {
        return reload(server, args, out);
    } else {
        return Err(args::subcommand_syntax(sub, "DEBUG"));
    }
    Ok(())
}

/// A `0` or `1` argument, read the way C reads one.
///
/// Which is `atoi`, so anything that is not a number at all is nought and the
/// gate goes off. That is worth reproducing rather than tidying up, because a
/// test suite that sends `DEBUG SET-ACTIVE-EXPIRE no` gets a server with the
/// sweep turned off on a real server and would get one with it left on here if
/// this refused what it could not read.
fn flag(value: &[u8]) -> u64 {
    let value = value.strip_prefix(b"-").unwrap_or(value);
    let digits = value
        .iter()
        .take_while(|b| b.is_ascii_digit())
        .fold(0u64, |n, b| {
            n.saturating_mul(10).saturating_add(u64::from(b - b'0'))
        });
    u64::from(digits != 0)
}

/// `DEBUG SLEEP <seconds>`, which stops this thread where it stands.
///
/// Decimals allowed and read with C's `strtod`, so a word is nought seconds and
/// a negative number is nought seconds, and both answer `OK` at once. There is
/// no upper bound, which is the point: a suite that wants a server that does not
/// answer for ten seconds asks for ten seconds.
///
/// On a server with one shard thread, which is the default, this is the whole
/// server, which is what it is on Redis. Above one thread it is the thread this
/// connection landed on and the others keep answering, which is D-129.
fn sleep(value: &[u8]) {
    let text = core::str::from_utf8(value).unwrap_or("");
    let seconds = leading_double(text);
    if seconds > 0.0 {
        std::thread::sleep(std::time::Duration::from_secs_f64(seconds));
    }
}

/// As much of the front of `text` as reads as a double, or nought.
///
/// `strtod` takes the longest prefix that is a number and stops, so `1.5s` is a
/// second and a half and `abc` is nothing. Rust's parser wants the whole string,
/// so the prefix is found here.
fn leading_double(text: &str) -> f64 {
    let mut end = 0;
    for (at, _) in text.char_indices() {
        if text[..=at].parse::<f64>().is_ok() {
            end = at + 1;
        }
    }
    text[..end].parse().unwrap_or(0.0)
}

/// `DEBUG QUICKLIST-PACKED-THRESHOLD <size>`.
fn packed(server: &Server, value: &[u8], out: &mut Out) -> Result<()> {
    let size = super::server::parse_memory(value).filter(|&n| n <= MAX_PACKED);
    let Some(size) = size else {
        return Err(Error::new(
            Code::Invalid,
            "argument must be a memory value bigger than 1 and smaller than 4gb",
        ));
    };
    // Nought is not a threshold of nothing, it is the word for putting the
    // default back, which is the one part of this subcommand that is not
    // guessable from its name.
    let size = if size == 0 { DEFAULT_PACKED } else { size };
    server.debug.packed.store(size, Relaxed);
    out.ok();
    Ok(())
}

/// What a reload says when the file did not come back.
///
/// One sentence for every way it can go wrong, which is the reference's answer
/// too. A client can do nothing with the difference between a bad checksum and a
/// file that stops halfway, and whoever can is reading the log, so that is where
/// the reason goes.
const LOAD_FAILED: &str = "Error trying to load the RDB dump, check server logs.";

/// `DEBUG RELOAD [MERGE] [NOFLUSH] [NOSAVE]`, the round trip a suite leans on.
///
/// Write the whole dataset out as an RDB, throw away what is in memory and build
/// it again out of the file. It is here because Redis's own suite calls it after
/// almost every case: a value that comes back the same way it went in has proved
/// its writer and its reader agree, and a value that does not has found a bug in
/// one of them without anybody having to say which.
///
/// The three options are the reference's three. `NOSAVE` skips the write and
/// reads whatever file is already on disk, which is how a suite loads a file it
/// put there itself. `NOFLUSH` keeps what is in memory and lets the file land on
/// top of it. `MERGE` is read and changes nothing here, and that is D-131: on a
/// real server it is what makes a key that is in the file and in memory legal,
/// and without it the server takes itself down with `Duplicated key found in RDB
/// file`. A key arriving over one that is already there is an ordinary import
/// here, so there is nothing for the word to turn on.
///
/// The other difference from a real server is the window. Redis forks for the
/// save and has one thread for the load, so nothing can write in between. Here
/// the save walks one stripe at a time and another connection can write to a
/// stripe that has already been walked, which is D-132 and is the same window
/// [`super::persist::build`] already has for `SAVE`.
fn reload(server: &Server, args: Args<'_>, out: &mut Out) -> Result<()> {
    let (mut save, mut flush) = (true, true);
    for i in 2..args.len() {
        let word = args.get(i);
        if is(word, b"NOSAVE") {
            save = false;
        } else if is(word, b"NOFLUSH") {
            flush = false;
        } else if !is(word, b"MERGE") {
            return Err(Error::new(
                Code::Invalid,
                "DEBUG RELOAD only supports the MERGE, NOFLUSH and NOSAVE options.",
            ));
        }
    }
    if save {
        if !persist::write_file(server) {
            // The bare line `SAVE` answers, for the reason it gives.
            out.error(b"ERR");
            return Ok(());
        }
        // A key with no RDB shape is not in the file that was just written, so
        // flushing and reading it back would be a way of deleting it. Nothing on
        // a real server can be in this position, which is why the sentence is
        // ours: the reply says which way out there is rather than leaving the
        // caller to find out from a `DBSIZE` that came back short.
        let lost = persist::skipped(server);
        if flush && lost > 0 {
            return Err(if lost == 1 {
                Error::new(
                    Code::Invalid,
                    "DEBUG RELOAD would drop 1 key with no RDB form, use NOFLUSH to keep it",
                )
            } else {
                Error::fmt(
                    Code::Invalid,
                    format_args!(
                        "DEBUG RELOAD would drop {lost} keys with no RDB form, use NOFLUSH to keep them"
                    ),
                )
            });
        }
    }
    if yo_alloc::allow(|| load_file(server, flush)) {
        out.ok();
        Ok(())
    } else {
        Err(Error::new(Code::Invalid, LOAD_FAILED))
    }
}

/// Read `dump.rdb` back over the keyspace, and say whether all of it landed.
///
/// The walk itself is [`Server::load_image`], which is shared with the restore
/// the tool does at startup. What is here is the two things that are this
/// command's own: the file it reads is always the one `SAVE` writes, and a
/// reason it could not is a line in the log rather than a sentence to a client,
/// for the reason [`LOAD_FAILED`] gives.
///
/// The libraries the file carries are counted and dropped rather than loaded.
/// They are already here, because the flush above takes the databases and not
/// the function registry, and loading a library that is already registered is an
/// error rather than a no op.
fn load_file(server: &Server, flush: bool) -> bool {
    let path = server.dir().join(persist::FILE);
    let image = match std::fs::read(&path) {
        Ok(image) => image,
        Err(e) => {
            eprintln!("yodb: DEBUG RELOAD: {}: {e}", path.display());
            return false;
        }
    };
    match server.load_image(&image, flush) {
        Ok(_) => true,
        Err(refused) => {
            eprintln!("yodb: DEBUG RELOAD: {refused}");
            false
        }
    }
}

/// `DEBUG POPULATE <count> [<prefix> [<size>]]`.
///
/// Keys are `<prefix>:<n>` counting from nought, with `key` as the prefix if
/// none was given, and each value is `value:<n>`. A size pads that with zero
/// bytes to exactly that many, or cuts it short, and a size of nought means the
/// value is left as it is rather than made empty.
///
/// A key that is already there is left alone, value and deadline both, which is
/// the surprising half and is what makes this safe to run twice. A real server
/// checks the dictionary and skips, and it does that because the whole point of
/// the subcommand is filling a database quickly, and quickly means not paying
/// for a delete of something it is about to write over. Here that falls out of
/// asking for the write the way `SET key value NX` asks for it.
///
/// Nothing is told about the keys this writes: no keyspace notification, no
/// index update. The notifications are the reference's choice, since it adds the
/// keys to the dictionary directly and never goes near the event code. The
/// indexes are this build's, and they are safe to leave out rather than merely
/// cheap: an index follows hashes or JSON documents and every key here is a
/// string, and a key that was already a document is one of the keys this skips.
fn populate(server: &Server, session: &mut Session, args: Args<'_>, out: &mut Out) -> Result<()> {
    let count = positive(args.get(2))?;
    let prefix = if args.len() >= 4 { args.get(3) } else { b"key" };
    let size = if args.len() == 5 {
        positive(args.get(4))? as usize
    } else {
        0
    };
    // Two buffers reused across the whole run rather than a pair of allocations
    // per key, since the count a suite passes here is routinely a hundred
    // thousand and every one of those is the same two shapes with a different
    // number on the end.
    let mut key = Vec::with_capacity(prefix.len() + 24);
    let mut value = Vec::with_capacity(size.max(32));
    let db = &server.dbs[session.db];
    for n in 0..count {
        key.clear();
        key.extend_from_slice(prefix);
        key.push(b':');
        push_int(&mut key, n);
        value.clear();
        value.extend_from_slice(b"value:");
        push_int(&mut value, n);
        if size != 0 {
            // Shorter than the name is a cut and longer is zero bytes on the
            // end, which is what the reference's `sdsgrowzero` does and is why
            // a size of five gives `value` and not `value:0` cut to five.
            value.resize(size, 0);
        }
        // One stripe held per key rather than one for the run, because the keys
        // are spread across every stripe by design and holding them all would
        // be holding the whole database against every other thread for as long
        // as the fill takes.
        db.hold(&key)
            .set(&key, &value, SetOptions::PLAIN.if_missing())?;
    }
    out.ok();
    Ok(())
}

/// A count argument, which has to be a whole number that is not negative.
///
/// The reference reads both of `POPULATE`'s numbers with the same call and says
/// the same thing about both, so a size that is not a number complains about a
/// range rather than about not being a number.
fn positive(value: &[u8]) -> Result<i64> {
    parse_i64(value)
        .filter(|&n| n >= 0)
        .ok_or_else(|| Error::new(Code::Invalid, "value is out of range, must be positive"))
}

/// A whole number, appended.
fn push_int(out: &mut Vec<u8>, mut n: i64) {
    let start = out.len();
    if n == 0 {
        out.push(b'0');
        return;
    }
    while n > 0 {
        out.push(b'0' + (n % 10) as u8);
        n /= 10;
    }
    out[start..].reverse();
}

/// `DEBUG PROTOCOL <type>`, which is one reply of each type RESP3 has.
///
/// This is the command a client library's own test suite points at itself to
/// find out whether it decodes the protocol, so every one of these was read off
/// the wire of an 8.10.1 rather than off the documentation, on both protocols.
/// Two of them are worth spelling out.
///
/// `attrib` on RESP3 sends an attribute and then a real reply behind it, and on
/// RESP2 sends only the reply, because RESP2 has no way to carry the attribute
/// and dropping it is what the other side does. `push` is the other way round:
/// on RESP3 the real reply goes out first and the push follows it, and on RESP2
/// the whole subcommand is an error, because a push on RESP2 would be an
/// ordinary array and a client would read it as the reply.
// The double the reference sends is 3.141, which is close enough to pi for the
// lint to think somebody meant pi and typed it badly. Nobody did: it is a test
// value chosen to have three decimal places, and rounding it to the real
// constant would change the bytes on the wire, which are the whole point.
#[allow(clippy::approx_constant)]
fn protocol(kind: &[u8], out: &mut Out) -> Result<()> {
    if is(kind, b"string") {
        out.bulk(b"Hello World");
    } else if is(kind, b"integer") {
        out.int(12345);
    } else if is(kind, b"double") {
        out.double(3.141);
    } else if is(kind, b"bignum") {
        out.big_number(b"1234567999999999999999999999999999999");
    } else if is(kind, b"null") {
        out.nil();
    } else if is(kind, b"array") {
        out.array(3);
        for n in 0..3 {
            out.int(n);
        }
    } else if is(kind, b"set") {
        out.set(3);
        for n in 0..3 {
            out.int(n);
        }
    } else if is(kind, b"map") {
        // The keys are numbers and the values are booleans, so a RESP2 client
        // sees three pairs flattened with the booleans as `:0` and `:1`, which
        // is the shape a RESP2 client already gets from every map here.
        out.map(3);
        for n in 0..3 {
            out.int(n);
            out.bool(n == 1);
        }
    } else if is(kind, b"attrib") {
        if out.proto().is_resp3() {
            out.attribute(1);
            out.bulk(b"key-popularity");
            out.array(2);
            out.bulk(b"key:123");
            out.int(90);
        }
        out.bulk(b"Some real reply following the attribute");
    } else if is(kind, b"push") {
        if !out.proto().is_resp3() {
            return Err(Error::new(
                Code::Invalid,
                "RESP2 is not supported by this command",
            ));
        }
        out.bulk(b"Some real reply following the push reply");
        out.push(2);
        out.bulk(b"server-cpu-usage");
        out.int(42);
    } else if is(kind, b"verbatim") {
        out.verbatim(b"txt", b"This is a verbatim\nstring");
    } else if is(kind, b"true") {
        out.bool(true);
    } else if is(kind, b"false") {
        out.bool(false);
    } else {
        return Err(Error::new(
            Code::Invalid,
            "Wrong protocol type name. Please use one of the following: string|integer|double|bignum|null|array|set|map|attrib|push|verbatim|true|false",
        ));
    }
    Ok(())
}

/// What `DEBUG HELP` says, which is what is here and not what Redis has.
const HELP: &[&str] = &[
    "DEBUG <subcommand> [<arg> [value] [opt] ...]. Subcommands are:",
    "DICT-RESIZING <0|1>",
    "    Enable or disable the background reclaim of room the store no longer",
    "    needs.",
    "ERROR <string>",
    "    Return a Redis protocol error with <string> as message. Useful for",
    "    clients unit tests to simulate Redis errors.",
    "LOG <message>",
    "    Write <message> to the server log.",
    "PAUSE-CRON <0|1>",
    "    Stop periodic cron job processing.",
    "POPULATE <count> [<prefix>] [<size>]",
    "    Create <count> string keys named key:<num>. If <prefix> is specified",
    "    then it is used instead of the 'key' prefix. A key that already exists",
    "    is left alone.",
    "PROTOCOL <type>",
    "    Reply with a test value of the specified type. <type> can be: string,",
    "    integer, double, bignum, null, array, set, map, attrib, push, verbatim,",
    "    true, false.",
    "QUICKLIST-PACKED-THRESHOLD <size>",
    "    Sets the threshold for elements to be inserted as plain vs packed nodes",
    "    Default value is 1GB, allows values up to 4GB. Setting to 0 restores to default.",
    "RELOAD [MERGE] [NOFLUSH] [NOSAVE]",
    "    Save the dataset to the RDB file and load it back. NOSAVE reads the file",
    "    that is already there, NOFLUSH keeps what is in memory and lets the file",
    "    land on top of it, and MERGE is accepted and does nothing.",
    "SET-ACTIVE-EXPIRE <0|1>",
    "    Setting it to 0 disables expiring keys in background when they are not",
    "    accessed (otherwise the Redis behavior). Setting it to 1 reenables back",
    "    the default.",
    "SET-SKIP-CHECKSUM-VALIDATION <0|1>",
    "    Enables or disables checksum checks for RESTORE's payload.",
    "SLEEP <seconds>",
    "    Stop the server for <seconds>. Decimals allowed.",
    "HELP",
    "    Print this help.",
];

#[cfg(test)]
mod tests {
    use super::{flag, leading_double};

    /// The flag reads what C reads out of the same bytes.
    #[test]
    fn a_flag_is_atoi_and_anything_unreadable_is_off() {
        for (text, want) in [
            (&b"0"[..], 0),
            (b"1", 1),
            (b"00", 0),
            (b"01", 1),
            (b"2", 1),
            (b"-1", 1),
            (b"-0", 0),
            (b"x", 0),
            (b"", 0),
            (b"1x", 1),
            (b"true", 0),
            (b"18446744073709551617", 1),
        ] {
            assert_eq!(flag(text), want, "{}", String::from_utf8_lossy(text));
        }
    }

    /// A sleep argument reads as much of itself as is a number.
    #[test]
    fn a_sleep_reads_the_longest_number_at_the_front() {
        for (text, want) in [
            ("0", 0.0),
            ("0.05", 0.05),
            ("-1", -1.0),
            ("abc", 0.0),
            ("", 0.0),
            ("1.5s", 1.5),
            ("2x3", 2.0),
        ] {
            assert!(
                (leading_double(text) - want).abs() < 1e-9,
                "{text} read as {}",
                leading_double(text)
            );
        }
    }
}
