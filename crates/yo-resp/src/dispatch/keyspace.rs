//! The keyspace commands, from the wire.
//!
//! These are the ones that do not care what a value is. `DEL` deletes a string
//! today and will delete a list without a line changing here, because the
//! question it asks is about the key and not about what is under it.
//!
//! Four of them are here rather than in M3 with the rest of the group, and the
//! reason is that Redis's own test suite cannot run a single file without them.
//! In external mode the suite says `FLUSHALL` before every `start_server` block
//! and gives up on the whole file if that fails, and most test bodies then say
//! `DEL` to get to a known state. A compatibility suite that cannot start is
//! not a compatibility suite, so these landed early.
//!
//! `TYPE` reads the tag in the record's meta byte, so it answers whatever the
//! key actually holds and does not grow a case each time a type lands. Today
//! that is `string`, `set` or `none`, and the day the hash lands it is `hash`
//! too without a line here changing.

use super::args::{self, Args};
use super::table::Spec;
use crate::reply::Out;
use yo_common::{Code, Error, Result};
use yo_kv::Keyspace;

/// What `OBJECT FREQ` says on a server that is not counting accesses.
///
/// Which is every server here, because there is no eviction yet, so this is the
/// only thing it can say. Redis says it too whenever the policy is not an LFU
/// one, and the second half about switching at runtime is upstream's wording
/// and not ours.
const NOT_LFU: &str = "An LFU maxmemory policy is not selected, access frequency not tracked. Please note that when switching between policies at runtime LRU and LFU data will take some time to adjust.";

/// The text `OBJECT HELP` prints, one line an entry.
const OBJECT_HELP: &[&str] = &[
    "OBJECT <subcommand> [<arg> [value] [opt] ...]. Subcommands are:",
    "ENCODING <key>",
    "    Return the kind of internal representation used in order to store the value",
    "    associated with a <key>.",
    "FREQ <key>",
    "    Return the access frequency index of the <key>. The returned integer is",
    "    proportional to the logarithm of the real access frequency.",
    "IDLETIME <key>",
    "    Return the idle time of the <key>, that is the approximated number of",
    "    seconds elapsed since the last access to the value.",
    "REFCOUNT <key>",
    "    Return the number of references of the value associated with the key.",
    "HELP",
    "    Print this help.",
];

/// Run one keyspace command.
pub(super) fn execute(db: &mut Keyspace, spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<()> {
    match spec.name {
        // `UNLINK` is `DEL` with the freeing moved to a background thread on a
        // real server. Ours frees on the spot, which is what `UNLINK` promises
        // a client: the key is gone from the keyspace when the reply arrives.
        // The promise is about visibility and not about which thread did the
        // work, so this is the same body rather than a divergence.
        "del" | "unlink" => {
            let mut gone = 0i64;
            for i in 1..args.len() {
                if db.del(args.get(i)) {
                    gone += 1;
                }
            }
            out.int(gone);
        }
        // A key named twice counts twice, which looks like a bug and is what
        // Redis does. `EXISTS k k` on one key answers two.
        "exists" => {
            let mut found = 0i64;
            for i in 1..args.len() {
                if db.exists(args.get(i)) {
                    found += 1;
                }
            }
            out.int(found);
        }
        "type" => {
            let name = match db.kind_of(args.get(1)) {
                Some(k) => k.name().as_bytes(),
                None => &b"none"[..],
            };
            // A simple string on both protocols, which is unusual enough to be
            // worth saying out loud: most replies that carry a word are bulk
            // strings and this one is not.
            out.simple(name);
        }
        "object" => object(db, args, out)?,
        other => unreachable!("keyspace command with no body: {other}"),
    }
    Ok(())
}

/// `OBJECT ENCODING | REFCOUNT | IDLETIME | FREQ key`, and `OBJECT HELP`.
///
/// `ENCODING` is the one worth having and the other three are here because
/// clients call them without thinking. It is the only window onto the size
/// ladder from outside, so it is the command that says whether the ladder is
/// really Redis's ladder: a set of three numbers has to say `intset` and a hash
/// of two hundred fields has to say `listpack`, at the same counts a real server
/// says them at.
///
/// A missing key answers nil rather than an error, on all four. That reads like
/// a bug and it is what 8.10.1 does, checked rather than assumed, and it used to
/// be `no such key` years ago which is where the confusion comes from.
fn object(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let sub = args.get(1);
    if args::is(sub, b"help") {
        if args.len() != 2 {
            return Err(args::unknown_subcommand(sub, "OBJECT"));
        }
        out.array(OBJECT_HELP.len());
        for line in OBJECT_HELP {
            out.simple(line.as_bytes());
        }
        return Ok(());
    }

    let named = ["encoding", "refcount", "idletime", "freq"]
        .into_iter()
        .find(|n| args::is(sub, n.as_bytes()));
    let Some(named) = named else {
        return Err(args::unknown_subcommand(sub, "OBJECT"));
    };
    // The arity in the table is a minimum, so the count of a subcommand that
    // takes exactly one key is checked here, and the name of the command it
    // complains about is the container and the subcommand joined by a pipe.
    if args.len() != 3 {
        return Err(args::wrong_arity_sub("object", named));
    }
    // The lookup happens before FREQ refuses, so `OBJECT FREQ missing` is a nil
    // and not the policy complaint. Same order as Redis, which reaches for the
    // key first and asks about the policy after.
    let key = args.get(2);
    if !db.exists(key) {
        out.nil();
        return Ok(());
    }
    match named {
        "encoding" => {
            let name = db
                .encoding_name(key)
                .expect("the key is there, so it has an encoding");
            out.bulk(name.as_bytes());
        }
        // One reference, always. Redis shares the small integers and answers a
        // huge number for those, or it did: 8.10.1 answers 1 for `SET k 123`
        // like it does for everything else, so 1 is the whole answer here.
        "refcount" => out.int(1),
        // Nothing is tracking access time, and zero is what a key just touched
        // would say anyway, which is every key by the time this reaches it.
        "idletime" => out.int(0),
        "freq" => return Err(Error::new(Code::Unsupported, NOT_LFU)),
        other => unreachable!("no body for object {other}"),
    }
    Ok(())
}
