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
//! `TYPE` answers `string` or `none` and nothing else yet, which is true rather
//! than incomplete: there is nothing else in the store to answer with. It grows
//! a case each time a type lands.

use super::args::Args;
use super::table::Spec;
use crate::reply::Out;
use yo_common::Result;
use yo_kv::Strings;

/// Run one keyspace command.
pub(super) fn execute(db: &mut Strings, spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<()> {
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
            let name: &[u8] = if db.exists(args.get(1)) {
                b"string"
            } else {
                b"none"
            };
            // A simple string on both protocols, which is unusual enough to be
            // worth saying out loud: most replies that carry a word are bulk
            // strings and this one is not.
            out.simple(name);
        }
        other => unreachable!("keyspace command with no body: {other}"),
    }
    Ok(())
}
