//! `FT.PROFILE`, which is a search or an aggregation with the working shown.
//!
//! The reply is the ordinary one with a second half bolted on, and the second
//! half is two things: the tree the query turned into, with a count on every
//! branch of it, and the list of steps the rows that answered went through.
//!
//! # The envelope
//!
//! RESP2 gets a two element array of the reply and the profile, where the
//! profile is a flat four element array of `Shards`, a list of one, then
//! `Coordinator` and an empty list. RESP3 gets a map of `Results` and `Profile`
//! and the profile is a map of the same two keys. There is one shard because
//! there is one server, and the coordinator is empty for the same reason.
//!
//! # What is measured and what is not
//!
//! The three times at the front of a shard are real: how long the words took to
//! read, how long the query took to turn into something walkable, and how long
//! the whole command took. The time on an iterator is real for the root of the
//! tree, which is the whole walk, and zero for everything under it, because
//! timing every step of a walk costs more than the walk does. The same goes for
//! the steps a row went through, where the counts are exact and the times are
//! not taken. D-90 has the whole of it.
//!
//! # `LIMITED`
//!
//! Folds the branches of a union an expansion made into a count of them, and
//! leaves a union a client wrote with a bar alone. Which is measured: `al*`
//! under `LIMITED` reports the number of terms it stood for and `alpha|beta`
//! under the same word reports both branches in full.

use std::time::{Duration, Instant};

use yo_common::Result;
use yo_search::walk::Ran;

use super::{Args, Watch, aggregated, searched};
use crate::dispatch::Server;
use crate::dispatch::args;
use crate::reply::Out;

/// Neither of the words that says what to profile was where one belongs.
const NO_MODE: &[u8] = b"No `SEARCH`, `AGGREGATE`, or `HYBRID` provided";

/// The word in front of the query was not `QUERY`.
const NO_QUERY: &[u8] = b"The QUERY keyword is expected";

/// A profile runs the whole query at once, so there is nothing to hold a cursor
/// over.
const NO_CURSOR: &[u8] = b"FT.PROFILE does not support cursor";

/// The name the arity line reports under.
const NAME: &str = "FT.PROFILE";

/// `FT.PROFILE index SEARCH|AGGREGATE [LIMITED] QUERY <query> [options]`
///
/// # Errors
///
/// The arity line, when the words run out before the query does. Everything
/// else is written here and answered as done, including whatever the search
/// underneath complained about.
pub(super) fn run(server: &Server, db: usize, args: Args<'_>, out: &mut Out) -> Result<()> {
    let mut at = 2;
    let searching = if args::is(args.get(at), b"SEARCH") {
        true
    } else if args::is(args.get(at), b"AGGREGATE") {
        false
    } else {
        // Named without a code in front of it, the way a real server names it.
        out.error(NO_MODE);
        return Ok(());
    };
    at += 1;
    // Between the word that says what to profile and the word that says the
    // query is next, which is the only place it goes.
    let limited = args::is(args.get(at), b"LIMITED");
    if limited {
        at += 1;
    }
    if !args::is(args.get(at), b"QUERY") {
        out.error(NO_QUERY);
        return Ok(());
    }
    at += 1;
    if at >= args.len() {
        return Err(args::wrong_arity(NAME));
    }
    // Asked before anything runs, because the whole answer is built before the
    // first byte of it goes out and a cursor is a promise to send the rest
    // later.
    if (at + 1..args.len()).any(|spot| args::is(args.get(spot), b"WITHCURSOR")) {
        out.error(NO_CURSOR);
        return Ok(());
    }
    let mut watch = Watch::default();
    let clock = Instant::now();
    // Written to the side so that a search that answers an error answers it on
    // its own, without an envelope around it that says the command worked.
    let mut inner = Out::with_capacity(out.proto(), 256);
    match searching {
        true => searched(server, db, args, at, Some(&mut watch), &mut inner)?,
        false => aggregated(server, db, args, at, Some(&mut watch), &mut inner)?,
    }
    let whole = clock.elapsed();
    if inner.as_slice().first() == Some(&b'-') {
        out.raw(inner.as_slice());
        return Ok(());
    }
    let three = out.proto().is_resp3();
    if three {
        out.map(2);
        out.simple(b"Results");
    } else {
        out.array(2);
    }
    out.raw(inner.as_slice());
    if three {
        out.simple(b"Profile");
    }
    out.map(2);
    out.simple(b"Shards");
    out.array(1);
    shard(&watch, whole, limited, out);
    out.simple(b"Coordinator");
    out.map(0);
    Ok(())
}

/// One shard's worth of profile, which is the whole of it here.
fn shard(watch: &Watch, whole: Duration, limited: bool, out: &mut Out) {
    out.map(7);
    out.simple(b"Total profile time");
    out.double(millis(whole));
    out.simple(b"Parsing time");
    out.double(millis(watch.parsing));
    // Nothing queues, because nothing is handed to a pool of workers. A real
    // server with its workers turned off reports the same nothing.
    out.simple(b"Workers queue time");
    out.double(0.0);
    out.simple(b"Pipeline creation time");
    out.double(millis(watch.creating));
    out.simple(b"Warning");
    out.array(1);
    out.simple(b"None");
    out.simple(b"Iterators profile");
    match &watch.ran {
        Some(ran) => tree(ran, millis(watch.walking), limited, out),
        // Nothing to report, which is a query that never reached the walk.
        None => out.map(0),
    }
    out.simple(b"Result processors profile");
    out.array(watch.steps.len());
    for (name, rows) in &watch.steps {
        out.map(3);
        out.simple(b"Type");
        out.simple(name);
        out.simple(b"Time");
        out.double(0.0);
        out.simple(b"Results processed");
        out.uint(*rows as u64);
    }
}

/// One step of the walk and everything under it.
///
/// The keys and their order are measured. A leaf says what it answered off and
/// how many it thought there would be, a union says what made it, and a branch
/// says neither and lists what is under it instead. A negation and an optional
/// hold one thing and name it in the singular, which is a different key and not
/// a list of one.
fn tree(node: &Ran, spent: f64, limited: bool, out: &mut Out) {
    let named = node.term.is_some();
    let branch = !matches!(
        node.kind,
        "TEXT" | "TAG" | "NUMERIC" | "GEO" | "WILDCARD" | "EMPTY"
    );
    out.map(3 + usize::from(named) * 2 + usize::from(node.about.is_some()) + usize::from(branch));
    out.simple(b"Type");
    out.simple(node.kind.as_bytes());
    if let Some(term) = &node.term {
        out.simple(b"Term");
        out.bulk(term);
    }
    if let Some(about) = &node.about {
        out.simple(b"Query type");
        // The bare word goes out as a status and the name of an expansion as a
        // string, which is measured and is not a distinction anything else in
        // this reply makes. An expansion is the one that says what it stood
        // for, so it is the one with a dash in the middle of it.
        match about.windows(3).any(|three| three == b" - ") {
            true => out.bulk(about),
            false => out.simple(about),
        }
    }
    out.simple(b"Time");
    out.double(spent);
    out.simple(b"Number of reading operations");
    out.uint(node.reads);
    if let Some(size) = node.size {
        out.simple(b"Estimated number of matches");
        out.uint(u64::from(size));
    }
    if !branch {
        return;
    }
    if node.alone {
        out.simple(b"Child iterator");
        tree(&node.under[0], 0.0, limited, out);
        return;
    }
    out.simple(b"Child iterators");
    if limited && node.folds {
        let mut line = b"The number of iterators in the union is ".to_vec();
        line.extend_from_slice(node.under.len().to_string().as_bytes());
        out.simple(&line);
        return;
    }
    out.array(node.under.len());
    for child in &node.under {
        tree(child, 0.0, limited, out);
    }
}

/// A span in milliseconds, which is the unit every time in a profile is in.
fn millis(span: Duration) -> f64 {
    span.as_secs_f64() * 1000.0
}
