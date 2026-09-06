//! `EVAL` and its four spellings, `SCRIPT`, and as much of `FUNCTION` as a
//! server with no libraries can answer.
//!
//! The interpreter is next door in [`lua`](super::lua) and this is the command
//! surface in front of it: work out which body to run, cut the arguments into
//! `KEYS` and `ARGV`, and hand the three of them over. Everything that is about
//! Lua is over there and everything that is about a command is here, and the
//! line between them is that nothing in this file mentions a Lua value.
//!
//! # The cache
//!
//! `EVAL` and `EVALSHA` are the same command told the body two different ways.
//! A client sends the body once, gets its digest back, and sends the digest
//! forever after, which is worth doing when the body is two kilobytes and the
//! command is run a hundred thousand times a second. `EVAL` puts the body in
//! the cache on the way past, so a client that sent the body can send the
//! digest next time without a `SCRIPT LOAD` in between, and that is a real
//! server's behaviour and the reason `SCRIPT LOAD` is a convenience rather than
//! a requirement.
//!
//! Nothing ever leaves the cache except on `SCRIPT FLUSH`. That is what a real
//! server does and it is a real cost on a client that builds a script per
//! request, which is a thing clients do and should not.
//!
//! # `FUNCTION`
//!
//! Still no libraries, so `LIST` is empty and `DELETE` reports that the library
//! is not there, both of which are the true answers. `LOAD` and the rest are
//! the next piece of work and are not here, so a client that asks for one gets
//! `unknown subcommand` rather than an `OK` that did nothing. D-16 says which
//! are which.

use super::args::{self, Args, is};
use super::lua;
use super::table::Spec;
use super::{Server, Session};
use crate::reply::Out;
use yo_common::{Code, Error, Result};

/// Run one scripting command.
pub(super) fn execute(
    server: &Server,
    session: &mut Session,
    spec: &Spec,
    args: Args<'_>,
    out: &mut Out,
) -> Result<()> {
    match spec.name {
        "eval" | "eval_ro" => eval(server, session, spec.name.ends_with("_ro"), args, out),
        "evalsha" | "evalsha_ro" => evalsha(server, session, spec.name.ends_with("_ro"), args, out),
        "script" => script(server, args, out),
        "function" => function(args, out),
        other => unreachable!("scripting command with no body: {other}"),
    }
}

/// `EVAL script numkeys [key ...] [arg ...]`.
fn eval(
    server: &Server,
    session: &mut Session,
    ro: bool,
    args: Args<'_>,
    out: &mut Out,
) -> Result<()> {
    let body = args.get(1).to_vec();
    let split = numkeys(&args)?;
    // Remembered whether it runs or not, and remembered before it runs, which
    // is a real server's order: a script that fails halfway is still in the
    // cache and can still be reached by its digest.
    let sha = yo_alloc::allow(|| server.scripts.lock().add(&body));
    let found = Found {
        body: &body,
        sha: &sha,
        ro,
        split,
    };
    run(server, session, &found, args, out);
    Ok(())
}

/// `EVALSHA sha1 numkeys [key ...] [arg ...]`.
fn evalsha(
    server: &Server,
    session: &mut Session,
    ro: bool,
    args: Args<'_>,
    out: &mut Out,
) -> Result<()> {
    let split = numkeys(&args)?;
    let Some(sha) = digest(args.get(1)) else {
        no_script(out);
        return Ok(());
    };
    // The lock is let go before the script runs, because the script is going to
    // call commands and those take locks of their own.
    let Some(body) = yo_alloc::allow(|| server.scripts.lock().body(&sha)) else {
        no_script(out);
        return Ok(());
    };
    let found = Found {
        body: &body,
        sha: &sha,
        ro,
        split,
    };
    run(server, session, &found, args, out);
    Ok(())
}

/// Everything the two spellings worked out before they agreed on what to run.
struct Found<'a> {
    /// The script itself, whether the client sent it or the cache held it.
    body: &'a [u8],
    /// Its digest, which is the name the failure message ends with.
    sha: &'a [u8; 40],
    /// Whether this was one of the `_RO` spellings.
    ro: bool,
    /// How many of the arguments after the count are keys.
    split: usize,
}

/// The part both spellings share, once there is a body to run.
fn run(server: &Server, session: &mut Session, found: &Found<'_>, args: Args<'_>, out: &mut Out) {
    yo_alloc::allow(|| {
        // Collected rather than borrowed because `Args` hands out one argument
        // at a time and the interpreter wants them all at once. A script's
        // arguments are a handful of short strings, and this is the one command
        // group where a vector of slices is not worth avoiding.
        let words: Vec<&[u8]> = (3..args.len()).map(|i| args.get(i)).collect();
        let (keys, argv) = words.split_at(found.split);
        let ask = lua::Ask {
            keys,
            argv,
            sha: found.sha,
            ro: found.ro,
        };
        lua::run(server, session, found.body, &ask, out);
    });
}

/// How many of the arguments after the body are keys.
///
/// Strict about the digits, which is where most of the failures a client sees
/// here come from: `EVAL 'return 1' 1.0 k` is refused, and so are `01`, ` 1`
/// and `+1`. That is `string2ll` and not a choice made here, and a client that
/// formats the count itself rather than letting its library do it is the one
/// that trips over it.
fn numkeys(args: &Args<'_>) -> Result<usize> {
    let n = args.int(2)?;
    if n < 0 {
        return Err(Error::new(
            Code::Invalid,
            "Number of keys can't be negative",
        ));
    }
    // `args.len() - 3` and not `- 2`, because the count itself is one of them.
    let usable = i64::try_from(args.len() - 3).unwrap_or(i64::MAX);
    if n > usable {
        return Err(Error::new(
            Code::Invalid,
            "Number of keys can't be greater than number of args",
        ));
    }
    Ok(usize::try_from(n).unwrap_or(0))
}

/// A forty character hex digest as the bytes a client sent, folded to lower
/// case, or `None` if it is not one.
///
/// Folded because a client that stored the digest and upper cased it somewhere
/// along the way still means the same script, and a real server folds too.
fn digest(arg: &[u8]) -> Option<[u8; 40]> {
    let mut sha = [0u8; 40];
    if arg.len() != sha.len() {
        return None;
    }
    for (slot, &b) in sha.iter_mut().zip(arg) {
        if !b.is_ascii_hexdigit() {
            return None;
        }
        *slot = b.to_ascii_lowercase();
    }
    Some(sha)
}

/// What a digest nobody has loaded answers.
///
/// The code is the whole message here. Every client library branches on it and
/// sends the body instead, which is what makes `EVALSHA` safe to try first.
fn no_script(out: &mut Out) {
    out.error_line(b"NOSCRIPT ", b"No matching script. Please use EVAL.");
}

/// `SCRIPT LOAD|EXISTS|FLUSH|KILL|DEBUG|HELP`.
fn script(server: &Server, args: Args<'_>, out: &mut Out) -> Result<()> {
    let sub = args.get(1);
    if is(sub, b"FLUSH") {
        // The mode is checked and then ignored, because there is nothing to
        // free either way: the cache is a map and dropping it is dropping it.
        // Redis's message here is its own and reads like a typo, "only support"
        // rather than "only supports", and the `FUNCTION` one further down says
        // "supports". Both are copied as they are.
        if args.len() > 3 || (args.len() == 3 && !mode(args.get(2))) {
            return Err(Error::new(
                Code::Invalid,
                "SCRIPT FLUSH only support SYNC|ASYNC option",
            ));
        }
        yo_alloc::allow(|| server.scripts.lock().wipe());
        out.ok();
    } else if is(sub, b"LOAD") {
        if args.len() != 3 {
            return Err(args::wrong_arity_sub("script", "load"));
        }
        // Compiled before it is kept, so that a client finds out its script does
        // not parse when it loads it rather than the first time it runs it. The
        // digest is of the body as it was sent, whitespace and all, so two
        // scripts that differ by a space are two scripts.
        let body = args.get(2).to_vec();
        if let Err(why) = lua::compiles(&body) {
            return Err(yo_alloc::allow(|| {
                Error::fmt(
                    Code::Invalid,
                    format_args!("Error compiling script (new function): {why}"),
                )
            }));
        }
        let sha = yo_alloc::allow(|| server.scripts.lock().add(&body));
        out.bulk(&sha);
    } else if is(sub, b"EXISTS") {
        if args.len() < 3 {
            return Err(args::wrong_arity_sub("script", "exists"));
        }
        // One answer per digest and in the order they were asked about, which
        // is how a client works out which of its scripts it still has to send.
        let held = server.scripts.lock();
        out.array(args.len() - 2);
        for i in 2..args.len() {
            let there = digest(args.get(i)).is_some_and(|sha| held.has(&sha));
            out.int(i64::from(there));
        }
    } else if is(sub, b"KILL") {
        if args.len() != 2 {
            return Err(args::wrong_arity_sub("script", "kill"));
        }
        // Always, because a script that could be killed would have to be running
        // on another thread while this command ran on this one, and a script
        // here runs to the end of the command that started it. There is no
        // timeout either, which is the same fact from the other side and is
        // registered as D-101: a script with a loop in it holds its own thread
        // and nothing else.
        out.error_line(b"NOTBUSY ", b"No scripts in execution right now.");
    } else if is(sub, b"DEBUG") {
        if args.len() != 3 {
            return Err(args::wrong_arity_sub("script", "debug"));
        }
        // There is no debugger, so the setting is checked and dropped, which is
        // D-104. It is checked because a client that asks for a mode that does
        // not exist has made a mistake whether or not anything would have used
        // the answer, and `YES` and `SYNC` are accepted because refusing them
        // would stop a client that only ever turns the thing off again.
        if !is(args.get(2), b"YES") && !is(args.get(2), b"SYNC") && !is(args.get(2), b"NO") {
            return Err(Error::new(Code::Invalid, "Use SCRIPT DEBUG YES/SYNC/NO"));
        }
        out.ok();
    } else if is(sub, b"HELP") {
        super::server::help(out, SCRIPT_HELP);
    } else {
        return Err(args::unknown_subcommand(sub, "SCRIPT"));
    }
    Ok(())
}

/// `FUNCTION FLUSH|LIST|DELETE|HELP`.
fn function(args: Args<'_>, out: &mut Out) -> Result<()> {
    let sub = args.get(1);
    if is(sub, b"FLUSH") {
        // Redis splits these two: a bad mode is a sentence about the mode, and
        // a second one after it is the generic subcommand error, because the
        // arity is checked before the argument is looked at.
        if args.len() > 3 {
            return Err(unknown_or_arity("flush"));
        }
        if args.len() == 3 && !mode(args.get(2)) {
            return Err(Error::new(
                Code::Invalid,
                "FUNCTION FLUSH only supports SYNC|ASYNC option",
            ));
        }
        out.ok();
    } else if is(sub, b"LIST") {
        // `LIBRARYNAME <pattern>` and `WITHCODE`, in any order and any number
        // of times, which is how Redis parses it. Every one of them narrows an
        // empty list to an empty list, so the parse exists to reject a word
        // that is not one of them.
        let mut i = 2;
        while i < args.len() {
            let a = args.get(i);
            if is(a, b"WITHCODE") {
                i += 1;
            } else if is(a, b"LIBRARYNAME") && i + 1 < args.len() {
                i += 2;
            } else {
                return Err(yo_alloc::allow(|| {
                    Error::fmt(
                        Code::Invalid,
                        format_args!("Unknown argument {}", String::from_utf8_lossy(a)),
                    )
                }));
            }
        }
        out.array(0);
    } else if is(sub, b"DELETE") {
        if args.len() != 3 {
            return Err(unknown_or_arity("delete"));
        }
        // Always, and truthfully. There are no libraries to find.
        return Err(Error::new(Code::Unsupported, "Library not found"));
    } else if is(sub, b"HELP") {
        super::server::help(out, FUNCTION_HELP);
    } else {
        // LOAD, RESTORE, KILL, DUMP and STATS are not here. STATS is the one
        // that looks answerable and is not: a real server lists LUA in its
        // engines map, and there is no engine here to list, so an empty map
        // would be a different answer rather than the same one.
        return Err(args::unknown_subcommand(sub, "FUNCTION"));
    }
    Ok(())
}

/// Whether an argument is `SYNC` or `ASYNC`.
fn mode(arg: &[u8]) -> bool {
    is(arg, b"SYNC") || is(arg, b"ASYNC")
}

/// `ERR unknown subcommand or wrong number of arguments for 'x'. Try FUNCTION
/// HELP.`
///
/// One sentence for two different mistakes, which is Redis's shape here and not
/// ours: `FUNCTION` reports a subcommand it does not know and a subcommand with
/// the wrong number of arguments the same way.
fn unknown_or_arity(sub: &str) -> Error {
    Error::fmt(
        Code::Unsupported,
        format_args!(
            "unknown subcommand or wrong number of arguments for '{sub}'. Try FUNCTION HELP."
        ),
    )
}

/// What `SCRIPT HELP` says, which is Redis's text and not ours.
const SCRIPT_HELP: &[&str] = &[
    "SCRIPT <subcommand> [<arg> [value] [opt] ...]. Subcommands are:",
    "DEBUG (YES|SYNC|NO)",
    "    Set the debug mode for subsequent scripts executed.",
    "EXISTS <sha1> [<sha1> ...]",
    "    Return information about the existence of the scripts in the script cache.",
    "FLUSH [ASYNC|SYNC]",
    "    Flush the Lua scripts cache. Very dangerous on replicas.",
    "    When called without the optional mode argument, the behavior is determined by the",
    "    lazyfree-lazy-user-flush configuration directive. Valid modes are:",
    "    * ASYNC: Asynchronously flush the scripts cache.",
    "    * SYNC: Synchronously flush the scripts cache.",
    "KILL",
    "    Kill the currently executing Lua script.",
    "LOAD <script>",
    "    Load a script into the scripts cache without executing it.",
    "HELP",
    "    Print this help.",
];

/// What `FUNCTION HELP` says, which is Redis's text and not ours.
const FUNCTION_HELP: &[&str] = &[
    "FUNCTION <subcommand> [<arg> [value] [opt] ...]. Subcommands are:",
    "LOAD [REPLACE] <FUNCTION CODE>",
    "    Create a new library with the given library name and code.",
    "DELETE <LIBRARY NAME>",
    "    Delete the given library.",
    "LIST [LIBRARYNAME PATTERN] [WITHCODE]",
    "    Return general information on all the libraries:",
    "    * Library name",
    "    * The engine used to run the Library",
    "    * Functions list",
    "    * Library code (if WITHCODE is given)",
    "    It also possible to get only function that matches a pattern using LIBRARYNAME argument.",
    "STATS",
    "    Return information about the current function running:",
    "    * Function name",
    "    * Command used to run the function",
    "    * Duration in MS that the function is running",
    "    If no function is running, return nil",
    "    In addition, returns a list of available engines.",
    "KILL",
    "    Kill the current running function.",
    "FLUSH [ASYNC|SYNC]",
    "    Delete all the libraries.",
    "    When called without the optional mode argument, the behavior is determined by the",
    "    lazyfree-lazy-user-flush configuration directive. Valid modes are:",
    "    * ASYNC: Asynchronously flush the libraries.",
    "    * SYNC: Synchronously flush the libraries.",
    "DUMP",
    "    Return a serialized payload representing the current libraries, can be restored using FUNCTION RESTORE command",
    "RESTORE <PAYLOAD> [FLUSH|APPEND|REPLACE]",
    "    Restore the libraries represented by the given payload, it is possible to give a restore policy to",
    "    control how to handle existing libraries (default APPEND):",
    "    * FLUSH: delete all existing libraries.",
    "    * APPEND: appends the restored libraries to the existing libraries. On collision, abort.",
    "    * REPLACE: appends the restored libraries to the existing libraries, On collision, replace the old",
    "      libraries with the new libraries (notice that even on this option there is a chance of failure",
    "      in case of functions name collision with another library).",
    "HELP",
    "    Print this help.",
];
