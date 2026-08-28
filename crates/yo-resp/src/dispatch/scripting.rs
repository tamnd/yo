//! `SCRIPT` and `FUNCTION`, as far as a server with no interpreter can go.
//!
//! There is no Lua here and there will not be one until M6, so nothing can be
//! loaded and nothing can run. What is here is the part of both containers that
//! is exactly right on a server whose script cache and library set are empty
//! and always will be for now: flushing nothing, listing nothing, and reporting
//! that a script or a library is not there.
//!
//! That is not a shim. `SCRIPT EXISTS` answering zero is the true answer to the
//! question it was asked, and so is `FUNCTION LIST` answering an empty array,
//! and so is `FUNCTION DELETE` answering that the library is not found. The
//! subcommands that would need an interpreter to answer honestly are not here
//! at all, and a client that asks for one gets `unknown subcommand` rather than
//! an `OK` that did nothing. D-16 says which are which.
//!
//! The reason these two land in M2 rather than with the rest of scripting is
//! the same reason the keyspace four did. Redis's own test suite says `FUNCTION
//! FLUSH` and `SCRIPT FLUSH` in the preamble that runs before every
//! `start_server` block in external mode, and gives up on the whole file when
//! either fails. Two subcommands stood between us and running any of it.

use super::args::{self, Args, is};
use super::table::Spec;
use crate::reply::Out;
use yo_common::{Code, Error, Result};

/// Run one scripting command.
pub(super) fn execute(spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<()> {
    match spec.name {
        "script" => script(args, out),
        "function" => function(args, out),
        other => unreachable!("scripting command with no body: {other}"),
    }
}

/// `SCRIPT FLUSH|EXISTS|HELP`.
fn script(args: Args<'_>, out: &mut Out) -> Result<()> {
    let sub = args.get(1);
    if is(sub, b"FLUSH") {
        // The mode is checked and then ignored, because there is nothing to
        // free either way. Redis's message here is its own and reads like a
        // typo, "only support" rather than "only supports", and the `FUNCTION`
        // one two functions down says "supports". Both are copied as they are.
        if args.len() > 3 || (args.len() == 3 && !mode(args.get(2))) {
            return Err(Error::new(
                Code::Invalid,
                "SCRIPT FLUSH only support SYNC|ASYNC option",
            ));
        }
        out.ok();
    } else if is(sub, b"EXISTS") {
        if args.len() < 3 {
            return Err(args::wrong_arity_sub("script", "exists"));
        }
        // Nothing is cached, so nothing is there, and a client that reads this
        // and sends the script body is doing exactly the right thing.
        out.array(args.len() - 2);
        for _ in 2..args.len() {
            out.int(0);
        }
    } else if is(sub, b"HELP") {
        super::server::help(out, SCRIPT_HELP);
    } else {
        // LOAD, KILL and DEBUG need an interpreter and are not here. The help
        // above still lists them, because it is Redis's text and a client that
        // prints it is printing what the command will do once M6 lands.
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
