//! The `redis` table, and the door from a script back into the command layer.
//!
//! Two kinds of thing live here. The constants and the plain functions are the
//! same on every thread for the life of the process and are built once. The two
//! that need the server, `redis.pcall` and `redis.setresp`, are built for every
//! `EVAL` and handed in through [`mlua::Lua::scope`], which is what lets a
//! borrow of the server reach a Lua callback without a raw pointer.
//!
//! # The nested call
//!
//! A script calling `redis.call('set', 'k', 'v')` does not reach into the
//! keyspace from here. The arguments are written out as RESP, read back with
//! the same decoder a connection uses, and handed to the same
//! [`resolved`](super::super::resolved) every client's command goes through.
//! That costs an encode and a decode per call, and it buys the one thing worth
//! having: there is no second path into a command. The name lookup, the arity
//! check, the type check, the memory limit and the reply are all the ones a
//! client would get, because they are the same code.
//!
//! The checks that happen before that are the ones a real server makes on the
//! way in, in the order it makes them, which is measurable and is not the order
//! anybody would guess: an argument of the wrong type is caught before the
//! command name is looked at, and a command with the wrong number of arguments
//! is caught before the flag that says a script may not call it.

use super::super::table::{Spec, arity_ok, lookup};
use super::super::{Args, Server, Session, resolved};
use super::{convert, nested, sha1};
use crate::frame;
use crate::proto::Limits;
use crate::reply::Out;
use crate::request::{Argv, Step};
use mlua::{Lua, MultiValue, Scope, Table, Value, ffi};
use std::cell::{Cell, RefCell};
use std::os::raw::c_int;

/// Everything one running script is allowed to reach.
pub(super) struct Ctx<'a> {
    server: &'a Server,
    /// The connection the script is running on, which a nested `SELECT` moves
    /// and a nested command reads. Behind a cell because Lua hands the same
    /// borrow to every callback and only one of them runs at a time.
    session: RefCell<&'a mut Session>,
    /// Whether this is one of the `_RO` spellings.
    ro: bool,
    /// What `redis.setresp` was last told, which is per script and starts at
    /// RESP2 whatever protocol the client is speaking.
    resp3: Cell<bool>,
}

impl<'a> Ctx<'a> {
    /// The context for one run.
    pub(super) fn new(server: &'a Server, session: &'a mut Session, ro: bool) -> Ctx<'a> {
        Ctx {
            server,
            session: RefCell::new(session),
            ro,
            resp3: Cell::new(false),
        }
    }
}

/// The `redis` table as it is on every thread, before a run lends it anything.
///
/// The names a script actually calls are not all here. `call`, `error_reply`,
/// `status_reply`, `set_repl` and the argument checking in front of `sha1hex`,
/// `log` and `setresp` are written in Lua in the prelude, because all of them
/// are about counting arguments and raising, and raising a value that is not a
/// string is something Lua does and Rust cannot ask it to do.
pub(super) fn table(lua: &Lua) -> mlua::Result<Table> {
    let t = lua.create_table()?;
    // The four log levels, which are the numbers a real server uses and not
    // ours, since a script is written against the numbers.
    t.raw_set("LOG_DEBUG", 0)?;
    t.raw_set("LOG_VERBOSE", 1)?;
    t.raw_set("LOG_NOTICE", 2)?;
    t.raw_set("LOG_WARNING", 3)?;
    // The replication targets. `REPL_SLAVE` and `REPL_REPLICA` are the same
    // number under two names and both are there, because scripts written before
    // the rename are still running.
    t.raw_set("REPL_NONE", 0)?;
    t.raw_set("REPL_AOF", 1)?;
    t.raw_set("REPL_SLAVE", 2)?;
    t.raw_set("REPL_REPLICA", 2)?;
    t.raw_set("REPL_ALL", 3)?;
    // The version a script branches on. It is the Redis version this server
    // answers as and not our own, for the same reason `INFO` reports that one:
    // a script that checks it is checking which commands it may use.
    t.raw_set("REDIS_VERSION", super::super::server::REPORTED_VERSION)?;
    t.raw_set("REDIS_VERSION_NUM", version_num())?;
    Ok(t)
}

/// The version as the single number `redis.REDIS_VERSION_NUM` reports.
///
/// One byte each for major, minor and patch, which is the packing a real server
/// uses, so `8.10.1` is `0x080a01`.
fn version_num() -> i64 {
    let mut parts = super::super::server::REPORTED_VERSION.split('.');
    let mut at = |shift: u32| -> i64 {
        let n: i64 = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0);
        (n & 0xff) << shift
    };
    at(16) | at(8) | at(0)
}

/// The functions that need nothing from the server, put where the prelude's
/// wrappers can find them.
pub(super) fn statics(lua: &Lua, raw: &Table) -> mlua::Result<()> {
    raw.raw_set(
        "sha1hex",
        lua.create_function(|_, value: Value| {
            // A string is hashed as it is, a number as the digits Lua would
            // print, and anything else as nothing at all, which is why
            // `redis.sha1hex(true)` answers the digest of the empty string.
            let bytes = match &value {
                Value::String(s) => s.as_bytes().to_vec(),
                Value::Integer(n) => n.to_string().into_bytes(),
                Value::Number(d) => lua_number(*d).into_bytes(),
                _ => Vec::new(),
            };
            Ok(String::from_utf8_lossy(&sha1::hex(&bytes)).into_owned())
        })?,
    )?;
    raw.raw_set(
        "log",
        lua.create_function(|_, (level, message): (i64, String)| {
            // The server's log is its standard error, so that is where this
            // goes. A real server would drop a line below its configured level
            // and this one has no such setting, which is registered as D-98.
            let word = match level {
                0 => "debug",
                1 => "verbose",
                3 => "warning",
                _ => "notice",
            };
            eprintln!("yodb script {word}: {message}");
            Ok(())
        })?,
    )?;
    raw.raw_set(
        "known",
        lua.create_function(|_, name: mlua::LuaString| Ok(lookup(&name.as_bytes()).is_some()))?,
    )?;
    raw.raw_set(
        "arity_ok",
        lua.create_function(|_, (name, count): (mlua::LuaString, usize)| {
            Ok(lookup(&name.as_bytes()).is_some_and(|spec| arity_ok(spec, count)))
        })?,
    )?;
    Ok(())
}

/// Call the Lua function this one carries as its first upvalue with the
/// arguments this one was called with, and answer everything it answered.
///
/// Nothing here catches anything. A failure inside the function jumps straight
/// out through this frame the way it would through any other C function, which
/// is the whole point: there are no Rust values on this frame that need
/// dropping, so there is nothing for the jump to skip.
///
/// # Safety
///
/// Lua calls this with a state whose stack holds only the arguments, which is
/// what the stack arithmetic below assumes, and only ever as a closure that
/// [`bridge`] built, which is where the upvalue comes from.
unsafe extern "C-unwind" fn forward(state: *mut mlua::lua_State) -> c_int {
    // SAFETY: the caller is Lua, so the state is live and the stack is the one
    // it just built for this call. Room for one more slot is asked for before
    // anything is pushed, and the count handed to `lua_call` is the count that
    // was on the stack before the function went underneath it.
    unsafe {
        let given = ffi::lua_gettop(state);
        if ffi::lua_checkstack(state, 1) == 0 {
            return 0;
        }
        ffi::lua_pushvalue(state, ffi::lua_upvalueindex(1));
        ffi::lua_insert(state, 1);
        ffi::lua_call(state, given, ffi::LUA_MULTRET);
        ffi::lua_gettop(state)
    }
}

/// The function the prelude puts in front of every library function it wrote.
///
/// Every library function is written in Lua and every one of them can fail, and
/// those two facts together are the reason for the indirection. Lua 5.1 throws
/// the caller's stack frame away on `return f(...)` when `f` is a Lua function
/// and keeps it when `f` is a C function. A real server's libraries are C, so
/// `return redis.call('get', KEYS[1])` still knows it was on line one of the
/// script when the call fails. Ours would have reported `(tail call)` and no
/// line at all, on the single most common line anybody writes.
///
/// So what comes back from here is a C function that does nothing but call the
/// Lua one it was handed. The frame survives, and because the jump out of a
/// failure goes through C rather than through Rust, what the script raised is
/// still the table it raised rather than something wrapped on the way past.
///
/// The Lua one rides along as an upvalue rather than sitting under a name
/// somewhere, which is what lets the prelude build a bridge whenever it likes
/// rather than only while the interpreter is being set up. `cjson.new()` needs
/// that, since it hands a script a fresh table of functions in the middle of a
/// script.
pub(super) fn bridge(lua: &Lua) -> mlua::Result<mlua::Function> {
    lua.create_function(|lua, body: mlua::Function| {
        // SAFETY: `exec_raw` pushes the one argument and hands over a stack
        // that holds nothing else, so the closure below takes that function off
        // and leaves the C one in its place, which is the single value the call
        // then reads back.
        unsafe {
            lua.exec_raw::<mlua::Function>(body, |state| {
                ffi::lua_pushcclosure(state, forward, 1);
            })
        }
    })
}

/// Put the two functions that need the server into the raw table for one run.
pub(super) fn lend<'scope, 'env: 'scope>(
    lua: &Lua,
    scope: &'scope Scope<'scope, 'env>,
    ctx: &'env Ctx<'env>,
) -> mlua::Result<()> {
    let raw: Table = lua.named_registry_value("yo_raw")?;
    raw.raw_set(
        "pcall",
        scope.create_function(move |lua, args: MultiValue| command(lua, ctx, args))?,
    )?;
    raw.raw_set(
        "setresp",
        scope.create_function(move |_, three: bool| {
            ctx.resp3.set(three);
            Ok(())
        })?,
    )?;
    Ok(())
}

/// `redis.pcall(name, arg...)`, which is also what `redis.call` is built on.
///
/// Never an `Err`. A command that failed comes back as the table a script reads
/// an `err` field out of, and turning that back into a failure is the prelude's
/// job, because the prelude can raise a table and this cannot.
fn command(lua: &Lua, ctx: &Ctx<'_>, args: MultiValue) -> mlua::Result<Value> {
    let words = match flatten(lua, &args) {
        Ok(w) => w,
        Err(msg) => return failed(lua, msg),
    };
    const REFUSED: &[u8] = b"ERR This Redis command is not allowed from script";
    let Some(spec) = lookup(&words[0]) else {
        // A name a real server has and refuses is refused here whether or not
        // this server has it yet, because the answer is the same either way and
        // a script that asks deserves to be told the real reason. The list is
        // in `shut`.
        if shut(&words) {
            return failed(lua, REFUSED);
        }
        return failed(lua, b"ERR Unknown Redis command called from script");
    };
    if !arity_ok(spec, words.len()) {
        return failed(
            lua,
            b"ERR Wrong number of args calling Redis command from script",
        );
    }
    if spec.flags.contains(&"noscript") || shut(&words) {
        return failed(lua, REFUSED);
    }
    // The full stop at the end of this one is a real server's and is the only
    // sentence in the group that has one.
    if ctx.ro && spec.flags.contains(&"write") {
        return failed(
            lua,
            b"ERR Write commands are not allowed from read-only scripts.",
        );
    }
    reply(lua, ctx, spec, &words)
}

/// Whether this is a command a script may not call.
///
/// Most of what a script is kept away from is the `noscript` flag on the spec
/// and is checked against the flag. Two sets are not.
///
/// The first is the commands a real server refuses that this one does not have
/// yet, which are the transactions, the subscriptions and the replication
/// plumbing. Reading them off the flag would mean telling a script that `MULTI`
/// is not a command, which is not why it cannot call it.
///
/// The second is the containers, where the refusal sits on the subcommand and
/// this table has no subcommand rows. The shape is the same for all of them:
/// everything under the name is out except `HELP`, which only prints. `CLUSTER`
/// is the exception a real server makes and would be the exception here too,
/// since only `CLUSTER RESET` is refused there.
fn shut(words: &[Vec<u8>]) -> bool {
    /// Every top level name a real server carries `noscript` on, whether or not
    /// this one has the command.
    const ALONE: [&[u8]; 39] = [
        b"auth",
        b"bgrewriteaof",
        b"bgsave",
        b"debug",
        b"discard",
        b"eval",
        b"eval_ro",
        b"evalsha",
        b"evalsha_ro",
        b"exec",
        b"failover",
        b"fcall",
        b"fcall_ro",
        b"hello",
        b"monitor",
        b"multi",
        b"psubscribe",
        b"psync",
        b"punsubscribe",
        b"quit",
        b"replconf",
        b"replicaof",
        b"reset",
        b"role",
        b"save",
        b"search.clusterinfo",
        b"search.clusterrefresh",
        b"search.clusterset",
        b"shutdown",
        b"slaveof",
        b"ssubscribe",
        b"subscribe",
        b"sunsubscribe",
        b"sync",
        b"timeseries.clusterset",
        b"timeseries.refreshcluster",
        b"unsubscribe",
        b"unwatch",
        b"watch",
    ];
    /// The containers whose every subcommand but `HELP` is out.
    const CONTAINERS: [&[u8]; 9] = [
        b"acl",
        b"backup",
        b"client",
        b"config",
        b"function",
        b"hotkeys",
        b"latency",
        b"module",
        b"script",
    ];
    let name = words[0].to_ascii_lowercase();
    if ALONE.contains(&name.as_slice()) {
        return true;
    }
    if !CONTAINERS.contains(&name.as_slice()) {
        return false;
    }
    !words
        .get(1)
        .is_some_and(|sub| sub.eq_ignore_ascii_case(b"help"))
}

/// Run the command and read its reply back as a Lua value.
fn reply(lua: &Lua, ctx: &Ctx<'_>, spec: &'static Spec, words: &[Vec<u8>]) -> mlua::Result<Value> {
    let wire = encode(words);
    let limits = Limits::default();
    let mut argv = Argv::new();
    match argv.decode(&wire, &limits) {
        Ok(Step::Command { .. }) => {}
        // The bytes were written here and read here, so the only way this
        // happens is an argument longer than the protocol allows.
        _ => return failed(lua, b"ERR Lua redis lib command arguments are too long"),
    }
    let mut scratch = Out::new(nested(ctx.resp3.get()));
    {
        let mut session = ctx.session.borrow_mut();
        resolved(
            ctx.server,
            &mut session,
            Some(spec),
            Args::new(&argv, &wire),
            &mut scratch,
        );
    }
    match frame::decode(scratch.as_slice(), &limits) {
        Ok(Some((f, _))) => convert::pull(lua, &f, ctx.resp3.get()),
        // A command that wrote nothing, which is one of the blocking six
        // finding nothing to take. A script never waits, so nothing is the
        // answer rather than the start of one.
        _ => Ok(Value::Boolean(false)),
    }
}

/// The arguments as the byte strings a command is made of.
///
/// The two things a real server accepts here are a string and a number, and a
/// number arrives as the digits Lua would print rather than as anything this
/// side chooses. Everything else is refused before the command name is even
/// looked at, which is why `redis.call('get', {})` complains about the argument
/// and not about `GET` taking one key.
fn flatten(_lua: &Lua, args: &MultiValue) -> Result<Vec<Vec<u8>>, &'static [u8]> {
    if args.is_empty() {
        return Err(b"ERR Please specify at least one argument for this redis lib call");
    }
    let mut words = Vec::with_capacity(args.len());
    for arg in args {
        words.push(match arg {
            Value::String(s) => s.as_bytes().to_vec(),
            Value::Integer(n) => n.to_string().into_bytes(),
            Value::Number(d) => lua_number(*d).into_bytes(),
            _ => return Err(b"ERR Lua redis lib command arguments must be strings or integers"),
        });
    }
    Ok(words)
}

/// A number the way Lua prints one, which is fourteen significant digits.
///
/// This is what a script's `1.5` becomes on the way into a command, so
/// `redis.call('set', 'k', 1.5)` stores `1.5` and not `1.500000`.
fn lua_number(d: f64) -> String {
    if d == d.trunc() && d.abs() < 1e15 {
        return format!("{}", d as i64);
    }
    format!("{d:.14}")
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_string()
}

/// The command as the bytes a connection would have sent.
fn encode(words: &[Vec<u8>]) -> Vec<u8> {
    let mut wire = format!("*{}\r\n", words.len()).into_bytes();
    for w in words {
        wire.extend_from_slice(format!("${}\r\n", w.len()).as_bytes());
        wire.extend_from_slice(w);
        wire.extend_from_slice(b"\r\n");
    }
    wire
}

/// The table a failed call answers with.
///
/// The second field is a real server's and is visible to the script. It says
/// this failure was the server's own rather than one a command replied with,
/// so a script that catches it with `pcall` does not also move the counters
/// `INFO errorstats` reports.
fn failed(lua: &Lua, msg: &[u8]) -> mlua::Result<Value> {
    let t = lua.create_table()?;
    t.raw_set("err", lua.create_string(msg)?)?;
    t.raw_set("ignore_error_stats_update", true)?;
    Ok(Value::Table(t))
}
