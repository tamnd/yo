//! The Lua 5.1 interpreter behind `EVAL`, and the sandbox it runs in.
//!
//! # Why 5.1
//!
//! Because that is what a script is written against. Every script anybody has
//! in production was written for the interpreter Redis embedded in 2010, and a
//! newer Lua would change `unpack`, integer division, `#` on a table with holes
//! and the behaviour of `pcall`. None of those are improvements to a server
//! whose job is to run somebody else's twelve line script the way it has always
//! run. So this is 5.1, built from source inside our own build rather than
//! found on the machine, and it is not a version anybody here chose.
//!
//! # One interpreter per thread
//!
//! A Lua state is not shared and not moved. Each thread that runs a command
//! builds its own the first time it is asked to and keeps it, so two clients
//! running scripts at once run in two states and never wait for each other. The
//! cost is one state per thread that has ever run a script, which is a few tens
//! of kilobytes, and the alternative is a global lock around every `EVAL`.
//!
//! What that means for a script is that a global it managed to set would only
//! be there for the next script on the same thread. It cannot set one: the
//! globals are readonly, which is Redis's rule and is the whole reason it is
//! Redis's rule.
//!
//! # Getting the server into a callback
//!
//! `redis.call` needs the server and the connection, and neither of those is
//! `'static`. [`mlua::Lua::scope`] is the safe way to lend a borrow to Lua for
//! the length of one call and have Lua give it back, so that is what is used,
//! and the two functions that need the borrow are made fresh for every `EVAL`.
//! A raw pointer in a thread local would save building two closures per script
//! and would be the wrong trade: `12` section 7 says in as many words that
//! scripting is not a performance path, and this is the one place in the crate
//! where that is the whole design.
//!
//! # The libraries that came with the interpreter
//!
//! A script gets more than the `redis` table. Redis links four libraries into
//! its Lua and a script written against Redis calls them without thinking about
//! where they came from, so they are part of the surface and not an extra. Each
//! one lives in its own module here, with the arithmetic in Rust and the
//! argument checking in the prelude, for the same reason the `redis` table
//! splits the same way: a message a script reads has to be raised from Lua.
//!
//! # Where the error handling lives
//!
//! In Lua, not here, because that is where a real server puts it. The prelude
//! installs `__redis__err__handler` under the same name Redis uses, and every
//! script runs under it. It is what turns a raised string into a table with an
//! `err` field, and it is what attaches the source and line a client reads at
//! the end of the message. Doing it in Rust would mean getting the raised Lua
//! value across the boundary, and a raised value can be any Lua value at all.
//!
//! # Functions are the same interpreter with different globals
//!
//! `FCALL` runs in the state this thread already has, not in one of its own. A
//! real server keeps a second Lua state for libraries and this keeps a second
//! set of globals, which comes to the same thing from a client's side: no
//! `KEYS`, no `ARGV`, no error handler on the global table, and a `redis` table
//! without the three names that only mean something inside `EVAL`. What it does
//! not come to the same thing on is a library's own locals, which are per
//! thread here and per process there, and that is D-110.

mod api;
mod argue;
mod bit;
mod cjson;
mod cmsgpack;
mod convert;
pub(in crate::dispatch) mod library;
mod sha1;
mod r#struct;

use super::{Server, Session};
use crate::proto::Proto;
use crate::reply::Out;
use mlua::{Lua, StdLib, Table, Value};
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    /// This thread's interpreter, built the first time it runs a script.
    ///
    /// Taken out for the length of a run and put back after, so a script that
    /// found a way to start another one would find the slot empty rather than
    /// finding a state that is already running. Nothing can: `EVAL` and its
    /// four relatives all carry `noscript` and `redis.call` refuses those.
    static VM: RefCell<Option<Lua>> = const { RefCell::new(None) };
}

/// Every script body a client has loaded, by the name it is loaded under.
///
/// One table for the server rather than one per connection, because a name is
/// what `EVALSHA` sends and a client that loaded a script on one connection
/// runs it on another. That is the whole point of the cache: `SCRIPT LOAD` on
/// startup, `EVALSHA` for the rest of the process's life.
#[derive(Default)]
pub(in crate::dispatch) struct Scripts {
    held: HashMap<[u8; 40], Box<[u8]>>,
}

impl Scripts {
    /// Remember a body and answer the name it is now under.
    pub(in crate::dispatch) fn add(&mut self, body: &[u8]) -> [u8; 40] {
        let sha = sha1::hex(body);
        self.held.entry(sha).or_insert_with(|| body.into());
        sha
    }

    /// The body under a name, if it is there.
    ///
    /// A copy, because the caller has to let the lock go before it runs the
    /// script: the script is going to call commands, and those take locks of
    /// their own. A script is a few hundred bytes and it is copied once per
    /// `EVALSHA`, which is the trade a lock held across an interpreter would
    /// otherwise be.
    pub(in crate::dispatch) fn body(&self, sha: &[u8; 40]) -> Option<Box<[u8]>> {
        self.held.get(sha).cloned()
    }

    /// Whether a name is one we know.
    pub(in crate::dispatch) fn has(&self, sha: &[u8; 40]) -> bool {
        self.held.contains_key(sha)
    }

    /// Forget every one of them, which is what `SCRIPT FLUSH` asks for.
    pub(in crate::dispatch) fn wipe(&mut self) {
        self.held.clear();
    }
}

/// What a script is given and what it is allowed to do.
pub(in crate::dispatch) struct Ask<'a> {
    /// `KEYS`, in the order the client wrote them.
    pub keys: &'a [&'a [u8]],
    /// `ARGV`, which is everything after the keys.
    pub argv: &'a [&'a [u8]],
    /// The name the message ends with. For a script that is the digest of the
    /// body, whether the client sent the body or the digest, and for a function
    /// it is the function's own name.
    pub name: &'a [u8],
    /// Set by the `_RO` spellings, and checked in `redis.call` rather than here
    /// so that a script that only reads still runs.
    pub ro: bool,
}

/// Run one script and write its reply.
///
/// Nothing comes back as an `Err`. A script that fails writes its own line,
/// because the line a script fails with is not one of ours: it carries the
/// script's own error code when it has one, it carries no code at all when the
/// script raised a table, and it ends with the name and the line the failure
/// was on. Handing that to the generic error writer would put `ERR` in front of
/// half of them.
pub(in crate::dispatch) fn run(
    server: &Server,
    session: &mut Session,
    body: &[u8],
    ask: &Ask<'_>,
    out: &mut Out,
) {
    // Everything from here down allocates, and says so. An interpreter is not
    // a command path: it parses a chunk, builds a table per reply and hands out
    // a string per argument, and there is no arrangement of those that does not
    // allocate. Y7 is about the commands underneath, and those are still held
    // to it when the script calls them.
    yo_alloc::allow(|| {
        let Some(lua) = VM.with(|vm| vm.borrow_mut().take().or_else(|| interpreter().ok())) else {
            out.error(b"ERR the script interpreter could not be started");
            return;
        };
        // Put back rather than cleared, because a script is not the only thing
        // that can be running on this connection: nothing nests today, and a
        // flag that is set and cleared rather than saved and restored is a flag
        // that will be wrong the first time something does.
        let outer = session.scripted;
        session.scripted = true;
        running(&lua, server, session, body, ask, out);
        session.scripted = outer;
        VM.with(|vm| *vm.borrow_mut() = Some(lua));
    });
}

/// Whether a body parses, and what Lua said about it if it does not.
///
/// `SCRIPT LOAD` asks this so that a client finds out its script has a typo in
/// it when it loads the script rather than the first time something runs it,
/// which on a client that loads everything at startup is the difference between
/// a failure at startup and a failure at three in the morning.
///
/// The message is the whole of what a client is shown, with no name and no line
/// added on the end, because there is no run for a line to be in.
pub(in crate::dispatch) fn compiles(body: &[u8]) -> Result<(), String> {
    yo_alloc::allow(|| {
        VM.with(|vm| {
            let mut held = vm.borrow_mut();
            if held.is_none() {
                *held = interpreter().ok();
            }
            let Some(lua) = held.as_ref() else {
                return Err("the script interpreter could not be started".to_string());
            };
            match lua.load(body).set_name("@user_script").into_function() {
                Ok(_) => Ok(()),
                Err(e) => Err(compile_message(&e)),
            }
        })
    })
}

/// The digest of a library's code, which is how one thread's copy is told apart
/// from another's.
pub(in crate::dispatch) fn fingerprint(code: &[u8]) -> [u8; 40] {
    sha1::hex(code)
}

/// Which library and which function inside it an `FCALL` is about.
///
/// Copied out of the registry rather than borrowed from it, because the lock
/// over the registry has to be let go before anything runs: a function calls
/// commands and those take locks of their own, and one of the commands it can
/// call is `FUNCTION LIST`.
pub(in crate::dispatch) struct Call<'a> {
    /// The library the function was registered by.
    pub library: &'a str,
    /// The digest of the library's code, which is how a thread works out
    /// whether the copy it compiled earlier is still the current one.
    pub sha: &'a [u8; 40],
    /// The library's code, from the newline after the shebang.
    pub body: &'a [u8],
    /// The function's own name, spelled the way it was registered rather than
    /// the way the client asked for it.
    pub function: &'a str,
}

/// Compile a library on this thread and run it once, so that it registers what
/// it registers.
///
/// This is `FUNCTION LOAD` on the thread the client sent it to, and it is also
/// what every other thread does the first time one of its clients calls into
/// the library. The failure is the whole sentence a client reads, with no code
/// in front of it, because the two it can be already read as sentences:
/// `Error compiling function: ...` and `Error registering functions: ...`.
pub(in crate::dispatch) fn install(
    name: &str,
    sha: &[u8; 40],
    body: &[u8],
) -> Result<Vec<library::Func>, String> {
    yo_alloc::allow(|| {
        VM.with(|vm| {
            let mut held = vm.borrow_mut();
            if held.is_none() {
                *held = interpreter().ok();
            }
            let Some(lua) = held.as_ref() else {
                return Err("the script interpreter could not be started".to_string());
            };
            compile(lua, name, sha, body)
        })
    })
}

/// Run one function and write its reply.
///
/// Nothing comes back as an `Err` here for the same reason nothing does out of
/// [`run`]: a function that fails writes a line that carries its own code and
/// ends with its own name and line.
pub(in crate::dispatch) fn fcall(
    server: &Server,
    session: &mut Session,
    call: &Call<'_>,
    ask: &Ask<'_>,
    out: &mut Out,
) {
    yo_alloc::allow(|| {
        let Some(lua) = VM.with(|vm| vm.borrow_mut().take().or_else(|| interpreter().ok())) else {
            out.error(b"ERR the script interpreter could not be started");
            return;
        };
        let outer = session.scripted;
        session.scripted = true;
        calling(&lua, server, session, call, ask, out);
        session.scripted = outer;
        VM.with(|vm| *vm.borrow_mut() = Some(lua));
    });
}

/// The body of [`install`], with the interpreter in hand.
fn compile(
    lua: &Lua,
    name: &str,
    sha: &[u8; 40],
    body: &[u8],
) -> Result<Vec<library::Func>, String> {
    let loaded: mlua::Result<(bool, Value)> = (|| {
        let loader: mlua::Function = lua.named_registry_value("yo_load")?;
        loader.call((name, lua.create_string(sha)?, lua.create_string(body)?))
    })();
    let (ok, value) = loaded.map_err(|e| e.to_string())?;
    if !ok {
        return Err(match &value {
            Value::String(s) => String::from_utf8_lossy(&s.as_bytes()).into_owned(),
            other => format!("{other:?}"),
        });
    }
    let Value::Table(list) = value else {
        return Err("the library did not say what it registered".to_string());
    };
    let mut found = Vec::new();
    for one in list.sequence_values::<Table>() {
        let one = one.map_err(|e| e.to_string())?;
        let name: mlua::LuaString = one.raw_get("name").map_err(|e| e.to_string())?;
        let desc: Option<mlua::LuaString> = one.raw_get("desc").map_err(|e| e.to_string())?;
        let flags: u32 = one.raw_get("flags").map_err(|e| e.to_string())?;
        found.push(library::Func {
            name: String::from_utf8_lossy(&name.as_bytes()).into(),
            desc: desc.map(|d| d.as_bytes().to_vec().into_boxed_slice()),
            flags,
        });
    }
    Ok(found)
}

/// The body of [`fcall`], with the interpreter in hand.
fn calling(
    lua: &Lua,
    server: &Server,
    session: &mut Session,
    call: &Call<'_>,
    ask: &Ask<'_>,
    out: &mut Out,
) {
    // Compiled here when this thread has never run anything out of this
    // library, which is every thread but the one the load arrived on, and again
    // after a `FUNCTION LOAD REPLACE` that some other thread took. The digest is
    // what tells the two apart.
    let held: mlua::Result<bool> = (|| {
        let holds: mlua::Function = lua.named_registry_value("yo_holds")?;
        holds.call((call.library, lua.create_string(call.sha)?))
    })();
    if !held.unwrap_or(false)
        && let Err(why) = compile(lua, call.library, call.sha, call.body)
    {
        out.error_line(b"ERR ", why.as_bytes());
        return;
    }

    let ctx = api::Ctx::new(server, session, ask.ro);
    let done = lua.scope(|scope| {
        api::lend(lua, scope, &ctx)?;
        let runner: mlua::Function = lua.named_registry_value("yo_fcall")?;
        runner.call::<(bool, Value)>((
            call.library,
            call.function,
            strings(lua, ask.keys)?,
            strings(lua, ask.argv)?,
        ))
    });

    match done {
        Ok((true, value)) => convert::push(out, &value),
        Ok((false, value)) => out.error(&failure(&value, ask.name)),
        Err(e) => {
            let mut line = b"ERR ".to_vec();
            line.extend_from_slice(e.to_string().as_bytes());
            out.error_line(b"", &line);
        }
    }
}

/// The body of [`run`], with the interpreter in hand.
fn running(
    lua: &Lua,
    server: &Server,
    session: &mut Session,
    body: &[u8],
    ask: &Ask<'_>,
    out: &mut Out,
) {
    // `@` in front of the name is Lua's way of saying the name is a file rather
    // than a snippet, and it is what makes the messages read `user_script:1:`
    // instead of quoting the whole script back at the client.
    let chunk = match lua.load(body).set_name("@user_script").into_function() {
        Ok(f) => f,
        Err(e) => {
            let text = compile_message(&e);
            out.error_line(
                b"ERR Error compiling script (new function): ",
                text.as_bytes(),
            );
            return;
        }
    };

    let ctx = api::Ctx::new(server, session, ask.ro);
    let done = lua.scope(|scope| {
        api::lend(lua, scope, &ctx)?;
        let globals = lua.globals();
        globals.raw_set("KEYS", strings(lua, ask.keys)?)?;
        globals.raw_set("ARGV", strings(lua, ask.argv)?)?;
        let runner: mlua::Function = lua.named_registry_value("yo_run")?;
        runner.call::<(bool, Value)>(chunk)
    });

    match done {
        Ok((true, value)) => convert::push(out, &value),
        Ok((false, value)) => out.error(&failure(&value, ask.name)),
        // The handler cannot fail and the runner cannot fail, so this is Lua
        // running out of memory or a bug here. Either way the client gets a
        // line rather than a connection that answered nothing.
        Err(e) => {
            let mut line = b"ERR ".to_vec();
            line.extend_from_slice(e.to_string().as_bytes());
            out.error_line(b"", &line);
        }
    }
}

/// The message a client reads when a script failed.
///
/// The handler has already turned whatever was raised into a table with an
/// `err` field and, when it could work out where the failure was, a `source`
/// and a `line`. The two halves are joined here because the name of the script
/// belongs on the end and the handler does not know it.
fn failure(value: &Value, called: &[u8]) -> Vec<u8> {
    let Value::Table(t) = value else {
        return b"ERR the script failed and said nothing".to_vec();
    };
    let msg = match t.raw_get::<Value>("err") {
        Ok(Value::String(s)) => s.as_bytes().to_vec(),
        _ => b"ERR unknown error".to_vec(),
    };
    let (Ok(Value::String(source)), Ok(line)) =
        (t.raw_get::<Value>("source"), t.raw_get::<Value>("line"))
    else {
        return scrub(&msg);
    };
    let n = match line {
        Value::Integer(n) => n,
        Value::Number(d) => d as i64,
        _ => return scrub(&msg),
    };
    let mut out = msg;
    out.extend_from_slice(b" script: ");
    out.extend_from_slice(called);
    out.extend_from_slice(b", on ");
    out.extend_from_slice(&source.as_bytes());
    out.extend_from_slice(format!(":{n}.").as_bytes());
    scrub(&out)
}

/// A message with the two bytes that would end an error line taken out.
fn scrub(msg: &[u8]) -> Vec<u8> {
    msg.iter()
        .map(|&b| if b == b'\r' || b == b'\n' { b' ' } else { b })
        .collect()
}

/// The part of a compile failure a client is shown.
///
/// mlua wraps the message Lua produced in a struct with the chunk name beside
/// it, and the chunk name is already in the message, so only the message goes
/// out.
fn compile_message(e: &mlua::Error) -> String {
    match e {
        mlua::Error::SyntaxError { message, .. } => message.clone(),
        other => other.to_string(),
    }
}

/// A Lua table of byte strings numbered from one, which is what `KEYS` is.
fn strings(lua: &Lua, words: &[&[u8]]) -> mlua::Result<Table> {
    let t = lua.create_table_with_capacity(words.len(), 0)?;
    for (i, w) in words.iter().enumerate() {
        t.raw_set(i + 1, lua.create_string(w)?)?;
    }
    Ok(t)
}

/// Which protocol a nested command should answer in.
///
/// `redis.setresp` is the only thing that moves it, and it moves it for the
/// script and not for the client: a script may read a map as a map and still
/// reply to a RESP2 client.
fn nested(resp3: bool) -> Proto {
    if resp3 { Proto::Resp3 } else { Proto::Resp2 }
}

/// Build this thread's interpreter and lock it down.
fn interpreter() -> mlua::Result<Lua> {
    // The debug library comes in so that the error handler can ask where a
    // failure happened, and goes out again in the first few lines of the
    // prelude, before anything a client wrote has run. A script that could
    // reach `debug` could reach the environment of any function on the stack,
    // and past that the sandbox is decoration. `os` comes in for `clock` alone
    // and leaves with only that.
    //
    // The constructor is the unsafe one because mlua will not load `debug` from
    // the safe one, and it is right not to: `debug.setupvalue` and its
    // neighbours can hand out a Rust value as the wrong type. What makes it
    // sound here is that the only Lua that ever sees the library is the prelude
    // below, which is ours and is a constant, and the library is gone before the
    // function this returns is ever called. A script cannot reach it, and there
    // is no order of commands that lets one try.
    // SAFETY: the paragraph above is the argument. The `debug` library is
    // loaded, used by the prelude and taken off the global table by the same
    // chunk, so no Lua a client wrote can ever see it.
    let lua = unsafe {
        Lua::unsafe_new_with(
            StdLib::TABLE | StdLib::STRING | StdLib::MATH | StdLib::OS | StdLib::DEBUG,
            mlua::LuaOptions::default(),
        )
    };
    lua.globals().raw_set("redis", api::table(&lua)?)?;
    // The prelude is handed the two things it cannot write for itself. One is
    // the C function that goes in front of every library function so that a
    // failure on a tail call still knows which line of the script it was on.
    // The other is the light userdata a JSON null decodes to.
    //
    // It hands back two things. `run` is the function every script is put
    // through. `raw` is the private table its wrappers reach us through, and it
    // is the reason any of this works: it is a local of the prelude chunk, so
    // it is reachable from the wrappers and from the registry and from nowhere
    // a script can get to. Rust fills it in after the chunk has run, which is
    // fine because nothing in there reads it until a script calls something.
    let boot: Table = lua
        .load(PRELUDE)
        .set_name("@lua_prelude")
        .call((api::bridge(&lua)?, cjson::null()))?;
    let raw: Table = boot.raw_get("raw")?;
    api::statics(&lua, &raw)?;
    bit::statics(&lua, &raw)?;
    cjson::statics(&lua, &raw)?;
    r#struct::statics(&lua, &raw)?;
    cmsgpack::statics(&lua, &raw)?;
    library::statics(&lua, &raw)?;
    for name in ["run", "load", "holds", "fcall"] {
        lua.set_named_registry_value(&format!("yo_{name}"), boot.raw_get::<mlua::Function>(name)?)?;
    }
    lua.set_named_registry_value("yo_raw", raw)?;
    Ok(lua)
}

/// The Lua half of the sandbox, run once per thread.
///
/// It takes away the globals a script cannot have, puts back the one from `os`
/// that it can, fills in the rest of the `redis` table, wraps `pcall` so that a
/// caught error reads the way a real server's does, installs the error handler
/// under the name a real server gives it, and makes both `redis` and the global
/// table readonly. It answers the function every script is then run through and
/// the private table Rust reaches its wrappers through.
///
/// Written in Lua rather than in Rust because every line of it is about Lua's
/// own machinery: what is reachable from `_G`, what `pcall` hands back, which
/// stack frame a failure happened on, and above all what gets raised. A Rust
/// callback can only fail with a message, and half the failures here are tables
/// that a script is expected to read fields out of.
///
/// Every one of these functions raises with a level of zero, which is Lua for
/// "do not put a file and a line in front of this". The position a client reads
/// at the end of the line is the script's own and is put there by the handler,
/// not by the raise, which is why `redis.sha1hex()` reports the script's line
/// and not a line in here.
const PRELUDE: &str = r#"
-- Two things Rust hands in because neither can be written in Lua. `bridge` is
-- the C function that goes in front of a library function, and the comment on
-- `bridge` in `api.rs` says what it is for. `null` is the light userdata a JSON
-- null decodes to, which has to come from Rust because Lua has no way to make
-- one.
local bridge, null = ...

-- Held here so that taking them off the global table does not take them away
-- from the sandbox itself.
local dbg = debug
local rawpcall, rawxpcall = pcall, xpcall
local error, type, tostring, tonumber = error, type, tostring, tonumber
local setmetatable, rawset, select, next = setmetatable, rawset, select, next
local getmetatable, setfenv, ipairs = getmetatable, setfenv, ipairs
local rawloadstring, rawload = loadstring, load
local rawunpack = unpack
local sub, find, concat = string.sub, string.find, table.concat
local lower = string.lower
local floor = math.floor

-- Where the functions Rust hands over live. A local of this chunk and an
-- upvalue of everything below, so a script cannot reach it and Rust can.
local raw = {}

-- The real `redis` table, with the constants Rust has already put on it. A
-- script never gets a reference to this one.
local lib = redis

-- Everything the base library brings that a script must not reach. `os` keeps
-- `clock` and loses the rest, because a script that can read the wall clock or
-- the environment is a script whose replies are not a function of the keyspace.
for _, name in ipairs({'print', 'dofile', 'loadfile', 'newproxy', 'getfenv', 'setfenv',
                       'require', 'module', 'package', 'io', 'debug'}) do
  rawset(_G, name, nil)
end
rawset(_G, 'os', {clock = os.clock})

-- What every guarded table does when something writes to it. Level two puts the
-- script's own line in front of the message, because the frame that wrote is
-- the one a client wants to be told about.
local function readonly()
  error('Attempt to modify a readonly table', 2)
end

-- A library as a script sees it is an empty table in front of the real one.
--
-- It has to be empty. Lua 5.1 only calls `__newindex` for a key a table does
-- not already have, so a guard on a table that holds its own functions guards
-- nothing: `redis.call = 1` would go straight through and the next script on
-- this thread would find `redis.call` was a number. A real server closes that
-- with a change to its own copy of Lua that vanilla 5.1 does not have. An empty
-- table in front costs one more lookup per call and closes it here. What it
-- would cost a script is `pairs(redis)`, and the three readers further down
-- give that back.
--
-- Every library gets one, so the list here is the list of tables a script can
-- read and cannot change.
local shielded = {}

-- An empty table in front of a real one, and the metatable that joins them.
--
-- `__yo_real` is how anything on the Rust side that walks a table raw finds the
-- real one, since the proxy itself is empty. It sits on the metatable, which a
-- script cannot reach.
local function guarded(real)
  local front = {__index = real, __newindex = readonly, __yo_real = real}
  return setmetatable({}, front), front
end

local function shield(name, real)
  local proxy, front = guarded(real)
  shielded[#shielded + 1] = {proxy = proxy, front = front, real = real}
  rawset(_G, name, proxy)
  return proxy
end

local proxy = shield('redis', lib)

-- `redis.pcall` is the one that reaches the server and it answers a table with
-- an `err` field when the command failed. `redis.call` is the same call with
-- that turned back into a failure, which is the only difference between them.
lib.pcall = function(...) return raw.pcall(...) end
lib.call = bridge(function(...)
  local reply = raw.pcall(...)
  if type(reply) == 'table' and reply.err ~= nil then
    error(reply)
  end
  return reply
end)

-- The digest of one value. Anything that is not a string or a number hashes as
-- nothing at all, so `redis.sha1hex({})` is the digest of the empty string, and
-- that is a real server's answer and not a shortcut taken here.
lib.sha1hex = bridge(function(...)
  if select('#', ...) ~= 1 then error('wrong number of arguments', 0) end
  return raw.sha1hex((select(1, ...)))
end)

-- Which protocol the replies a script reads are written in. It moves what the
-- script sees and not what the client is answered in, so a script can read a
-- map as a map while still replying to a RESP2 client.
lib.setresp = bridge(function(...)
  if select('#', ...) ~= 1 then error('redis.setresp() requires one argument.', 0) end
  local version = select(1, ...)
  if version ~= 2 and version ~= 3 then error('RESP version must be 2 or 3.', 0) end
  raw.setresp(version == 3)
end)

-- A line in the server's log. Everything after the level is joined with spaces,
-- which is why `redis.log(redis.LOG_WARNING, 'k', key)` reads the way it does.
lib.log = bridge(function(...)
  local n = select('#', ...)
  if n < 2 then error('redis.log() requires two arguments or more.', 0) end
  local level = select(1, ...)
  if type(level) ~= 'number' then error('First argument must be a number (log level).', 0) end
  local parts = {}
  for i = 2, n do parts[i - 1] = tostring((select(i, ...))) end
  raw.log(level, concat(parts, ' '))
end)

-- Which of a script's effects reach the replica and the log. Accepted and then
-- ignored, because every effect this server produces goes to both. Kept because
-- a script that narrows replication and then widens it again would otherwise
-- fail on the first call, and registered as D-99.
lib.set_repl = bridge(function(...)
  if select('#', ...) ~= 1 then error('redis.set_repl() requires one argument.', 0) end
  -- A value that is not a number counts as zero rather than as a mistake, which
  -- is why `redis.set_repl('x')` is accepted and `redis.set_repl(9)` is not.
  local flags = tonumber((select(1, ...))) or 0
  if flags < 0 or flags > 3 then
    error('Invalid replication flags. Use REPL_AOF, REPL_REPLICA, REPL_ALL or REPL_NONE.', 0)
  end
end)

-- Whether the connection may run a command. There are no users here yet, so the
-- only thing this can answer is whether the command exists and whether it was
-- handed the right number of arguments, and it says yes to everything that gets
-- past those two. Registered as D-100.
lib.acl_check_cmd = bridge(function(...)
  local n = select('#', ...)
  if n < 1 then
    error('Please specify at least one argument for this redis lib call', 0)
  end
  local name = select(1, ...)
  if type(name) ~= 'string' or not raw.known(name) then
    error('Invalid command passed to redis.acl_check_cmd()', 0)
  end
  -- The command name and its arguments are counted together, which is why
  -- `redis.acl_check_cmd('get')` is one short rather than complete.
  if not raw.arity_ok(name, n) then
    error('Wrong number of args for redis.acl_check_cmd()', 0)
  end
  return true
end)

-- The two that build a reply rather than sending one. Neither raises, even on
-- arguments that make no sense: a bad call answers a table with its own
-- complaint in it, and a script that hands that straight back sends that
-- complaint to the client.
local function only_string(n, value)
  return n == 1 and type(value) == 'string'
end

lib.error_reply = function(...)
  local msg = select(1, ...)
  if not only_string(select('#', ...), msg) then
    return {err = 'ERR wrong number or type of arguments'}
  end
  -- One leading minus is taken off, and then a message with no space in it gets
  -- `ERR ` in front. The space is how a real server guesses whether the message
  -- already starts with a code of its own, and it guesses rather than checking,
  -- so `redis.error_reply('lower case')` goes out with no code at all.
  if sub(msg, 1, 1) == '-' then msg = sub(msg, 2) end
  if not find(msg, ' ', 1, true) then msg = 'ERR ' .. msg end
  return {err = msg}
end

lib.status_reply = function(...)
  local msg = select(1, ...)
  if not only_string(select('#', ...), msg) then
    return {err = 'ERR wrong number or type of arguments'}
  end
  return {ok = msg}
end

-- Effects replication has been the only kind since 7.0, so this says yes and
-- does nothing. A script written for 3.2 calls it before its first write and
-- would stop there if it got anything else.
lib.replicate_commands = function() return true end

-- The debugger, which this server does not have. False rather than a failure,
-- because both of these are what a script calls when it is being stepped
-- through and neither should stop a script that is not.
lib.breakpoint = function() return false end
lib.debug = function() return false end

-- The `bit` library, which is Mike Pall's and is what a script reaches for when
-- it has to work on a word rather than on a number. Lua 5.1 has doubles and
-- nothing else, so without this a script cannot and two flags together.
--
-- The arithmetic is in Rust and the checking is here, because a Rust callback
-- cannot raise the sentence a real server raises and this can. What counts as a
-- number is what Lua counts as one, so the string '0x10' is sixteen and the
-- boolean true is a mistake. The position in front of the message is the
-- script's own line, which is what a level of three works out to from inside a
-- helper called by a wrapper called by the script.
local bitlib = {}

-- What a bad argument reads like, for every library that raises one.
--
-- The name is the one the call site used rather than the one the function was
-- defined under, which is why `pcall(bit.band, 'x')` complains about a function
-- called `?`. `up` is how many frames stand between here and the wrapper the
-- script called: one when a wrapper raises this itself, two when it goes through
-- a helper of its own. Everything above that is the C function the bridge put in
-- front and then the script, which is the frame a client wants named.
local function argue(up, i, why)
  local at = dbg.getinfo(up + 2, 'n')
  local name = (at and at.name) or '?'
  error("bad argument #" .. i .. " to '" .. name .. "' (" .. why .. ")", up + 3)
end

-- A library failure that is not about an argument, raised from the wrapper
-- itself so that the script's own line goes in front of it.
local function fault(why)
  error(why, 4)
end

-- A failure with no position in front of it at all, which is what the runtime
-- raises from inside a C function since there is no line to point at.
local function bare(why)
  error(why, 0)
end

-- One argument as the word the arithmetic works on.
local function word(i, n, value)
  local why
  if i > n then
    why = 'no value'
  else
    local x = tonumber(value)
    if x ~= nil then return x end
    why = type(value)
  end
  argue(2, i, 'number expected, got ' .. why)
end

-- The three that take as many arguments as a script cares to hand them. The
-- first is a word on its own, so `bit.band(x)` is `bit.tobit(x)`, and the rest
-- are folded into it one pair at a time.
for _, name in ipairs({'band', 'bor', 'bxor'}) do
  local key = 'bit_' .. name
  bitlib[name] = bridge(function(...)
    local n = select('#', ...)
    local acc = raw.bit_tobit(word(1, n, (select(1, ...))))
    for i = 2, n do
      acc = raw[key](acc, word(i, n, (select(i, ...))))
    end
    return acc
  end)
end

-- The three that take one value and nothing else.
for _, name in ipairs({'tobit', 'bnot', 'bswap'}) do
  local key = 'bit_' .. name
  bitlib[name] = bridge(function(...)
    return raw[key](word(1, select('#', ...), (select(1, ...))))
  end)
end

-- The five that take a value and a count. Only the low five bits of the count
-- are read, which is what the hardware does, so a shift of thirty two moves
-- nothing and `bit.lshift(1, 33)` is two.
for _, name in ipairs({'lshift', 'rshift', 'arshift', 'rol', 'ror'}) do
  local key = 'bit_' .. name
  bitlib[name] = bridge(function(...)
    local n = select('#', ...)
    local x = word(1, n, (select(1, ...)))
    return raw[key](x, word(2, n, (select(2, ...))))
  end)
end

-- The one with a default. Eight digits unless a count says otherwise, and a
-- negative count asks for upper case rather than for a different number of
-- them.
bitlib.tohex = bridge(function(...)
  local n = select('#', ...)
  local x = word(1, n, (select(1, ...)))
  local digits = 8
  if n > 1 then digits = word(2, n, (select(2, ...))) end
  return raw.bit_tohex(x, digits)
end)

shield('bit', bitlib)

-- The `cjson` library, which is Mark Pulford's and is the one a real script
-- reaches for more than any other, because a Redis value is a byte string and
-- JSON is how anything with a shape gets into one.
--
-- The encoding and the decoding are in Rust and the settings and the checking
-- are here. What the settings are worth is deliberately plain numbers rather
-- than booleans, because that is what the C holds and because
-- `encode_invalid_numbers` has three states rather than two.
local cjson_defaults = {
  encode_sparse_convert = 0,
  encode_sparse_ratio = 2,
  encode_sparse_safe = 10,
  encode_max_depth = 1000,
  decode_max_depth = 1000,
  encode_invalid_numbers = 0,
  decode_invalid_numbers = 1,
  encode_keep_buffer = 1,
  encode_number_precision = 14,
  decode_array_with_array_mt = 0,
}

-- The largest number a setting will take, which is a C `int` and not anything
-- Lua would have picked.
local cjson_max = 2147483647
local cjson_flags = {'off', 'on'}

-- A whole module table, its own settings and all. This is `cjson.new`, and it
-- is also how the one a script finds under the name is built, because on a real
-- server they are the same function.
local function cjson_new()
  local cfg = {}
  for name, value in next, cjson_defaults do cfg[name] = value end
  local mod = {}

  -- One setting that is a number in a range, checked the way Lua checks an
  -- integer argument, which is to throw away everything past the point. The
  -- range complaint says argument one whichever argument it was, which is the
  -- library's own quirk and shows up in `cjson.encode_sparse_array`.
  local function whole(index, key, low, high, given)
    if given ~= nil then
      local value = tonumber(given)
      if value == nil then argue(2, index, 'number expected, got ' .. type(given)) end
      if value < 0 then value = -floor(-value) else value = floor(value) end
      if value < low or value > high then
        argue(2, 1, 'expected integer between ' .. low .. ' and ' .. high)
      end
      cfg[key] = value
    end
    return cfg[key]
  end

  -- One setting that is a choice. A boolean is the choice by number, a string
  -- is the choice by name, and a number is a string as far as this is
  -- concerned, so `cjson.encode_invalid_numbers(1)` complains about an option
  -- called `1` rather than turning anything on.
  local function flag(index, key, options, given)
    if given ~= nil then
      if type(given) == 'boolean' then
        cfg[key] = given and 1 or 0
      else
        local name = given
        if type(name) == 'number' then name = tostring(name) end
        if type(name) ~= 'string' then
          argue(2, index, 'string expected, got ' .. type(given))
        end
        local found
        for i = 1, #options do
          if options[i] == name then found = i - 1 end
        end
        if found == nil then argue(2, index, "invalid option '" .. name .. "'") end
        cfg[key] = found
      end
    end
    local value = cfg[key]
    if value == 0 or value == 1 then return value == 1 end
    return options[value + 1]
  end

  mod.encode = bridge(function(...)
    if select('#', ...) ~= 1 then argue(1, 1, 'expected 1 argument') end
    local ok, out = raw.cjson_encode((select(1, ...)), cfg)
    if not ok then fault(out) end
    return out
  end)

  mod.decode = bridge(function(...)
    if select('#', ...) ~= 1 then argue(1, 1, 'expected 1 argument') end
    local text = select(1, ...)
    -- A number is a string here, because the C reads the argument with the
    -- checker that coerces one, so `cjson.decode(1)` decodes the text `1`.
    if type(text) == 'number' then text = tostring(text) end
    if type(text) ~= 'string' then
      argue(1, 1, 'string expected, got ' .. type((select(1, ...))))
    end
    local ok, out = raw.cjson_decode(text, cfg)
    if not ok then fault(out) end
    return out
  end)

  mod.encode_sparse_array = bridge(function(...)
    if select('#', ...) > 3 then argue(1, 4, 'found too many arguments') end
    local convert = flag(1, 'encode_sparse_convert', cjson_flags, (select(1, ...)))
    local ratio = whole(2, 'encode_sparse_ratio', 0, cjson_max, (select(2, ...)))
    local safe = whole(3, 'encode_sparse_safe', 0, cjson_max, (select(3, ...)))
    return convert, ratio, safe
  end)

  mod.encode_invalid_numbers = bridge(function(...)
    if select('#', ...) > 1 then argue(1, 2, 'found too many arguments') end
    -- Not a tail call, because the frame this sits in is the one a raise
    -- below counts back from.
    local answer = flag(1, 'encode_invalid_numbers', {'off', 'on', 'null'}, (select(1, ...)))
    return answer
  end)

  mod.encode_max_depth = bridge(function(...)
    if select('#', ...) > 1 then argue(1, 2, 'found too many arguments') end
    -- Not a tail call, because the frame this sits in is the one a raise
    -- below counts back from.
    local answer = whole(1, 'encode_max_depth', 1, cjson_max, (select(1, ...)))
    return answer
  end)

  mod.decode_max_depth = bridge(function(...)
    if select('#', ...) > 1 then argue(1, 2, 'found too many arguments') end
    -- Not a tail call, because the frame this sits in is the one a raise
    -- below counts back from.
    local answer = whole(1, 'decode_max_depth', 1, cjson_max, (select(1, ...)))
    return answer
  end)

  mod.encode_number_precision = bridge(function(...)
    if select('#', ...) > 1 then argue(1, 2, 'found too many arguments') end
    -- Not a tail call, because the frame this sits in is the one a raise
    -- below counts back from.
    local answer = whole(1, 'encode_number_precision', 1, 14, (select(1, ...)))
    return answer
  end)

  -- Whether the encoder keeps its scratch buffer between calls, which this
  -- server does not have to decide because Rust builds a fresh one every time.
  -- Taken and remembered anyway, so that a script that reads it back gets what
  -- it set, which is the whole of what the setting is observable through.
  mod.encode_keep_buffer = bridge(function(...)
    if select('#', ...) > 1 then argue(1, 2, 'found too many arguments') end
    -- Not a tail call, because the frame this sits in is the one a raise
    -- below counts back from.
    local answer = flag(1, 'encode_keep_buffer', cjson_flags, (select(1, ...)))
    return answer
  end)

  mod.decode_invalid_numbers = bridge(function(...)
    if select('#', ...) > 1 then argue(1, 2, 'found too many arguments') end
    -- Not a tail call, because the frame this sits in is the one a raise
    -- below counts back from.
    local answer = flag(1, 'decode_invalid_numbers', cjson_flags, (select(1, ...)))
    return answer
  end)

  mod.decode_array_with_array_mt = bridge(function(...)
    if select('#', ...) > 1 then argue(1, 2, 'found too many arguments') end
    -- Not a tail call, because the frame this sits in is the one a raise
    -- below counts back from.
    local answer = flag(1, 'decode_array_with_array_mt', cjson_flags, (select(1, ...)))
    return answer
  end)

  -- A whole other module with settings of its own, which is a plain table and
  -- not a guarded one, because that is what a real server hands back. Only the
  -- first of the two return values goes out, since the settings are ours.
  mod.new = bridge(function() return (cjson_new()) end)
  -- The value a `null` in the text decodes to. Not `nil`, because `t[k] = nil`
  -- takes the key out of the table and a decoded object would quietly lose
  -- every null field it had.
  mod.null = null
  mod._NAME = 'cjson'
  mod._VERSION = '2.1.0'
  return mod, cfg
end

local cjsonlib, cjson_cfg = cjson_new()
shield('cjson', cjsonlib)

-- The `struct` library, which is Roberto Ierusalimschy's and is the answer to
-- the thing Lua 5.1 is worst at: a string is a byte string, so a script can
-- hold a packed record perfectly well, and the language gives it no way to
-- build one or take one apart.
--
-- All three of these are thinner than the wrappers in the other libraries,
-- because the format is what decides how many arguments there are and what
-- each one has to be, so Rust does the checking and hands back which argument
-- was wrong along with why. Nought means it worked, one is a plain failure and
-- two is a complaint about an argument.
local structlib = {}

structlib.pack = bridge(function(...)
  local kind, value, why = raw.struct_pack(...)
  if kind == 1 then fault(value) end
  if kind == 2 then argue(1, value, why) end
  return value
end)

structlib.unpack = bridge(function(...)
  local kind, value, why = raw.struct_unpack(...)
  if kind == 1 then fault(value) end
  if kind == 2 then argue(1, value, why) end
  return rawunpack(value, 1, value.n)
end)

structlib.size = bridge(function(...)
  local kind, value, why = raw.struct_size(...)
  if kind == 1 then fault(value) end
  if kind == 2 then argue(1, value, why) end
  return value
end)

shield('struct', structlib)

-- cmsgpack, which is Salvatore Sanfilippo's lua-cmsgpack 0.4.0. Same shape as
-- struct: Rust reads the arguments and says which one was wrong, and the raise
-- happens here so the script's own line goes in front of it. The third kind of
-- answer is the one the runtime rather than the library raises, which is a bad
-- table key and comes out with no position on it at all.
local cmsgpacklib = {
  _NAME = 'cmsgpack',
  _VERSION = 'lua-cmsgpack 0.4.0',
  _COPYRIGHT = 'Copyright (C) 2012, Salvatore Sanfilippo',
  _DESCRIPTION = 'MessagePack C implementation for Lua',
}

cmsgpacklib.pack = bridge(function(...)
  local kind, value, why = raw.cmsgpack_pack(...)
  if kind == 1 then fault(value) end
  if kind == 2 then argue(1, value, why) end
  if kind == 3 then bare(value) end
  return value
end)

local function unpacker(name)
  return bridge(function(...)
    local kind, value, why = raw[name](...)
    if kind == 1 then fault(value) end
    if kind == 2 then argue(1, value, why) end
    if kind == 3 then bare(value) end
    return rawunpack(value, 1, value.n)
  end)
end

cmsgpacklib.unpack = unpacker('cmsgpack_unpack')
cmsgpacklib.unpack_one = unpacker('cmsgpack_unpack_one')
cmsgpacklib.unpack_limit = unpacker('cmsgpack_unpack_limit')

shield('cmsgpack', cmsgpacklib)

-- ---------------------------------------------------------------------------
-- Libraries, which is what `FUNCTION LOAD` compiles and `FCALL` runs.
--
-- A library is a chunk that runs once, at load time, and everything it does
-- that outlives that run is a call to `redis.register_function`. What it
-- registers are closures, so a library's own locals are alive for as long as
-- the library is, and reading one is what a library is for.
--
-- A library sees two different worlds. While it is loading it can reach one
-- table with one name on it, and the eight names on that, and nothing else at
-- all: no `tostring`, no `error`, not even `redis.call`. Once it is loaded, its
-- callbacks can reach everything a script can reach except `KEYS`, `ARGV` and
-- the error handler, and the `redis` table they get is missing the three names
-- that only mean something inside `EVAL`. That is one swap of one `__index` on
-- one environment, and it is the same swap a real server does to its own
-- globals metatable.

-- Every library this thread has compiled, by name, with the digest of the code
-- it was compiled from beside it. A library that was replaced by a client on
-- another thread is found here under the old digest and compiled again.
local libs = {}

-- The library being loaded right now, or nil when none is.
local loading = nil

-- The `redis` table a callback gets. The same one a script gets, without the
-- three that answer for a debugger and for a replication mode that has not
-- existed since 7.0, none of which a function has any business calling.
local flib = {}
for name, value in next, lib do flib[name] = value end
flib.breakpoint = nil
flib.debug = nil
flib.replicate_commands = nil

local fproxy, ffront = guarded(flib)

-- The five flags a function can be registered with, as the bits Rust reads
-- them back as.
local flagbit = {
  ['no-writes'] = 1,
  ['allow-oom'] = 2,
  ['allow-stale'] = 4,
  ['no-cluster'] = 8,
  ['allow-cross-slot-keys'] = 16,
}

-- One argument the way `luaGetStringSds` reads one, which takes a number as the
-- digits Lua would print it as and takes nothing else at all. It is why
-- `redis.register_function(1, f)` registers a function called `1`.
local function sdsarg(value)
  local kind = type(value)
  if kind == 'string' then return value end
  if kind == 'number' then return tostring(value) end
  return nil
end

-- The flags table as a number, or nil if it holds anything that is not one of
-- the five. The walk stops at the first hole rather than at the last key, so
-- `{'no-writes', nil, 'allow-oom'}` is one flag and not two.
local function flagmask(t)
  local mask = 0
  local i = 1
  while true do
    local value = t[i]
    if value == nil then return mask end
    local name = sdsarg(value)
    if name == nil then return nil end
    local one = flagbit[lower(name)]
    if one == nil then return nil end
    if mask % (one + one) < one then mask = mask + one end
    i = i + 1
  end
end

-- What a bad call to `redis.register_function` raises, which is a table rather
-- than a string so that no position ends up in front of it. A real server
-- builds the same table for every failure it reports out of a C function, and
-- the `ERR` is on the front for the same reason: the sentence has to carry a
-- code by the time it reaches a client, and this is the only place that knows
-- it does not already have one.
local function reject(why)
  error({err = 'ERR ' .. why}, 0)
end

local function register(...)
  -- Checked first, and reachable, because a library can keep this function in
  -- an upvalue at load time and call it from a callback later.
  if loading == nil then
    reject('redis.register_function can only be called on FUNCTION LOAD command')
  end
  local n = select('#', ...)
  if n < 1 or n > 2 then
    reject('wrong number of arguments to redis.register_function')
  end
  local name, desc, callback, mask
  if n == 1 then
    local named = select(1, ...)
    if type(named) ~= 'table' then
      reject('calling redis.register_function with a single argument is only ' ..
             'applicable to Lua table (representing named arguments).')
    end
    mask = 0
    for key, value in next, named do
      if type(key) ~= 'string' and type(key) ~= 'number' then
        reject('unknown argument given to redis.register_function')
      end
      local which = lower(key)
      if which == 'function_name' then
        name = sdsarg(value)
        if name == nil then
          reject('function_name argument given to redis.register_function must be a string')
        end
      elseif which == 'description' then
        desc = sdsarg(value)
        if desc == nil then
          reject('description argument given to redis.register_function must be a string')
        end
      elseif which == 'callback' then
        if type(value) ~= 'function' then
          reject('callback argument given to redis.register_function must be a function')
        end
        callback = value
      elseif which == 'flags' then
        if type(value) ~= 'table' then
          reject('flags argument to redis.register_function must be a table ' ..
                 'representing function flags')
        end
        mask = flagmask(value)
        if mask == nil then reject('unknown flag given') end
      else
        reject('unknown argument given to redis.register_function')
      end
    end
    if name == nil then reject('redis.register_function must get a function name argument') end
    if callback == nil then reject('redis.register_function must get a callback argument') end
  else
    name = sdsarg((select(1, ...)))
    if name == nil then
      reject('first argument to redis.register_function must be a string')
    end
    callback = select(2, ...)
    if type(callback) ~= 'function' then
      reject('second argument to redis.register_function must be a function')
    end
    mask = 0
  end
  -- The sentence says library where it means function, which is a real server's
  -- own slip: it checks both names with the same function and hands back the
  -- same string.
  if not raw.named(name) then
    reject('Library names can only contain letters, numbers, or underscores(_) ' ..
           'and must be at least one character long')
  end
  if loading.byname[name] ~= nil then
    reject('Function already exists in the library')
  end
  loading.byname[name] = callback
  loading.order[#loading.order + 1] = {name = name, desc = desc, flags = mask}
end

-- The whole of what a library can reach while it is loading. Reading a name
-- that is not on it is the same mistake as reading a global that is not there,
-- which is why `redis.call` during a load complains about a nonexistent global
-- variable called `call` rather than about `redis`.
local loadredis = {
  register_function = register,
  log = lib.log,
  LOG_DEBUG = lib.LOG_DEBUG,
  LOG_VERBOSE = lib.LOG_VERBOSE,
  LOG_NOTICE = lib.LOG_NOTICE,
  LOG_WARNING = lib.LOG_WARNING,
  REDIS_VERSION = lib.REDIS_VERSION,
  REDIS_VERSION_NUM = lib.REDIS_VERSION_NUM,
}

-- What reading a name that is not there says. Only the sentence, because the
-- raise itself has to happen inside whichever metamethod was asked, so that the
-- position on the front of the message is the line that did the reading and not
-- a line in here.
local function missing(name)
  return "Script attempted to access nonexistent global variable '" .. tostring(name) .. "'"
end

local loadfront = {
  __index = function(_, name)
    local value = loadredis[name]
    if value ~= nil then return value end
    error(missing(name), 2)
  end,
  __newindex = readonly,
  __yo_real = loadredis,
}
local loadproxy = setmetatable({}, loadfront)

-- The two tables above go through the same readers and writers the script side
-- goes through, so a function that walks `redis` sees the real names and a
-- function that writes to it is turned away.
local fshielded = {
  {proxy = fproxy, front = ffront, real = flib},
  {proxy = loadproxy, front = loadfront, real = loadredis},
}

local loadapi = {redis = loadproxy}

-- A failure that is a table with an `err` field reaches a script as the string
-- inside it rather than as the table. That is what a real server does and it is
-- what every script that prints the error it caught depends on.
local function flatten(ok, err, ...)
  if ok == false and type(err) == 'table' and type(err.err) == 'string' then
    return false, err.err
  end
  return ok, err, ...
end
rawset(_G, 'pcall', function(...) return flatten(rawpcall(...)) end)
rawset(_G, 'xpcall', function(...) return flatten(rawxpcall(...)) end)

-- The handler every script runs under, under the name a real server gives it.
-- Anything that is not already a table becomes one, which is where the `ERR` in
-- front of a plain `error('boom')` comes from, and the source and line are
-- taken from the stack rather than from the table, so a script that sets them
-- itself does not get to say where it failed.
--
-- The frames that are skipped are the ones that are not the script's. There are
-- up to two of them: `error` itself, which is a C function and is on the stack
-- whenever anything raised on purpose, and the library function that called it,
-- which is a line in here. A real server skips one C frame and is done, because
-- over there the library function is the C frame. What a client wants to know
-- either way is where in its own script the call was.
local function handler(err)
  local i
  for level = 2, 8 do
    local at = dbg.getinfo(level, 'nSl')
    if not at then break end
    if at.what ~= 'C' and at.source ~= '@lua_prelude' then
      i = at
      break
    end
  end
  if type(err) ~= 'table' then
    err = {err = 'ERR ' .. tostring(err)}
  end
  if i then
    err.source = i.source
    err.line = i.currentline
  end
  return err
end
rawset(_G, '__redis__err__handler', handler)

-- The globals a script writes to, which is an empty table that stays empty.
--
-- Reading falls through to the real global table, so `KEYS` and `redis` and
-- `tonumber` are all there. Writing never falls through, because the key is
-- never present, so `x = 1` and `redis = 1` and `pcall = 1` all end at the
-- guard. That last one is the case a metatable on the real global table cannot
-- catch on its own, and it is the case a script that wants to break the next
-- script on this thread would reach for.
local env = setmetatable({}, {__index = _G, __newindex = readonly})

-- A chunk a script compiles for itself runs in the same sandbox the script
-- does. Lua 5.1 would otherwise hand it the real global table, which is one
-- `loadstring` away from everything the paragraph above is about.
rawset(_G, 'loadstring', function(...)
  local f, why = rawloadstring(...)
  if f then setfenv(f, env) end
  return f, why
end)
rawset(_G, 'load', function(...)
  local f, why = rawload(...)
  if f then setfenv(f, env) end
  return f, why
end)

-- Reading a global that is not there is a mistake rather than a nil, which is a
-- real server's rule and catches the misspelled name that would otherwise make
-- a script quietly do nothing. The guard on writing is the one the sandbox
-- above already covers, and is here for the `_G.x = 1` spelling.
--
-- It catches a name that is not there and not a name that is, because
-- `__newindex` only fires for keys a table does not have. The `redis` table a
-- few paragraphs up closes that hole by hiding behind an empty proxy, and this
-- table cannot, because it is the environment every script runs in and a proxy
-- would put a metatable lookup in front of every read of every global. So
-- `_G.pcall = 1` lands, which is D-103, and `restore` below takes it back out
-- before the next script sees the table.
local guard = {
  __index = function(_, name) error(missing(name), 2) end,
  __newindex = readonly,
}
setmetatable(_G, guard)

-- The three base functions that write to a table without asking the table.
--
-- A guard in a metatable is the only kind Lua 5.1 offers and `rawset` exists to
-- walk past it, so a script that has read the manual gets to `rawset(redis,
-- 'call', 1)` in one line. `setmetatable` is the same hole one step further
-- back: take the guard off and everything under it is writable. Both of them
-- are functions on the global table, and the global table is ours.
--
-- `getmetatable` is here for what it hands out rather than what it changes. A
-- real server answers `nil` for the `redis` table and a table for `_G`, and the
-- one it answers for `_G` cannot be written to either, so what goes out here is
-- an empty table with a guard of its own rather than the real one.
local shadow = setmetatable({}, {__newindex = readonly})
local hidden = {}
hidden[env] = false
hidden[shadow] = false
hidden[_G] = shadow
for _, one in ipairs(shielded) do hidden[one.proxy] = false end
for _, one in ipairs(fshielded) do hidden[one.proxy] = false end

local rawrawset, rawgetmeta, rawsetmeta = rawset, getmetatable, setmetatable
rawset(_G, 'rawset', function(t, name, value)
  if hidden[t] ~= nil then error('Attempt to modify a readonly table', 0) end
  return rawrawset(t, name, value)
end)
rawset(_G, 'setmetatable', function(t, meta)
  if hidden[t] ~= nil then error('Attempt to modify a readonly table', 0) end
  return rawsetmeta(t, meta)
end)
rawset(_G, 'getmetatable', function(t)
  local answer = hidden[t]
  if answer ~= nil then
    if answer == false then return nil end
    return answer
  end
  return rawgetmeta(t)
end)

-- The three base functions that read a table without asking the table.
--
-- The empty proxy that closes `redis.call = 1` closes `pairs(redis)` with it,
-- since a traversal of an empty table finds nothing where a real server's
-- traversal finds every name, and the same goes for `next` and `rawget`. It
-- does not have to. The proxy is ours and the table behind it is ours, so a
-- read that walks round a metatable can be pointed at the real table while
-- every write still lands on the guard. A script gets back exactly what a real
-- server hands it and can still do nothing with it.
--
-- Only these three, because they are the only readers in the base library that
-- skip `__index`. Everything else already goes through the metatable and
-- already sees the real names.
local mirror = {}
for _, one in ipairs(shielded) do mirror[one.proxy] = one.real end
for _, one in ipairs(fshielded) do mirror[one.proxy] = one.real end

local rawrawget, rawnext, rawpairs = rawget, next, pairs
rawset(_G, 'rawget', function(t, name)
  return rawrawget(mirror[t] or t, name)
end)
rawset(_G, 'next', function(t, name)
  return rawnext(mirror[t] or t, name)
end)
rawset(_G, 'pairs', function(t)
  local real = mirror[t]
  if real == nil then return rawpairs(t) end
  -- The raw one, so that a script cannot make the loop lie by replacing the
  -- global `next` it would otherwise have been handed.
  return rawnext, real, nil
end)

-- What the global table looks like before any script has run. Taken last, so it
-- holds every name above, and used to put the table back afterwards.
local pristine = {}
for name, value in next, _G do pristine[name] = value end

-- Empty a table that is supposed to be empty already.
--
-- `rawset` is not something the guards can see, so `rawset(redis, 'call', 1)`
-- and `rawset(_G, 'x', 1)` both land. They land somewhere that is thrown away
-- at the end of the script rather than somewhere the next script reads.
local function wipe(t)
  local name = next(t)
  while name ~= nil do
    rawset(t, name, nil)
    name = next(t)
  end
end

-- The global table as it was, which is thirty odd comparisons and almost never
-- a write. `KEYS` and `ARGV` are left alone because Rust writes them fresh for
-- every script anyway, and the two metatables go back on because a script that
-- found a way past the wrappers above would leave the next one without them.
local function restore()
  local name, value = next(_G)
  while name ~= nil do
    local want = pristine[name]
    if want ~= nil then
      if value ~= want then rawset(_G, name, want) end
    elseif name ~= 'KEYS' and name ~= 'ARGV' then
      rawset(_G, name, nil)
    end
    name, value = next(_G, name)
  end
  setmetatable(_G, guard)
  for _, one in ipairs(shielded) do setmetatable(one.proxy, one.front) end
  -- The `cjson` settings go back to what they ship as. A real server keeps them
  -- for the life of the process because it has one interpreter, and this one
  -- has an interpreter for every thread, so keeping them would mean a script
  -- got whatever the last script on the same thread happened to leave. Handing
  -- every script the defaults is the only answer that is the same twice, and it
  -- is D-105.
  for name, value in next, cjson_defaults do cjson_cfg[name] = value end
end

local function run(chunk)
  setfenv(chunk, env)
  local ok, result = rawxpcall(chunk, handler)
  wipe(env)
  for _, one in ipairs(shielded) do wipe(one.proxy) end
  restore()
  return ok, result
end

-- ---------------------------------------------------------------------------
-- The second set of globals, which is what makes a function a function.
--
-- Everything above this line is one global table with one set of names on it. A
-- function does not run against that table. It runs against this one, which is
-- a copy of it taken before any script ran, without the error handler a script
-- needs and with a `redis` table that is missing the three names only `EVAL`
-- answers for. `KEYS` and `ARGV` are not on it and never will be, because a
-- function is handed its keys and its arguments as the two arguments of its
-- callback and reading a global called `KEYS` is a mistake a real server
-- reports as one.
--
-- A real server gets here by keeping a whole second interpreter. This gets here
-- by keeping a second table, which costs one table and gives the same answers,
-- and which matters because a library's callbacks and a script's chunks then
-- share a garbage collector and a string table instead of doubling both.
local fglobals = {}
for name, value in next, pristine do fglobals[name] = value end
fglobals.__redis__err__handler = nil
fglobals.redis = fproxy
setmetatable(fglobals, guard)
hidden[fglobals] = shadow

-- `_G` is a proxy for the same reason the `redis` table is one. On the script
-- side `_G` cannot be, because it is the table every global read goes through
-- and a proxy would put a metatable lookup in front of all of them, which is
-- why `_G.pcall = 1` lands there and is D-103. Here nothing reads through it:
-- a function reads its globals through the environment below, and `_G` is only
-- ever the long way round. So it can be empty, and `_G.pcall = 1` ends at a
-- guard the way a real server ends it.
local gproxy, gfront = guarded(fglobals)
fglobals._G = gproxy
fshielded[#fshielded + 1] = {proxy = gproxy, front = gfront, real = fglobals}
hidden[gproxy] = shadow
mirror[gproxy] = fglobals

local fpristine = {}
for name, value in next, fglobals do fpristine[name] = value end

-- The environment a library chunk and every callback it makes runs in.
--
-- One table, and an `__index` that answers from a different place depending on
-- whether a load is running. That is the whole of the difference between load
-- time and call time: during a load the only name there is is `redis`, and the
-- only names on that are the eight a library needs to describe itself. After
-- the load the same environment answers from the globals above, which is what
-- lets a callback that was written during the load call `redis.call` when it is
-- finally called.
--
-- It has to be one table rather than two, because a chunk's environment is
-- fixed by `setfenv` at load time and every closure the chunk makes inherits
-- it. Swapping what the environment resolves against is the only way to give
-- the closures a different world than the chunk that made them, and it is what
-- a real server does to its own globals metatable for the same reason.
local libenv = setmetatable({}, {
  __index = function(_, name)
    local value = rawrawget(loading and loadapi or fglobals, name)
    if value ~= nil then return value end
    error(missing(name), 2)
  end,
  __newindex = readonly,
})
hidden[libenv] = false

-- The function globals as they were, for the same reason `restore` exists: the
-- `_G.pcall = 1` spelling lands because `__newindex` does not fire for a key
-- that is already there, and the next call on this thread must not find it.
local function frestore()
  local name, value = next(fglobals)
  while name ~= nil do
    local want = fpristine[name]
    if want ~= nil then
      if value ~= want then rawset(fglobals, name, want) end
    else
      rawset(fglobals, name, nil)
    end
    name, value = next(fglobals, name)
  end
  setmetatable(fglobals, guard)
  for _, one in ipairs(fshielded) do setmetatable(one.proxy, one.front) end
  for name, value in next, cjson_defaults do cjson_cfg[name] = value end
end

local function tidy()
  wipe(libenv)
  for _, one in ipairs(fshielded) do wipe(one.proxy) end
  frestore()
end

-- Compile a library and run it once, which is the only time its own code runs.
--
-- The digest of the code is kept beside the callbacks so that a library another
-- thread replaced is noticed here: the caller asks `holds` first, and a digest
-- that does not match means this thread compiles the new code before it calls
-- anything.
local function loadlib(name, sha, code)
  local chunk, why = rawloadstring(code, '@user_function')
  if not chunk then
    tidy()
    return false, 'Error compiling function: ' .. tostring(why)
  end
  setfenv(chunk, libenv)
  local outer = loading
  loading = {byname = {}, order = {}}
  local ok, err = rawpcall(chunk)
  local built = loading
  loading = outer
  if not ok then
    tidy()
    -- A table with an `err` field is one of the sentences `register` turned the
    -- call away with, and it already carries a code. A string is Lua's own,
    -- with the line it happened on already on the front of it, and it needs
    -- one. That is the whole reason a registration failure reads `ERR ERR` and
    -- a runtime failure does not.
    local said
    if type(err) == 'table' and type(err.err) == 'string' then
      said = err.err
    else
      said = 'ERR ' .. tostring(err)
    end
    return false, 'Error registering functions: ' .. said
  end
  tidy()
  if #built.order == 0 then
    return false, 'No functions registered'
  end
  libs[name] = {sha = sha, byname = built.byname}
  return true, built.order
end

-- Whether this thread has the library the server is holding, rather than an
-- older one under the same name.
local function holds(name, sha)
  local one = libs[name]
  return one ~= nil and one.sha == sha
end

local function fcall(name, fname, keys, args)
  local callback = libs[name].byname[fname]
  -- `xpcall` in 5.1 takes no arguments for the function it calls, so the call
  -- goes inside a closure. The result goes through a local on the way out
  -- because `return callback(...)` would be a tail call, and a tail call has no
  -- frame for the error handler to read a line number off.
  local ok, result = rawxpcall(function()
    local answer = callback(keys, args)
    return answer
  end, handler)
  tidy()
  return ok, result
end

return {run = run, load = loadlib, holds = holds, fcall = fcall, raw = raw}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// The names on a table, sorted, as one line.
    fn names(t: &Table) -> String {
        let mut found: Vec<String> = t
            .pairs::<String, Value>()
            .map(|pair| pair.expect("pair").0)
            .collect();
        found.sort();
        found.join(" ")
    }

    #[test]
    fn the_interpreter_builds() {
        interpreter().expect("prelude");
    }

    #[test]
    fn the_global_table_holds_what_a_script_is_allowed_to_reach() {
        let lua = interpreter().expect("prelude");
        // `KEYS` and `ARGV` are the two a run puts there, so a fresh
        // interpreter is this list and a running script is this list plus two.
        assert_eq!(
            names(&lua.globals()),
            "_G _VERSION __redis__err__handler assert bit cjson cmsgpack collectgarbage \
             coroutine error gcinfo getmetatable ipairs load loadstring math next os pairs \
             pcall rawequal rawget rawset redis select setmetatable string struct table \
             tonumber tostring type unpack xpcall"
        );
        // The libraries that reach outside the process are not there at all
        // rather than there and stubbed, so a script that wants one finds out.
        for gone in [
            "io",
            "os.execute",
            "package",
            "require",
            "dofile",
            "loadfile",
        ] {
            assert!(
                lua.globals().raw_get::<Value>(gone).expect("read").is_nil(),
                "{gone} is still reachable"
            );
        }
        // `os` is one clock and nothing else, which is what a script uses it
        // for and is the only part of it that cannot see the filesystem.
        let os: Table = lua.globals().get("os").expect("os");
        assert_eq!(names(&os), "clock");
        assert_eq!(
            lua.globals().get::<String>("_VERSION").expect("version"),
            "Lua 5.1"
        );
    }

    #[test]
    fn the_redis_table_holds_the_names_a_script_calls() {
        let lua = interpreter().expect("prelude");
        // The table under the name is the empty guard with the real one behind
        // it, and mlua walks it with the raw `lua_next` rather than the wrapped
        // `pairs` the prelude puts on the global table, so this is the one view
        // in the world that finds nothing. A script gets the other one, and the
        // end to end tests are where that is checked.
        let seen: Table = lua.globals().get("redis").expect("redis");
        assert_eq!(names(&seen), "");
        let front = seen.metatable().expect("front");
        let lib: Table = front.get("__index").expect("index");
        assert_eq!(
            names(&lib),
            "LOG_DEBUG LOG_NOTICE LOG_VERBOSE LOG_WARNING REDIS_VERSION \
             REDIS_VERSION_NUM REPL_ALL REPL_AOF REPL_NONE REPL_REPLICA REPL_SLAVE \
             acl_check_cmd breakpoint call debug error_reply log pcall replicate_commands \
             set_repl setresp sha1hex status_reply"
        );
    }

    #[test]
    fn a_body_is_checked_for_being_lua_before_it_is_remembered() {
        assert!(compiles(b"return 1").is_ok());
        assert!(compiles(b"").is_ok());
        let why = compiles(b"this is not lua").expect_err("refused");
        assert_eq!(why, "user_script:1: '=' expected near 'is'");
        // The name in the message is the one a client would see the failure
        // under later, not the name of anything of ours.
        assert!(
            !compiles(b"return {")
                .expect_err("refused")
                .contains("prelude")
        );
    }

    #[test]
    fn the_cache_is_keyed_by_the_digest_of_the_body() {
        let mut held = Scripts::default();
        let sha = held.add(b"return 1");
        assert_eq!(&sha, b"e0e1f9fabfc9d4800c877a703b823ac0578ff8db");
        assert!(held.has(&sha));
        assert_eq!(held.body(&sha).as_deref(), Some(&b"return 1"[..]));
        // Adding the same body twice is the same name and one entry.
        assert_eq!(held.add(b"return 1"), sha);
        let other = held.add(b"return 2");
        assert_ne!(other, sha);
        assert!(held.has(&other));

        held.wipe();
        assert!(!held.has(&sha));
        assert!(held.body(&sha).is_none());
    }

    #[test]
    fn a_line_that_would_end_early_is_scrubbed() {
        assert_eq!(scrub(b"plain"), b"plain");
        assert_eq!(scrub(b"a\r\nb"), b"a  b");
        assert_eq!(scrub(b"\n"), b" ");
    }
}
