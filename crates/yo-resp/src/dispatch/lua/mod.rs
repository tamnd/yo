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
//! # Where the error handling lives
//!
//! In Lua, not here, because that is where a real server puts it. The prelude
//! installs `__redis__err__handler` under the same name Redis uses, and every
//! script runs under it. It is what turns a raised string into a table with an
//! `err` field, and it is what attaches the source and line a client reads at
//! the end of the message. Doing it in Rust would mean getting the raised Lua
//! value across the boundary, and a raised value can be any Lua value at all.

mod api;
mod convert;
mod sha1;

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
    /// The name the message ends with, which is the digest of the body whether
    /// the client sent the body or the name.
    pub sha: &'a [u8; 40],
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
        Ok((false, value)) => out.error(&failure(&value, ask.sha)),
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
fn failure(value: &Value, sha: &[u8; 40]) -> Vec<u8> {
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
    out.extend_from_slice(sha);
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
    // The prelude hands back four things. `run` is the function every script is
    // put through. `raw` is the private table its wrappers reach us through,
    // and it is the reason any of this works: it is a local of the prelude
    // chunk, so it is reachable from the wrappers and from the registry and
    // from nowhere a script can get to. `lib` is the real `redis` table, which
    // by then is behind a proxy. `written` holds the library functions that go
    // out through a C function of ours rather than straight onto `lib`.
    let boot: Table = lua.load(PRELUDE).set_name("@lua_prelude").call(())?;
    let raw: Table = boot.raw_get("raw")?;
    api::statics(&lua, &raw)?;
    let lib: Table = boot.raw_get("lib")?;
    let written: Table = boot.raw_get("written")?;
    api::bridges(&lua, &lib, &written)?;
    lua.set_named_registry_value("yo_run", boot.raw_get::<mlua::Function>("run")?)?;
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
-- Held here so that taking them off the global table does not take them away
-- from the sandbox itself.
local dbg = debug
local rawpcall, rawxpcall = pcall, xpcall
local error, type, tostring, tonumber = error, type, tostring, tonumber
local setmetatable, rawset, select, next = setmetatable, rawset, select, next
local getmetatable, setfenv, ipairs = getmetatable, setfenv, ipairs
local rawloadstring, rawload = loadstring, load
local sub, find, concat = string.sub, string.find, table.concat

-- Where the functions Rust hands over live. A local of this chunk and an
-- upvalue of everything below, so a script cannot reach it and Rust can.
local raw = {}

-- The library functions that go out through a C function of ours rather than
-- straight onto the table. Every one of them can fail, and that is the whole
-- reason for the detour: see the comment on `bridges` in `api.rs`.
local written = {}

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

-- `redis` as a script sees it is an empty table in front of the real one.
--
-- It has to be empty. Lua 5.1 only calls `__newindex` for a key a table does
-- not already have, so a guard on a table that holds its own functions guards
-- nothing: `redis.call = 1` would go straight through and the next script on
-- this thread would find `redis.call` was a number. A real server closes that
-- with a change to its own copy of Lua that vanilla 5.1 does not have. An empty
-- table in front costs one more lookup per call and closes it here. What it
-- would cost a script is `pairs(redis)`, and the three readers further down
-- give that back.
local front = {__index = lib, __newindex = readonly}
local proxy = setmetatable({}, front)
rawset(_G, 'redis', proxy)

-- `redis.pcall` is the one that reaches the server and it answers a table with
-- an `err` field when the command failed. `redis.call` is the same call with
-- that turned back into a failure, which is the only difference between them.
lib.pcall = function(...) return raw.pcall(...) end
written.call = function(...)
  local reply = raw.pcall(...)
  if type(reply) == 'table' and reply.err ~= nil then
    error(reply)
  end
  return reply
end

-- The digest of one value. Anything that is not a string or a number hashes as
-- nothing at all, so `redis.sha1hex({})` is the digest of the empty string, and
-- that is a real server's answer and not a shortcut taken here.
written.sha1hex = function(...)
  if select('#', ...) ~= 1 then error('wrong number of arguments', 0) end
  return raw.sha1hex((select(1, ...)))
end

-- Which protocol the replies a script reads are written in. It moves what the
-- script sees and not what the client is answered in, so a script can read a
-- map as a map while still replying to a RESP2 client.
written.setresp = function(...)
  if select('#', ...) ~= 1 then error('redis.setresp() requires one argument.', 0) end
  local version = select(1, ...)
  if version ~= 2 and version ~= 3 then error('RESP version must be 2 or 3.', 0) end
  raw.setresp(version == 3)
end

-- A line in the server's log. Everything after the level is joined with spaces,
-- which is why `redis.log(redis.LOG_WARNING, 'k', key)` reads the way it does.
written.log = function(...)
  local n = select('#', ...)
  if n < 2 then error('redis.log() requires two arguments or more.', 0) end
  local level = select(1, ...)
  if type(level) ~= 'number' then error('First argument must be a number (log level).', 0) end
  local parts = {}
  for i = 2, n do parts[i - 1] = tostring((select(i, ...))) end
  raw.log(level, concat(parts, ' '))
end

-- Which of a script's effects reach the replica and the log. Accepted and then
-- ignored, because every effect this server produces goes to both. Kept because
-- a script that narrows replication and then widens it again would otherwise
-- fail on the first call, and registered as D-99.
written.set_repl = function(...)
  if select('#', ...) ~= 1 then error('redis.set_repl() requires one argument.', 0) end
  -- A value that is not a number counts as zero rather than as a mistake, which
  -- is why `redis.set_repl('x')` is accepted and `redis.set_repl(9)` is not.
  local flags = tonumber((select(1, ...))) or 0
  if flags < 0 or flags > 3 then
    error('Invalid replication flags. Use REPL_AOF, REPL_REPLICA, REPL_ALL or REPL_NONE.', 0)
  end
end

-- Whether the connection may run a command. There are no users here yet, so the
-- only thing this can answer is whether the command exists and whether it was
-- handed the right number of arguments, and it says yes to everything that gets
-- past those two. Registered as D-100.
written.acl_check_cmd = function(...)
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
end

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
  __index = function(_, name)
    error("Script attempted to access nonexistent global variable '" .. tostring(name) .. "'", 2)
  end,
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
hidden[proxy] = false
hidden[env] = false
hidden[shadow] = false
hidden[_G] = shadow

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
mirror[proxy] = lib

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
  setmetatable(proxy, front)
end

local function run(chunk)
  setfenv(chunk, env)
  local ok, result = rawxpcall(chunk, handler)
  wipe(env)
  wipe(proxy)
  restore()
  return ok, result
end

return {run = run, raw = raw, lib = lib, written = written}
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
            "_G _VERSION __redis__err__handler assert collectgarbage coroutine error \
             gcinfo getmetatable ipairs load loadstring math next os pairs pcall rawequal \
             rawget rawset redis select setmetatable string table tonumber tostring type \
             unpack xpcall"
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
