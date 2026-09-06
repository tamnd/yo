//! The bits `struct` and `cmsgpack` share, which is how a failure gets back to
//! the prelude and how an argument is read the way the C reads one.
//!
//! Both of those libraries do their argument checking in Rust rather than in the
//! prelude, because in both the first argument is what decides how many other
//! arguments there are and what each one has to be. A format string says a
//! number and then a string, and a message pack stream says whatever it says.
//! So Rust reads the arguments, works out which one was wrong, and hands the
//! prelude a small answer it can turn into the raise. Doing the raise in Lua is
//! what puts the script's own line in front of the message, which is the part a
//! Rust frame cannot do for itself.

use super::cjson::typename;
use mlua::{Lua, Value};

/// What went wrong, in the shape the prelude needs to raise it.
pub(super) enum Stop {
    /// A plain failure, which comes out with the script's position in front.
    /// This is what `luaL_error` gives.
    Plain(Vec<u8>),
    /// A complaint about one argument, which also names the position and the
    /// function the call site used. This is what `luaL_argerror` gives.
    Arg(usize, Vec<u8>),
    /// A failure with nothing in front of it at all, which is what the runtime
    /// raises from inside a C function since there is no line to point at.
    Bare(Vec<u8>),
}

pub(super) type Answer<T> = Result<T, Stop>;

/// What Lua says when an argument is the wrong type.
///
/// `missing` is what the message calls an argument that is not there at all,
/// and it is not the same everywhere: `struct.pack` pushes a nil of its own
/// before it starts reading, so the first argument it runs out of is a nil
/// rather than nothing, and it says so.
pub(super) fn expected(want: &str, what: &str) -> Vec<u8> {
    format!("{want} expected, got {what}").into_bytes()
}

/// One argument as a string, the way `luaL_checklstring` reads one.
///
/// A number counts as a string, because Lua converts one on the way in.
pub(super) fn text(lua: &Lua, args: &[Value], i: usize, missing: &str) -> Answer<Vec<u8>> {
    let Some(value) = args.get(i - 1) else {
        return Err(Stop::Arg(i, expected("string", missing)));
    };
    match lua.coerce_string(value.clone()) {
        Ok(Some(s)) => Ok(s.as_bytes().to_vec()),
        _ => Err(Stop::Arg(i, expected("string", typename(value)))),
    }
}

/// One argument as a number, the way `luaL_checknumber` reads one.
///
/// A string that reads as a number counts, which is the same courtesy Lua
/// extends everywhere else.
pub(super) fn number(lua: &Lua, args: &[Value], i: usize, missing: &str) -> Answer<f64> {
    let Some(value) = args.get(i - 1) else {
        return Err(Stop::Arg(i, expected("number", missing)));
    };
    match lua.coerce_number(value.clone()) {
        Ok(Some(n)) => Ok(n),
        _ => Err(Stop::Arg(i, expected("number", typename(value)))),
    }
}

/// The answer the prelude reads, where nought is a result, one is a plain
/// failure, two is a complaint about a numbered argument and three is a failure
/// with no position in front of it.
pub(super) fn answer(lua: &Lua, given: Answer<Value>) -> mlua::Result<(i64, Value, Value)> {
    match given {
        Ok(value) => Ok((0, value, Value::Nil)),
        Err(Stop::Plain(why)) => Ok((1, Value::String(lua.create_string(&why)?), Value::Nil)),
        Err(Stop::Arg(at, why)) => Ok((
            2,
            Value::Integer(at as i64),
            Value::String(lua.create_string(&why)?),
        )),
        Err(Stop::Bare(why)) => Ok((3, Value::String(lua.create_string(&why)?), Value::Nil)),
    }
}

/// The real table behind a library's guard, when there is one.
///
/// Every library a script can see is an empty table with a metatable pointing
/// at the real one, so that a write can be turned away. From Lua that is
/// invisible, because `next`, `pairs` and `rawget` all step through it. From
/// Rust it is not, so anything that walks a table the way the C walks one has
/// to step through it here or it sees a whole library as empty. A real server
/// has no guard and no proxy, so the answer to `cjson.encode(bit)` there is a
/// complaint about a function rather than an empty object.
///
/// The mark is on the metatable rather than on the table, so a script cannot
/// see it and cannot put one on a table of its own.
pub(super) fn behind(t: &mlua::Table) -> Option<mlua::Table> {
    t.metatable()?
        .raw_get::<Option<mlua::Table>>("__yo_real")
        .ok()
        .flatten()
}
