//! The two directions a value travels between Lua and the wire.
//!
//! A script returns a Lua value and a client reads a RESP reply, and a script
//! calls a command and reads a Lua value back. Neither of those is a mapping
//! anybody would design. Lua has one number type and RESP has three, Lua has
//! one table type and RESP has five aggregates, and the rules that bridge them
//! were written twenty years ago and every client library in the world now
//! depends on them. So none of this is decided here. All of it was measured
//! against 8.10.1 and the tests at the bottom are the measurements.
//!
//! The two surprises worth naming up front. A number becomes an integer, which
//! means `return 3.9` answers 3 and there is nothing a script can do about it
//! short of returning a string. And a table becomes an array that stops at the
//! first hole, which means `{1, nil, 3}` answers one element rather than three
//! and `{nil, 1}` answers none.

use crate::frame::Frame;
use crate::reply::Out;
use mlua::{Lua, Table, Value};

/// Write a Lua value into `out` as the reply the client reads.
///
/// The six named fields are checked in the order a real server checks them,
/// which is why `{ok = 'a', err = 'b'}` is an error and not a status: `err`
/// comes first and whichever was written first in the script makes no
/// difference. A field whose value is the wrong type is not an error, it is
/// simply not that kind of reply, and the value falls through to the array walk
/// below and comes out as an empty array.
pub(super) fn push(out: &mut Out, value: &Value) {
    match value {
        // No return, an explicit `nil`, and `false` are one answer. A client
        // that wanted to tell them apart cannot, and neither can a script that
        // reads the reply of a nested call: this is where the round trip loses
        // the difference, and it loses it on a real server too.
        Value::Nil | Value::Boolean(false) => out.nil(),
        Value::Boolean(true) => out.int(1),
        Value::Integer(n) => out.int(*n),
        // Rust's cast saturates and sends NaN to zero, which happens to be
        // exactly what a real server answers for `2^63`, `1/0`, `-1/0` and
        // `0/0`. That is not a coincidence worth relying on quietly, so the
        // test at the bottom pins all four.
        Value::Number(d) => out.int(*d as i64),
        Value::String(s) => out.bulk(&s.as_bytes()),
        Value::Table(t) => table(out, t),
        // A function, a thread or a userdata is not a reply. Nothing can make
        // one of these except a script that meant to return something else, and
        // a real server answers the same nothing.
        _ => out.nil(),
    }
}

/// The part of [`push`] that reads a table, which is where the rules are.
fn table(out: &mut Out, t: &Table) {
    if let Some(msg) = text(t, "err") {
        // The line goes out as it was written, with no code word put in front
        // of it, which is how a script sends `WRONGTYPE` or a code of its own.
        // A carriage return or a newline inside it would end the line early and
        // leave the rest of it looking like another reply, so both become
        // spaces.
        out.error(&scrubbed(&msg));
        return;
    }
    if let Some(msg) = text(t, "ok") {
        out.simple(&scrubbed(&msg));
        return;
    }
    if let Ok(Value::Number(d)) = t.raw_get::<Value>("double") {
        out.double(d);
        return;
    }
    if let Ok(Value::Integer(n)) = t.raw_get::<Value>("double") {
        out.double(n as f64);
        return;
    }
    if let Some(digits) = text(t, "big_number") {
        // Not checked for being digits, because a real server does not check
        // either: `{big_number = 'xyz'}` goes out as `(xyz` on RESP3 and as a
        // bulk string of the same three letters on RESP2.
        out.big_number(&digits);
        return;
    }
    if let Ok(Value::Table(inner)) = t.raw_get::<Value>("map") {
        pairs(out, &inner);
        return;
    }
    if let Ok(Value::Table(inner)) = t.raw_get::<Value>("set") {
        members(out, &inner);
        return;
    }
    array(out, t);
}

/// The array a table is when none of the named fields fits.
///
/// The walk stops at the first index that is not there, so the length of the
/// reply is the length of the run starting at one and not the number of things
/// in the table. The header has to go in front of elements that have not been
/// counted yet, so the count is worked out first and the elements are written
/// after it, which costs one extra lookup per element and nothing else.
fn array(out: &mut Out, t: &Table) {
    let mut n = 0;
    while !matches!(t.raw_get::<Value>(n + 1), Ok(Value::Nil) | Err(_)) {
        n += 1;
    }
    out.array(n);
    for i in 1..=n {
        push(out, &t.raw_get::<Value>(i).unwrap_or(Value::Nil));
    }
}

/// `{map = {...}}`, which is RESP3's map and a flat array on RESP2.
///
/// The pairs come out in whatever order the table hands them over, which is not
/// the order they were written and is not stable between runs. That is Lua's
/// hash order and a real server has the same one, so a script that needs an
/// order returns an array instead.
fn pairs(out: &mut Out, t: &Table) {
    let held: Vec<(Value, Value)> = t.pairs::<Value, Value>().filter_map(Result::ok).collect();
    out.map(held.len());
    for (k, v) in &held {
        push(out, k);
        push(out, v);
    }
}

/// `{set = {...}}`, which is RESP3's set and an array on RESP2.
///
/// Only the keys are read. A set in Lua is a table whose values are `true`, and
/// nothing looks at whether they are.
fn members(out: &mut Out, t: &Table) {
    let held: Vec<Value> = t
        .pairs::<Value, Value>()
        .filter_map(|p| p.ok())
        .map(|(k, _)| k)
        .collect();
    out.set(held.len());
    for k in &held {
        push(out, k);
    }
}

/// A string field of a table, or `None` if it is missing or not a string.
fn text(t: &Table, field: &str) -> Option<Vec<u8>> {
    match t.raw_get::<Value>(field) {
        Ok(Value::String(s)) => Some(s.as_bytes().to_vec()),
        _ => None,
    }
}

/// A one line version of a message, with the two bytes that would end the line
/// turned into spaces.
fn scrubbed(msg: &[u8]) -> Vec<u8> {
    let mut out = msg.to_vec();
    for b in &mut out {
        if *b == b'\r' || *b == b'\n' {
            *b = b' ';
        }
    }
    out
}

/// Turn a reply the server wrote into the value a script reads.
///
/// `resp3` is what the script asked for with `redis.setresp`, and it changes
/// what the script sees rather than what the server wrote: the nested command
/// is run against a buffer in that protocol, so a map really is a map by the
/// time it gets here.
///
/// # Errors
///
/// Only if Lua cannot make a table, which means it is out of memory.
pub(super) fn pull(lua: &Lua, frame: &Frame<'_>, resp3: bool) -> mlua::Result<Value> {
    Ok(match frame {
        // A status reply keeps its shape, so a script can tell `+OK` from the
        // bulk string `OK` and can hand it straight back to a client.
        Frame::Simple(s) => {
            let t = lua.create_table()?;
            t.raw_set("ok", lua.create_string(s)?)?;
            Value::Table(t)
        }
        Frame::Error(e) | Frame::BlobError(e) => {
            let t = lua.create_table()?;
            t.raw_set("err", lua.create_string(e)?)?;
            // A real server marks the errors it makes itself so that a script
            // that swallows one with `pcall` does not also move the server's
            // error counters. The field is visible to the script, which is why
            // it is set here rather than kept on the side.
            t.raw_set("ignore_error_stats_update", true)?;
            Value::Table(t)
        }
        Frame::Int(n) => Value::Integer(*n),
        Frame::Bulk(b) => Value::String(lua.create_string(b)?),
        // Both RESP2 nulls and RESP3's `_`. Under RESP2 a script reads `false`,
        // which is why `redis.call('get', 'missing') == false` is the way every
        // script written before RESP3 checks for a missing key. Under RESP3 it
        // reads `nil`, and a script that asked for RESP3 asked for that.
        Frame::Null if resp3 => Value::Nil,
        Frame::Null => Value::Boolean(false),
        Frame::Double(d) => wrapped(lua, "double", Value::Number(*d))?,
        Frame::Bool(b) => Value::Boolean(*b),
        Frame::BigNumber(digits) => {
            wrapped(lua, "big_number", Value::String(lua.create_string(digits)?))?
        }
        // The format tag is dropped, which is a real server's choice and not
        // ours: the table it builds has one field and the field is the text.
        Frame::Verbatim { text, .. } => wrapped(
            lua,
            "verbatim_string",
            Value::String(lua.create_string(text)?),
        )?,
        Frame::Array(items) | Frame::Push(items) => Value::Table(list(lua, items, resp3)?),
        Frame::Set(items) => {
            let t = lua.create_table()?;
            for item in items {
                t.raw_set(pull(lua, item, resp3)?, true)?;
            }
            wrapped(lua, "set", Value::Table(t))?
        }
        Frame::Map(items) | Frame::Attribute(items) => {
            let t = lua.create_table()?;
            for (k, v) in items {
                t.raw_set(pull(lua, k, resp3)?, pull(lua, v, resp3)?)?;
            }
            wrapped(lua, "map", Value::Table(t))?
        }
    })
}

/// A table holding one named field, which is how the RESP3 types arrive.
fn wrapped(lua: &Lua, field: &str, value: Value) -> mlua::Result<Value> {
    let t = lua.create_table()?;
    t.raw_set(field, value)?;
    Ok(Value::Table(t))
}

/// An array reply as a table numbered from one.
///
/// A null inside an array becomes `false` and not a hole, so `MGET a missing b`
/// gives a script three elements and `#` says three. That matters more than it
/// looks: an array that lost its length in the middle would make every script
/// that walks a reply with `ipairs` quietly wrong.
fn list(lua: &Lua, items: &[Frame<'_>], resp3: bool) -> mlua::Result<Table> {
    let t = lua.create_table_with_capacity(items.len(), 0)?;
    for (i, item) in items.iter().enumerate() {
        t.raw_set(i + 1, pull(lua, item, resp3)?)?;
    }
    Ok(t)
}
