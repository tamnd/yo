//! `cmsgpack`, the fourth and last of the libraries a script gets.
//!
//! This is Salvatore Sanfilippo's lua-cmsgpack 0.4.0, the same file Redis
//! carries in `deps/lua/src/lua_cmsgpack.c`, and it is a port of that rather
//! than of the MessagePack specification. The two are close but not the same,
//! and where they differ this follows the C, because what a script gets back
//! today is what it has to keep getting back.
//!
//! `cmsgpack.pack(...)` packs each argument on its own and hands back all of
//! them stuck together, so packing three values gives one string a later
//! `cmsgpack.unpack` reads three values out of. `unpack` reads the whole
//! string, `unpack_one` reads one value and tells you where to carry on from,
//! and `unpack_limit` reads up to a count.
//!
//! # A table is an array or it is a map
//!
//! An array is a table whose keys are exactly one up to the number of them,
//! with no hole and nothing else in it, and the key has to fit in a C `int`.
//! Everything else is a map, including a table with a zero key, a negative key,
//! a key too big for an `int` or any key that is not a number. An empty table
//! is an array, since nought keys running from one to nought is a run with no
//! hole in it.
//!
//! # Sixteen levels and then nothing
//!
//! A table nested sixteen deep is packed as a nil rather than refused, which is
//! how the C survives a table that points at itself. It is not an error and
//! there is no way for a script to tell it happened other than by unpacking and
//! looking.
//!
//! # A number is an integer when it can be
//!
//! Lua 5.1 has one number type, so the packer decides for itself: a number that
//! survives a trip through a signed sixty four bit integer goes out as an
//! integer, and everything else goes out as a float when a `f32` holds it
//! exactly and a double when it does not. That last rule is why `1.5` packs
//! into five bytes and `0.1` into nine.
//!
//! # The two places the C is wrong and this is wrong the same way
//!
//! `unpack_limit` and `unpack_one` complain about a negative offset with a
//! sentence that says "and limit of N", where N is the length of the input
//! rather than the limit. That is a mixed up argument in the C's format call
//! and it has been there since 2014.
//!
//! An unsigned sixty four bit integer is read into a signed one and then into a
//! double, so a message pack `uint 64` above about nine quintillion comes back
//! negative rather than large. Reading a stream this library wrote never gets
//! there, since the packer only writes a `uint 64` for a number it read as a
//! positive signed one.

use super::argue::{Answer, Stop, answer, behind, number, text};
use mlua::{Lua, MultiValue, Table, Value};

/// How deep a table can nest before it is packed as a nil instead.
const MAX_NESTING: u32 = 16;

/// The first double a signed sixty four bit integer cannot hold.
const TWO_63: f64 = 9_223_372_036_854_775_808.0;

/// A string, with the shortest of the four headers that fits it.
fn bytes(out: &mut Vec<u8>, s: &[u8]) {
    let len = s.len();
    if len < 32 {
        out.push(0xa0 | (len as u8 & 0x1f));
    } else if len <= 0xff {
        out.push(0xd9);
        out.push(len as u8);
    } else if len <= 0xffff {
        out.push(0xda);
        out.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        out.push(0xdb);
        out.extend_from_slice(&(len as u32).to_be_bytes());
    }
    out.extend_from_slice(s);
}

/// A number that is not a whole one, as a float when that is exact and a double
/// when it is not.
fn real(out: &mut Vec<u8>, d: f64) {
    let narrow = d as f32;
    if d == f64::from(narrow) {
        out.push(0xca);
        out.extend_from_slice(&narrow.to_be_bytes());
    } else {
        out.push(0xcb);
        out.extend_from_slice(&d.to_be_bytes());
    }
}

/// A whole number, in the shortest of the ten forms that holds it.
fn integer(out: &mut Vec<u8>, n: i64) {
    if n >= 0 {
        if n <= 127 {
            out.push(n as u8 & 0x7f);
        } else if n <= 0xff {
            out.push(0xcc);
            out.push(n as u8);
        } else if n <= 0xffff {
            out.push(0xcd);
            out.extend_from_slice(&(n as u16).to_be_bytes());
        } else if n <= 0xffff_ffff {
            out.push(0xce);
            out.extend_from_slice(&(n as u32).to_be_bytes());
        } else {
            out.push(0xcf);
            out.extend_from_slice(&(n as u64).to_be_bytes());
        }
    } else if n >= -32 {
        out.push(n as u8);
    } else if n >= -128 {
        out.push(0xd0);
        out.push(n as u8);
    } else if n >= -32768 {
        out.push(0xd1);
        out.extend_from_slice(&(n as i16).to_be_bytes());
    } else if n >= -2_147_483_648 {
        out.push(0xd2);
        out.extend_from_slice(&(n as i32).to_be_bytes());
    } else {
        out.push(0xd3);
        out.extend_from_slice(&n.to_be_bytes());
    }
}

/// How many things follow, with the tag that says which kind they are.
fn count(out: &mut Vec<u8>, n: usize, fix: u8, wide: u8, wider: u8) {
    if n <= 15 {
        out.push(fix | (n as u8 & 0xf));
    } else if n <= 65535 {
        out.push(wide);
        out.extend_from_slice(&(n as u16).to_be_bytes());
    } else {
        out.push(wider);
        out.extend_from_slice(&(n as u32).to_be_bytes());
    }
}

/// A number the C would happily put in a signed sixty four bit integer and get
/// the same value back out of.
///
/// The C writes this as a cast and a comparison, which is undefined once the
/// number is out of range, so the range test is here instead. On the machines
/// anyone runs this on the two agree.
fn whole(n: f64) -> Option<i64> {
    if !n.is_finite() || !(-TWO_63..TWO_63).contains(&n) {
        return None;
    }
    let held = n as i64;
    if held as f64 == n { Some(held) } else { None }
}

/// A number the C would put in a C `int` the same way, which is what a table
/// key has to survive to count as an array index.
fn small(n: f64) -> bool {
    n.is_finite() && f64::from(n as i32) == n
}

/// Whether a table is keyed by one up to however many things are in it.
///
/// The rule is the C's: every key has to be a number, above zero and small
/// enough for a C `int`, and the largest one has to be the count. Since a table
/// cannot hold the same key twice, those two together say there is no hole.
fn listed(t: &Table) -> mlua::Result<bool> {
    let mut seen: i64 = 0;
    let mut top: i64 = 0;
    for pair in t.pairs::<Value, Value>() {
        let (key, _) = pair?;
        let n = match key {
            Value::Integer(i) => i as f64,
            Value::Number(f) => f,
            _ => return Ok(false),
        };
        if n <= 0.0 || !small(n) {
            return Ok(false);
        }
        top = top.max(n as i64);
        seen += 1;
    }
    Ok(top == seen)
}

/// One Lua value, with `level` counting how many tables we are already inside.
fn encode(out: &mut Vec<u8>, v: &Value, level: u32) -> mlua::Result<()> {
    match v {
        Value::String(s) => bytes(out, &s.as_bytes()),
        Value::Boolean(b) => out.push(if *b { 0xc3 } else { 0xc2 }),
        Value::Integer(i) => integer(out, *i),
        Value::Number(f) => match whole(*f) {
            Some(held) => integer(out, held),
            None => real(out, *f),
        },
        Value::Table(t) if level < MAX_NESTING => {
            // A library table is a guard in front of the real one, and the real
            // one is what a server with no guard would be walking here.
            let held;
            let t = match behind(t) {
                Some(real) => {
                    held = real;
                    &held
                }
                None => t,
            };
            if listed(t)? {
                // The length is the one the `#` operator gives, which for a
                // table that got this far is the count of what is in it.
                let len = t.raw_len();
                count(out, len, 0x90, 0xdc, 0xdd);
                for j in 1..=len {
                    let item: Value = t.get(j)?;
                    encode(out, &item, level + 1)?;
                }
            } else {
                // The C walks the table twice, once to count and once to
                // write, because message pack wants the count up front.
                let mut len = 0usize;
                for pair in t.pairs::<Value, Value>() {
                    pair?;
                    len += 1;
                }
                count(out, len, 0x80, 0xde, 0xdf);
                for pair in t.pairs::<Value, Value>() {
                    let (key, held) = pair?;
                    encode(out, &key, level + 1)?;
                    encode(out, &held, level + 1)?;
                }
            }
        }
        // A table too deep to follow lands here along with everything that has
        // no message pack shape at all, which is a nil, a function and the two
        // kinds of userdata.
        _ => out.push(0xc0),
    }
    Ok(())
}

/// Where the reading has got to in a packed string.
struct Cur<'a> {
    data: &'a [u8],
    at: usize,
}

/// What the C sets on the cursor when the string ran out.
fn short() -> Stop {
    Stop::Plain(b"Missing bytes in input.".to_vec())
}

/// What the C sets on the cursor when the tag means nothing.
fn garbled() -> Stop {
    Stop::Plain(b"Bad data format in input.".to_vec())
}

impl Cur<'_> {
    fn left(&self) -> usize {
        self.data.len() - self.at
    }

    fn need(&self, len: usize) -> Answer<()> {
        if self.left() < len {
            Err(short())
        } else {
            Ok(())
        }
    }

    fn byte(&self, i: usize) -> u8 {
        self.data[self.at + i]
    }

    fn take(&mut self, len: usize) {
        self.at += len;
    }

    /// The next `len` bytes as a number, big endian, which is the only order
    /// message pack has.
    fn wide(&self, from: usize, len: usize) -> u64 {
        let mut held = 0u64;
        for i in 0..len {
            held = (held << 8) | u64::from(self.byte(from + i));
        }
        held
    }
}

/// A string of `len` bytes starting `from` bytes into what is left.
fn string(lua: &Lua, c: &mut Cur, from: usize, len: usize) -> Answer<Value> {
    c.need(from + len)?;
    let held = lua
        .create_string(&c.data[c.at + from..c.at + from + len])
        .map_err(|e| Stop::Plain(e.to_string().into_bytes()))?;
    c.take(from + len);
    Ok(Value::String(held))
}

/// A run of values, keyed one upwards.
fn array(lua: &Lua, c: &mut Cur, len: usize) -> Answer<Value> {
    let out = lua
        .create_table()
        .map_err(|e| Stop::Plain(e.to_string().into_bytes()))?;
    for i in 1..=len {
        let held = value(lua, c)?;
        out.raw_set(i, held)
            .map_err(|e| Stop::Plain(e.to_string().into_bytes()))?;
    }
    Ok(Value::Table(out))
}

/// A run of key and value pairs.
///
/// A key that cannot be one is the one place this raises something with no
/// position in front of it, because in the C it is the runtime rather than the
/// library that says so and the runtime is inside a C function at the time.
fn hash(lua: &Lua, c: &mut Cur, len: usize) -> Answer<Value> {
    let out = lua
        .create_table()
        .map_err(|e| Stop::Plain(e.to_string().into_bytes()))?;
    for _ in 0..len {
        let key = value(lua, c)?;
        let held = value(lua, c)?;
        match key {
            Value::Nil => return Err(Stop::Bare(b"table index is nil".to_vec())),
            Value::Number(n) if n.is_nan() => {
                return Err(Stop::Bare(b"table index is NaN".to_vec()));
            }
            _ => {}
        }
        out.raw_set(key, held)
            .map_err(|e| Stop::Plain(e.to_string().into_bytes()))?;
    }
    Ok(Value::Table(out))
}

/// One packed value.
fn value(lua: &Lua, c: &mut Cur) -> Answer<Value> {
    c.need(1)?;
    let tag = c.byte(0);
    let number = |c: &mut Cur, len: usize, held: f64| -> Answer<Value> {
        c.take(len);
        Ok(Value::Number(held))
    };
    match tag {
        // The unsigned widths go through a signed sixty four bit integer on
        // the way to a double, which is where a very large `uint 64` turns
        // negative.
        0xcc => {
            c.need(2)?;
            let held = c.wide(1, 1) as f64;
            number(c, 2, held)
        }
        0xcd => {
            c.need(3)?;
            let held = c.wide(1, 2) as f64;
            number(c, 3, held)
        }
        0xce => {
            c.need(5)?;
            let held = c.wide(1, 4) as f64;
            number(c, 5, held)
        }
        0xcf => {
            c.need(9)?;
            let held = c.wide(1, 8) as i64 as f64;
            number(c, 9, held)
        }
        0xd0 => {
            c.need(2)?;
            let held = f64::from(c.byte(1) as i8);
            number(c, 2, held)
        }
        0xd1 => {
            c.need(3)?;
            let held = f64::from(c.wide(1, 2) as u16 as i16);
            number(c, 3, held)
        }
        0xd2 => {
            c.need(5)?;
            let held = f64::from(c.wide(1, 4) as u32 as i32);
            number(c, 5, held)
        }
        0xd3 => {
            c.need(9)?;
            let held = c.wide(1, 8) as i64 as f64;
            number(c, 9, held)
        }
        0xc0 => {
            c.take(1);
            Ok(Value::Nil)
        }
        0xc2 | 0xc3 => {
            let held = tag == 0xc3;
            c.take(1);
            Ok(Value::Boolean(held))
        }
        0xca => {
            c.need(5)?;
            let held = f64::from(f32::from_bits(c.wide(1, 4) as u32));
            number(c, 5, held)
        }
        0xcb => {
            c.need(9)?;
            let held = f64::from_bits(c.wide(1, 8));
            number(c, 9, held)
        }
        0xd9 => {
            c.need(2)?;
            let len = c.byte(1) as usize;
            string(lua, c, 2, len)
        }
        0xda => {
            c.need(3)?;
            let len = c.wide(1, 2) as usize;
            string(lua, c, 3, len)
        }
        0xdb => {
            c.need(5)?;
            let len = c.wide(1, 4) as usize;
            string(lua, c, 5, len)
        }
        0xdc => {
            c.need(3)?;
            let len = c.wide(1, 2) as usize;
            c.take(3);
            array(lua, c, len)
        }
        0xdd => {
            c.need(5)?;
            let len = c.wide(1, 4) as usize;
            c.take(5);
            array(lua, c, len)
        }
        0xde => {
            c.need(3)?;
            let len = c.wide(1, 2) as usize;
            c.take(3);
            hash(lua, c, len)
        }
        0xdf => {
            c.need(5)?;
            let len = c.wide(1, 4) as usize;
            c.take(5);
            hash(lua, c, len)
        }
        // The rest carry their length or their value in the tag itself.
        _ => {
            if tag & 0x80 == 0 {
                let held = f64::from(tag);
                number(c, 1, held)
            } else if tag & 0xe0 == 0xe0 {
                let held = f64::from(tag as i8);
                number(c, 1, held)
            } else if tag & 0xe0 == 0xa0 {
                let len = (tag & 0x1f) as usize;
                string(lua, c, 1, len)
            } else if tag & 0xf0 == 0x90 {
                let len = (tag & 0xf) as usize;
                c.take(1);
                array(lua, c, len)
            } else if tag & 0xf0 == 0x80 {
                let len = (tag & 0xf) as usize;
                c.take(1);
                hash(lua, c, len)
            } else {
                Err(garbled())
            }
        }
    }
}

/// Every argument packed on its own and stuck together.
fn pack(lua: &Lua, args: &[Value]) -> Answer<Value> {
    if args.is_empty() {
        return Err(Stop::Arg(0, b"MessagePack pack needs input.".to_vec()));
    }
    let mut out = Vec::new();
    for held in args {
        encode(&mut out, held, 0).map_err(|e| Stop::Plain(e.to_string().into_bytes()))?;
    }
    let held = lua
        .create_string(&out)
        .map_err(|e| Stop::Plain(e.to_string().into_bytes()))?;
    Ok(Value::String(held))
}

/// The reading all three of the unpack functions do.
///
/// A limit and an offset of nought together mean read everything and say
/// nothing about where it stopped, which is what plain `unpack` asks for. Any
/// other pair means read up to the limit and put the next offset in front of
/// the values, or a minus one when there is nothing left.
fn unpack(lua: &Lua, args: &[Value], limit: i64, offset: i64) -> Answer<(Vec<Value>, Option<i64>)> {
    let data = text(lua, args, 1, "no value")?;
    let len = data.len() as i64;
    let everything = limit == 0 && offset == 0;
    if offset < 0 || limit < 0 {
        // The second number is the length of the input and not the limit,
        // which is a mixed up argument in the C and is kept on purpose.
        return Err(Stop::Plain(
            format!(
                "Invalid request to unpack with offset of {} and limit of {}.",
                offset as i32, len as i32
            )
            .into_bytes(),
        ));
    }
    if offset > len {
        return Err(Stop::Plain(
            format!(
                "Start offset {} greater than input length {}.",
                offset as i32, len as i32
            )
            .into_bytes(),
        ));
    }
    let limit = if everything {
        i64::from(i32::MAX)
    } else {
        limit
    };
    let mut c = Cur {
        data: &data[offset as usize..],
        at: 0,
    };
    let mut found = Vec::new();
    while c.left() > 0 && (found.len() as i64) < limit {
        found.push(value(lua, &mut c)?);
    }
    if everything {
        return Ok((found, None));
    }
    let stopped = if c.left() == 0 {
        -1
    } else {
        len - c.left() as i64
    };
    Ok((found, Some(stopped)))
}

/// The values and the offset laid out for the prelude, with `n` saying how many
/// there are since a nil among them makes the length operator no use.
fn listing(lua: &Lua, given: Answer<(Vec<Value>, Option<i64>)>) -> Answer<Value> {
    let (found, stopped) = given?;
    let build = || -> mlua::Result<Value> {
        let out = lua.create_table()?;
        let mut at = 0;
        if let Some(stopped) = stopped {
            at += 1;
            out.raw_set(at, stopped)?;
        }
        for held in found {
            at += 1;
            out.raw_set(at, held)?;
        }
        out.raw_set("n", at)?;
        Ok(Value::Table(out))
    };
    build().map_err(|e| Stop::Plain(e.to_string().into_bytes()))
}

/// An optional integer argument, the way `luaL_optinteger` reads one.
fn maybe(lua: &Lua, args: &[Value], i: usize, default: i64) -> Answer<i64> {
    match args.get(i - 1) {
        None | Some(Value::Nil) => Ok(default),
        Some(_) => Ok(number(lua, args, i, "no value")? as i64),
    }
}

/// Put the four functions on the private table the prelude reaches us through.
pub(super) fn statics(lua: &Lua, raw: &Table) -> mlua::Result<()> {
    raw.raw_set(
        "cmsgpack_pack",
        lua.create_function(|lua, args: MultiValue| {
            let held: Vec<Value> = args.into_iter().collect();
            answer(lua, pack(lua, &held))
        })?,
    )?;
    raw.raw_set(
        "cmsgpack_unpack",
        lua.create_function(|lua, args: MultiValue| {
            let held: Vec<Value> = args.into_iter().collect();
            answer(lua, listing(lua, unpack(lua, &held, 0, 0)))
        })?,
    )?;
    raw.raw_set(
        "cmsgpack_unpack_one",
        lua.create_function(|lua, args: MultiValue| {
            let held: Vec<Value> = args.into_iter().collect();
            let read = maybe(lua, &held, 2, 0).and_then(|at| unpack(lua, &held, 1, at));
            answer(lua, listing(lua, read))
        })?,
    )?;
    raw.raw_set(
        "cmsgpack_unpack_limit",
        lua.create_function(|lua, args: MultiValue| {
            let read = (|| {
                let held: Vec<Value> = args.into_iter().collect();
                let limit = number(lua, &held, 2, "no value")? as i64;
                let at = maybe(lua, &held, 3, 0)?;
                unpack(lua, &held, limit, at)
            })();
            answer(lua, listing(lua, read))
        })?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{integer, real, whole};

    /// The ten integer forms, picked by size the way the C picks them.
    #[test]
    fn a_whole_number_takes_the_shortest_form_that_holds_it() {
        for (n, want) in [
            (0i64, vec![0x00]),
            (127, vec![0x7f]),
            (128, vec![0xcc, 0x80]),
            (255, vec![0xcc, 0xff]),
            (256, vec![0xcd, 0x01, 0x00]),
            (65535, vec![0xcd, 0xff, 0xff]),
            (65536, vec![0xce, 0x00, 0x01, 0x00, 0x00]),
            (-1, vec![0xff]),
            (-32, vec![0xe0]),
            (-33, vec![0xd0, 0xdf]),
            (-128, vec![0xd0, 0x80]),
            (-129, vec![0xd1, 0xff, 0x7f]),
            (-32768, vec![0xd1, 0x80, 0x00]),
            (-32769, vec![0xd2, 0xff, 0xff, 0x7f, 0xff]),
        ] {
            let mut out = Vec::new();
            integer(&mut out, n);
            assert_eq!(out, want, "{n}");
        }
    }

    /// A float when one holds the number exactly and a double when it does not.
    #[test]
    fn a_fraction_goes_out_narrow_when_that_loses_nothing() {
        let mut out = Vec::new();
        real(&mut out, 1.5);
        assert_eq!(out, vec![0xca, 0x3f, 0xc0, 0x00, 0x00]);
        out.clear();
        real(&mut out, 0.1);
        assert_eq!(out[0], 0xcb);
        assert_eq!(out.len(), 9);
    }

    /// The range test, including the edge the C reaches by casting out of range.
    #[test]
    fn a_number_is_whole_only_when_an_integer_holds_it_and_gives_it_back() {
        assert_eq!(whole(1.0), Some(1));
        assert_eq!(whole(-1.0), Some(-1));
        assert_eq!(whole(0.5), None);
        assert_eq!(whole(f64::INFINITY), None);
        assert_eq!(whole(f64::NAN), None);
        assert_eq!(whole(-9_223_372_036_854_775_808.0), Some(i64::MIN));
        assert_eq!(whole(9_223_372_036_854_775_808.0), None);
        assert_eq!(whole(1e30), None);
    }
}
