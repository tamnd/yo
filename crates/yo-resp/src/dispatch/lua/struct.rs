//! `struct`, the library a script gets when it needs bytes rather than text.
//!
//! This is Roberto Ierusalimschy's `struct` from lua.org, the same file Redis
//! carries in `deps/lua/src/lua_struct.c`, and it is the answer to the thing
//! Lua 5.1 is worst at. A Lua string is a byte string, so a script can hold a
//! packed record perfectly well, but the language gives it no way to build one
//! or take one apart. `string.byte` and `string.char` get you a byte at a time
//! and nothing gets you a float.
//!
//! A format is a string of options read left to right, one per value.
//! `struct.pack('>i4i4', 1, 2)` is eight bytes big endian, `struct.unpack` reads
//! them back and hands over the position it stopped at, and `struct.size` says
//! how long a fixed format is without packing anything.
//!
//! # The sizes are the machine's, not the format's
//!
//! `l` is a C `long`, so it is eight bytes here and four on a thirty two bit
//! build. `T` is a `size_t`, `i` with no number after it is a C `int` and so is
//! four. A script that wants a width it can count on writes `i8` or `i2` and
//! says so. This is a port of the C, so the sizes here are the ones the C picks
//! on the platform Redis is built for, which is little endian and sixty four
//! bit, and they do not follow the machine this happens to be running on, which
//! is D-107.
//!
//! # Alignment is off until it is asked for
//!
//! Every option is packed hard against the last one unless the format has a `!`
//! in it. That is the opposite of what a C struct does, so `struct.size('bi4')`
//! is five and not eight, and `!4` is how you ask for the C layout.
//!
//! # The two places it is quietly lenient
//!
//! `struct.size` ignores an option it does not know as long as it is a letter
//! or a digit, so `struct.size('A')` is zero rather than a failure, while
//! `struct.pack('A', 'x')` refuses the same option. And an option that takes a
//! count reads it as decimal digits right after the letter, so `c10` is ten
//! bytes and `c` on its own is one.
//!
//! # Where the argument checking is
//!
//! Here, unlike the other three libraries, because the format is what decides
//! how many arguments there are and what each one has to be. The prelude cannot
//! know that `>i4c8` wants a number and then a string without reading the
//! format itself. So this reads the arguments and says which one was wrong, and
//! the prelude does nothing but turn that into the raise.

use super::cjson::typename;
use mlua::{Lua, MultiValue, Table, Value};

/// What went wrong, in the shape the prelude needs to raise it.
enum Stop {
    /// A plain failure, which comes out with the script's position in front.
    Plain(Vec<u8>),
    /// A complaint about one argument, which also names the position and the
    /// function the call site used.
    Arg(usize, Vec<u8>),
}

type Answer<T> = Result<T, Stop>;

/// Where the reading has got to in a format string.
struct Fmt<'a> {
    text: &'a [u8],
    at: usize,
}

/// How the options that follow are laid out.
struct Header {
    big: bool,
    align: usize,
}

/// A C `long` and a C `size_t`, which is what `l`, `L` and `T` mean, and eight
/// on the platform this is a port of.
const LONG: usize = 8;

/// A C `int`, which is what `i` with no number after it means.
const INT: usize = 4;

/// A C `float` and a C `double`, which are the only two widths `f` and `d`
/// have ever had.
const FLOAT: usize = 4;
const DOUBLE: usize = 8;

/// The widest an `i` or an `I` is allowed to be.
const MAXINT: usize = 32;

/// What `!` sets the alignment to when no number follows it, which is the
/// alignment a `double` wants.
const MAXALIGN: usize = 8;

impl Fmt<'_> {
    /// The byte under the cursor, or a zero past the end.
    fn byte(&self) -> u8 {
        self.text.get(self.at).copied().unwrap_or(0)
    }

    /// The decimal number right after an option letter, or the default.
    ///
    /// The overflow check is the C's, which is done a digit at a time against
    /// the largest `int` rather than afterwards, so the failure comes before
    /// anything wraps.
    fn num(&mut self, default: usize) -> Answer<usize> {
        if !self.byte().is_ascii_digit() {
            return Ok(default);
        }
        let mut a: i32 = 0;
        while self.byte().is_ascii_digit() {
            let digit = i32::from(self.byte() - b'0');
            if a > i32::MAX / 10 || a * 10 > i32::MAX - digit {
                return Err(Stop::Plain(b"integral size overflow".to_vec()));
            }
            a = a * 10 + digit;
            self.at += 1;
        }
        Ok(a as usize)
    }
}

/// How many bytes one option takes, or zero when it is not a value at all.
fn optsize(opt: u8, fmt: &mut Fmt<'_>) -> Answer<usize> {
    match opt {
        b'b' | b'B' | b'x' => Ok(1),
        b'h' | b'H' => Ok(2),
        b'l' | b'L' | b'T' => Ok(LONG),
        b'f' => Ok(FLOAT),
        b'd' => Ok(DOUBLE),
        b'c' => fmt.num(1),
        b'i' | b'I' => {
            let size = fmt.num(INT)?;
            if size > MAXINT {
                return Err(Stop::Plain(
                    format!("integral size {size} is larger than limit of {MAXINT}").into_bytes(),
                ));
            }
            Ok(size)
        }
        // Everything else is either a layout option or a mistake, and neither
        // has a size. `s` lands here too, since how long it is depends on the
        // string rather than on the format.
        _ => Ok(0),
    }
}

/// How many zero bytes go in front of an option to line it up.
///
/// The rule is the C's, including that it only makes sense for a size that is a
/// power of two and is applied to any size anyway. With the alignment left at
/// one, which is where it starts, this is always zero.
fn to_align(len: usize, h: &Header, opt: u8, size: usize) -> usize {
    if size == 0 || opt == b'c' {
        return 0;
    }
    let size = size.min(h.align);
    (size - (len & (size - 1))) & (size - 1)
}

/// What an option that is not a value does, which is set the layout or fail.
fn control(opt: u8, fmt: &mut Fmt<'_>, h: &mut Header) -> Answer<()> {
    match opt {
        b' ' => Ok(()),
        b'>' => {
            h.big = true;
            Ok(())
        }
        b'<' => {
            h.big = false;
            Ok(())
        }
        b'!' => {
            let want = fmt.num(MAXALIGN)?;
            if want == 0 || want & (want - 1) != 0 {
                return Err(Stop::Plain(
                    format!("alignment {want} is not a power of 2").into_bytes(),
                ));
            }
            h.align = want;
            Ok(())
        }
        _ => {
            let mut why = b"invalid format option '".to_vec();
            why.push(opt);
            why.push(b'\'');
            Err(Stop::Arg(1, why))
        }
    }
}

/// What Lua says when an argument is the wrong type.
///
/// `missing` is what the message calls an argument that is not there at all,
/// and it is not the same for all three functions: `pack` pushes a nil of its
/// own before it starts reading, so the first argument it runs out of is a nil
/// rather than nothing, and it says so.
fn expected(want: &str, what: &str) -> Vec<u8> {
    format!("{want} expected, got {what}").into_bytes()
}

/// One argument as a string, the way `luaL_checklstring` reads one.
///
/// A number counts as a string, because Lua converts one on the way in.
fn text(lua: &Lua, args: &[Value], i: usize, missing: &str) -> Answer<Vec<u8>> {
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
fn number(lua: &Lua, args: &[Value], i: usize, missing: &str) -> Answer<f64> {
    let Some(value) = args.get(i - 1) else {
        return Err(Stop::Arg(i, expected("number", missing)));
    };
    match lua.coerce_number(value.clone()) {
        Ok(Some(n)) => Ok(n),
        _ => Err(Stop::Arg(i, expected("number", typename(value)))),
    }
}

/// A format string as the C sees one, which is up to the first zero byte.
///
/// The C takes it as a pointer and stops at the terminator, so a zero in the
/// middle ends the format early rather than being an option nobody knows.
fn until_zero(text: &[u8]) -> &[u8] {
    match text.iter().position(|&b| b == 0) {
        Some(end) => &text[..end],
        None => text,
    }
}

/// A number written out as the bytes of an integer that wide.
///
/// Past eight bytes the value has run out and the rest are zeros, which means a
/// negative number packed as `i16` is not sign extended. That is what the C
/// does, since it shifts an unsigned long right until nothing is left.
fn put_integer(out: &mut Vec<u8>, n: f64, big: bool, size: usize) {
    let mut value = if n < 0.0 { (n as i64) as u64 } else { n as u64 };
    let at = out.len();
    out.resize(at + size, 0);
    for i in 0..size {
        let byte = (value & 0xff) as u8;
        out[if big { at + size - 1 - i } else { at + i }] = byte;
        value >>= 8;
    }
}

/// The bytes of an integer that wide read back as a number.
fn get_integer(data: &[u8], big: bool, signed: bool, size: usize) -> f64 {
    let mut value = 0u64;
    for i in 0..size {
        let byte = data[if big { i } else { size - 1 - i }];
        value = (value << 8) | u64::from(byte);
    }
    if !signed {
        return value as f64;
    }
    // The sign is spread up from the top bit of the width that was asked for.
    // The shift is taken modulo sixty four, which is what the C ends up doing
    // on the machine it is built for once the width goes past eight bytes.
    let mask = (!0u64) << ((size * 8 - 1) % 64);
    if value & mask != 0 {
        value |= mask;
    }
    (value as i64) as f64
}

/// Build the bytes a format and its values come to.
fn pack(lua: &Lua, args: &[Value]) -> Answer<Vec<u8>> {
    let held = text(lua, args, 1, "no value")?;
    let mut fmt = Fmt {
        text: until_zero(&held),
        at: 0,
    };
    let mut h = Header {
        big: false,
        align: 1,
    };
    let mut arg = 2;
    let mut total = 0usize;
    let mut out = Vec::new();
    while fmt.at < fmt.text.len() {
        let opt = fmt.text[fmt.at];
        fmt.at += 1;
        let mut size = optsize(opt, &mut fmt)?;
        let pad = to_align(total, &h, opt, size);
        total += pad;
        out.resize(out.len() + pad, 0);
        match opt {
            b'b' | b'B' | b'h' | b'H' | b'l' | b'L' | b'T' | b'i' | b'I' => {
                // Every argument this runs out of reads as a nil rather than as
                // nothing, because the C pushes one before it starts.
                let n = number(lua, args, arg, "nil")?;
                arg += 1;
                put_integer(&mut out, n, h.big, size);
            }
            b'x' => out.push(0),
            b'f' => {
                let n = number(lua, args, arg, "nil")? as f32;
                arg += 1;
                let bytes = if h.big {
                    n.to_be_bytes()
                } else {
                    n.to_le_bytes()
                };
                out.extend_from_slice(&bytes);
            }
            b'd' => {
                let n = number(lua, args, arg, "nil")?;
                arg += 1;
                let bytes = if h.big {
                    n.to_be_bytes()
                } else {
                    n.to_le_bytes()
                };
                out.extend_from_slice(&bytes);
            }
            b'c' | b's' => {
                let s = text(lua, args, arg, "nil")?;
                arg += 1;
                // A count of zero means the whole string, which is what `s`
                // always asks for and what `c0` asks for on purpose.
                if size == 0 {
                    size = s.len();
                }
                if s.len() < size {
                    // The C reads the string with `arg++` and then complains
                    // about `arg`, so the number it prints is one past the
                    // argument it is actually unhappy with. Scripts have seen
                    // that number for years, so keep it.
                    return Err(Stop::Arg(arg, b"string too short".to_vec()));
                }
                out.extend_from_slice(&s[..size]);
                if opt == b's' {
                    out.push(0);
                    size += 1;
                }
            }
            _ => control(opt, &mut fmt, &mut h)?,
        }
        total += size;
    }
    Ok(out)
}

/// Read a format's worth of values out of a string.
///
/// The last thing handed back is where the reading stopped, one past the last
/// byte taken and counting from one, so it can be passed straight back in as
/// the offset for the next call.
fn unpack(lua: &Lua, args: &[Value]) -> Answer<(Vec<Value>, f64)> {
    let held = text(lua, args, 1, "no value")?;
    let data = text(lua, args, 2, "no value")?;
    let start = match args.get(2) {
        None | Some(Value::Nil) => 1.0,
        Some(_) => number(lua, args, 3, "no value")?,
    };
    // The C reads the offset into a `size_t`, so a negative one comes out as a
    // very large positive one and only zero is turned away here. Everything
    // else falls through to the length check below and is told the string is
    // too short, which is the answer scripts have always had.
    let counted = start as i64;
    if counted == 0 {
        return Err(Stop::Arg(3, b"offset must be 1 or greater".to_vec()));
    }
    let long = data.len();
    // Anything past the end of the string is held one byte past it, so the
    // length check below turns it into the failure the C gives rather than
    // into arithmetic that wraps.
    let mut pos = (counted as u64).wrapping_sub(1).min(long as u64 + 1) as usize;
    let mut fmt = Fmt {
        text: until_zero(&held),
        at: 0,
    };
    let mut h = Header {
        big: false,
        align: 1,
    };
    let mut found: Vec<Value> = Vec::new();
    let short = || Stop::Arg(2, b"data string too short".to_vec());
    while fmt.at < fmt.text.len() {
        let opt = fmt.text[fmt.at];
        fmt.at += 1;
        let mut size = optsize(opt, &mut fmt)?;
        pos = pos.saturating_add(to_align(pos, &h, opt, size));
        if size > long || pos > long - size {
            return Err(short());
        }
        match opt {
            b'b' | b'B' | b'h' | b'H' | b'l' | b'L' | b'T' | b'i' | b'I' => {
                // A lower case letter is the signed spelling and an upper case
                // one is not, which is the whole of the difference.
                let signed = opt.is_ascii_lowercase();
                let n = get_integer(&data[pos..], h.big, signed, size);
                found.push(Value::Number(n));
            }
            b'x' => {}
            b'f' => {
                let raw = [data[pos], data[pos + 1], data[pos + 2], data[pos + 3]];
                let n = if h.big {
                    f32::from_be_bytes(raw)
                } else {
                    f32::from_le_bytes(raw)
                };
                found.push(Value::Number(f64::from(n)));
            }
            b'd' => {
                let mut raw = [0u8; 8];
                raw.copy_from_slice(&data[pos..pos + 8]);
                let n = if h.big {
                    f64::from_be_bytes(raw)
                } else {
                    f64::from_le_bytes(raw)
                };
                found.push(Value::Number(n));
            }
            b'c' => {
                // `c0` takes its length from the value read just before it,
                // which is how a length prefixed field is spelled.
                if size == 0 {
                    let last = found
                        .last()
                        .and_then(|v| lua.coerce_number(v.clone()).ok().flatten());
                    let Some(n) = last else {
                        return Err(Stop::Plain(b"format 'c0' needs a previous size".to_vec()));
                    };
                    found.pop();
                    size = if n < 0.0 { usize::MAX } else { n as usize };
                    if size > long || pos > long - size {
                        return Err(short());
                    }
                }
                found.push(string(lua, &data[pos..pos + size])?);
            }
            b's' => {
                let Some(end) = data[pos..].iter().position(|&b| b == 0) else {
                    return Err(Stop::Plain(b"unfinished string in data".to_vec()));
                };
                found.push(string(lua, &data[pos..pos + end])?);
                size = end + 1;
            }
            _ => control(opt, &mut fmt, &mut h)?,
        }
        pos += size;
    }
    Ok((found, (pos + 1) as f64))
}

/// How many bytes a format comes to, when that does not depend on the values.
fn sizeof(lua: &Lua, args: &[Value]) -> Answer<f64> {
    let held = text(lua, args, 1, "no value")?;
    let mut fmt = Fmt {
        text: until_zero(&held),
        at: 0,
    };
    let mut h = Header {
        big: false,
        align: 1,
    };
    let mut pos = 0usize;
    while fmt.at < fmt.text.len() {
        let opt = fmt.text[fmt.at];
        fmt.at += 1;
        let size = optsize(opt, &mut fmt)?;
        pos += to_align(pos, &h, opt, size);
        if opt == b's' {
            return Err(Stop::Arg(1, b"option 's' has no fixed size".to_vec()));
        }
        if opt == b'c' && size == 0 {
            return Err(Stop::Arg(1, b"option 'c0' has no fixed size".to_vec()));
        }
        // A letter or a digit that means nothing here is passed over rather
        // than refused, which is why `struct.size('A')` is zero while
        // `struct.pack('A', 'x')` is a failure.
        if !opt.is_ascii_alphanumeric() {
            control(opt, &mut fmt, &mut h)?;
        }
        pos += size;
    }
    Ok(pos as f64)
}

/// Some bytes as a Lua string, or the allocation failure as a plain one.
fn string(lua: &Lua, bytes: &[u8]) -> Answer<Value> {
    lua.create_string(bytes)
        .map(Value::String)
        .map_err(|e| Stop::Plain(e.to_string().into_bytes()))
}

/// Turn an answer into the three values the prelude reads.
///
/// The first says which of the three it is, which is nought for a result, one
/// for a plain failure and two for a complaint about an argument. Doing it this
/// way rather than raising from Rust is the same split the other libraries
/// make: a message a script reads has to be raised from Lua so that the line in
/// front of it is the script's own.
fn answer(lua: &Lua, given: Answer<Value>) -> mlua::Result<(i64, Value, Value)> {
    match given {
        Ok(value) => Ok((0, value, Value::Nil)),
        Err(Stop::Plain(why)) => Ok((1, Value::String(lua.create_string(&why)?), Value::Nil)),
        Err(Stop::Arg(at, why)) => Ok((
            2,
            Value::Integer(at as i64),
            Value::String(lua.create_string(&why)?),
        )),
    }
}

/// Put the three functions on the private table the prelude reaches us through.
pub(super) fn statics(lua: &Lua, raw: &Table) -> mlua::Result<()> {
    raw.raw_set(
        "struct_pack",
        lua.create_function(|lua, args: MultiValue| {
            let held: Vec<Value> = args.into_iter().collect();
            let packed = pack(lua, &held).and_then(|out| string(lua, &out));
            answer(lua, packed)
        })?,
    )?;
    raw.raw_set(
        "struct_unpack",
        lua.create_function(|lua, args: MultiValue| {
            let held: Vec<Value> = args.into_iter().collect();
            // The results come back in a table with the count on it, because a
            // list of them cannot ride alongside the two values that say
            // whether this worked.
            let read = unpack(lua, &held).and_then(|(found, at)| {
                let build = || -> mlua::Result<Value> {
                    let out = lua.create_table()?;
                    let count = found.len() + 1;
                    for (i, value) in found.into_iter().enumerate() {
                        out.raw_set(i + 1, value)?;
                    }
                    out.raw_set(count, at)?;
                    out.raw_set("n", count)?;
                    Ok(Value::Table(out))
                };
                build().map_err(|e| Stop::Plain(e.to_string().into_bytes()))
            });
            answer(lua, read)
        })?,
    )?;
    raw.raw_set(
        "struct_size",
        lua.create_function(|lua, args: MultiValue| {
            let held: Vec<Value> = args.into_iter().collect();
            answer(lua, sizeof(lua, &held).map(Value::Number))
        })?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{get_integer, put_integer};

    /// The widths and the two orders, plus the two edges the C reaches by
    /// running an unsigned long out of bits.
    #[test]
    fn an_integer_goes_out_and_comes_back_the_way_the_c_library_moves_one() {
        for (n, size, big, want) in [
            (1.0, 1, false, vec![1]),
            (255.0, 1, false, vec![255]),
            (-1.0, 1, false, vec![255]),
            (258.0, 2, false, vec![2, 1]),
            (258.0, 2, true, vec![1, 2]),
            (-1.0, 4, false, vec![255, 255, 255, 255]),
            // Past eight bytes the value has run out, so a negative number is
            // not spread into the rest.
            (
                -1.0,
                9,
                false,
                vec![255, 255, 255, 255, 255, 255, 255, 255, 0],
            ),
            (1.0, 3, true, vec![0, 0, 1]),
        ] {
            let mut out = Vec::new();
            put_integer(&mut out, n, big, size);
            assert_eq!(out, want, "{n} in {size} bytes");
        }
        for (bytes, size, big, signed, want) in [
            (vec![255u8], 1, false, true, -1.0),
            (vec![255u8], 1, false, false, 255.0),
            (vec![2u8, 1], 2, false, true, 258.0),
            (vec![2u8, 1], 2, true, true, 513.0),
            (vec![0u8, 0, 0, 128], 4, false, true, -2147483648.0),
            (vec![0u8, 0, 0, 128], 4, false, false, 2147483648.0),
        ] {
            assert_eq!(
                get_integer(&bytes, big, signed, size),
                want,
                "{bytes:?} {size} {big} {signed}",
            );
        }
    }
}
