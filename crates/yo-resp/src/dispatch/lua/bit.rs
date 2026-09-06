//! `bit`, the library a script gets when it needs to work on words.
//!
//! Lua 5.1 has one number type and it is a double, so a script that wants to
//! and two flags together, pack four bytes into an integer or hash something by
//! hand has nothing to do it with. Mike Pall's LuaBitOp is what filled that gap
//! and it is what Redis embeds, so it is what a script written against Redis
//! calls, and this is that library rather than a set of operations that happen
//! to have the same names.
//!
//! Everything here works on a signed thirty two bit word. A number goes in, is
//! taken down to thirty two bits, is operated on and comes back out as a Lua
//! number in the range a signed word covers, which is why `bit.tohex(-1)` is
//! `ffffffff` and `bit.bnot(0)` is minus one.
//!
//! # The conversion is the library
//!
//! `tobit` is not a truncation and it is not a modulo. It is the one LuaBitOp
//! does: add the double whose mantissa starts at the units place, let the
//! hardware round the result to an integer, and read the low word straight out
//! of the bit pattern. That gives round to nearest with ties to even rather
//! than rounding toward zero, so `bit.tobit(1.5)` is two and not one, and it
//! wraps at two to the thirty two for free. Anything else here would answer
//! differently from a real server on a number a script did arithmetic on, which
//! is most of them.
//!
//! # Where the argument checking is
//!
//! Not here. Every function in this file takes numbers and cannot fail, and the
//! prelude wraps each one in the Lua that checks its arguments and raises the
//! sentence a real server raises. That is the same split the rest of the
//! scripting code makes and for the same reason: a raise has to come from Lua
//! to read the way a script expects.

use mlua::{Lua, Table};

/// The double whose last mantissa bit is the units place, which is two to the
/// fifty two plus two to the fifty one.
const MAGIC: f64 = 6_755_399_441_055_744.0;

/// A Lua number as the word every operation below works on.
///
/// Adding `MAGIC` lands the value in the one place where a double's mantissa
/// and an integer line up, so the rounding the hardware does on the way is the
/// rounding LuaBitOp specifies and the low word of the result is the answer.
/// This reads the low word off the little end, which is the only end this
/// server is ever built for.
fn tobit(x: f64) -> i32 {
    (x + MAGIC).to_bits() as u32 as i32
}

/// `x` as up to eight hex digits, lowest first, upper case when `n` is negative.
///
/// The count goes through the same conversion the value does, because Lua 5.1
/// reads an integer argument with the same rounding trick, so `bit.tohex(1, 2.7)`
/// asks for three digits and not for two.
fn tohex(x: i32, n: i32) -> String {
    let upper = n < 0;
    let digits = usize::try_from(n.saturating_abs()).unwrap_or(8).min(8);
    let word = x as u32;
    let mut out = String::with_capacity(digits);
    // The digits a real server keeps are the low ones, so a count under eight
    // takes the tail of the word rather than the head of the text.
    for i in (0..digits).rev() {
        let nibble = (word >> (i * 4)) & 0xf;
        let digit = char::from_digit(nibble, 16).unwrap_or('0');
        out.push(if upper {
            digit.to_ascii_uppercase()
        } else {
            digit
        });
    }
    out
}

/// Put the arithmetic on the private table the prelude reaches us through.
///
/// The names carry a prefix because that table is shared with the `redis`
/// library's own helpers and a plain `band` would say nothing about where it
/// belongs.
pub(super) fn statics(lua: &Lua, raw: &Table) -> mlua::Result<()> {
    // One argument.
    raw.raw_set("bit_tobit", lua.create_function(|_, x: f64| Ok(tobit(x)))?)?;
    raw.raw_set("bit_bnot", lua.create_function(|_, x: f64| Ok(!tobit(x)))?)?;
    raw.raw_set(
        "bit_bswap",
        lua.create_function(|_, x: f64| Ok(tobit(x).swap_bytes()))?,
    )?;
    raw.raw_set(
        "bit_tohex",
        lua.create_function(|_, (x, n): (f64, f64)| Ok(tohex(tobit(x), tobit(n))))?,
    )?;

    // Two arguments, folded from the left by the wrapper when a script passes
    // more than two.
    raw.raw_set(
        "bit_band",
        lua.create_function(|_, (a, b): (f64, f64)| Ok(tobit(a) & tobit(b)))?,
    )?;
    raw.raw_set(
        "bit_bor",
        lua.create_function(|_, (a, b): (f64, f64)| Ok(tobit(a) | tobit(b)))?,
    )?;
    raw.raw_set(
        "bit_bxor",
        lua.create_function(|_, (a, b): (f64, f64)| Ok(tobit(a) ^ tobit(b)))?,
    )?;

    // A shift of thirty two is a shift of nothing, because only the low five
    // bits of the count are read. That is what the hardware does and what
    // LuaBitOp documents, and it is why `bit.lshift(1, 32)` is one again.
    raw.raw_set(
        "bit_lshift",
        lua.create_function(|_, (x, n): (f64, f64)| Ok(((tobit(x) as u32) << count(n)) as i32))?,
    )?;
    raw.raw_set(
        "bit_rshift",
        lua.create_function(|_, (x, n): (f64, f64)| Ok(((tobit(x) as u32) >> count(n)) as i32))?,
    )?;
    raw.raw_set(
        "bit_arshift",
        lua.create_function(|_, (x, n): (f64, f64)| Ok(tobit(x) >> count(n)))?,
    )?;
    raw.raw_set(
        "bit_rol",
        lua.create_function(|_, (x, n): (f64, f64)| Ok(tobit(x).rotate_left(count(n))))?,
    )?;
    raw.raw_set(
        "bit_ror",
        lua.create_function(|_, (x, n): (f64, f64)| Ok(tobit(x).rotate_right(count(n))))?,
    )?;
    Ok(())
}

/// How far a shift actually moves, which is the low five bits of what was asked.
fn count(n: f64) -> u32 {
    tobit(n) as u32 & 31
}

#[cfg(test)]
mod tests {
    use super::{tobit, tohex};

    /// The rounding is the part nothing else would get right, so these are the
    /// cases either side of a tie and either side of the wrap.
    #[test]
    fn a_number_comes_down_to_a_word_the_way_lua_bit_op_does_it() {
        for (given, want) in [
            (1.0, 1),
            (-1.0, -1),
            (0.0, 0),
            (4_294_967_295.0, -1),
            (4_294_967_296.0, 0),
            (4_294_967_297.0, 1),
            (2_147_483_648.0, i32::MIN),
            (-2_147_483_648.0, i32::MIN),
            (9_007_199_254_740_992.0, 0),
            // Ties go to the even neighbour, which is the whole reason the
            // conversion is written the way it is.
            (1.5, 2),
            (2.5, 2),
            (0.5, 0),
            (-0.5, 0),
            (-1.5, -2),
            (1.4, 1),
            (1.6, 2),
        ] {
            assert_eq!(tobit(given), want, "{given}");
        }
    }

    /// The count is clamped rather than refused, and a negative one asks for
    /// upper case rather than for a count of its own.
    #[test]
    fn a_word_is_written_out_as_the_digits_that_were_asked_for() {
        for (word, digits, want) in [
            (1, 8, "00000001"),
            (-1, 8, "ffffffff"),
            (255, 2, "ff"),
            (255, -8, "000000FF"),
            (-1, -8, "FFFFFFFF"),
            (0x8765_4321_u32 as i32, 4, "4321"),
            (1, 0, ""),
            (1, 9, "00000001"),
            (1, -9, "00000001"),
            (1, i32::MIN, "00000001"),
        ] {
            assert_eq!(tohex(word, digits), want, "{word} {digits}");
        }
    }
}
