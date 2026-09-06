//! `FT.CONFIG GET|SET|HELP`, the three ways at the search module's settings.
//!
//! The table behind this lives in [`yo_search::config`] and so does everything
//! about what a setting will take. What is here is the wire: which rows a
//! client asked for, and the two shapes they come back in.
//!
//! # A dump is a list of pairs and a map, depending
//!
//! `GET` answers a list of two element lists on RESP2 and a map on RESP3, and
//! `HELP` answers a list of five element lists on RESP2 and a map of two entry
//! maps on RESP3. The names go out as simple strings and the values as bulk
//! strings, which is a mix nothing else in the group sends and is what a real
//! server writes.
//!
//! # Nothing globs, and one word means everything
//!
//! `*` on its own is every row, in the order the table holds them. Any other
//! spelling is one name matched whole, ignoring case, and a name nobody knows
//! is an empty answer rather than an error. Words after the name are stepped
//! over: `FT.CONFIG GET timeout extra more` answers about `TIMEOUT` and says
//! nothing about the rest.
//!
//! # Only `SET` can go wrong
//!
//! And it checks things in an order that shows. The name first, then whether it
//! can move at all, then the value, and only after all three have passed does
//! it notice there were too many words. That last one is not an error either:
//! it is `+EXCESSARGS`, a status reply, and the value has already been written
//! by the time it is sent.

use yo_common::Result;
use yo_search::Registry;
use yo_search::config::{Config, EXCESS, Kind, TABLE, UNKNOWN};

use super::Args;
use crate::dispatch::args;
use crate::reply::Out;

/// The name the arity lines and the unknown subcommand line report under.
const NAME: &str = "FT.CONFIG";

/// `FT.CONFIG GET|SET|HELP ...`
///
/// # Errors
///
/// The arity of a subcommand and the unknown subcommand line, which are the two
/// the dispatcher writes. Everything a setting itself complains about is
/// written here and answered as done.
pub(super) fn run(reg: &mut Registry, args: Args<'_>, out: &mut Out) -> Result<()> {
    let sub = args.get(1);
    let wanted = |what: &str| args::wrong_arity_sub(NAME, what);
    match sub {
        _ if args::is(sub, b"GET") => {
            if args.len() < 3 {
                return Err(wanted("GET"));
            }
            dump(&reg.config, args.get(2), out, false);
        }
        _ if args::is(sub, b"HELP") => {
            if args.len() < 3 {
                return Err(wanted("HELP"));
            }
            dump(&reg.config, args.get(2), out, true);
        }
        _ if args::is(sub, b"SET") => {
            if args.len() < 3 {
                return Err(wanted("SET"));
            }
            set(&mut reg.config, args, out);
        }
        _ => return Err(args::unknown_subcommand(sub, NAME)),
    }
    Ok(())
}

/// Writes the rows a name asked for, with the help text or without it.
fn dump(config: &Config, name: &[u8], out: &mut Out, helping: bool) {
    // The one place a name is not a name. Every other spelling is matched
    // whole, so this is a comparison and not a glob.
    let rows: Vec<usize> = if name == b"*" {
        (0..TABLE.len()).collect()
    } else {
        Config::find(name).into_iter().collect()
    };
    let three = out.proto().is_resp3();
    if three {
        out.map(rows.len());
    } else {
        out.array(rows.len());
    }
    for at in rows {
        if !three {
            out.array(if helping { 5 } else { 2 });
        }
        out.simple(TABLE[at].name.as_bytes());
        if helping {
            if three {
                out.map(2);
            }
            out.simple(b"Description");
            out.simple(TABLE[at].help.as_bytes());
            out.simple(b"Value");
        }
        match config.value(at) {
            Some(value) => out.bulk(&value),
            None => out.nil(),
        }
    }
}

/// `FT.CONFIG SET name [value]`, and everything that can be said about it.
fn set(config: &mut Config, args: Args<'_>, out: &mut Out) {
    let Some(at) = Config::find(args.get(2)) else {
        out.error(UNKNOWN.as_bytes());
        return;
    };
    // A setting that takes no value swallows one word and every other setting
    // swallows two, so this is where the count of words a good `SET` uses up
    // comes from, and anything past it is the excess.
    let takes = if matches!(TABLE[at].kind, Kind::Bare) {
        3
    } else {
        4
    };
    let value = (args.len() > 3).then(|| args.get(3));
    if let Err(why) = config.set(at, value) {
        out.error(&why);
        return;
    }
    if args.len() > takes {
        out.simple(EXCESS);
        return;
    }
    out.ok();
}
