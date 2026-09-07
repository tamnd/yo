//! The HyperLogLog commands, from the wire.
//!
//! Five commands over the string values [`strings`](super::strings) writes, and
//! the same two rules as that file and as [`bits`](super::bits): nothing is
//! written to the reply until the arguments are known to be good, and nothing
//! allocates.
//!
//! The keyspace layer is where the interesting decisions are and this is mostly
//! shape. Two pieces of shape are worth naming, though, because both were read
//! off a running 8.10.1 and neither is what the documentation implies.
//!
//! `PFDEBUG` has its own sentence for a subcommand it does not know, "Unknown
//! PFDEBUG subcommand 'X'", rather than the "unknown subcommand 'x'. Try X
//! HELP." every other container command uses, and it quotes the word exactly as
//! the client spelled it. `PFDEBUG` on a key that is not there is an error where
//! all three of the real commands treat a missing key as an empty sketch.
//!
//! `PFDEBUG ENCODING` answers a simple status and `PFDEBUG DECODE` answers a
//! bulk string, so a client cannot tell them apart by reply type by accident.

use super::args::{self, Args, is};
use super::notify::{self, class};
use super::table::Spec;
use crate::reply::Out;
use yo_common::{Code, Error, Result};
use yo_kv::Db;
use yo_kv::hll;

/// What Redis says about a `PFDEBUG` subcommand it does not know.
const UNKNOWN_SUB: &str = "Unknown PFDEBUG subcommand";

/// Run one HyperLogLog command.
pub(super) fn execute(
    db: &Db,
    on: usize,
    spec: &Spec,
    args: Args<'_>,
    out: &mut Out,
) -> Result<()> {
    match spec.name {
        "pfadd" => {
            let eles = (2..args.len()).map(|i| args.get(i));
            let key = args.get(1);
            let updated = db.hold(key).pfadd(key, eles)?;
            out.int(i64::from(updated));
            // Only when a register moved or the key was created, which is the
            // same answer the reply gives, so `PFADD h` on a sketch that is
            // already there says nothing and the same call on a name that is
            // free says it. The class is the string one, since that is what a
            // sketch is stored as.
            if updated {
                notify::fire(on, class::STRING, "pfadd", key);
            }
        }
        "pfcount" => {
            let keys = (1..args.len()).map(|i| args.get(i));
            out.uint(db.pfcount(keys)?);
        }
        "pfmerge" => {
            let srcs = (2..args.len()).map(|i| args.get(i));
            let dest = args.get(1);
            db.pfmerge(dest, srcs)?;
            out.ok();
            // Always, where `PFADD` asks first, and under `PFADD`'s name rather
            // than one of its own. Redis calls a merge a mass add and says so
            // whatever it merged, including `PFMERGE d` with no sources at all,
            // which touches nothing and still says it.
            notify::fire(on, class::STRING, "pfadd", dest);
        }
        "pfdebug" => debug(db, args, out)?,
        // Redis runs a few thousand additions and checks the estimate is within
        // its bounds. Ours are checked in `cargo test` instead, where a failure
        // stops a release rather than a client, so this is the OK that a
        // compatibility suite expects and nothing more.
        "pfselftest" => out.ok(),
        _ => return Err(args::syntax()),
    }
    Ok(())
}

/// `PFDEBUG subcommand key`, which is four subcommands and no more.
fn debug(db: &Db, args: Args<'_>, out: &mut Out) -> Result<()> {
    let (sub, key) = (args.get(1), args.get(2));
    if is(sub, b"getreg") {
        // Sixteen kibibytes of registers, filled before a byte of the reply is
        // written, since the fill can still fail on a sketch that is corrupt.
        let mut regs = [0u8; hll::REGISTERS];
        db.hold(key).pfgetreg(key, &mut regs)?;
        out.array(regs.len());
        for &val in &regs {
            out.int(i64::from(val));
        }
    } else if is(sub, b"decode") {
        db.hold(key).pfdecode(key, |text| out.bulk(text))?;
    } else if is(sub, b"encoding") {
        let enc = db.hold(key).pfencoding(key)?;
        out.simple(enc.name().as_bytes());
    } else if is(sub, b"todense") {
        out.int(i64::from(db.hold(key).pftodense(key)?));
    } else {
        return Err(unknown(sub));
    }
    Ok(())
}

/// `ERR Unknown PFDEBUG subcommand 'NOPE'`, quoting it as the client sent it.
fn unknown(sub: &[u8]) -> Error {
    yo_alloc::allow(|| {
        Error::fmt(
            Code::Unsupported,
            format_args!("{UNKNOWN_SUB} '{}'", String::from_utf8_lossy(sub)),
        )
    })
}
