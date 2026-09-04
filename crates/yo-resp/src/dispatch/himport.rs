//! `HIMPORT`, and the fieldsets a connection prepares to feed it.
//!
//! Redis 8.10 added this as a container of four subcommands over one idea. A
//! client that is about to write a million hashes with the same shape tells the
//! connection the shape once, with `HIMPORT PREPARE`, and then sends only the
//! values for each key. The field names travel over the wire once instead of a
//! million times, which is the whole saving on the client side, and on the
//! server side it is what lets Redis store the names once for the family rather
//! than once per key.
//!
//! # What is here and what is not
//!
//! The command surface is all here and it is exact. The storage saving is not,
//! because the size ladder in [`yo_kv::hash`] already answers the question a
//! different way and adding a second answer to it is a storage project rather
//! than a command. So a key written by `HIMPORT SET` is an ordinary hash here,
//! it reads back byte for byte the same as it does on a real server, and the one
//! thing that differs is the word `OBJECT ENCODING` puts on it. That is D-45.
//!
//! # Where the fieldsets live
//!
//! On the connection, which is the surprising part and is the reference's
//! design rather than a choice made here. A fieldset prepared on one connection
//! is invisible to every other one, `RESET` throws them all away and `SELECT`
//! does not, and a key built from a fieldset outlives the fieldset and the
//! connection both. So the registry hangs off [`Session`] next to the database
//! number and the client name, and no part of the keyspace knows it exists.
//!
//! [`Session`]: super::Session

use yo_common::{Code, Error, Result};
use yo_kv::{Db, Keyspace};

use super::args::{self, Args};
use crate::reply::Out;

/// What a real server says when one `PREPARE` names the same field twice.
const DUPLICATE: &str = "duplicate field name in fieldset";
/// And when `SET` names a fieldset this connection never prepared.
const NO_FIELDSET: &str = "no such fieldset";
/// And when the values do not fill the fieldset exactly, either way.
const BAD_COUNT: &str = "value count does not match fieldset field count";

/// One prepared fieldset: a name, and the shape it stands for.
struct Fieldset {
    /// The name `HIMPORT SET` will ask for, compared byte for byte.
    name: Vec<u8>,
    /// The fields in the order they are written, each carrying the position of
    /// the value that fills it.
    ///
    /// Two orders are in play and the pair is what keeps them apart. The values
    /// on a `SET` line arrive in the order the fields were declared in, and the
    /// hash is built in sorted order, so the field that sorts first is not
    /// generally the one the first value belongs to. Storing the declared
    /// position alongside the name means the sort happens once, here, rather
    /// than once per key on the write path.
    fields: Vec<(Vec<u8>, usize)>,
}

/// Everything one connection has prepared.
///
/// A vector and a walk rather than a map. The reference documents `DISCARD` as
/// linear in the number of fieldsets, so it is a walk there too, and a
/// connection holding enough of these for the difference to show is a connection
/// that has misunderstood the command: the point of it is a handful of shapes
/// reused across a great many keys.
#[derive(Default)]
pub(super) struct Fieldsets(Vec<Fieldset>);

impl Fieldsets {
    /// `HIMPORT PREPARE name field [field ...]`.
    ///
    /// Built to one side and only put in place at the end, because a duplicate
    /// field is an error and a `PREPARE` that fails has to leave the name
    /// pointing at whatever it pointed at before. Measured against 8.10.1
    /// rather than assumed, since the other reading, that a failed prepare
    /// clears the name, is just as plausible from the outside.
    fn prepare(&mut self, args: Args<'_>) -> Result<()> {
        let mut fields: Vec<(Vec<u8>, usize)> = yo_alloc::allow(|| {
            (3..args.len())
                .map(|i| (args.get(i).to_vec(), i - 3))
                .collect()
        });
        // Length first and bytes second, which is not the ordering anything
        // else in this codebase sorts names with and is what the reference
        // does: `HKEYS` on a key built from `b` and `aa` answers `b` then `aa`,
        // where a plain byte comparison would answer the other way round.
        fields.sort_by(|(a, _), (b, _)| (a.len(), a.as_slice()).cmp(&(b.len(), b.as_slice())));
        if fields.windows(2).any(|w| w[0].0 == w[1].0) {
            return Err(Error::new(Code::Invalid, DUPLICATE));
        }

        let name = args.get(2);
        yo_alloc::allow(|| match self.0.iter_mut().find(|f| f.name == name) {
            Some(old) => old.fields = fields,
            None => self.0.push(Fieldset {
                name: name.to_vec(),
                fields,
            }),
        });
        Ok(())
    }

    /// The fieldset under this name, if this connection prepared one.
    fn get(&self, name: &[u8]) -> Option<&Fieldset> {
        self.0.iter().find(|f| f.name == name)
    }

    /// `HIMPORT DISCARD name`. Answers whether there was one.
    fn discard(&mut self, name: &[u8]) -> bool {
        let Some(at) = self.0.iter().position(|f| f.name == name) else {
            return false;
        };
        self.0.swap_remove(at);
        true
    }

    /// `HIMPORT DISCARDALL`. Answers how many there were.
    fn discard_all(&mut self) -> usize {
        let had = self.0.len();
        self.0.clear();
        had
    }

    /// Throw everything away, for `RESET`.
    pub(super) fn clear(&mut self) {
        self.0.clear();
    }
}

/// Run one `HIMPORT` subcommand.
///
/// The container's arity is a minimum of two, so the table has already refused a
/// bare `HIMPORT`, and each subcommand checks its own count here. The name of
/// the command an arity complaint carries is the container and the subcommand
/// joined by a pipe, the same as `OBJECT` and `CONFIG`.
pub(super) fn execute(
    db: &mut Db,
    sets: &mut Fieldsets,
    args: Args<'_>,
    out: &mut Out,
) -> Result<()> {
    let sub = args.get(1);
    if args::is(sub, b"prepare") {
        // At least one field, so the shortest legal line is four words.
        if args.len() < 4 {
            return Err(args::wrong_arity_sub("himport", "prepare"));
        }
        sets.prepare(args)?;
        out.ok();
    } else if args::is(sub, b"set") {
        if args.len() < 5 {
            return Err(args::wrong_arity_sub("himport", "set"));
        }
        // The only subcommand that touches the keyspace, and it names its key
        // third, so that key's stripe is what it is given.
        set(db.at(args.get(2)), sets, args, out)?;
    } else if args::is(sub, b"discard") {
        if args.len() != 3 {
            return Err(args::wrong_arity_sub("himport", "discard"));
        }
        out.int(i64::from(sets.discard(args.get(2))));
    } else if args::is(sub, b"discardall") {
        if args.len() != 2 {
            return Err(args::wrong_arity_sub("himport", "discardall"));
        }
        out.int(i64::try_from(sets.discard_all()).unwrap_or(i64::MAX));
    } else {
        // The reference points at a `HIMPORT HELP` that does not exist: sending
        // it gets this same sentence back. Copied as it is, because the message
        // is what a client sees and an accurate one here would be the odd one
        // out among every other container.
        return Err(args::unknown_subcommand(sub, "HIMPORT"));
    }
    Ok(())
}

/// `HIMPORT SET key fieldset value [value ...]`.
fn set(db: &mut Keyspace, sets: &Fieldsets, args: Args<'_>, out: &mut Out) -> Result<()> {
    let key = args.get(2);
    // The type of the key is asked first and beats both of the complaints
    // below it, so `HIMPORT SET str nope 1` against a string is a WRONGTYPE
    // and not a missing fieldset even though the fieldset really is missing.
    // That ordering is the reference's and it is the reason this lookup is
    // here rather than being left to `hreplace` further down.
    db.hlen(key)?;
    let Some(fs) = sets.get(args.get(3)) else {
        return Err(Error::new(Code::Invalid, NO_FIELDSET));
    };
    // One message for too few and too many alike, which is unusual and is what
    // the reference says on both.
    if args.len() - 4 != fs.fields.len() {
        return Err(Error::new(Code::Invalid, BAD_COUNT));
    }
    // Nothing is collected: the pairs are the sorted field list walked in place
    // against the argument buffer, so a fieldset of a thousand fields writes a
    // thousand pairs and allocates none of them.
    db.hreplace(
        key,
        fs.fields
            .iter()
            .map(|(field, at)| (field.as_slice(), args.get(at + 4))),
    )?;
    out.ok();
    Ok(())
}
