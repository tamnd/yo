//! `_FT.DEBUG`, the window on what an index is actually holding.
//!
//! Eight of the sixty two subcommands a real server registers, and they are the
//! eight that read the structures rather than driving the collector or the
//! background scan: the term dictionary, one term's posting list, one tag
//! field's values, one numeric field's documents, what the table knows about
//! one document, and the three questions about document numbers. Everything
//! here answers about one index and none of it changes anything.
//!
//! # A real server keeps these behind a flag and this one does not
//!
//! Redis refuses every one of them with `Debug commands are disabled, please
//! follow the redis configuration guide to enable them` unless the server was
//! started with `enable-debug-command`. That gate is on the container command
//! and not on the module, so it hides the whole family. Here they always
//! answer, which is D-95: nothing in them can change the keyspace, and a gate
//! that has to be turned on at startup makes a debugging aid useless at the
//! moment somebody needs it.
//!
//! # The error lines have no code word
//!
//! Six of the seven are the module's own sentences with nothing in front of
//! them, `Can not create a search ctx` for an index nobody made and five more,
//! so they are written straight to the buffer rather than going through
//! [`Error`](yo_common::Error), which always writes a code. Only the two the
//! dispatcher owns, the arity of a subcommand and an unknown subcommand, come
//! back as errors and they are the two that carry `ERR`.
//!
//! # A dump reads a name the way a query does
//!
//! `DUMP_TAGIDX i num` over `SCHEMA n AS num TAG` is the tag index of `n`, and
//! `DUMP_TAGIDX i n` over the same schema is a field the index spec does not
//! have. So these take the attribute, the same as `FT.SEARCH` does, and not the
//! identifier the value is read from. Measured, and it is the only sensible
//! reading: the identifier is where the bytes come from and the attribute is
//! what the index calls the thing it built out of them.
//!
//! # A number that means nothing is still in the lists
//!
//! A document that has been deleted, or written again under a number of its
//! own, leaves its old number in every posting list and in every numeric index
//! it was in, because nothing walks back over them to take it out. So a dump
//! answers numbers that [`IDTODOCID`](run) says were removed, which is what a
//! real server does for exactly the same reason and is the whole point of
//! having both subcommands.

use yo_common::Result;
use yo_common::num::parse_i64;
use yo_search::Registry;
use yo_search::field::{Field, Kind};
use yo_search::index::Index;
use yo_search::posts::Id;
use yo_search::sorted::Sorted;

use super::{Args, twelve};
use crate::dispatch::args;
use crate::reply::Out;

/// The name the arity lines and the unknown subcommand line report under.
const NAME: &str = "_FT.DEBUG";

/// No index goes by that name, which the module reports as a failure to build
/// the context it would have read the index through.
const NO_CTX: &[u8] = b"Can not create a search ctx";
/// No document in the index holds that term.
const NO_INVIDX: &[u8] = b"Can not find the inverted index";
/// The index has no field a query would call that, or it has one of the wrong
/// kind, which are one sentence and not two.
const NO_FIELD: &[u8] = b"Could not find given field in index spec";
/// A document number that is not a number at all.
const BAD_ID: &[u8] = b"bad id given";
/// A document number that no key is under any more, which covers a number that
/// was never handed out and one whose key has been deleted or rewritten.
const GONE: &[u8] = b"document was removed";
/// The index is not following that key.
const NO_DOC: &[u8] = b"Document not found in index";
/// `DOCINFO` was given something other than its one keyword.
const BAD_MODE: &[u8] = b"Invalid argument. Expected REVEAL or OBFUSCATE as the last argument";

/// The subcommands this answers, in the order a real server lists its own.
///
/// A real `_FT.DEBUG HELP` names all sixty two whether or not the caller can
/// use them. This names the eight that are here, because a list is worth
/// reading only if the things on it work, which is D-97.
const NAMES: &[&str] = &[
    "DUMP_INVIDX",
    "DUMP_NUMIDX",
    "DUMP_TAGIDX",
    "IDTODOCID",
    "DOCIDTOID",
    "DOCINFO",
    "DUMP_TERMS",
    "GET_MAX_DOC_ID",
];

/// `_FT.DEBUG subcommand index [more]`
///
/// # Errors
///
/// The arity of a subcommand and the unknown subcommand line. Everything an
/// index or a field or a document number complains about is written here and
/// answered as done.
pub(super) fn run(reg: &mut Registry, args: Args<'_>, out: &mut Out) -> Result<()> {
    let sub = args.get(1);
    let wanted = |what: &str| args::wrong_arity_sub(NAME, what);
    match sub {
        // The one subcommand that is not about an index, so it is the one that
        // steps over whatever came after it rather than counting it.
        _ if args::is(sub, b"HELP") => help(out),
        _ if args::is(sub, b"DUMP_TERMS") => {
            if args.len() != 3 {
                return Err(wanted("DUMP_TERMS"));
            }
            if let Some(index) = open(reg, args.get(2), out) {
                terms(index, out);
            }
        }
        _ if args::is(sub, b"DUMP_INVIDX") => {
            if args.len() != 4 {
                return Err(wanted("DUMP_INVIDX"));
            }
            if let Some(index) = open(reg, args.get(2), out) {
                postings(index, args.get(3), out);
            }
        }
        _ if args::is(sub, b"DUMP_TAGIDX") => {
            if args.len() != 4 {
                return Err(wanted("DUMP_TAGIDX"));
            }
            if let Some(index) = open(reg, args.get(2), out) {
                tags(index, args.get(3), out);
            }
        }
        _ if args::is(sub, b"DUMP_NUMIDX") => {
            if args.len() != 4 {
                return Err(wanted("DUMP_NUMIDX"));
            }
            if let Some(index) = open(reg, args.get(2), out) {
                numbers(index, args.get(3), out);
            }
        }
        // The keyword is read at the one place it belongs and anything after it
        // is stepped over, which is why the count here is a floor and not the
        // exact match the rest of them use. The line that complains about the
        // keyword says last argument, but a real server does not look there.
        _ if args::is(sub, b"DOCINFO") => {
            if args.len() < 5 {
                return Err(wanted("DOCINFO"));
            }
            if let Some(index) = open(reg, args.get(2), out) {
                about(index, args.get(3), args.get(4), out);
            }
        }
        _ if args::is(sub, b"IDTODOCID") => {
            if args.len() != 4 {
                return Err(wanted("IDTODOCID"));
            }
            if let Some(index) = open(reg, args.get(2), out) {
                keyed(index, args.get(3), out);
            }
        }
        _ if args::is(sub, b"DOCIDTOID") => {
            if args.len() != 4 {
                return Err(wanted("DOCIDTOID"));
            }
            if let Some(index) = open(reg, args.get(2), out) {
                numbered(index, args.get(3), out);
            }
        }
        _ if args::is(sub, b"GET_MAX_DOC_ID") => {
            if args.len() != 3 {
                return Err(wanted("GET_MAX_DOC_ID"));
            }
            if let Some(index) = open(reg, args.get(2), out) {
                out.uint(u64::from(index.held.docs.last()));
            }
        }
        _ => return Err(args::unknown_subcommand(sub, NAME)),
    }
    Ok(())
}

/// The index a name means, having written the failure line if there is none.
///
/// Opened rather than looked up, so an alias resolves and the open is counted:
/// a real server's `number_of_uses` moves by one for each of these, the same as
/// for an `FT.INFO`, because reading an index is reading an index whoever is
/// asking.
fn open<'a>(reg: &'a mut Registry, name: &[u8], out: &mut Out) -> Option<&'a Index> {
    // Looked for first and opened second, because the borrow the open hands
    // back has to outlive this and one that failed cannot be dropped inside a
    // match arm without the borrow checker keeping it alive over the error.
    if reg.get(name).is_none() {
        out.error(NO_CTX);
        return None;
    }
    reg.open(name).map(|index| &*index)
}

/// Every subcommand that is here, one bulk string each.
fn help(out: &mut Out) {
    out.array(NAMES.len());
    for name in NAMES {
        out.bulk(name.as_bytes());
    }
}

/// `DUMP_TERMS index`, the whole term dictionary in byte order.
///
/// Stems and synonym groups are in it beside the words they stand for, spelled
/// the way the dictionary holds them, so a document holding `running` puts both
/// `running` and `+run` on this list.
fn terms(index: &Index, out: &mut Out) {
    out.array(index.held.words());
    for term in index.held.terms() {
        out.bulk(term);
    }
}

/// `DUMP_INVIDX index term`, the documents one term is in.
///
/// The term is matched byte for byte and nothing folds it, so a query that
/// would have found `RUNNING` by folding it first does not mean this will.
fn postings(index: &Index, term: &[u8], out: &mut Out) {
    let Some(posts) = index.held.posts(term) else {
        out.error(NO_INVIDX);
        return;
    };
    out.array(posts.len() as usize);
    let mut reader = posts.read();
    while let Some(post) = reader.step() {
        out.uint(u64::from(post.id));
    }
}

/// `DUMP_TAGIDX index field`, every value the field holds and who holds it.
///
/// A pair per value in byte order, and the values are the ones the index made:
/// folded unless the field is case sensitive, and split on the separator.
fn tags(index: &Index, attribute: &[u8], out: &mut Out) {
    if !kind(index, attribute, |kind| matches!(kind, Kind::Tag(_))) {
        out.error(NO_FIELD);
        return;
    }
    let Some(held) = index.held.values(attribute) else {
        out.array(0);
        return;
    };
    out.array(held.len());
    for (value, ids) in held.all() {
        out.array(2);
        out.bulk(value);
        out.array(ids.len());
        for id in ids {
            out.uint(u64::from(*id));
        }
    }
}

/// `DUMP_NUMIDX index field`, the documents a numeric field holds.
///
/// A list of lists, because a real server splits a numeric field into ranges
/// once it is worth splitting and answers one list per range. This keeps one
/// ordered list per field and so answers one range, which is D-96, and a field
/// nobody has written to answers no ranges at all rather than one empty one.
///
/// A `GEO` field answers here too, since a point is a number underneath on both
/// servers.
fn numbers(index: &Index, attribute: &[u8], out: &mut Out) {
    let numeric = kind(index, attribute, |kind| matches!(kind, Kind::Numeric));
    let geo = kind(index, attribute, |kind| matches!(kind, Kind::Geo));
    if !numeric && !geo {
        out.error(NO_FIELD);
        return;
    }
    let ids = match numeric {
        true => index.held.numbers(attribute).map(|held| held.ids()),
        false => index.held.places(attribute).map(|held| held.ids()),
    };
    let Some(ids) = ids.filter(|ids| !ids.is_empty()) else {
        out.array(0);
        return;
    };
    out.array(1);
    out.array(ids.len());
    for id in ids {
        out.uint(u64::from(id));
    }
}

/// `DOCINFO index key REVEAL|OBFUSCATE`, the document table's row for one key.
///
/// The key is looked up before the keyword is read, which is a real server's
/// order and is visible: a bad keyword over a key the index is not following
/// answers about the key. `OBFUSCATE` replaces every field name with its place
/// in the schema, which is what a server logging one of these writes when it
/// has been told to keep user data out of its logs.
fn about(index: &Index, key: &[u8], mode: &[u8], out: &mut Out) {
    let Some(id) = index.held.docs.id(key) else {
        out.error(NO_DOC);
        return;
    };
    let hide = if args::is(mode, b"OBFUSCATE") {
        true
    } else if args::is(mode, b"REVEAL") {
        false
    } else {
        out.error(BAD_MODE);
        return;
    };
    let Some(doc) = index.held.docs.get(id) else {
        out.error(NO_DOC);
        return;
    };
    // Which sortable fields the schema has and what this document put in them.
    // A document written before an `FT.ALTER` added a sortable field has
    // nothing in the slot that field took, which reads the same as a document
    // that simply had nothing to put there.
    let sortable: Vec<(usize, &Field, Option<&Sorted>)> = index
        .schema
        .iter()
        .enumerate()
        .filter(|(_, field)| field.sortable)
        .enumerate()
        .map(|(slot, (at, field))| (at, field, doc.sorted(slot)))
        .collect();
    let carried = sortable.iter().any(|(_, _, value)| value.is_some());
    let pairs = 6 + usize::from(carried);
    if out.proto().is_resp3() {
        out.map(pairs);
    } else {
        out.array(pairs * 2);
    }
    out.simple(b"internal_id");
    out.uint(u64::from(id));
    out.bulk(b"flags");
    out.bulk(&flags(index, doc.payload.is_some(), carried));
    out.simple(b"score");
    out.double(doc.score);
    out.simple(b"num_tokens");
    out.uint(u64::from(doc.tokens));
    out.simple(b"max_freq");
    out.uint(u64::from(doc.top));
    // A real server counts the handles open on the row and nothing here holds
    // one, so this is the number a caller reading it back would see: the one
    // the table itself has.
    out.simple(b"refcount");
    out.uint(1);
    if !carried {
        return;
    }
    out.simple(b"sortables");
    out.array(sortable.len());
    for (slot, (at, field, value)) in sortable.iter().enumerate() {
        out.array(6);
        out.simple(b"index");
        out.uint(slot as u64);
        out.bulk(b"field");
        out.bulk(&named(field, *at, hide));
        out.bulk(b"value");
        match value {
            Some(Sorted::Number(number)) => out.bulk(twelve(*number).as_bytes()),
            Some(Sorted::Text(text)) => out.bulk(text),
            None => out.nil(),
        }
    }
}

/// The word for a document's flags, which is the number and then the names.
///
/// Three bits of the four a real server has: the fourth marks a row it has
/// collected and nothing here collects yet. The offsets bit follows what the
/// index was created with rather than what the write path did, the same way it
/// does there.
fn flags(index: &Index, payload: bool, sortable: bool) -> Vec<u8> {
    let offsets = !index.definition.options.nooffsets;
    let bits = (u8::from(payload) << 1) | (u8::from(sortable) << 2) | (u8::from(offsets) << 3);
    let mut out = format!("(0x{bits:x}):").into_bytes();
    for (held, name) in [
        (payload, "HasPayload,"),
        (sortable, "HasSortVector,"),
        (offsets, "HasOffsetVector,"),
    ] {
        if held {
            out.extend_from_slice(name.as_bytes());
        }
    }
    out
}

/// What `DOCINFO` calls one field, which is where it reads from and what a
/// query calls it, or the two places it sits in the schema when the caller
/// asked for the names to be kept back.
fn named(field: &Field, at: usize, hide: bool) -> Vec<u8> {
    let mut out = Vec::new();
    if hide {
        out.extend_from_slice(format!("FieldPath@{at} AS Field@{at}").as_bytes());
        return out;
    }
    out.extend_from_slice(&field.identifier);
    out.extend_from_slice(b" AS ");
    out.extend_from_slice(&field.attribute);
    out
}

/// `IDTODOCID index id`, the key a document number belongs to.
///
/// The number is read the way Redis reads every integer, so a leading zero, a
/// leading plus and a space in front are all refused, and one that reads fine
/// and names nothing answers that the document was removed. That covers a
/// number nobody ever handed out as well as one whose key has gone, since the
/// table cannot tell those apart and neither can a real server.
fn keyed(index: &Index, arg: &[u8], out: &mut Out) {
    let Some(number) = parse_i64(arg) else {
        out.error(BAD_ID);
        return;
    };
    let id = Id::try_from(number).ok();
    match id.and_then(|id| index.held.docs.key(id)) {
        Some(key) => out.bulk(key),
        None => out.error(GONE),
    }
}

/// `DOCIDTOID index key`, the number a key is indexed under.
///
/// Nought for a key the index is not following, which is why the numbers start
/// at one: there has to be a number that means no document.
fn numbered(index: &Index, key: &[u8], out: &mut Out) {
    out.uint(u64::from(index.held.docs.id(key).unwrap_or(0)));
}

/// Whether the index has a field a query would call this name and it is the
/// kind the caller wants.
fn kind(index: &Index, attribute: &[u8], want: impl Fn(&Kind) -> bool) -> bool {
    index
        .field(attribute)
        .is_some_and(|field| want(&field.kind))
}
