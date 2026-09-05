//! The search index commands, on the wire (`09` section 5).
//!
//! Sixteen of them, and all sixteen are about the index itself rather than
//! about what is in it: making one, changing its schema, taking it away,
//! describing it, listing them and naming them. `FT.SEARCH` and everything that
//! reads documents comes after this, on top of the same [`Registry`].
//!
//! # There is no key here
//!
//! Every other group in this file's neighbourhood is handed one [`Keyspace`]
//! and works on a key inside it. This one is handed the registry off the server
//! and works on a name that is not a key: an index does not appear in `KEYS *`,
//! `TYPE` has nothing to say about it, and `DEL` cannot remove one. That is
//! Redis's model and not a shortcut, and it is why the registry hangs off
//! [`Server`](super::Server) rather than off a database. `SELECT 1` followed by
//! `FT._LIST` lists the indexes made on database zero, on a real server and on
//! this one.
//!
//! Emptying the keyspace does empty it, though, and from any database:
//! `FLUSHDB` on database nine drops an index that only ever followed keys on
//! database zero. That is measured against a real server rather than reasoned
//! about, and it falls out of the module hanging its callback on the flush
//! event without looking at which database flushed.
//!
//! # The errors are the module's own
//!
//! Nothing here goes through [`Error`](yo_common::Error), which is the type the
//! rest of dispatch answers with and which always comes out under `ERR`,
//! `WRONGTYPE` or `INVALIDOBJ`. The search module has its own code words,
//! `SEARCH_INDEX_EXISTS` and `SEARCH_PARSE_ARGS` and six more, and two of its
//! errors have no code word at all. A client branches on the first word of an
//! error line, so those are copied exactly, down to `Alias does not exist`
//! having no prefix and `Tag separator must be a single character. Got `%s``
//! carrying an unexpanded format specifier that a real server never fills in.
//!
//! That is what [`Fail`] is for. It holds the line in the pieces it is built
//! from, so the one piece that came from the client is the one piece that gets
//! its line endings taken out, and so nothing is allocated to answer a bad
//! argument.
//!
//! # The field option grammar is two loops
//!
//! `SCHEMA t TEXT WITHSUFFIXTRIE SORTABLE` is accepted and
//! `SCHEMA t TEXT SORTABLE WITHSUFFIXTRIE` is not, which is not a typo in
//! either the test or the server. A field is read by two loops, one after the
//! other. The first takes the options that belong to the type, plus
//! `WITHSUFFIXTRIE`, `INDEXEMPTY` and `INDEXMISSING`, in any order. The second
//! takes `SORTABLE [UNF]` and `NOINDEX`, in any order and repeatable, and it
//! never goes back. So a first loop keyword arriving after a second loop
//! keyword is not an option at all: the field is over, and the word is read as
//! the name of the next field, which then has no type after it and says so.
//!
//! This was worked out by measuring a real server rather than by reading its
//! source, and every accept and reject observed follows from it.
//!
//! # What `FT.INFO` cannot say yet
//!
//! Two dozen of its fields are RediSearch's own internals: how many megabytes
//! its inverted index is holding, how many blocks it is in, what its garbage
//! collector has collected. There is no inverted index under this yet and the
//! ones there will be are not theirs, so those fields are reported as zero
//! rather than as an invented number. The four averages are `nan`, which is
//! what a real server answers for an index with no documents in it too, because
//! all four are a division by the document count. D-58 has the whole list.

use yo_common::num::parse_f64;
use yo_common::{Result, parse_i64};
use yo_search::field::{self, Algo, Coords, Kind, Tag, Text, Vector, Width};
use yo_search::follow::Errors;
use yo_search::index::{Definition, Source};
use yo_search::query::{self, Ask, Bad, Pair};
use yo_search::{Clash, Field, Index, Registry};
use yo_shape::Metric;

use super::args::{self, Args};
use super::table::Spec;
use crate::reply::Out;

/// The languages a document may be stemmed in, in the spelling `FT.INFO`
/// reports them in.
///
/// The client's spelling is matched against these without regard to case and
/// the entry here is what gets stored, so `LANGUAGE FRENCH` and
/// `LANGUAGE French` both come back as `french`.
const LANGUAGES: &[&[u8]] = &[
    b"arabic",
    b"armenian",
    b"chinese",
    b"danish",
    b"dutch",
    b"english",
    b"finnish",
    b"french",
    b"german",
    b"greek",
    b"hungarian",
    b"indonesian",
    b"irish",
    b"italian",
    b"lithuanian",
    b"nepali",
    b"norwegian",
    b"portuguese",
    b"romanian",
    b"russian",
    b"serbian",
    b"spanish",
    b"swedish",
    b"tamil",
    b"turkish",
    b"yiddish",
];

/// The phonetic matchers a `TEXT` field may ask for.
///
/// Four letters, a two letter algorithm and a two letter language with a colon
/// between them. Only one algorithm exists, so the first half is always `dm`,
/// and the second half is one of these four.
const PHONETICS: &[&[u8]] = &[b"dm:en", b"dm:fr", b"dm:pt", b"dm:es"];

/// One error line, in the pieces it is built from.
///
/// The middle piece is the client's own bytes and the two around it are the
/// server's text, which is the only arrangement any of these lines take. A line
/// with nothing to quote has an empty middle and an empty tail, which writes
/// the head on its own.
struct Fail<'a> {
    head: &'static str,
    word: &'a [u8],
    tail: &'static str,
}

impl<'a> Fail<'a> {
    /// A line that says the same thing whatever the client sent.
    const fn plain(head: &'static str) -> Fail<'a> {
        Fail {
            head,
            word: b"",
            tail: "",
        }
    }

    /// A line that ends with a word the client sent.
    const fn naming(head: &'static str, word: &'a [u8]) -> Fail<'a> {
        Fail {
            head,
            word,
            tail: "",
        }
    }

    /// A line with a word the client sent in the middle of it.
    const fn about(head: &'static str, word: &'a [u8], tail: &'static str) -> Fail<'a> {
        Fail { head, word, tail }
    }

    /// Writes it out.
    fn write(&self, out: &mut Out) {
        out.error_about(self.head.as_bytes(), self.word, self.tail.as_bytes());
    }
}

/// What a command answers when it did not work.
type Answer<'a> = core::result::Result<(), Fail<'a>>;

const EXISTS: &str = "SEARCH_INDEX_EXISTS Index already exists";
/// What `FT.CREATE` says anywhere but on database zero.
///
/// The only sentence in the group with no code word in front of it, which reads
/// like an oversight and is what a real server sends. Only the two creates
/// refuse. `FT.ALTER` and `FT.DROPINDEX` work from any database, and the rest
/// answer the same everywhere, since the registry is one table for the server.
const NOT_DB_ZERO: &str = "Cannot create index on db != 0";
const ALIAS_EXISTS: &str = "SEARCH_INDEX_EXISTS Alias already exists";
const MISSING: &str = "SEARCH_INDEX_NOT_FOUND Index not found: ";
const CONFLICT: &str = "SEARCH_ALIAS_CONFLICT Alias conflicts with an existing index name";
const NOT_MINE: &str = "SEARCH_INDEX_NOT_FOUND Alias does not belong to provided spec";
const NO_ALIAS: &str = "Alias does not exist";
/// What the two `ALIASADD` spellings answer when the index is not there, which
/// is not the line the rest of the group uses for the same thing.
const NO_TARGET: &str = "SEARCH_INDEX_NOT_FOUND Unknown index name (or name is an alias itself)";
const NO_SCHEMA: &str = "SEARCH_PARSE_ARGS No schema found";
const NO_FIELDS: &str = "SEARCH_PARSE_ARGS Fields arguments are missing";
const AFTER_ALTER: &str = "ALTER must be followed by SCHEMA";
const ALTER_ACTION: &str = "Unknown action passed to ALTER SCHEMA";
const UNKNOWN: &str = "SEARCH_ARG_UNRECOGNIZED Unknown argument `";
const UNKNOWN_BARE: &str = "SEARCH_ARG_UNRECOGNIZED Unknown argument";
const NO_TYPE: &str = "SEARCH_PARSE_ARGS Field `";
const NO_TYPE_END: &str = "` does not have a type";
const BAD_TYPE: &str = "SEARCH_PARSE_ARGS Invalid field type for field `";
const DUPLICATE: &str = "SEARCH_QUERY_BAD Duplicate field in schema - ";
const RULE: &str = "SEARCH_ADD_ARGS Invalid rule type: ";
const LANGUAGE: &str = "SEARCH_ADD_ARGS Invalid language";
const SCORE: &str = "SEARCH_ADD_ARGS Invalid score";
const BOTH: &str =
    "SEARCH_PARSE_ARGS 'Field cannot be defined with both `NOINDEX` and `INDEXMISSING` `";
const SEPARATOR: &str = "SEARCH_PARSE_ARGS Tag separator must be a single character. Got `%s`";
const MATCHER: &str = "SEARCH_QUERY_BAD Matcher Format: <2 chars algorithm>:<2 chars language>. Support algorithms: double metaphone (dm). Supported languages: English (en), French (fr), Portuguese (pt) and Spanish (es)";
const AS_ARG: &str = "SEARCH_PARSE_ARGS AS requires an argument";
const TOO_MANY_TEXT: &str = "SEARCH_QUERY_BAD MAXTEXTFIELDS cannot be used with NOFIELDS";
const BAD_ARGS: &str = "SEARCH_PARSE_ARGS Bad arguments for ";
const NOT_ENOUGH: &str =
    "SEARCH_PARSE_ARGS Bad arguments for vector similarity: not enough arguments";
/// The two halves of the vector grammar that name no keyword of their own, which
/// a real server writes as though they were keywords.
const ALGO_WORD: &str = "vector similarity algorithm";
const COUNT_WORD: &str = "vector similarity number of parameters";
/// What `TRAINING_THRESHOLD` and `REDUCE` answer when they were sent without the
/// compression they only mean something with.
const NO_TRAINING: &str =
    "SEARCH_PARSE_ARGS TRAINING_THRESHOLD is irrelevant when compression was not requested";
const NO_REDUCE: &str =
    "SEARCH_PARSE_ARGS REDUCE is irrelevant when compression is not of type LeanVec";
const SMALL_TRAINING: &str =
    "SEARCH_PARSE_ARGS Invalid TRAINING_THRESHOLD: cannot be lower than DEFAULT_BLOCK_SIZE (1024)";

/// The arity line the two `DROPINDEX` spellings answer with, which names the
/// command with an underscore in front of it. That is not a typo here: a real
/// server answers `'_FT.DROPINDEX'` and `'_FT._DROPINDEXIFX'`, because the
/// module registers the pair under those names and the coordinator wraps them.
const ARITY: &str = "ERR wrong number of arguments for '_";
/// The same line without the underscore, which is what everything else uses.
const ARITY_PLAIN: &str = "ERR wrong number of arguments for '";
/// The tail of both.
const ARITY_END: &str = "' command";

/// The three ways a value can be wrong, in the words a real server ends the
/// `Bad arguments for X` lines with.
const NOT_A_NUMBER: &str = ": Could not convert argument to expected type";
const OUT_OF_RANGE: &str = ": Value is outside acceptable bounds";
const NOT_THERE: &str = ": Expected an argument, but none provided";
const UNKNOWN_WORD: &str = ": Unknown argument";

/// The same four, for the vector lines that quote the keyword in backticks. The
/// closing one belongs to the tail because a line is a head, a client word and a
/// tail, and the keyword is the client's word.
const V_NOT_A_NUMBER: &str = "`: Could not convert argument to expected type";
const V_OUT_OF_RANGE: &str = "`: Value is outside acceptable bounds";
const V_NOT_THERE: &str = "`: Expected an argument, but none provided";
const V_UNKNOWN: &str = "`: Unknown argument";

/// `Bad arguments for X: why`, where `X` is the keyword the value belonged to.
fn bad(what: &'static str, why: &'static str) -> Fail<'static> {
    Fail::about(BAD_ARGS, what.as_bytes(), why)
}

pub(super) fn execute<'a>(
    reg: &mut Registry,
    db: usize,
    spec: &Spec,
    args: Args<'a>,
    out: &mut Out,
) -> Result<Option<&'a [u8]>> {
    // The name of an index that was made here, so the caller can run the scan
    // over the keys that were already there. It stays `None` for the other
    // sixteen commands, and for a create that answered that the name is taken.
    let mut made = None;
    let done = match spec.name {
        "FT.CREATE" => create(reg, db, args, out, false).map(|name| made = name),
        "FT._CREATEIFNX" => create(reg, db, args, out, true).map(|name| made = name),
        "FT.ALTER" => alter(reg, args, out, false),
        "FT._ALTERIFNX" => alter(reg, args, out, true),
        "FT.DROPINDEX" => drop_index(reg, spec, args, out, false, true),
        "FT._DROPINDEXIFX" => drop_index(reg, spec, args, out, true, true),
        "FT.DROP" => drop_index(reg, spec, args, out, false, false),
        "FT._DROPIFX" => drop_index(reg, spec, args, out, true, false),
        "FT.INFO" => info(reg, args, out),
        "FT._LIST" => list(reg, spec, args, out),
        "FT.ALIASADD" => alias_add(reg, args, out, false),
        "FT._ALIASADDIFNX" => alias_add(reg, args, out, true),
        "FT.ALIASDEL" => alias_del(reg, args, out, false),
        "FT._ALIASDELIFX" => alias_del(reg, args, out, true),
        "FT.ALIASUPDATE" => alias_update(reg, args, out),
        "FT.ALIASLIST" => alias_list(reg, args, out),
        "FT.EXPLAIN" => explain(reg, args, out, false),
        "FT.EXPLAINCLI" => explain(reg, args, out, true),
        other => unreachable!("{other} is not a search command"),
    };
    if let Err(f) = done {
        f.write(out);
        return Ok(None);
    }
    Ok(made)
}

/// `FT.CREATE index [options] SCHEMA field type [options] ...`
///
/// The name is checked before the arguments are, which is a real server's order
/// and is visible: `FT.CREATE i BOGUS SCHEMA t TEXT` over an index that already
/// exists answers that it exists rather than that `BOGUS` is not an argument.
///
/// The name comes back when an index was made, because the keys that already
/// match its prefix have to be read into it and this is the only place that
/// knows one was made. `FT._CREATEIFNX` over a name that is taken answers `OK`
/// and hands back nothing, since there is nothing new to fill.
fn create<'a>(
    reg: &mut Registry,
    db: usize,
    args: Args<'a>,
    out: &mut Out,
    ifnx: bool,
) -> core::result::Result<Option<&'a [u8]>, Fail<'a>> {
    let name = args.get(1);
    // The `IFNX` shortcut comes first and the database comes second, which is
    // the order a real server checks them in and is visible: `FT._CREATEIFNX`
    // over a name that is taken answers `OK` from database one, while
    // `FT.CREATE` over the same name from database one refuses on the database
    // rather than on the name.
    if ifnx && reg.named(name).is_some() {
        out.ok();
        return Ok(None);
    }
    if db != 0 {
        return Err(Fail::plain(NOT_DB_ZERO));
    }
    if reg.named(name).is_some() {
        return Err(Fail::plain(EXISTS));
    }

    let (definition, at) = definition(args, 2)?;
    let mut schema = Vec::new();
    fields(args, at, &mut schema)?;

    // The name is free and the arguments parsed, so nothing below can fail and
    // leave half an index behind.
    let _ = reg.create(Index::new(name, definition, schema));
    out.ok();
    Ok(Some(name))
}

/// The options in front of `SCHEMA`, and where the schema starts.
///
/// Any order, which is what a real server takes. Every keyword that carries a
/// count takes exactly that many words after it whatever they say, so
/// `PREFIX 2 a: SCHEMA t TEXT` reads `SCHEMA` as the second prefix and then
/// trips over `t`, which is the error a real server answers too.
fn definition(args: Args<'_>, from: usize) -> core::result::Result<(Definition, usize), Fail<'_>> {
    let mut d = Definition::default();
    let mut prefixes: Option<Vec<Box<[u8]>>> = None;
    let mut at = from;
    loop {
        let Some(a) = args.opt(at) else {
            return Err(Fail::plain(NO_SCHEMA));
        };
        at += 1;
        if args::is(a, b"schema") {
            break;
        } else if args::is(a, b"on") {
            let v = value(args, &mut at, "ON")?;
            d.on = if args::is(v, b"hash") {
                Source::Hash
            } else if args::is(v, b"json") {
                Source::Json
            } else {
                return Err(Fail::naming(RULE, v));
            };
        } else if args::is(a, b"prefix") {
            let n = count(args, &mut at, "PREFIX")?;
            let mut list = Vec::with_capacity(n);
            for _ in 0..n {
                list.push(value(args, &mut at, "PREFIX")?.into());
            }
            prefixes = Some(list);
        } else if args::is(a, b"filter") {
            d.filter = Some(value(args, &mut at, "FILTER")?.into());
        } else if args::is(a, b"language") {
            let v = value(args, &mut at, "LANGUAGE")?;
            d.language = Some(language(v).ok_or(Fail::plain(LANGUAGE))?.into());
        } else if args::is(a, b"language_field") {
            d.language_field = Some(value(args, &mut at, "LANGUAGE_FIELD")?.into());
        } else if args::is(a, b"score") {
            let v = value(args, &mut at, "SCORE")?;
            // Bounded to a fraction, both ends included, and anything else is
            // the same error as text that is not a number at all.
            d.score = match parse_f64(v) {
                Some(s) if (0.0..=1.0).contains(&s) => s,
                _ => return Err(Fail::plain(SCORE)),
            };
        } else if args::is(a, b"score_field") {
            d.score_field = Some(value(args, &mut at, "SCORE_FIELD")?.into());
        } else if args::is(a, b"payload_field") {
            d.payload_field = Some(value(args, &mut at, "PAYLOAD_FIELD")?.into());
        } else if args::is(a, b"temporary") {
            d.temporary = Some(count(args, &mut at, "TEMPORARY")? as u64);
        } else if args::is(a, b"stopwords") {
            let n = count(args, &mut at, "STOPWORDS")?;
            let mut list = Vec::with_capacity(n);
            for _ in 0..n {
                list.push(value(args, &mut at, "STOPWORDS")?.into());
            }
            d.stopwords = Some(list);
        } else if args::is(a, b"maxtextfields") {
            d.options.maxtextfields = true;
        } else if args::is(a, b"nooffsets") {
            d.options.nooffsets = true;
        } else if args::is(a, b"nohl") {
            d.options.nohl = true;
        } else if args::is(a, b"nofields") {
            d.options.nofields = true;
        } else if args::is(a, b"nofreqs") {
            d.options.nofreqs = true;
        } else if args::is(a, b"skipinitialscan") {
            d.skip_initial_scan = true;
        } else {
            return Err(Fail::about(UNKNOWN, a, "`"));
        }
    }
    // Room for more text fields than the sixteen a header has bits for, next to
    // an order to keep no field bits at all, is a pair that cannot both be
    // honoured. Either order is refused and the sentence names the first one.
    if d.options.maxtextfields && d.options.nofields {
        return Err(Fail::plain(TOO_MANY_TEXT));
    }
    // An index that named no prefixes and one that named none out of a count of
    // zero are the same index, and both follow every key. The one empty prefix
    // is what `FT.INFO` reports for both.
    if let Some(list) = prefixes
        && !list.is_empty()
    {
        d.prefixes = list;
    }
    Ok((d, at))
}

/// The language this spelling means, in the spelling that gets stored.
fn language(v: &[u8]) -> Option<&'static [u8]> {
    LANGUAGES.iter().copied().find(|l| args::is(v, l))
}

/// The next word, or the line that says the keyword ran out of arguments.
fn value<'a>(
    args: Args<'a>,
    at: &mut usize,
    what: &'static str,
) -> core::result::Result<&'a [u8], Fail<'a>> {
    let v = args.opt(*at).ok_or_else(|| bad(what, NOT_THERE))?;
    *at += 1;
    Ok(v)
}

/// The next word as a count of the words after it.
fn count<'a>(
    args: Args<'a>,
    at: &mut usize,
    what: &'static str,
) -> core::result::Result<usize, Fail<'a>> {
    let v = value(args, at, what)?;
    let n = parse_i64(v).ok_or_else(|| bad(what, NOT_A_NUMBER))?;
    usize::try_from(n).map_err(|_| bad(what, OUT_OF_RANGE))
}

/// Every field after `SCHEMA`, appended to what is already there.
///
/// `into` arrives holding the fields the index already has, which is empty for
/// `FT.CREATE` and the whole current schema for `FT.ALTER`, so a field that
/// clashes with an old one and a field that clashes with a new one are found
/// the same way and in the order the client wrote them.
fn fields<'a>(args: Args<'a>, from: usize, into: &mut Vec<Field>) -> Answer<'a> {
    if args.opt(from).is_none() {
        return Err(Fail::plain(NO_FIELDS));
    }
    let mut at = from;
    while at < args.len() {
        let f = one_field(args, &mut at, into)?;
        into.push(f);
    }
    Ok(())
}

/// One field: where it reads from, what a query calls it, what it holds and
/// what may be asked of it.
fn one_field<'a>(
    args: Args<'a>,
    at: &mut usize,
    have: &[Field],
) -> core::result::Result<Field, Fail<'a>> {
    let identifier = args.get(*at);
    *at += 1;
    let attribute = if args.opt(*at).is_some_and(|a| args::is(a, b"as")) {
        *at += 1;
        let named = args.opt(*at).ok_or(Fail::plain(AS_ARG))?;
        *at += 1;
        named
    } else {
        identifier
    };

    let kind = kind(args, at, attribute)?;
    let mut f = Field::new(identifier, kind).named(attribute);
    if have.iter().any(|o| o.attribute == f.attribute) {
        return Err(Fail::naming(DUPLICATE, attribute));
    }

    // The first loop: everything that belongs to the type, plus the three that
    // belong to any of them. It stops at the first word it does not know,
    // which the second loop then gets a look at.
    while let Some(a) = args.opt(*at) {
        if args::is(a, b"withsuffixtrie") && f.kind.takes_empty() {
            f.suffix_trie = true;
        } else if args::is(a, b"indexempty") && f.kind.takes_empty() {
            f.index_empty = true;
        } else if args::is(a, b"indexmissing") {
            f.index_missing = true;
        } else if !type_option(args, at, &mut f)? {
            break;
        }
        // One step past whatever was taken. A bare word leaves the cursor on
        // itself and a pair leaves it on its value, so the same step works for
        // both and the loop cannot stand still.
        *at += 1;
    }

    // The second loop, which never hands anything back to the first one. A
    // `WITHSUFFIXTRIE` down here is not an option, it is the name of the next
    // field.
    while let Some(a) = args.opt(*at) {
        if args::is(a, b"sortable") {
            f.sortable = true;
            *at += 1;
            if args.opt(*at).is_some_and(|n| args::is(n, b"unf")) {
                f.unf = true;
                *at += 1;
            }
        } else if args::is(a, b"noindex") {
            f.noindex = true;
            *at += 1;
        } else {
            break;
        }
    }

    // A field nobody indexes cannot record which documents are missing it,
    // because recording that is indexing it.
    if f.noindex && f.index_missing {
        return Err(Fail::about(BOTH, attribute, "` '"));
    }
    Ok(f)
}

/// What the field holds, read off the word after its name.
fn kind<'a>(
    args: Args<'a>,
    at: &mut usize,
    attribute: &'a [u8],
) -> core::result::Result<Kind, Fail<'a>> {
    let Some(t) = args.opt(*at) else {
        return Err(Fail::about(NO_TYPE, attribute, NO_TYPE_END));
    };
    *at += 1;
    if args::is(t, b"text") {
        Ok(Kind::Text(Text::default()))
    } else if args::is(t, b"tag") {
        Ok(Kind::Tag(Tag::default()))
    } else if args::is(t, b"numeric") {
        Ok(Kind::Numeric)
    } else if args::is(t, b"geo") {
        Ok(Kind::Geo)
    } else if args::is(t, b"geoshape") {
        // The one type whose own option is a bare word rather than a pair, and
        // the only one where leaving it out picks the more expensive default.
        let coords = match args.opt(*at) {
            Some(c) if args::is(c, b"flat") => {
                *at += 1;
                Coords::Flat
            }
            Some(c) if args::is(c, b"spherical") => {
                *at += 1;
                Coords::Spherical
            }
            _ => Coords::Spherical,
        };
        Ok(Kind::GeoShape(coords))
    } else if args::is(t, b"vector") {
        vector(args, at).map(Kind::Vector)
    } else {
        Err(Fail::about(BAD_TYPE, attribute, "`"))
    }
}

/// One option that only means something for one kind of field.
///
/// Answers whether it took the word, so the loop above can tell an option it
/// does not know from an option it does. The cursor is left on the keyword and
/// moved by the caller, which is what keeps the two kinds of option in one
/// loop.
fn type_option<'a>(
    args: Args<'a>,
    at: &mut usize,
    f: &mut Field,
) -> core::result::Result<bool, Fail<'a>> {
    let a = args.get(*at);
    match &mut f.kind {
        Kind::Text(t) => {
            if args::is(a, b"nostem") {
                t.nostem = true;
            } else if args::is(a, b"weight") {
                let mut next = *at + 1;
                let v = value(args, &mut next, "weight")?;
                t.weight = parse_f64(v).ok_or_else(|| bad("weight", NOT_A_NUMBER))?;
                *at = next - 1;
            } else if args::is(a, b"phonetic") {
                let mut next = *at + 1;
                let v = value(args, &mut next, "PHONETIC")?;
                if !PHONETICS.iter().any(|p| args::is(v, p)) {
                    return Err(Fail::plain(MATCHER));
                }
                t.phonetic = Some(v.into());
                *at = next - 1;
            } else {
                return Ok(false);
            }
            Ok(true)
        }
        Kind::Tag(t) => {
            if args::is(a, b"casesensitive") {
                t.casesensitive = true;
            } else if args::is(a, b"separator") {
                let mut next = *at + 1;
                let v = value(args, &mut next, "SEPARATOR")?;
                // One character and not one byte, which for a separator is the
                // same thing: a real server takes the first byte and refuses
                // anything longer, so a multi byte character is refused too.
                let [c] = v else {
                    return Err(Fail::naming(SEPARATOR, v));
                };
                t.separator = *c;
                *at = next - 1;
            } else {
                return Ok(false);
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// `VECTOR algorithm count key value ...`
///
/// The count is a count of words rather than of pairs, and it is the only thing
/// that decides where the vector options end. Words past it belong to the next
/// field and words the schema meant for the next field are read as options when
/// the count reaches over them, which is why an eight over six real options
/// answers about the field name that followed rather than about the count.
///
/// An odd count is not an error of its own. The last keyword it covers is simply
/// left without a value, and that is the line a real server answers with.
fn vector<'a>(args: Args<'a>, at: &mut usize) -> core::result::Result<Vector, Fail<'a>> {
    let a = args.opt(*at).ok_or_else(|| bad(ALGO_WORD, NOT_THERE))?;
    *at += 1;
    let (algo, label) = if args::is(a, b"flat") {
        (Algo::Flat, "FLAT")
    } else if args::is(a, b"hnsw") {
        (Algo::Hnsw, "HNSW")
    } else if args::is(a, b"svs-vamana") {
        (Algo::Svs, "SVS-VAMANA")
    } else {
        return Err(bad(ALGO_WORD, UNKNOWN_WORD));
    };

    let c = args.opt(*at).ok_or_else(|| bad(COUNT_WORD, NOT_THERE))?;
    *at += 1;
    let n = parse_i64(c).ok_or_else(|| bad(COUNT_WORD, NOT_A_NUMBER))?;
    let words = usize::try_from(n).map_err(|_| bad(COUNT_WORD, OUT_OF_RANGE))?;
    if args.len() - *at < words {
        return Err(Fail::plain(NOT_ENOUGH));
    }

    let mut width = None;
    let mut dim = None;
    let mut metric = None;
    let mut training: Option<u64> = None;
    let mut reduce = false;
    let mut v = Vector::new(algo, Width::Float32, 0, Metric::L2);
    let mut left = words;
    while left > 0 {
        let key = args.get(*at);
        *at += 1;
        left -= 1;
        if left == 0 {
            return Err(vbad(label, key, V_NOT_THERE));
        }
        let val = args.get(*at);
        *at += 1;
        left -= 1;

        // The three every algorithm needs, then the ones that belong to one of
        // them. A keyword another algorithm would have taken is not an unknown
        // word, it is a word in the wrong place, and it gets its own line.
        if args::is(key, b"type") {
            width = Some(self::width(val).ok_or_else(|| vbad(label, key, V_UNKNOWN))?);
        } else if args::is(key, b"dim") {
            let d = parse_i64(val).ok_or_else(|| vbad(label, key, V_NOT_A_NUMBER))?;
            if d <= 0 {
                return Err(vbad(label, key, V_OUT_OF_RANGE));
            }
            dim = Some(d as u64);
        } else if args::is(key, b"distance_metric") {
            metric = Some(self::metric(val).ok_or_else(|| vbad(label, key, V_UNKNOWN))?);
        } else if args::is(key, b"initial_cap") && algo != Algo::Svs {
            v.initial_cap = Some(whole(label, key, val)?);
        } else if args::is(key, b"block_size") && algo == Algo::Flat {
            v.block_size = Some(whole(label, key, val)?);
        } else if args::is(key, b"m") && algo == Algo::Hnsw {
            v.m = whole(label, key, val)?;
        } else if args::is(key, b"ef_construction") && algo == Algo::Hnsw {
            v.ef_construction = whole(label, key, val)?;
        } else if args::is(key, b"ef_runtime") && algo == Algo::Hnsw {
            v.ef_runtime = whole(label, key, val)?;
        } else if args::is(key, b"epsilon") && algo != Algo::Flat {
            v.epsilon = parse_f64(val).ok_or_else(|| vbad(label, key, V_NOT_A_NUMBER))?;
        } else if args::is(key, b"graph_max_degree") && algo == Algo::Svs {
            v.graph_max_degree = whole(label, key, val)?;
        } else if args::is(key, b"construction_window_size") && algo == Algo::Svs {
            v.construction_window = whole(label, key, val)?;
        } else if args::is(key, b"search_window_size") && algo == Algo::Svs {
            // Taken, checked and dropped, which is what a real server does with
            // it: nothing it changes reaches `FT.INFO`.
            let _ = whole(label, key, val)?;
        } else if args::is(key, b"compression") && algo == Algo::Svs {
            let c = compression(val).ok_or_else(|| vbad(label, key, V_UNKNOWN))?;
            v.compression = Some(c.as_bytes().into());
        } else if args::is(key, b"training_threshold") && algo == Algo::Svs {
            training = Some(whole(label, key, val)?);
        } else if args::is(key, b"reduce") && algo == Algo::Svs {
            let _ = whole(label, key, val)?;
            reduce = true;
        } else {
            return Err(unwanted(label, key));
        }
    }

    // Two of the SVS options only mean something next to a compression, and one
    // of those only next to a compression of one kind. Both are checked after
    // the loop because the order the client wrote them in does not matter.
    if let Some(t) = training {
        if v.compression.is_none() {
            return Err(Fail::plain(NO_TRAINING));
        }
        if t < field::MIN_TRAINING {
            return Err(Fail::plain(SMALL_TRAINING));
        }
        v.training_threshold = Some(t);
    }
    if reduce && !v.compression.as_deref().is_some_and(is_leanvec) {
        return Err(Fail::plain(NO_REDUCE));
    }

    // All three are mandatory and a real server names the first one missing in
    // the order they are listed here.
    v.width = width.ok_or_else(|| missing(label, "TYPE"))?;
    v.dim = dim.ok_or_else(|| missing(label, "DIM"))?;
    v.metric = metric.ok_or_else(|| missing(label, "DISTANCE_METRIC"))?;
    Ok(v)
}

/// The compression schemes the Vamana form takes, in the spelling `FT.INFO`
/// gives back. The client's own case is not kept, because a real server takes
/// `lvq8` and reports it in the spelling it knows it by.
const COMPRESSIONS: &[&str] = &[
    "LVQ8",
    "LVQ4",
    "LVQ4x4",
    "LVQ4x8",
    "LeanVec4x8",
    "LeanVec8x8",
];

/// The scheme a word names, or `None` for a word that names none.
fn compression(v: &[u8]) -> Option<&'static str> {
    COMPRESSIONS
        .iter()
        .copied()
        .find(|c| args::is(v, c.as_bytes()))
}

/// Whether a scheme is one of the two that project the vector down first, which
/// are the only two `REDUCE` means anything beside.
fn is_leanvec(c: &[u8]) -> bool {
    c.starts_with(b"LeanVec")
}

/// `Bad arguments for algorithm HNSW: EF_RUNTIME`, which is the line for a
/// keyword that is real and belongs to another algorithm, and the line for a
/// keyword that is not real at all. A real server does not tell the two apart.
fn unwanted<'a>(label: &'static str, key: &'a [u8]) -> Fail<'a> {
    Fail::naming(
        match label {
            "FLAT" => "SEARCH_PARSE_ARGS Bad arguments for algorithm FLAT: ",
            "HNSW" => "SEARCH_PARSE_ARGS Bad arguments for algorithm HNSW: ",
            _ => "SEARCH_PARSE_ARGS Bad arguments for algorithm SVS-VAMANA: ",
        },
        key,
    )
}

/// `Bad arguments for vector similarity FLAT index `DIM`: why`.
fn vbad<'a>(label: &'static str, key: &'a [u8], why: &'static str) -> Fail<'a> {
    // The label and the key are two pieces and a line has room for one, so the
    // key is the piece that came from the client and the label is folded into
    // the head by the caller's choice of constant. There are three labels and
    // they are all `'static`, so this is a match rather than a format.
    Fail::about(
        match label {
            "FLAT" => "SEARCH_PARSE_ARGS Bad arguments for vector similarity FLAT index `",
            "HNSW" => "SEARCH_PARSE_ARGS Bad arguments for vector similarity HNSW index `",
            _ => "SEARCH_PARSE_ARGS Bad arguments for vector similarity SVS-VAMANA index `",
        },
        key,
        why,
    )
}

/// `Missing mandatory parameter: cannot create FLAT index without specifying
/// DIM argument`.
fn missing(label: &'static str, what: &'static str) -> Fail<'static> {
    Fail::about(
        match label {
            "FLAT" => {
                "SEARCH_PARSE_ARGS Missing mandatory parameter: cannot create FLAT index without specifying "
            }
            "HNSW" => {
                "SEARCH_PARSE_ARGS Missing mandatory parameter: cannot create HNSW index without specifying "
            }
            _ => {
                "SEARCH_PARSE_ARGS Missing mandatory parameter: cannot create SVS-VAMANA index without specifying "
            }
        },
        what.as_bytes(),
        " argument",
    )
}

/// A vector parameter that has to be a whole number.
fn whole<'a>(
    label: &'static str,
    key: &'a [u8],
    val: &[u8],
) -> core::result::Result<u64, Fail<'a>> {
    let n = parse_i64(val).ok_or_else(|| vbad(label, key, V_NOT_A_NUMBER))?;
    u64::try_from(n).map_err(|_| vbad(label, key, V_OUT_OF_RANGE))
}

/// How wide one coordinate is, from the word after `TYPE`.
fn width(v: &[u8]) -> Option<Width> {
    [
        Width::Int8,
        Width::Uint8,
        Width::Float16,
        Width::BFloat16,
        Width::Float32,
        Width::Float64,
    ]
    .into_iter()
    .find(|w| args::is(v, w.token().as_bytes()))
}

/// What the index measures, from the word after `DISTANCE_METRIC`.
fn metric(v: &[u8]) -> Option<Metric> {
    if args::is(v, b"l2") {
        Some(Metric::L2)
    } else if args::is(v, b"ip") {
        Some(Metric::Ip)
    } else if args::is(v, b"cosine") {
        Some(Metric::Cosine)
    } else {
        None
    }
}

/// `FT.ALTER index [SKIPINITIALSCAN] SCHEMA ADD field type ...`
///
/// Several fields at once, which is what the grammar allows even though the
/// documentation shows one.
fn alter<'a>(reg: &mut Registry, args: Args<'a>, out: &mut Out, ifnx: bool) -> Answer<'a> {
    let name = args.get(1);
    // The name is resolved before the grammar after it is read, which is why a
    // call that goes on to be refused has still counted as a use of the index.
    // The lookup does not get to decide the answer though, only to count: a name
    // that is not there next to a word that is not SCHEMA still answers about the
    // word.
    reg.touch(name);
    let mut at = 2;
    if args
        .opt(at)
        .is_some_and(|a| args::is(a, b"skipinitialscan"))
    {
        at += 1;
    }
    if !args.opt(at).is_some_and(|a| args::is(a, b"schema")) {
        return Err(Fail::plain(AFTER_ALTER));
    }
    at += 1;
    if !args.opt(at).is_some_and(|a| args::is(a, b"add")) {
        return Err(Fail::plain(ALTER_ACTION));
    }
    at += 1;

    // The index is looked up after the shape of the command is checked, which
    // is a real server's order: `FT.ALTER nope BLAH ADD q NUMERIC` answers that
    // ALTER must be followed by SCHEMA rather than that there is no such index.
    let Some(index) = reg.get(name) else {
        return Err(Fail::naming(MISSING, name));
    };
    let mut schema = index.schema.clone();
    match fields(args, at, &mut schema) {
        Ok(()) => {}
        // The quiet form swallows a field that is already there and nothing
        // else, so a schema with a bad type in it is still an error.
        Err(f) if ifnx && f.head == DUPLICATE => {
            out.ok();
            return Ok(());
        }
        Err(f) => return Err(f),
    }
    if let Some(index) = reg.get_mut(name) {
        index.schema = schema;
    }
    out.ok();
    Ok(())
}

/// `FT.DROPINDEX index [DD]`, and the three other spellings of it.
///
/// `DD` says to delete the documents the index followed as well as the index
/// itself. There are no documents under an index yet, so it is taken and does
/// nothing, which is the right answer for an empty index either way. Only the
/// two `DROPINDEX` spellings take it: `FT.DROP i DD` is an unknown argument on
/// a real server, which is the sort of thing that only turns up by asking.
///
/// The index is looked up before the arguments after it are, so
/// `FT.DROP nope junk` answers that there is no such index and `FT.DROP i junk`
/// answers that `junk` is not an argument.
fn drop_index<'a>(
    reg: &mut Registry,
    spec: &Spec,
    args: Args<'a>,
    out: &mut Out,
    ifx: bool,
    dd: bool,
) -> Answer<'a> {
    // All four spellings count their own arguments, which is why the table
    // cannot do it for them. Which name goes in the line depends on which end
    // the count went wrong at: too few names the command plainly and too many
    // names it with an underscore in front, because the plain form is the one a
    // client called and the underscore form is the one the coordinator hands the
    // longer call on to. That is not a typo, it is the module's registration
    // showing through, and a client that matches on the whole line sees it.
    if args.len() < 2 {
        return Err(Fail::about(ARITY_PLAIN, spec.name.as_bytes(), ARITY_END));
    }
    if args.len() > 3 {
        return Err(Fail::about(ARITY, spec.name.as_bytes(), ARITY_END));
    }
    let name = args.get(1);
    if !reg.touch(name) {
        if ifx {
            out.ok();
            return Ok(());
        }
        return Err(Fail::naming(MISSING, name));
    }
    if let Some(a) = args.opt(2)
        && !(dd && args::is(a, b"dd"))
    {
        return Err(Fail::plain(UNKNOWN_BARE));
    }
    let _ = reg.drop(name);
    out.ok();
    Ok(())
}

/// `FT._LIST`, every index by name.
///
/// A set on RESP3 and an array on RESP2, of simple strings on both, which is
/// not the shape `FT.ALIASLIST` answers with even though the two look like the
/// same question asked twice.
///
/// One argument after the name is taken and ignored and two are too many, which
/// is the arity a real server enforces even though `COMMAND INFO` reports this
/// as taking any number at all.
fn list<'a>(reg: &Registry, spec: &Spec, args: Args<'a>, out: &mut Out) -> Answer<'a> {
    if args.len() > 2 {
        return Err(Fail::about(ARITY_PLAIN, spec.name.as_bytes(), ARITY_END));
    }
    out.set(reg.len());
    for index in reg.iter() {
        word(out, &index.name);
    }
    Ok(())
}

/// `FT.ALIASADD alias index`.
///
/// The suffix on `FT._ALIASADDIFNX` forgives less than the name suggests. It
/// forgives the one case where the alias is already pointing at the index the
/// client is asking for, which is the call that has nothing left to do, and it
/// refuses an alias pointing at another index just as loudly as the plain form
/// does. An index that is not there is refused by both, and the two do not say
/// it the same way: the plain form answers the not found line the rest of the
/// group uses and the suffixed one answers the line about a name that might be
/// an alias, which is the wrapper around it doing its own lookup first. An index
/// name that is really an alias gets that longer line from both of them, because
/// this is the one command in the group that will not follow an alias and so the
/// one place where the difference between a name and an alias is worth a
/// sentence of its own.
fn alias_add<'a>(reg: &mut Registry, args: Args<'a>, out: &mut Out, ifnx: bool) -> Answer<'a> {
    let alias = args.get(1);
    let name = args.get(2);
    // The index on the right is looked up first and counts as a use of it, even
    // on the calls that go on to refuse the alias on the left. It is looked up
    // by its own name and never through an alias, which is why the count only
    // moves for a real index name.
    if reg.named(name).is_some() {
        reg.touch(name);
    }
    if ifnx
        && let Some(at) = reg.target(alias)
        && reg.named(name).is_some_and(|i| *i.name == *at)
    {
        out.ok();
        return Ok(());
    }
    match reg.alias(alias, name) {
        Ok(()) => out.ok(),
        Err(Clash::IsIndex) => return Err(Fail::plain(CONFLICT)),
        Err(Clash::Aliased) => return Err(Fail::plain(ALIAS_EXISTS)),
        // A target that is itself an alias reads the same from both spellings,
        // and it is the only case where the plain form says the longer sentence.
        Err(Clash::IsAlias) => return Err(Fail::plain(NO_TARGET)),
        Err(_) if ifnx => return Err(Fail::plain(NO_TARGET)),
        Err(_) => return Err(Fail::naming(MISSING, name)),
    }
    Ok(())
}

/// `FT.ALIASUPDATE alias index`, which moves an alias that is already pointing
/// somewhere and adds one that is not.
fn alias_update<'a>(reg: &mut Registry, args: Args<'a>, out: &mut Out) -> Answer<'a> {
    let alias = args.get(1);
    let name = args.get(2);
    // Where the alias was pointing, read before the move takes it away.
    let old: Option<Box<[u8]>> = reg.target(alias).map(Into::into);
    match reg.realias(alias, name) {
        Ok(()) => {
            // Both ends count, so moving an alias from one index to another
            // counts a use on the one it left as well as on the one it landed
            // on. Only a move that happens counts: this is the one command in
            // the group that counts nothing at all when it refuses, where the
            // rest have all counted by the time they work out the answer.
            if let Some(old) = old {
                reg.touch(&old);
            }
            reg.touch(name);
            out.ok();
        }
        // Not the conflict `FT.ALIASADD` answers with for the same argument,
        // which is a real server's inconsistency and is copied because a client
        // branches on the code word in front of it.
        Err(Clash::IsIndex) => return Err(Fail::plain(NOT_MINE)),
        Err(Clash::IsAlias) => return Err(Fail::plain(NO_TARGET)),
        Err(_) => return Err(Fail::naming(MISSING, name)),
    }
    Ok(())
}

/// `FT.ALIASDEL alias`.
///
/// The name of an index is refused by both forms, the one with the suffix
/// included, and with the sentence about ownership rather than the one about a
/// missing alias. The suffix only forgives a name that is nothing at all, and an
/// index name is something, it is just not an alias.
fn alias_del<'a>(reg: &mut Registry, args: Args<'a>, out: &mut Out, ifx: bool) -> Answer<'a> {
    let alias = args.get(1);
    // The suffixed form resolves the name itself before it hands the call on to
    // the plain one, which then resolves it again, so a name that resolves at all
    // counts twice under the suffixed form and once under the plain one.
    if ifx {
        reg.touch(alias);
    }
    reg.touch(alias);
    if reg.named(alias).is_some() {
        return Err(Fail::plain(NOT_MINE));
    }
    match reg.unalias(alias) {
        Ok(()) => out.ok(),
        Err(_) if ifx => out.ok(),
        Err(_) => return Err(Fail::plain(NO_ALIAS)),
    }
    Ok(())
}

/// `FT.ALIASLIST index`, the aliases pointing at one index.
///
/// The argument is an index and not an alias, so asking this about an alias
/// answers that there is no index by that name.
fn alias_list<'a>(reg: &mut Registry, args: Args<'a>, out: &mut Out) -> Answer<'a> {
    let name = args.get(1);
    if reg.named(name).is_none() {
        return Err(Fail::naming(MISSING, name));
    }
    // Only an index name gets this far, and finding it counts as a use of it the
    // same way every other lookup by name does.
    let real: Box<[u8]> = name.into();
    reg.open(&real);
    // Bulk strings where `FT._LIST` writes simple ones, and a set on RESP3 the
    // same way.
    let n = reg.aliases_of(&real).count();
    out.set(n);
    for alias in reg.aliases_of(&real) {
        out.bulk(alias);
    }
    Ok(())
}

/// A string in a reply, as a simple string when it can be one.
///
/// `FT.INFO` writes nearly everything as a simple string, the index name and
/// the key prefixes included, and those are the client's own bytes. A real
/// server writes them straight out and a prefix with a newline in it breaks the
/// stream it goes into. A bulk string carries the same bytes and says how many
/// there are, so the value that cannot be a simple string goes out as one and
/// every ordinary value is byte for byte what a real server sends.
fn word(out: &mut Out, s: &[u8]) {
    if s.contains(&b'\r') || s.contains(&b'\n') {
        out.bulk(s);
    } else {
        out.simple(s);
    }
}

/// A name and a simple string value.
fn pair(out: &mut Out, name: &str, value: &[u8]) {
    out.simple(name.as_bytes());
    word(out, value);
}

/// A name and a number that is written as a double on RESP3 and as a bulk
/// string of the same digits on RESP2.
fn number(out: &mut Out, name: &str, value: f64) {
    out.simple(name.as_bytes());
    out.double(value);
}

/// A name and a count, which is an integer on both protocols.
fn tally(out: &mut Out, name: &str, value: u64) {
    out.simple(name.as_bytes());
    out.uint(value);
}

/// `FT.INFO index`, everything the server knows about one index.
fn info<'a>(reg: &mut Registry, args: Args<'a>, out: &mut Out) -> Answer<'a> {
    let name = args.get(1);
    // Counted before it is reported, so the first `FT.INFO` after a create says
    // one rather than nought. An alias counts once and not twice, because the
    // alias is resolved on the way to the one lookup rather than being a lookup
    // of its own.
    let Some(index) = reg.open(name) else {
        return Err(Fail::naming(MISSING, name));
    };

    let d = &index.definition;
    let stopwords = d.stopwords.as_ref();
    out.map(34 + usize::from(stopwords.is_some()));

    pair(out, "index_name", &index.name);
    out.simple(b"index_options");
    let tokens = d.options.tokens();
    out.array(tokens.len());
    for t in &tokens {
        out.simple(t.as_bytes());
    }

    out.simple(b"index_definition");
    let mut pairs = 4;
    pairs += usize::from(d.filter.is_some());
    pairs += usize::from(d.language.is_some());
    pairs += usize::from(d.language_field.is_some());
    pairs += usize::from(d.score_field.is_some());
    pairs += usize::from(d.payload_field.is_some());
    out.map(pairs);
    pair(out, "key_type", d.on.token().as_bytes());
    out.simple(b"prefixes");
    out.array(d.prefixes.len());
    for p in &d.prefixes {
        word(out, p);
    }
    if let Some(f) = &d.filter {
        pair(out, "filter", f);
    }
    if let Some(l) = &d.language {
        pair(out, "default_language", l);
    }
    if let Some(l) = &d.language_field {
        pair(out, "language_field", l);
    }
    number(out, "default_score", d.score);
    if let Some(s) = &d.score_field {
        pair(out, "score_field", s);
    }
    if let Some(p) = &d.payload_field {
        pair(out, "payload_field", p);
    }
    // Always false, on a real server as well as here. Every index that ought to
    // report true reports false there, including one created with an explicit
    // empty prefix and no filter, so this is measured rather than computed.
    pair(out, "indexes_all", b"false");

    out.simple(b"attributes");
    out.array(index.schema.len());
    for f in &index.schema {
        attribute(out, f);
    }

    // Nothing is indexed yet, so every one of these is nought rather than a
    // number this server has no way to produce. D-58 is the register entry.
    let docs = index.held.docs.len() as u64;
    let records = index.held.records();
    tally(out, "num_docs", docs);
    tally(out, "max_doc_id", u64::from(index.held.docs.last()));
    tally(out, "num_terms", index.held.words() as u64);
    tally(out, "num_records", records);
    number(out, "inverted_sz_mb", 0.0);
    number(out, "vector_index_sz_mb", 0.0);
    tally(out, "total_inverted_index_blocks", 0);
    number(out, "offset_vectors_sz_mb", 0.0);
    number(out, "doc_table_size_mb", 0.0);
    number(out, "sortable_values_size_mb", 0.0);
    number(out, "key_table_size_mb", 0.0);
    number(out, "tag_overhead_sz_mb", 0.0);
    number(out, "text_overhead_sz_mb", 0.0);
    number(out, "total_index_memory_sz_mb", 0.0);
    number(out, "geoshapes_sz_mb", 0.0);
    // Four averages, and the three after the first are over byte counts this
    // build does not keep yet. All four are a division by nought on an empty
    // index, and a real server answers `nan` there for the same reason rather
    // than as a placeholder.
    //
    // The division is done in single precision and printed in double, which is
    // what a real server does and is visible: seventeen records over three
    // documents comes out as 5.666666507720947 and not 5.666666666666667. The
    // counters are doubles by the time they reach the reply and the arithmetic
    // behind them is not, so a client comparing the two builds sees the same
    // digits.
    number(
        out,
        "records_per_doc_avg",
        f64::from(records as f32 / docs as f32),
    );
    number(out, "bytes_per_record_avg", f64::NAN);
    number(out, "offsets_per_term_avg", f64::NAN);
    number(out, "offset_bits_per_record_avg", f64::NAN);
    tally(
        out,
        "hash_indexing_failures",
        index.trouble.whole().failures(),
    );
    number(out, "total_indexing_time", 0.0);
    tally(out, "indexing", 0);
    number(out, "percent_indexed", 1.0);
    tally(out, "number_of_uses", index.uses);
    tally(out, "cleaning", 0);

    out.simple(b"gc_stats");
    out.map(7);
    number(out, "bytes_collected", 0.0);
    number(out, "total_ms_run", 0.0);
    number(out, "total_cycles", 0.0);
    number(out, "average_cycle_time_ms", f64::NAN);
    number(out, "last_run_time_ms", 0.0);
    number(out, "gc_numeric_trees_missed", 0.0);
    number(out, "gc_blocks_denied", 0.0);

    out.simple(b"cursor_stats");
    out.map(4);
    tally(out, "global_idle", 0);
    tally(out, "global_total", 0);
    tally(out, "index_capacity", CURSOR_CAPACITY);
    tally(out, "index_total", 0);

    if let Some(list) = stopwords {
        out.simple(b"stopwords_list");
        out.array(list.len());
        for w in list {
            out.bulk(w);
        }
    }

    out.simple(b"dialect_stats");
    out.map(4);
    for n in 1..=4 {
        out.simple(match n {
            1 => b"dialect_1".as_slice(),
            2 => b"dialect_2",
            3 => b"dialect_3",
            _ => b"dialect_4",
        });
        out.uint(0);
    }

    out.simple(b"Index Errors");
    out.map(4);
    errors(out, index.trouble.whole());
    out.simple(b"background indexing status");
    out.simple(b"OK");

    out.simple(b"field statistics");
    out.array(index.schema.len());
    for f in &index.schema {
        statistics(out, f, index.trouble.field(&f.attribute));
    }
    Ok(())
}

/// How many cursors an index has room for, which is what `FT.INFO` reports
/// before a single one has been opened.
const CURSOR_CAPACITY: u64 = 128;

/// One schema field, as `FT.INFO` describes it.
///
/// The order is fixed and is not the order the client wrote the options in: a
/// field declared `SORTABLE NOSTEM` and one declared `NOSTEM SORTABLE` are one
/// field and describe themselves the same way.
///
/// The two protocols disagree about the shape. RESP2 writes a flat array with
/// the flag words on the end of it, and RESP3 writes a map with the flags
/// gathered into an array under `flags`, which is there even when it is empty.
fn attribute(out: &mut Out, f: &Field) {
    let mut flags: Vec<&str> = Vec::new();
    if f.sortable {
        flags.push("SORTABLE");
    }
    if f.is_unf() {
        flags.push("UNF");
    }
    if let Kind::Text(t) = &f.kind
        && t.nostem
    {
        flags.push("NOSTEM");
    }
    if let Kind::Tag(t) = &f.kind
        && t.casesensitive
    {
        flags.push("CASESENSITIVE");
    }
    if f.suffix_trie {
        flags.push("WITHSUFFIXTRIE");
    }
    if f.index_empty {
        flags.push("INDEXEMPTY");
    }
    if f.index_missing {
        flags.push("INDEXMISSING");
    }
    if f.noindex {
        flags.push("NOINDEX");
    }

    // A tag reports its separator and a text its weight where the flags go, in
    // front of them, so the pairs are counted before the shape is chosen.
    let pairs = 3 + match &f.kind {
        Kind::Text(_) | Kind::Tag(_) => 1,
        Kind::GeoShape(_) => 1,
        Kind::Vector(v) => match v.algo {
            Algo::Flat => 4,
            Algo::Hnsw => 7,
            // The Vamana form reports what it compresses to always and how many
            // vectors it trains that compression over only when there is one to
            // train, so a compressed field carries one pair more than a plain
            // one does.
            Algo::Svs => 7 + usize::from(v.compression.is_some()),
        },
        Kind::Numeric | Kind::Geo => 0,
    };
    if out.proto().is_resp3() {
        out.map(pairs + 1);
    } else {
        out.array(pairs * 2 + flags.len());
    }

    pair(out, "identifier", &f.identifier);
    pair(out, "attribute", &f.attribute);
    pair(out, "type", f.kind.token().as_bytes());
    match &f.kind {
        Kind::Text(t) => number(out, "WEIGHT", t.weight),
        Kind::Tag(t) => {
            out.simple(b"SEPARATOR");
            word(out, &[t.separator]);
        }
        Kind::GeoShape(c) => pair(out, "coord_system", c.token().as_bytes()),
        Kind::Vector(v) => {
            pair(out, "algorithm", v.algo.token().as_bytes());
            pair(out, "data_type", v.width.token().as_bytes());
            tally(out, "dim", v.dim);
            pair(out, "distance_metric", v.metric_token().as_bytes());
            match v.algo {
                Algo::Flat => {}
                Algo::Hnsw => {
                    tally(out, "M", v.m);
                    tally(out, "ef_construction", v.ef_construction);
                    tally(out, "ef_runtime", v.ef_runtime);
                }
                Algo::Svs => {
                    tally(out, "graph_max_degree", v.graph_max_degree);
                    tally(out, "construction_window_size", v.construction_window);
                    pair(
                        out,
                        "compression",
                        v.compression
                            .as_deref()
                            .unwrap_or(field::NO_COMPRESSION.as_bytes()),
                    );
                    if v.compression.is_some() {
                        tally(
                            out,
                            "training_threshold",
                            v.training_threshold.unwrap_or(field::TRAINING_THRESHOLD),
                        );
                    }
                }
            }
        }
        Kind::Numeric | Kind::Geo => {}
    }

    if out.proto().is_resp3() {
        out.simple(b"flags");
        out.array(flags.len());
    }
    for flag in &flags {
        out.simple(flag.as_bytes());
    }
}

/// The three lines an error block starts with, which are the same for the index
/// and for each of its fields.
///
/// The sentence is a simple string and the key is a bulk, which is not a
/// consistent pair and is what a real server writes, `N/A` included.
fn errors(out: &mut Out, e: &Errors) {
    tally(out, "indexing failures", e.failures());
    out.simple(b"last indexing error");
    out.simple(e.sentence());
    out.simple(b"last indexing error key");
    out.bulk(e.about());
}

/// One field's own error counters, which are all nought until a key it could
/// not read has been written.
///
/// A vector field carries four more than the rest, and they are the four a
/// client watching an index fill up would read.
fn statistics(out: &mut Out, f: &Field, e: &Errors) {
    let vector = matches!(f.kind, Kind::Vector(_));
    out.map(3 + if vector { 4 } else { 0 });
    pair(out, "identifier", &f.identifier);
    pair(out, "attribute", &f.attribute);
    out.simple(b"Index Errors");
    out.map(3);
    errors(out, e);
    if vector {
        tally(out, "memory", 0);
        tally(out, "marked_deleted", 0);
        tally(out, "direct_hnsw_insertions", 0);
        tally(out, "flat_buffer_size", 0);
    }
}

/// The most recent grammar a client may ask for.
const NEWEST: u8 = 4;

const BAD_DIALECT: &str = "SEARCH_PARSE_ARGS DIALECT requires a non negative integer >=1 and <= 4";
const NEED_ARG: &str = "SEARCH_PARSE_ARGS Need an argument for ";
const ODD_PARAMS: &str = "SEARCH_ADD_ARGS Parameters must be specified in PARAM VALUE pairs";
const NOT_MAIN: &str = "` at position ";
const NOT_MAIN_END: &str = " for <main>";

/// What a client asked for beside the query, and where the reading of it stops.
///
/// `FT.EXPLAIN` shares its argument list with `FT.SEARCH`, so most of what can
/// appear here is about a result set that this command never produces. The ones
/// that change the tree are taken and the ones that change the rows are taken
/// and dropped, which is what a real server does with them too when all it is
/// being asked for is the tree.
struct Asked {
    dialect: u8,
    params: Vec<Pair>,
    verbatim: bool,
    stopwords: bool,
}

impl Default for Asked {
    fn default() -> Asked {
        Asked {
            dialect: 1,
            params: Vec::new(),
            verbatim: false,
            stopwords: true,
        }
    }
}

/// The keywords that mean something to a result set and nothing to a tree,
/// with how many words each one carries.
const IGNORED: &[(&[u8], usize)] = &[
    (b"NOCONTENT", 0),
    (b"WITHSCORES", 0),
    (b"WITHPAYLOADS", 0),
    (b"WITHSORTKEYS", 0),
    (b"EXPLAINSCORE", 0),
    (b"LIMIT", 2),
    (b"TIMEOUT", 1),
    (b"SLOP", 1),
    (b"INORDER", 0),
    (b"LANGUAGE", 1),
    (b"EXPANDER", 1),
    (b"SCORER", 1),
    (b"PAYLOAD", 1),
];

/// Reads the arguments after the query.
///
/// The error text carries a position, which is why this hands back bytes rather
/// than a `Fail`: every other error line in this module is three static pieces
/// around a word the client sent, and this one has a number in it.
fn options(args: Args<'_>, from: usize) -> core::result::Result<Asked, Vec<u8>> {
    let mut asked = Asked::default();
    let mut at = from;
    while at < args.len() {
        let word = args.get(at);
        if args::is(word, b"DIALECT") {
            let Some(value) = args.opt(at + 1) else {
                return Err(line(NEED_ARG, b"DIALECT", ""));
            };
            let Some(dialect) = parse_i64(value).filter(|d| (1..=i64::from(NEWEST)).contains(d))
            else {
                return Err(BAD_DIALECT.as_bytes().to_vec());
            };
            asked.dialect = u8::try_from(dialect).unwrap_or(1);
            at += 2;
            continue;
        }
        if args::is(word, b"PARAMS") {
            at = params(args, at, &mut asked)?;
            continue;
        }
        if args::is(word, b"VERBATIM") {
            asked.verbatim = true;
            at += 1;
            continue;
        }
        if args::is(word, b"NOSTOPWORDS") {
            asked.stopwords = false;
            at += 1;
            continue;
        }
        if let Some((_, takes)) = IGNORED.iter().find(|(k, _)| args::is(word, k)) {
            if args.opt(at + takes).is_none() {
                return Err(line(BAD_ARGS, word, NOT_THERE));
            }
            at += takes + 1;
            continue;
        }
        return Err(unknown(word, at - from + 1));
    }
    Ok(asked)
}

/// `PARAMS n name value ...`, which is where a `$name` in a query comes from.
fn params(args: Args<'_>, at: usize, asked: &mut Asked) -> core::result::Result<usize, Vec<u8>> {
    let Some(count) = args.opt(at + 1) else {
        return Err(line(BAD_ARGS, b"PARAMS", NOT_THERE));
    };
    let Some(count) = parse_i64(count) else {
        return Err(line(BAD_ARGS, b"PARAMS", NOT_A_NUMBER));
    };
    let count = usize::try_from(count).unwrap_or(0);
    if count == 0 || count % 2 != 0 {
        return Err(ODD_PARAMS.as_bytes().to_vec());
    }
    for step in 0..count / 2 {
        let name = args.opt(at + 2 + step * 2);
        let value = args.opt(at + 3 + step * 2);
        let (Some(name), Some(value)) = (name, value) else {
            return Err(line(BAD_ARGS, b"PARAMS", NOT_THERE));
        };
        asked.params.push((name.into(), value.into()));
    }
    Ok(at + 2 + count)
}

/// An error line built from a head, a word the client sent and a tail.
fn line(head: &str, word: &[u8], tail: &str) -> Vec<u8> {
    let mut out = head.as_bytes().to_vec();
    out.extend_from_slice(word);
    out.extend_from_slice(tail.as_bytes());
    out
}

/// The line for an argument nobody knows, which counts from the query.
fn unknown(word: &[u8], position: usize) -> Vec<u8> {
    let mut out = UNKNOWN.as_bytes().to_vec();
    out.extend_from_slice(word);
    out.extend_from_slice(NOT_MAIN.as_bytes());
    out.extend_from_slice(position.to_string().as_bytes());
    out.extend_from_slice(NOT_MAIN_END.as_bytes());
    out
}

/// The line a refused query answers with.
fn refused(bad: &Bad) -> Vec<u8> {
    match bad {
        Bad::Syntax { at, near } => spot("SEARCH_SYNTAX Syntax error at offset ", *at, near),
        Bad::Unknown { at, near } => named(
            "SEARCH_SYNTAX Unknown field at offset ",
            *at,
            near.as_deref(),
        ),
        Bad::Wrong { kind, at, near } => {
            let head = format!("SEARCH_SYNTAX Expected a {kind} field at offset ");
            named(&head, *at, near.as_deref())
        }
        Bad::Attribute(name) => line("SEARCH_OPTION_INVALID Invalid attribute ", name, ""),
        Bad::Value { name, value } => {
            let mut out = b"SEARCH_SYNTAX Invalid value (".to_vec();
            out.extend_from_slice(value);
            out.extend_from_slice(b") for `");
            out.extend_from_slice(name);
            out.push(b'`');
            out
        }
        Bad::Missing(name) => {
            let mut out = b"SEARCH_PARAM_NOT_FOUND Parameter not found `".to_vec();
            out.extend_from_slice(name);
            out.push(b'`');
            out
        }
        Bad::Taken(name) => {
            let mut out = b"SEARCH_INDEX_EXISTS Property `".to_vec();
            out.extend_from_slice(name);
            out.extend_from_slice(b"` already exists in schema");
            out
        }
        Bad::Plain(text) => line("SEARCH_SYNTAX ", text.as_bytes(), ""),
        Bad::Refused(text) => line("SEARCH_QUERY_BAD ", text.as_bytes(), ""),
    }
}

/// The same for the two errors about a field, which leave the `near` off
/// altogether when there is no word to put in it rather than trailing an empty
/// one the way a syntax error does.
fn named(head: &str, at: usize, near: Option<&[u8]>) -> Vec<u8> {
    let Some(near) = near else {
        let mut out = head.as_bytes().to_vec();
        out.extend_from_slice(at.to_string().as_bytes());
        return out;
    };
    spot(head, at, near)
}

/// `... at offset 7 near the`, which is how the parser points at a query.
fn spot(head: &str, at: usize, near: &[u8]) -> Vec<u8> {
    let mut out = head.as_bytes().to_vec();
    out.extend_from_slice(at.to_string().as_bytes());
    out.extend_from_slice(b" near ");
    out.extend_from_slice(near);
    out
}

/// `FT.EXPLAIN index query [options]` and `FT.EXPLAINCLI` beside it.
///
/// The same work with two shapes of reply. `FT.EXPLAIN` sends the printout as
/// one bulk string and `FT.EXPLAINCLI` sends it split on newlines, including
/// the empty piece after the last one, so a client that joins the array back
/// with newlines gets exactly what the other command would have sent.
fn explain<'a>(reg: &mut Registry, args: Args<'a>, out: &mut Out, cli: bool) -> Answer<'a> {
    let name = args.get(1);
    let query = args.get(2);
    let asked = match options(args, 3) {
        Ok(asked) => asked,
        Err(text) => {
            out.error(&text);
            return Ok(());
        }
    };
    // Counted, like every other command that resolves a name, and counted after
    // the arguments rather than before: an unreadable argument list means the
    // index was never opened.
    let Some(index) = reg.open(name) else {
        return Err(Fail::naming(MISSING, name));
    };
    let ask = Ask {
        dialect: asked.dialect,
        params: &asked.params,
        verbatim: asked.verbatim,
        stopwords: asked.stopwords,
    };
    let node = match query::parse(query, index, &ask) {
        Ok(node) => node,
        Err(bad) => {
            out.error(&refused(&bad));
            return Ok(());
        }
    };
    let printed = query::explain(&node, index);
    if !cli {
        out.bulk(&printed);
        return Ok(());
    }
    let lines = query::explain::lines(&printed);
    out.array(lines.len());
    for line in lines {
        out.simple(line);
    }
    Ok(())
}
