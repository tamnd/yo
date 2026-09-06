//! The five commands from before a document was simply a hash.
//!
//! `FT.ADD`, `FT.SAFEADD`, `FT.GET`, `FT.MGET` and `FT.DEL` are the original
//! document interface, from the years when the module kept its own copy of every
//! document and a client had to hand it over field by field. Nothing works that
//! way any more. A document is a key, an index follows a prefix, and the whole
//! group is deprecated. They are all still there on 8.10.1 and they all still
//! work, so they are here, and what they do now is what they do there: write or
//! read or delete an ordinary hash.
//!
//! Which makes the interesting part the seams rather than the commands.
//!
//! # The key is the document id and nothing is prefixed
//!
//! `FT.ADD i a1 1.0 FIELDS t alpha` over an index following `i:` writes a hash
//! called `a1` and the index never sees it, because `a1` does not match the
//! prefix. The command does not refuse and does not warn. So the write goes
//! through the ordinary keyspace path and the ordinary indexing notification
//! picks it up or does not, exactly as an `HSET` of the same key would.
//!
//! # The score, the language and the payload are written into the hash
//!
//! There is nowhere else for them to go now that a document is a key, so `FT.ADD`
//! writes them into the hash beside the client's own fields, under `__score`,
//! `__language` and `__payload` in that order. `__score` goes in when the score
//! is not one, and always on a `PARTIAL`, which reads as an accident of how the
//! merge is built rather than anything meant. The other two go in whenever the
//! option was given. All three keep the client's own spelling, so `0.50` comes
//! back as `0.50` and `FrEnCh` comes back as `FrEnCh`.
//!
//! What makes them work rather than only be a record is the second half of the
//! write: an `FT.ADD` that puts one of the three in also sets the matching field
//! on the index definition, so `SCORE_FIELD __score`, `LANGUAGE_FIELD __language`
//! or `PAYLOAD_FIELD __payload`. That is a permanent change to the index and it
//! shows in `FT.INFO`. From then on every document the index reads takes its
//! score off the hash, including documents nothing to do with `FT.ADD`, which is
//! why a plain `HSET` writing `__score` does nothing on a fresh index and does
//! something on an index one `FT.ADD` has already touched. Documents already in
//! the index are left where they are and pick the new reading up whenever they
//! are next written.
//!
//! So the write here is an ordinary hash write with an ordinary indexing
//! notification behind it, and the score and the payload arrive through the same
//! path they arrive through for an `HSET`.
//!
//! # `IF` is asked of the document that is already there
//!
//! Only with `REPLACE`, and only when the key exists. A key that is not there is
//! a plain add and the expression is never even parsed. A key that is there and
//! is not in the index refuses with `SEARCH_DOC_NOT_FOUND` and an empty message,
//! also before the expression is parsed. Only a document the index actually
//! holds gets as far as reading the expression, and then a false answer is the
//! status `NOADD` rather than an error.

use yo_kv::value::Kind;
use yo_search::Registry;
use yo_search::expr::{Expr, Value};
use yo_search::field::Kind as Type;

use super::super::Server;
use super::super::args::{self, Args};
use super::super::indexing::{self, Change, Document};
use super::super::table::Spec;
use super::{Fail, LANGUAGES, MISSING, NOT_LOADED, QUOTE_END, line};
use crate::reply::Out;
use yo_common::Result;

/// The field a document's score is written into, and the one the index is told
/// to read it back out of.
const SCORE_FIELD: &[u8] = b"__score";
/// The same for the language it was written in.
const LANGUAGE_FIELD: &[u8] = b"__language";
/// And for the payload.
const PAYLOAD_FIELD: &[u8] = b"__payload";

/// What every document is worth when nobody said otherwise.
const PLAIN: f64 = 1.0;

const NO_FIELDS: &str = "SEARCH_ADD_ARGS No field list found";
const ODD_FIELDS: &str = "SEARCH_ADD_ARGS Fields must be specified in FIELD VALUE pairs";
const BAD_SCORE: &str = "SEARCH_ADD_ARGS Could not parse document score";
const SCORE_RANGE: &str = "SEARCH_ADD_ARGS Score must be between 0 and 1";
const BAD_LANGUAGE: &str = "SEARCH_ADD_ARGS Unsupported language";
const UNKNOWN_WORD: &str = "SEARCH_ADD_ARGS Unknown keyword `";
const UNKNOWN_END: &str = "` provided";
/// The head and tail of the line an option with nothing after it answers, which
/// names the option in capitals whatever case the client wrote it in.
const NO_ARGUMENT: &str = "SEARCH_ADD_ARGS Parsing error for document option ";
const NO_ARGUMENT_END: &str = ": Expected an argument, but none provided";
const HERE_ALREADY: &str = "SEARCH_DOCUMENT_EXISTS Document already exists";
const BAD_KEY: &str = "SEARCH_REDIS_KEY_TYPE_BAD Invalid Redis key";
/// The one line in the search surface with a code and a space and nothing else
/// after it. That is not a gap here, it is what a real server sends.
const NO_DOCUMENT: &str = "SEARCH_DOC_NOT_FOUND ";

/// What an `FT.ADD` was asked to do, once the words in front of `FIELDS` have
/// been read.
struct Adding<'a> {
    /// Whether a document that is already there may be written over.
    replace: bool,
    /// Whether the fields are merged into what is there rather than replacing
    /// it. Only means anything beside a `REPLACE`.
    partial: bool,
    /// The language the client named, spelled the way the client spelled it.
    language: Option<&'a [u8]>,
    /// The payload to record on the document.
    payload: Option<&'a [u8]>,
    /// The condition the document already there has to answer true to.
    condition: Option<&'a [u8]>,
    /// The first field name, which is one past `FIELDS`.
    at: usize,
}

pub(in crate::dispatch) fn execute(
    server: &Server,
    db: usize,
    spec: &Spec,
    args: Args<'_>,
    out: &mut Out,
) -> Result<()> {
    match spec.name {
        "FT.ADD" | "FT.SAFEADD" => add(server, db, spec, args, out),
        "FT.GET" => get(server, db, spec, args, out),
        "FT.MGET" => mget(server, db, spec, args, out),
        "FT.DEL" => del(server, db, spec, args, out),
        other => unreachable!("{other} is not a document command"),
    }
}

/// `FT.ADD index key score [REPLACE] [PARTIAL] [LANGUAGE l] [PAYLOAD p] [IF e]
/// FIELDS field value ...`, and `FT.SAFEADD` which is the same command.
///
/// The arity in the table is the module's own `-1`, so the four word minimum is
/// checked here rather than by the dispatcher.
fn add(server: &Server, db: usize, spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<()> {
    if args.len() < 4 {
        return Err(args::wrong_arity(spec.name));
    }
    let name = args.get(1);
    let key = args.get(2);
    // The index comes first, ahead of the score and ahead of the options, which
    // is a real server's order and is visible: `FT.ADD nope k abc FIELDS t a`
    // answers about the index and not about the score.
    let Some(canon) = resolved(&mut server.search.lock(), name) else {
        Fail::naming(MISSING, name).write(out);
        return Ok(());
    };
    let worth = args.get(3);
    let Some(score) = yo_common::num::parse_f64(worth) else {
        Fail::plain(BAD_SCORE).write(out);
        return Ok(());
    };
    if !(0.0..=PLAIN).contains(&score) {
        Fail::plain(SCORE_RANGE).write(out);
        return Ok(());
    }
    let asked = match options(args) {
        Ok(asked) => asked,
        Err(fail) => {
            fail.write(out);
            return Ok(());
        }
    };

    // Everything above reads the argument list and everything below touches the
    // keyspace. A key of another type is refused before either the document
    // already there or the condition is looked at.
    let kind = server.dbs[db].hold(key).kind_of(key);
    if !matches!(kind, None | Some(Kind::Hash)) {
        Fail::plain(BAD_KEY).write(out);
        return Ok(());
    }
    if kind.is_some() {
        if !asked.replace {
            Fail::plain(HERE_ALREADY).write(out);
            return Ok(());
        }
        if let Some(src) = asked.condition {
            match keeps(server, db, &canon, key, src) {
                Ok(true) => {}
                Ok(false) => {
                    out.simple(b"NOADD");
                    return Ok(());
                }
                Err(text) => {
                    out.error(&text);
                    return Ok(());
                }
            }
        }
    }

    let mut pairs: Vec<(&[u8], &[u8])> = Vec::new();
    let mut at = asked.at;
    while let (Some(field), Some(value)) = (args.opt(at), args.opt(at + 1)) {
        pairs.push((field, value));
        at += 2;
    }
    let recorded = asked.partial || score != PLAIN;
    if recorded {
        pairs.push((SCORE_FIELD, worth));
    }
    if let Some(language) = asked.language {
        pairs.push((LANGUAGE_FIELD, language));
    }
    if let Some(payload) = asked.payload {
        pairs.push((PAYLOAD_FIELD, payload));
    }
    // Told before the write rather than after it, so the indexing notification
    // the write raises is the one that reads the three fields back off the hash.
    {
        let mut reg = server.search.lock();
        if let Some(index) = reg.get_mut(&canon) {
            if recorded {
                index.definition.score_field = Some(SCORE_FIELD.into());
            }
            if asked.language.is_some() {
                index.definition.language_field = Some(LANGUAGE_FIELD.into());
            }
            if asked.payload.is_some() {
                index.definition.payload_field = Some(PAYLOAD_FIELD.into());
            }
        }
    }
    {
        let mut held = server.dbs[db].hold(key);
        // Without `PARTIAL` the document is what was sent and nothing else, so
        // what is there goes first. With it the fields are merged into what is
        // there, which is an ordinary `HSET`.
        if !asked.partial {
            held.del(key);
        }
        if !pairs.is_empty() {
            held.hset(key, pairs.iter().copied())?;
        }
    }
    // The whole key was written rather than a field of it, so a key that ended
    // up with nothing in it is erased rather than indexed as an empty document.
    // Measured: an add of no fields at all moves nothing in `FT.INFO`.
    indexing::changed(server, db, key, Change::Key);
    out.ok();
    Ok(())
}

/// Reads the words between the score and the field list.
///
/// Any order and any case, and `FIELDS` ends it, so a word after `FIELDS` that
/// spells an option is an ordinary field name.
fn options<'a>(args: Args<'a>) -> core::result::Result<Adding<'a>, Fail<'a>> {
    let mut asked = Adding {
        replace: false,
        partial: false,
        language: None,
        payload: None,
        condition: None,
        at: 0,
    };
    let mut at = 4;
    loop {
        let Some(word) = args.opt(at) else {
            return Err(Fail::plain(NO_FIELDS));
        };
        at += 1;
        if args::is(word, b"FIELDS") {
            break;
        }
        if args::is(word, b"REPLACE") {
            asked.replace = true;
            continue;
        }
        if args::is(word, b"PARTIAL") {
            asked.partial = true;
            continue;
        }
        let taking = match () {
            () if args::is(word, b"LANGUAGE") => "LANGUAGE",
            () if args::is(word, b"PAYLOAD") => "PAYLOAD",
            () if args::is(word, b"IF") => "IF",
            () => return Err(Fail::about(UNKNOWN_WORD, word, UNKNOWN_END)),
        };
        let Some(value) = args.opt(at) else {
            return Err(Fail::about(NO_ARGUMENT, taking.as_bytes(), NO_ARGUMENT_END));
        };
        at += 1;
        match taking {
            // Checked against the list and then kept as the client wrote it,
            // since the spelling is what goes into the hash.
            "LANGUAGE" => {
                if !LANGUAGES.iter().any(|known| args::is(value, known)) {
                    return Err(Fail::plain(BAD_LANGUAGE));
                }
                asked.language = Some(value);
            }
            "PAYLOAD" => asked.payload = Some(value),
            _ => asked.condition = Some(value),
        }
    }
    if !(args.len() - at).is_multiple_of(2) {
        return Err(Fail::plain(ODD_FIELDS));
    }
    asked.at = at;
    Ok(asked)
}

/// Whether the document already under a key answers a condition true.
///
/// The error is the whole reply when there is one, which is three different
/// lines: the document is not one the index holds, the expression will not read,
/// or it names a property the schema has never heard of. The first of those is
/// checked before the expression is even parsed, so a nonsense condition over a
/// key the index does not hold answers about the key.
fn keeps(
    server: &Server,
    db: usize,
    canon: &[u8],
    key: &[u8],
    src: &[u8],
) -> core::result::Result<bool, Vec<u8>> {
    {
        let reg = server.search.lock();
        let held = reg
            .named(canon)
            .and_then(|index| index.held.docs.id(key))
            .is_some();
        if !held {
            return Err(NO_DOCUMENT.as_bytes().to_vec());
        }
    }
    // Read with the registry let go, which is the lock order every other path
    // through the indexes uses.
    let doc = indexing::read(&server.dbs[db], key).unwrap_or_default();
    let reg = server.search.lock();
    let Some(index) = reg.named(canon) else {
        return Err(NO_DOCUMENT.as_bytes().to_vec());
    };
    // Every property in the expression is a schema field, and the row is built
    // as the binder walks it, so a document that has nothing at a field the
    // schema does have is `Missing` rather than a refusal. That is what makes
    // `IF @n>3` false rather than an error once a plain `REPLACE` has taken `n`
    // off the hash.
    let mut row: Vec<Value> = Vec::new();
    let mut names: Vec<Box<[u8]>> = Vec::new();
    let mut expr = Expr::parse(src)?;
    expr.bind(&mut |name| {
        if let Some(at) = names.iter().position(|held| **held == *name) {
            return Some(at);
        }
        let field = index.field(name)?;
        names.push(name.into());
        row.push(valued(&doc, &field.identifier, &field.kind));
        Some(row.len() - 1)
    })
    .map_err(|missing| line(NOT_LOADED, &missing.0, QUOTE_END))?;
    Ok(expr.eval(&row)?.truth())
}

/// One field of a document as an expression sees it.
fn valued(doc: &Document, from: &[u8], kind: &Type) -> Value {
    let Some(value) = doc.held(from) else {
        return Value::Missing;
    };
    match kind {
        Type::Numeric => match yo_common::num::parse_f64(value) {
            Some(number) => Value::Number(number),
            None => Value::Text(value.into()),
        },
        _ => Value::Text(value.into()),
    }
}

/// `FT.GET index key`, and the hash under it when the index holds it.
fn get(server: &Server, db: usize, spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<()> {
    if args.len() != 3 {
        return Err(args::wrong_arity(spec.name));
    }
    let name = args.get(1);
    let Some(canon) = resolved(&mut server.search.lock(), name) else {
        Fail::naming(MISSING, name).write(out);
        return Ok(());
    };
    shown(server, db, &canon, args.get(2), out);
    Ok(())
}

/// `FT.MGET index key [key ...]`, which is the same answer once per key.
fn mget(server: &Server, db: usize, spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<()> {
    if args.len() < 3 {
        return Err(args::wrong_arity(spec.name));
    }
    let name = args.get(1);
    let Some(canon) = resolved(&mut server.search.lock(), name) else {
        Fail::naming(MISSING, name).write(out);
        return Ok(());
    };
    out.array(args.len() - 2);
    for at in 2..args.len() {
        shown(server, db, &canon, args.get(at), out);
    }
    Ok(())
}

/// One document, or nothing at all when the index is not holding it.
///
/// Held by the index and not merely covered by its prefix, which is the same
/// answer nearly always and is not the same question: an index with a `FILTER`
/// answers nothing for a key under its prefix that the filter left out.
///
/// A document the index holds whose key is not a hash is an empty list rather
/// than nothing, which is what an index `ON JSON` answers for every document in
/// it, since the reading here is a hash read and a JSON key is not one.
fn shown(server: &Server, db: usize, canon: &[u8], key: &[u8], out: &mut Out) {
    let recorded = {
        let reg = server.search.lock();
        let held = reg
            .named(canon)
            .filter(|index| index.held.docs.id(key).is_some());
        let Some(index) = held else {
            out.nil();
            return;
        };
        [
            index.definition.score_field.clone(),
            index.definition.language_field.clone(),
            index.definition.payload_field.clone(),
        ]
    };
    let Some(doc) = indexing::read(&server.dbs[db], key) else {
        out.array(0);
        return;
    };
    // The fields the score, the language and the payload are read out of are the
    // index's own bookkeeping, so they come off the reply. Which fields those are
    // is whatever the index was told, so an index naming `sc` hides `sc` and
    // shows a `__score` a client wrote by hand.
    let pairs: Vec<(&[u8], &[u8])> = doc
        .pairs()
        .into_iter()
        .filter(|(field, _)| {
            !recorded
                .iter()
                .flatten()
                .any(|name| name.as_ref() == *field)
        })
        .collect();
    out.array(pairs.len() * 2);
    for (field, value) in pairs {
        out.bulk(field);
        out.bulk(value);
    }
}

/// `FT.DEL index key [DD]`, and whether the key was there.
///
/// The word after the key is read and thrown away, whatever it says. `DD` used
/// to mean delete the document as well as the entry and `KEEPDOCS` used to mean
/// the opposite, and now the key goes either way, so the only thing the word
/// still does is take up the fourth place in the arity.
///
/// The key goes whether or not the index was holding it, and whether or not it
/// is a hash. This really is a `DEL` with an index name in front of it.
fn del(server: &Server, db: usize, spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<()> {
    if args.len() != 3 && args.len() != 4 {
        return Err(args::wrong_arity(spec.name));
    }
    let name = args.get(1);
    if resolved(&mut server.search.lock(), name).is_none() {
        Fail::naming(MISSING, name).write(out);
        return Ok(());
    }
    let key = args.get(2);
    let gone = server.dbs[db].hold(key).del(key);
    if gone {
        indexing::changed(server, db, key, Change::Key);
    }
    out.int(i64::from(gone));
    Ok(())
}

/// The name an index is really held under, counted as one use of it.
fn resolved(reg: &mut Registry, name: &[u8]) -> Option<Box<[u8]>> {
    reg.open(name).map(|index| index.name.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::tests::encode;
    use crate::proto::Limits;
    use crate::request::Argv;

    /// Reads an argument list the way the dispatcher would and hands the result
    /// of the option parser to the check. Every line these tests expect was read
    /// off Redis 8.10.1.
    fn read(words: &[&[u8]], check: impl FnOnce(core::result::Result<Adding<'_>, Fail<'_>>)) {
        let wire = encode(words);
        let mut argv = Argv::new();
        argv.decode(&wire, &Limits::default()).expect("it decodes");
        check(options(Args::new(&argv, &wire)));
    }

    #[test]
    fn the_option_words_come_in_any_order_and_any_case() {
        read(
            &[
                b"FT.ADD", b"i", b"k", b"1.0", b"partial", b"LaNgUaGe", b"FrEnCh", b"REPLACE",
                b"PAYLOAD", b"pp", b"IF", b"@n>1", b"FIELDS", b"t", b"a",
            ],
            |got| {
                let asked = got.ok().expect("it reads");
                assert!(asked.replace);
                assert!(asked.partial);
                // The language is kept the way the client spelled it, because
                // that spelling is what goes into the hash.
                assert_eq!(asked.language, Some(b"FrEnCh".as_slice()));
                assert_eq!(asked.payload, Some(b"pp".as_slice()));
                assert_eq!(asked.condition, Some(b"@n>1".as_slice()));
                assert_eq!(asked.at, 13);
            },
        );
    }

    #[test]
    fn fields_ends_the_option_list() {
        read(
            &[b"FT.ADD", b"i", b"k", b"1.0", b"FIELDS", b"REPLACE", b"x"],
            |got| {
                let asked = got.ok().expect("it reads");
                assert!(!asked.replace);
                assert_eq!(asked.at, 5);
            },
        );
    }

    #[test]
    fn an_option_with_nothing_after_it_names_itself_in_capitals() {
        for (word, named) in [
            (b"language".as_slice(), "LANGUAGE"),
            (b"PaYlOaD", "PAYLOAD"),
            (b"if", "IF"),
        ] {
            read(&[b"FT.ADD", b"i", b"k", b"1.0", word], |got| {
                let fail = got.err().expect("it refuses");
                assert_eq!(fail.head, NO_ARGUMENT);
                assert_eq!(fail.word, named.as_bytes());
                assert_eq!(fail.tail, NO_ARGUMENT_END);
            });
        }
    }

    #[test]
    fn an_unknown_word_comes_back_as_the_client_wrote_it() {
        // `NOSAVE` was an option once and is not one now, so it lands here with
        // everything else the parser has never heard of.
        for word in [b"bogus".as_slice(), b"NOSAVE"] {
            read(
                &[b"FT.ADD", b"i", b"k", b"1.0", word, b"FIELDS", b"t", b"a"],
                |got| {
                    let fail = got.err().expect("it refuses");
                    assert_eq!(fail.head, UNKNOWN_WORD);
                    assert_eq!(fail.word, word);
                    assert_eq!(fail.tail, UNKNOWN_END);
                },
            );
        }
    }

    #[test]
    fn a_field_list_is_required_and_comes_in_pairs() {
        read(&[b"FT.ADD", b"i", b"k", b"1.0", b"REPLACE"], |got| {
            assert_eq!(got.err().expect("it refuses").head, NO_FIELDS);
        });
        read(&[b"FT.ADD", b"i", b"k", b"1.0", b"FIELDS", b"t"], |got| {
            assert_eq!(got.err().expect("it refuses").head, ODD_FIELDS);
        });
        // An empty list is a pair list with no pairs in it, and is allowed.
        read(&[b"FT.ADD", b"i", b"k", b"1.0", b"FIELDS"], |got| {
            assert_eq!(got.ok().expect("it reads").at, 5);
        });
    }

    #[test]
    fn only_a_language_the_module_knows() {
        read(
            &[
                b"FT.ADD", b"i", b"k", b"1.0", b"LANGUAGE", b"klingon", b"FIELDS", b"t", b"a",
            ],
            |got| {
                assert_eq!(got.err().expect("it refuses").head, BAD_LANGUAGE);
            },
        );
    }
}
