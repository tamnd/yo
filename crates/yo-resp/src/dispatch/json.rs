//! The JSON commands, on the wire.
//!
//! A key holds one document, encoded as YOJB, and every command here is a path
//! against it. The path grammar is [`yo_doc::Path`], the text at both doors is
//! [`yo_doc::from_json`] and [`Value::write_json_at`], and a write is
//! [`yo_doc::edit`] over the offsets the path matched. There is very little
//! left for this file to do beyond deciding what shape the reply is, which is
//! most of what RedisJSON compatibility actually consists of.
//!
//! # The two path syntaxes and why they are half the work
//!
//! RedisJSON has two. A path that starts with `$` is a JSONPath and answers a
//! set, so every reply is an array with one entry per match and no match is an
//! empty array. Anything else is what it calls a legacy path, answers at most
//! one value, and a path that matched nothing is an error rather than an empty
//! answer. The two are matched by the same code and only the reply differs,
//! which is why [`Path::legacy`] exists and why nearly every command here ends
//! in a branch on it.
//!
//! It matters more than it looks. A client that sends `.` and one that sends
//! `$` are asking the same question and get different types back, so a
//! compatibility layer that picks one shape is wrong for half of the clients in
//! the world. The default when no path is given is the legacy root and not `$`,
//! so `JSON.GET key` answers the document and `JSON.GET key $` answers a one
//! element array holding it.
//!
//! # Two errors that go out without a prefix
//!
//! Every error this server sends is written by `write_error`, which puts `ERR`
//! or `WRONGTYPE` in front of the message because that prefix is what a client
//! branches on. RedisJSON sends two lines that have neither: a key of the wrong
//! type is `Existing key has wrong Redis type`, and a path that cannot create
//! what it names is `Err wrong static path`. They are odd, and they are what
//! `yo-compat` compares against byte for byte, so the two places that send them
//! write the line themselves and answer `Ok`.
//!
//! # What holds the document
//!
//! [`JsonBody`], through [`yo_kv::Foreign`], the same escape the graph and the
//! vector set go through. The payoff is the same too: `DEL`, `EXISTS`, `TYPE`,
//! `KEYS`, `SCAN`, `RANDOMKEY`, `EXPIRE`, `DBSIZE` and `FLUSHDB` all work on a
//! JSON key without a line here.
//!
//! `TYPE` answers `ReJSON-RL`, which is the name RedisJSON registers and which
//! clients test for by string.
//!
//! # An object comes back in key order
//!
//! Members are stored sorted by length and then by bytes, because that is what
//! makes a lookup a binary search, so the order a client wrote them in is not
//! kept anywhere. RedisJSON keeps insertion order. Every reply that writes an
//! object out shows it, and it is D-34 in the register.

use yo_common::{Code, Error, Result};
use yo_doc::{Builder, Edit, Format, Kind, Path, Step, Value, edit};
use yo_kv::{Foreign, Keyspace};

use super::args::{self, Args};
use super::table::Spec;
use crate::reply::Out;

/// What a key holding something that is not a document gets. No prefix, which
/// is RedisJSON's and not a slip here.
const WRONG_TYPE: &[u8] = b"Existing key has wrong Redis type";
/// What a path that would have to invent a place gets. Also unprefixed, and the
/// capitalisation is theirs.
const STATIC_PATH: &[u8] = b"Err wrong static path";
/// What a client gets for asking to create something below a key that is not
/// there.
const NOT_AT_ROOT: &str = "new objects must be created at the root";
/// The path when the command allows one and the client did not give one. The
/// legacy root and not `$`, which decides the shape of most of the replies
/// here.
const ROOT: &[u8] = b".";

/// One document under a key.
#[derive(Debug, Default)]
pub(super) struct JsonBody {
    /// The whole document, as one encoded value.
    ///
    /// One buffer and not a tree, because every read is a path walk over these
    /// bytes and every write rebuilds them. Empty only between the moment
    /// `JSON.DEL $` empties it and the moment the keyspace reaps the key.
    doc: Vec<u8>,
}

impl Foreign for JsonBody {
    fn type_name(&self) -> &'static str {
        "ReJSON-RL"
    }

    fn encoding(&self) -> &'static str {
        // RedisJSON reports `raw` here whatever it is holding, so a client that
        // reads `OBJECT ENCODING` off a document key gets the same word from
        // both servers even though neither one is telling it anything.
        "raw"
    }

    fn memory_bytes(&self) -> usize {
        self.doc.capacity()
    }

    fn is_empty(&self) -> bool {
        self.doc.is_empty()
    }
}

/// Run one JSON command.
pub(super) fn execute(db: &mut Keyspace, spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<()> {
    match spec.name {
        "json.set" => set(db, args, out),
        "json.get" => get(db, args, out),
        "json.mget" => mget(db, args, out),
        "json.del" | "json.forget" => del(db, args, out),
        "json.type" => kind(db, args, out),
        "json.toggle" => toggle(db, args, out),
        "json.clear" => clear(db, args, out),
        other => unreachable!("{other} is not a JSON command"),
    }
}

// ------------------------------------------------------------------ the writes

/// `JSON.SET key path value [NX|XX]`.
fn set(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let (key, raw, text) = (args.get(1), args.get(2), args.get(3));
    let mut only = Only::Either;
    if args.len() == 5 {
        if args::is(args.get(4), b"nx") {
            only = Only::Missing;
        } else if args::is(args.get(4), b"xx") {
            only = Only::Present;
        } else {
            return Err(args::syntax());
        }
    } else if args.len() != 4 {
        return Err(args::wrong_arity("json.set"));
    }
    let path = Path::parse(raw)?;
    // The document is parsed before the key is touched, so a client that sent
    // text that is not JSON changes nothing.
    let value = yo_doc::from_json(text)?;

    let body = match doc_mut(db, key, out)? {
        Doc::Wrong => return Ok(()),
        Doc::Gone => {
            // Nothing is there. The only path that says where a whole document
            // goes is the root, and this is checked before `NX` and `XX`
            // because RedisJSON checks it there.
            if !path.is_root() {
                return Err(Error::new(Code::Invalid, NOT_AT_ROOT));
            }
            if only == Only::Present {
                out.nil();
                return Ok(());
            }
            db.put_foreign(key, Box::new(JsonBody { doc: value }));
            out.ok();
            return Ok(());
        }
        Doc::Here(body) => body,
    };

    let after = {
        let root = readable(&body.doc)?;
        let mut hits = Vec::new();
        path.select(&root, &mut hits);
        if hits.is_empty() {
            if only == Only::Present {
                out.nil();
                return Ok(());
            }
            // A path that matched nothing can still say where a value goes, but
            // only if it says exactly one place. A wildcard or a descent would
            // have to invent one, and that is the unprefixed error.
            if !path.is_definite() {
                out.error(STATIC_PATH);
                return Ok(());
            }
            let Some(at) = grow(&root, &path, &value)? else {
                // The container that would hold it is not there, or is not an
                // object. Not an error on either syntax, just a write that did
                // not happen.
                out.nil();
                return Ok(());
            };
            edit(&root, &at)?
        } else {
            if only == Only::Missing {
                out.nil();
                return Ok(());
            }
            let at: Vec<_> = offsets(&root, &hits)?
                .into_iter()
                .map(|off| (off, Edit::Set(&value)))
                .collect();
            edit(&root, &at)?
        }
    };
    body.doc = after;
    out.ok();
    Ok(())
}

/// Whether a `NX` or an `XX` was given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Only {
    Either,
    Missing,
    Present,
}

/// Where a path that matched nothing would put a value, if anywhere.
///
/// Only ever called for a definite path, so the parent names at most one place
/// and the answer is at most one edit. `$.a.b` on a document with an `a` and no
/// `b` is the case this exists for.
fn grow<'v>(
    root: &Value<'_>,
    path: &Path<'v>,
    value: &'v [u8],
) -> Result<Option<Vec<(usize, Edit<'v>)>>> {
    let Some((parent, step)) = path.split_last() else {
        return Ok(None);
    };
    let Step::Key(name) = step else {
        // An index into an array, and the array either is not there or does not
        // reach that far. RedisJSON does not append there and neither does
        // this: `JSON.ARRAPPEND` is the command that appends and it says so in
        // its name.
        return Err(Error::new(Code::Invalid, "array index out of range"));
    };
    let Some(holder) = parent.first(root) else {
        return Ok(None);
    };
    if holder.kind() != Kind::Object {
        return Ok(None);
    }
    Ok(Some(vec![(offset(root, &holder)?, Edit::Put(name, value))]))
}

/// `JSON.DEL key [path]` and `JSON.FORGET`, which is the same command.
fn del(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let key = args.get(1);
    // Anything past the path is ignored rather than refused, which is
    // RedisJSON's behaviour and not an accident of the arity: it registers the
    // command as taking one path and never looks at the rest.
    let raw = args.opt(2).unwrap_or(ROOT);
    let path = Path::parse(raw)?;
    let body = match doc_mut(db, key, out)? {
        Doc::Wrong => return Ok(()),
        Doc::Gone => {
            out.int(0);
            return Ok(());
        }
        Doc::Here(body) => body,
    };
    // The root is the whole key, and a document with no value in it is not a
    // document, so this is the keyspace's business rather than the editor's.
    if path.is_root() {
        body.doc.clear();
        db.reap_foreign(key);
        out.int(1);
        return Ok(());
    }
    let (after, gone) = {
        let root = readable(&body.doc)?;
        let mut hits = Vec::new();
        path.select(&root, &mut hits);
        if hits.is_empty() {
            out.int(0);
            return Ok(());
        }
        let at: Vec<_> = offsets(&root, &hits)?
            .into_iter()
            .map(|off| (off, Edit::Remove))
            .collect();
        (edit(&root, &at)?, at.len())
    };
    // A delete that empties the root container empties the key, which is the
    // same rule the sets and the hashes follow and is why `EXISTS` on a list
    // somebody popped the last element off answers zero. It is a delete rule
    // and not a shape rule: a document that was written as an empty object by
    // `JSON.SET` stays, because nothing removed anything from it.
    let empty_root = matches!(
        readable(&after).map(|v| (v.kind(), v.is_empty())),
        Ok((Kind::Object | Kind::Array, true))
    );
    body.doc = after;
    if empty_root {
        body.doc.clear();
        db.reap_foreign(key);
    }
    out.int(gone as i64);
    Ok(())
}

/// `JSON.TOGGLE key path`, which flips every boolean the path names.
fn toggle(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let (key, raw) = (args.get(1), args.get(2));
    let path = Path::parse(raw)?;
    let body = match doc_mut(db, key, out)? {
        Doc::Wrong => return Ok(()),
        Doc::Gone => return Err(no_key()),
        Doc::Here(body) => body,
    };
    let (after, flipped) = {
        let root = readable(&body.doc)?;
        let mut hits = Vec::new();
        path.select(&root, &mut hits);
        // The two encoded booleans, kept here so the edits can borrow them.
        let (yes, no) = (yo_doc::from_json(b"true")?, yo_doc::from_json(b"false")?);
        let mut at = Vec::new();
        let mut flipped = Vec::new();
        for v in &hits {
            match v.as_bool() {
                Some(was) => {
                    let now: &[u8] = if was { &no } else { &yes };
                    at.push((offset(&root, v)?, Edit::Set(now)));
                    flipped.push(Some(!was));
                }
                // A path that matched something that is not a boolean is a hole
                // in the array for a JSONPath. A legacy path cannot say that,
                // so it says the same thing it says for a path that matched
                // nothing at all, in one sentence that covers both.
                None if path.legacy() => return Err(not_a_bool()),
                None => flipped.push(None),
            }
        }
        if hits.is_empty() && path.legacy() {
            return Err(not_a_bool());
        }
        (edit(&root, &at)?, flipped)
    };
    body.doc = after;
    if path.legacy() {
        match flipped.first().and_then(|f| *f) {
            Some(now) => out.bulk(if now { b"true" } else { b"false" }),
            None => out.nil(),
        }
        return Ok(());
    }
    out.array(flipped.len());
    for f in flipped {
        match f {
            Some(now) => out.int(i64::from(now)),
            None => out.nil(),
        }
    }
    Ok(())
}

/// `JSON.CLEAR key [path]`, which empties containers and zeroes numbers.
///
/// A string, a boolean and a null are left alone, which is RedisJSON's rule and
/// not an obvious one: clearing a string to `""` would be just as defensible.
fn clear(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let key = args.get(1);
    let raw = args.opt(2).unwrap_or(ROOT);
    let path = Path::parse(raw)?;
    let body = match doc_mut(db, key, out)? {
        Doc::Wrong => return Ok(()),
        Doc::Gone => return Err(no_key()),
        Doc::Here(body) => body,
    };
    let (after, cleared) = {
        let root = readable(&body.doc)?;
        let mut hits = Vec::new();
        path.select(&root, &mut hits);
        let (empty_object, empty_array, zero) = (
            yo_doc::from_json(b"{}")?,
            yo_doc::from_json(b"[]")?,
            yo_doc::from_json(b"0")?,
        );
        let mut at = Vec::new();
        for v in &hits {
            let to: &[u8] = match v.kind() {
                // Already empty is already clear, and rewriting it would count
                // a change that did not happen. The count is what the client
                // sees, so this is the whole difference.
                Kind::Object if v.is_empty() => continue,
                Kind::Array if v.is_empty() => continue,
                Kind::Object => &empty_object,
                Kind::Array => &empty_array,
                Kind::Int if v.as_int() == Some(0) => continue,
                Kind::Int | Kind::Float => &zero,
                _ => continue,
            };
            at.push((offset(&root, v)?, Edit::Set(to)));
        }
        (edit(&root, &at)?, at.len())
    };
    body.doc = after;
    out.int(cleared as i64);
    Ok(())
}

// ------------------------------------------------------------------- the reads

/// `JSON.GET key [INDENT s] [NEWLINE s] [SPACE s] [path ...]`.
fn get(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let key = args.get(1);
    let (f, from) = format(args, 2)?;
    let raws: Vec<&[u8]> = if from < args.len() {
        (from..args.len()).map(|i| args.get(i)).collect()
    } else {
        vec![ROOT]
    };
    // Parsed before the key is read, so a path that does not parse is the same
    // error whether or not the key is there.
    let paths: Vec<Path<'_>> = raws.iter().map(|r| Path::parse(r)).collect::<Result<_>>()?;

    let body = match doc(db, key, out)? {
        Doc::Wrong => return Ok(()),
        Doc::Gone => {
            out.nil();
            return Ok(());
        }
        Doc::Here(body) => body,
    };
    let root = readable(&body.doc)?;
    let mut text = Vec::new();
    if paths.len() == 1 {
        one(&root, &paths[0], &f, &mut text, 0)?;
    } else {
        // More than one path answers an object keyed by the paths as they were
        // written, so a client that asked three questions can tell which answer
        // is which. The keys come back in the order they were asked in, which
        // RedisJSON does not promise because it builds the reply out of a hash
        // map, and that is the same D-34 the document's own key order is.
        text.push(b'{');
        for (i, (path, raw)) in paths.iter().zip(&raws).enumerate() {
            if i > 0 {
                text.push(b',');
            }
            line(&f, &mut text, 1);
            quote(raw, &mut text);
            text.push(b':');
            text.extend_from_slice(f.space);
            one(&root, path, &f, &mut text, 1)?;
        }
        line(&f, &mut text, 0);
        text.push(b'}');
    }
    out.bulk(&text);
    Ok(())
}

/// One path's answer, as the text it contributes, written as if it sat `depth`
/// levels inside whatever wrapper the caller has already opened.
fn one(
    root: &Value<'_>,
    path: &Path<'_>,
    f: &Format<'_>,
    text: &mut Vec<u8>,
    depth: usize,
) -> Result<()> {
    let mut hits = Vec::new();
    path.select(root, &mut hits);
    if path.legacy() {
        let Some(v) = hits.first() else {
            return Err(missing());
        };
        return v.write_json_at(f, text, depth);
    }
    let laid_out = !f.is_plain() && !hits.is_empty();
    text.push(b'[');
    for (i, v) in hits.iter().enumerate() {
        if i > 0 {
            text.push(b',');
        }
        if laid_out {
            line(f, text, depth + 1);
        }
        v.write_json_at(f, text, depth + 1)?;
    }
    if laid_out {
        line(f, text, depth);
    }
    text.push(b']');
    Ok(())
}

/// End a line and indent the next one, for the wrapper this file builds itself.
///
/// The same rule the document writer follows, which is that the whole thing is
/// one line unless the client asked for something.
fn line(f: &Format<'_>, text: &mut Vec<u8>, depth: usize) {
    if f.is_plain() {
        return;
    }
    text.extend_from_slice(f.newline);
    for _ in 0..depth {
        text.extend_from_slice(f.indent);
    }
}

/// `JSON.MGET key [key ...] path`, one path against many documents.
fn mget(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let last = args.len() - 1;
    let path = Path::parse(args.get(last))?;
    let f = Format::default();
    out.array(last - 1);
    let mut text = Vec::new();
    for i in 1..last {
        let key = args.get(i);
        // A key that is not there and a key holding something that is not a
        // document are both a hole in the array. `MGET` does the same with a
        // key holding a hash, and for the same reason: one bad key in a hundred
        // should not lose the other ninety nine answers.
        let Ok(Some(body)) = db.foreign(key) else {
            out.nil();
            continue;
        };
        let Some(body) = body.downcast_ref::<JsonBody>() else {
            out.nil();
            continue;
        };
        let root = readable(&body.doc)?;
        text.clear();
        match one(&root, &path, &f, &mut text, 0) {
            Ok(()) => out.bulk(&text),
            Err(_) => out.nil(),
        }
    }
    Ok(())
}

/// `JSON.TYPE key [path]`.
fn kind(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let key = args.get(1);
    let raw = args.opt(2).unwrap_or(ROOT);
    let path = Path::parse(raw)?;
    let body = match doc(db, key, out)? {
        Doc::Wrong => return Ok(()),
        Doc::Gone => {
            out.nil();
            return Ok(());
        }
        Doc::Here(body) => body,
    };
    let root = readable(&body.doc)?;
    let mut hits = Vec::new();
    path.select(&root, &mut hits);
    if path.legacy() {
        // The one command here where a legacy path that matched nothing is a
        // nil rather than an error, which lines up with the key that is not
        // there answering nil too.
        match hits.first() {
            Some(v) => out.bulk(word(v)),
            None => out.nil(),
        }
        return Ok(());
    }
    out.array(hits.len());
    for v in &hits {
        out.bulk(word(v));
    }
    Ok(())
}

/// The word `JSON.TYPE` answers for a value.
///
/// `integer` and `number` are two words for what the encoding stores as two
/// kinds, which is the one place RedisJSON's type names line up with ours
/// exactly rather than by translation.
fn word(v: &Value<'_>) -> &'static [u8] {
    match v.kind() {
        Kind::Null => b"null",
        Kind::Bool => b"boolean",
        Kind::Int => b"integer",
        Kind::Float => b"number",
        Kind::Text => b"string",
        Kind::Array => b"array",
        Kind::Object => b"object",
    }
}

// ----------------------------------------------------------------- the plumbing

/// What a lookup of a document key found.
///
/// Three answers and not two, because the wrong type is a line this file writes
/// itself rather than an error it returns, and the caller has to know that the
/// reply is already written.
enum Doc<B> {
    /// The key holds a document.
    Here(B),
    /// The key is not there, which every command answers differently.
    Gone,
    /// The key holds something else and the error line is already out.
    Wrong,
}

/// The body under `key` for writing.
fn doc_mut<'d>(db: &'d mut Keyspace, key: &[u8], out: &mut Out) -> Result<Doc<&'d mut JsonBody>> {
    match db.foreign_mut(key) {
        Ok(Some(body)) => match body.downcast_mut::<JsonBody>() {
            Some(body) => Ok(Doc::Here(body)),
            None => {
                out.error(WRONG_TYPE);
                Ok(Doc::Wrong)
            }
        },
        Ok(None) => Ok(Doc::Gone),
        Err(e) if e.code() == Code::WrongType => {
            out.error(WRONG_TYPE);
            Ok(Doc::Wrong)
        }
        Err(e) => Err(e),
    }
}

/// The same, for reading.
fn doc<'d>(db: &'d mut Keyspace, key: &[u8], out: &mut Out) -> Result<Doc<&'d JsonBody>> {
    match db.foreign(key) {
        Ok(Some(body)) => match body.downcast_ref::<JsonBody>() {
            Some(body) => Ok(Doc::Here(body)),
            None => {
                out.error(WRONG_TYPE);
                Ok(Doc::Wrong)
            }
        },
        Ok(None) => Ok(Doc::Gone),
        Err(e) if e.code() == Code::WrongType => {
            out.error(WRONG_TYPE);
            Ok(Doc::Wrong)
        }
        Err(e) => Err(e),
    }
}

/// `INDENT`, `NEWLINE` and `SPACE` in any order, and where the paths start.
///
/// Redis takes these three as strings rather than as counts, so a client can
/// ask for tabs, and the default for all three is empty, which is one line.
fn format(args: Args<'_>, from: usize) -> Result<(Format<'_>, usize)> {
    let mut f = Format::default();
    let mut i = from;
    while i + 1 < args.len() {
        let arg = args.get(i);
        if args::is(arg, b"indent") {
            f.indent = args.get(i + 1);
        } else if args::is(arg, b"newline") {
            f.newline = args.get(i + 1);
        } else if args::is(arg, b"space") {
            f.space = args.get(i + 1);
        } else {
            break;
        }
        i += 2;
    }
    Ok((f, i))
}

/// A path as a JSON string, for the key of the object more than one path
/// answers.
fn quote(raw: &[u8], out: &mut Vec<u8>) {
    let mut b = Builder::new();
    // A path is bytes off the wire and could hold anything, so it goes through
    // the same escaping every other string does rather than being copied
    // between two quotes.
    if b.text_bytes(raw).is_ok()
        && let Ok(bytes) = b.finish()
        && let Some(v) = Value::new(bytes)
        && v.write_json(out).is_ok()
    {
        return;
    }
    out.extend_from_slice(b"\"\"");
}

/// The offsets of a set of matches inside the document they came out of.
fn offsets(root: &Value<'_>, hits: &[Value<'_>]) -> Result<Vec<usize>> {
    hits.iter().map(|v| offset(root, v)).collect()
}

/// The offset of one match.
fn offset(root: &Value<'_>, v: &Value<'_>) -> Result<usize> {
    v.offset_in(root).ok_or_else(|| {
        Error::new(
            Code::Invalid,
            "a path matched a value that is not in the document it was matched against",
        )
    })
}

/// The document under a key, as a value.
fn readable(doc: &[u8]) -> Result<Value<'_>> {
    Value::new(doc).ok_or_else(|| Error::new(Code::Invalid, "the document stored here is damaged"))
}

/// What a legacy path that named nothing gets.
///
/// The path is not in it. RedisJSON quotes the path in some of its other
/// messages and not in this one, and this is the one clients see.
fn missing() -> Error {
    Error::new(Code::Invalid, "Path does not exist")
}

/// What a legacy `JSON.TOGGLE` gets for a path that matched nothing and for a
/// path that matched something that is not a boolean, which is one message.
fn not_a_bool() -> Error {
    Error::new(Code::Invalid, "Path does not exist or not a bool")
}

/// What the commands that need a document get when there is not one.
fn no_key() -> Error {
    Error::new(
        Code::Invalid,
        "could not perform this operation on a key that doesn't exist",
    )
}
