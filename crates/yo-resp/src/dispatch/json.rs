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
        "json.mset" => mset(db, args, out),
        "json.merge" => merge(db, args, out),
        "json.get" => get(db, args, out),
        "json.resp" => resp(db, args, out),
        "json.debug" => debug(db, args, out),
        "json.mget" => mget(db, args, out),
        "json.del" | "json.forget" => del(db, args, out),
        "json.type" => kind(db, args, out),
        "json.toggle" => toggle(db, args, out),
        "json.clear" => clear(db, args, out),
        "json.arrlen" => sized(db, args, out, Asked::ArrayLen),
        "json.objlen" => sized(db, args, out, Asked::ObjectLen),
        "json.strlen" => sized(db, args, out, Asked::TextLen),
        "json.objkeys" => sized(db, args, out, Asked::ObjectKeys),
        "json.arrappend" => arrappend(db, args, out),
        "json.arrinsert" => arrinsert(db, args, out),
        "json.arrtrim" => arrtrim(db, args, out),
        "json.arrpop" => arrpop(db, args, out),
        "json.arrindex" => arrindex(db, args, out),
        "json.numincrby" => arith(db, args, out, Arith::Add),
        "json.nummultby" => arith(db, args, out, Arith::Mul),
        "json.numpowby" => arith(db, args, out, Arith::Pow),
        "json.strappend" => strappend(db, args, out),
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
    let value = match yo_doc::from_json(text) {
        Ok(value) => value,
        Err(e) => {
            unprefixed(&e, out);
            return Ok(());
        }
    };
    match plan_one(db, key, &path, &value, only, out)? {
        Plan::Store(doc) => {
            store(db, key, doc);
            out.ok();
        }
        Plan::Nothing => out.nil(),
        Plan::Refused => {}
    }
    Ok(())
}

/// What one `key path value` write would do, worked out without doing it.
///
/// Three answers and not two, because the line a path that cannot say where a
/// value goes gets has no prefix and so is written where it happens rather than
/// returned, and the caller has to know the reply is already out.
enum Plan {
    /// Put this document under the key.
    Store(Vec<u8>),
    /// The path named nowhere the value could go.
    Nothing,
    /// Nothing is to be written and the reply is already out.
    Refused,
}

/// One `key path value` write, which is all of `JSON.SET` and one of
/// `JSON.MSET`'s triples.
///
/// It answers the document rather than storing it because `JSON.MSET` has to
/// know that every triple works before the first one is written.
fn plan_one<'v>(
    db: &mut Keyspace,
    key: &[u8],
    path: &Path<'v>,
    value: &'v [u8],
    only: Only,
    out: &mut Out,
) -> Result<Plan> {
    let body = match doc(db, key, out)? {
        Doc::Wrong => return Ok(Plan::Refused),
        Doc::Gone => {
            // Nothing is there. The only path that says where a whole document
            // goes is the root, and this is checked before `NX` and `XX`
            // because RedisJSON checks it there.
            if !path.is_root() {
                return Err(Error::new(Code::Invalid, NOT_AT_ROOT));
            }
            if only == Only::Present {
                return Ok(Plan::Nothing);
            }
            return Ok(Plan::Store(value.to_vec()));
        }
        Doc::Here(body) => body,
    };

    let root = readable(&body.doc)?;
    let mut hits = Vec::new();
    path.select(&root, &mut hits);
    if hits.is_empty() {
        if only == Only::Present {
            return Ok(Plan::Nothing);
        }
        // A path that matched nothing can still say where a value goes, but
        // only if it says exactly one place. A wildcard or a descent would
        // have to invent one, and that is the unprefixed error.
        if !path.is_definite() {
            out.error(STATIC_PATH);
            return Ok(Plan::Refused);
        }
        let Some(at) = grow(&root, path, value)? else {
            // The container that would hold it is not there, or is not an
            // object. Not an error on either syntax, just a write that did
            // not happen.
            return Ok(Plan::Nothing);
        };
        Ok(Plan::Store(edit(&root, &at)?))
    } else {
        if only == Only::Missing {
            return Ok(Plan::Nothing);
        }
        let at: Vec<_> = offsets(&root, &hits)?
            .into_iter()
            .map(|off| (off, Edit::Set(value)))
            .collect();
        Ok(Plan::Store(edit(&root, &at)?))
    }
}

/// Put a document under a key, whether or not one is there already.
///
/// Only ever called for a key a [`Plan`] was worked out against, so the key
/// either holds a document or holds nothing, and the wrong type is somebody
/// else's error to write.
fn store(db: &mut Keyspace, key: &[u8], doc: Vec<u8>) {
    if let Ok(Some(body)) = db.foreign_mut(key)
        && let Some(body) = body.downcast_mut::<JsonBody>()
    {
        body.doc = doc;
        return;
    }
    db.put_foreign(key, Box::new(JsonBody { doc }));
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

/// `JSON.MSET key path value [key path value ...]`.
///
/// Every triple is worked out against the keyspace as it was before the command
/// and nothing is written until all of them are known to work, so a client that
/// saw an error knows that nothing happened. What is not an error is a path that
/// names nowhere to put its value: that triple is skipped, the others are still
/// written, and the reply is a nil instead of `OK`. Read off 8.10.1 both ways
/// round, with the failing triple first and last, because a naive loop and this
/// are the same for one order and not for the other.
///
/// Working every triple out against the state the command started in is also
/// what the reference does, so two triples on one key do not see each other and
/// the last one written is the one that stays.
fn mset(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    if args.len() < 4 || !(args.len() - 1).is_multiple_of(3) {
        return Err(args::wrong_arity("json.mset"));
    }
    let mut jobs = Vec::with_capacity((args.len() - 1) / 3);
    for i in (1..args.len()).step_by(3) {
        let path = Path::parse(args.get(i + 1))?;
        let value = match yo_doc::from_json(args.get(i + 2)) {
            Ok(value) => value,
            Err(e) => {
                unprefixed(&e, out);
                return Ok(());
            }
        };
        jobs.push((args.get(i), path, value));
    }
    let mut plans = Vec::with_capacity(jobs.len());
    for (key, path, value) in &jobs {
        match plan_one(db, key, path, value, Only::Either, out)? {
            Plan::Refused => return Ok(()),
            plan => plans.push(plan),
        }
    }
    let mut all = true;
    for ((key, _, _), plan) in jobs.iter().zip(plans) {
        match plan {
            Plan::Store(doc) => store(db, key, doc),
            Plan::Nothing => all = false,
            // Every one of these was turned into an early return above.
            Plan::Refused => unreachable!("a refused triple is answered before any is written"),
        }
    }
    if all {
        out.ok();
    } else {
        out.nil();
    }
    Ok(())
}

/// `JSON.MERGE key path value`.
///
/// The value is a merge patch, RFC 7386: a patch that is not an object replaces
/// what it is merged onto, and an object patch is applied member by member with
/// a `null` deleting the member of that name. It is the one write here whose
/// argument is not the value being stored.
fn merge(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    if args.len() != 4 {
        return Err(args::syntax());
    }
    let (key, raw, text) = (args.get(1), args.get(2), args.get(3));
    let path = Path::parse(raw)?;
    let patch = match yo_doc::from_json(text) {
        Ok(patch) => patch,
        Err(e) => {
            unprefixed(&e, out);
            return Ok(());
        }
    };
    let body = match doc_mut(db, key, out)? {
        Doc::Wrong => return Ok(()),
        Doc::Gone => {
            if !path.is_root() {
                return Err(Error::new(Code::Invalid, NOT_AT_ROOT));
            }
            // Stored as it stands, nulls and all. A merge patch says a null
            // deletes the member of that name, and RFC 7386 goes on to say that
            // a patch applied to nothing has its nulls dropped, but RedisJSON
            // only does that where there is something to merge onto: a key that
            // is not there and a member the path is about to create both keep
            // the null. Read off 8.10.1 with `{"x":null,"y":1}` on both.
            db.put_foreign(key, Box::new(JsonBody { doc: patch }));
            out.ok();
            return Ok(());
        }
        Doc::Here(body) => body,
    };

    // The merged values, held out here because the edits below borrow them.
    let made: Vec<Vec<u8>>;
    let after = {
        let root = readable(&body.doc)?;
        let patched = readable(&patch)?;
        let mut hits = Vec::new();
        path.select(&root, &mut hits);
        if hits.is_empty() {
            // The same rule `JSON.SET` follows for a path that matched nothing,
            // down to the unprefixed line for a path that would have to invent
            // the place it names.
            if !path.is_definite() {
                out.error(STATIC_PATH);
                return Ok(());
            }
            let Some(at) = grow(&root, &path, &patch)? else {
                out.nil();
                return Ok(());
            };
            edit(&root, &at)?
        } else {
            made = folded(&root, &hits, &patched)?;
            let at: Vec<_> = offsets(&root, &hits)?
                .into_iter()
                .zip(&made)
                .map(|(off, bytes)| (off, Edit::Set(bytes)))
                .collect();
            edit(&root, &at)?
        }
    };
    body.doc = after;
    out.ok();
    Ok(())
}

/// The patch merged onto every match, innermost match first.
///
/// A `$..` path is the only one that matches a value and something inside that
/// same value, and the two are not independent: the reference merges the inner
/// match first and then merges the outer one onto the result, so the inner
/// change is still there at the end. `$..*` with `{"m":1}` on `{"a":{"b":1}}`
/// comes out as `{"a":{"b":{"m":1},"m":1}}` and not as `{"a":{"b":1,"m":1}}`.
/// Working outermost first would throw the inner one away, which is what
/// [`edit`] does with an edit inside an edit and is right for `JSON.SET`.
///
/// The answers line up with `hits`. Sorting by offset the other way round is
/// what puts an inner match before the outer one, since a match is always
/// further into the document than anything that holds it.
fn folded(root: &Value<'_>, hits: &[Value<'_>], patch: &Value<'_>) -> Result<Vec<Vec<u8>>> {
    let spans = hits
        .iter()
        .map(|v| {
            let at = offset(root, v)?;
            Ok((at, at + v.encoded_len().ok_or_else(damaged)?))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut order: Vec<usize> = (0..hits.len()).collect();
    order.sort_by_key(|&i| core::cmp::Reverse(spans[i].0));

    let mut made: Vec<Vec<u8>> = vec![Vec::new(); hits.len()];
    for &i in &order {
        let (at, end) = spans[i];
        // Everything already merged that sits inside this match. The ones
        // nested inside another of these are dropped by `edit`, which is right,
        // because the one that holds them already has them folded in.
        let inside: Vec<_> = (0..hits.len())
            .filter(|&j| j != i && spans[j].0 > at && spans[j].0 < end)
            .map(|j| Ok((offset(&hits[i], &hits[j])?, Edit::Set(made[j].as_slice()))))
            .collect::<Result<Vec<_>>>()?;
        let with = if inside.is_empty() {
            None
        } else {
            Some(edit(&hits[i], &inside)?)
        };
        let target = match &with {
            Some(bytes) => readable(bytes)?,
            None => hits[i],
        };
        let mut b = Builder::new();
        merged(Some(&target), patch, &mut b)?;
        made[i] = b.finish()?.to_vec();
    }
    Ok(made)
}

/// `patch` merged onto `target`, which is RFC 7386 and nothing more.
///
/// A patch that is not an object replaces the target outright, whatever the
/// target was. An object patch merged onto anything that is not an object
/// starts from an empty object, which is what makes `{"x":null,"y":1}` onto a
/// `7` come out as `{"y":1}`: the deletion has nothing to delete and is dropped
/// rather than kept as a null.
fn merged(target: Option<&Value<'_>>, patch: &Value<'_>, b: &mut Builder) -> Result<()> {
    if patch.kind() != Kind::Object {
        return b.embed(patch);
    }
    let was = target.filter(|t| t.kind() == Kind::Object);
    b.begin_object()?;
    if let Some(t) = was {
        for i in 0..t.len() {
            // A member the patch names is written by the loop below, or is a
            // deletion and is written by nobody.
            let key = t.key_at(i).ok_or_else(damaged)?;
            if patch.get(key).is_some() {
                continue;
            }
            b.key(key)?;
            b.embed(&t.at(i).ok_or_else(damaged)?)?;
        }
    }
    for i in 0..patch.len() {
        let key = patch.key_at(i).ok_or_else(damaged)?;
        let v = patch.at(i).ok_or_else(damaged)?;
        if v.is_null() {
            continue;
        }
        b.key(key)?;
        merged(was.and_then(|t| t.get(key)).as_ref(), &v, b)?;
    }
    b.end_object()
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
                // in the array for a JSONPath, and for a legacy path it is a
                // match this skips over on the way to the next one.
                None => flipped.push(None),
            }
        }
        // Nothing to flip anywhere is the error, and it is one sentence for a
        // path that matched nothing and a path that matched no boolean, because
        // a legacy reply has no room to say which.
        if path.legacy() && !flipped.iter().any(Option::is_some) {
            return Err(not_a_bool());
        }
        (edit(&root, &at)?, flipped)
    };
    body.doc = after;
    if path.legacy() {
        match Pick::Last.of(&flipped) {
            Some(now) => out.bulk(if *now { b"true" } else { b"false" }),
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

// ------------------------------------------------------------- the array family

/// Which of `JSON.ARRLEN`, `JSON.OBJLEN`, `JSON.STRLEN` and `JSON.OBJKEYS` is
/// being run.
///
/// The four are one command with four sets of answers, and the answers do not
/// follow a pattern. A legacy path that matched nothing is an error for
/// `JSON.ARRLEN` and `JSON.STRLEN` and a nil for the other two. A legacy path
/// that matched the wrong kind of value is an `ERR` for `JSON.ARRLEN` and
/// `JSON.OBJKEYS` and a `WRONGTYPE` for the other two. A JSONPath against a key
/// that is not there is `could not perform this operation on a key that doesn't
/// exist` for three of them and `Path does not exist or not an object` for
/// `JSON.OBJLEN`, which is a line about a path being sent for a missing key.
///
/// None of that is a simplification of RedisJSON and none of it is a guess. It
/// was read off a running module one command at a time, because a client that
/// branches on the error text is the entire reason this file exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Asked {
    ArrayLen,
    ObjectLen,
    TextLen,
    ObjectKeys,
}

impl Asked {
    /// The kind of value the path is supposed to name.
    fn kind(self) -> Kind {
        match self {
            Asked::ArrayLen => Kind::Array,
            Asked::ObjectLen | Asked::ObjectKeys => Kind::Object,
            Asked::TextLen => Kind::Text,
        }
    }

    /// Whether a legacy path that matched nothing answers nil rather than an
    /// error.
    fn quiet(self) -> bool {
        matches!(self, Asked::ObjectLen | Asked::ObjectKeys)
    }

    /// What a legacy path that matched the wrong kind of value gets.
    fn wrong(self) -> Error {
        match self {
            Asked::ArrayLen => Error::new(Code::Invalid, "Path does not exist or not an array"),
            Asked::ObjectKeys => Error::new(Code::Invalid, "Path does not exist or not an object"),
            Asked::ObjectLen => Error::new(
                Code::WrongType,
                "wrong type of path value - expected object",
            ),
            Asked::TextLen => Error::new(
                Code::WrongType,
                "wrong type of path value - expected string",
            ),
        }
    }

    /// What a JSONPath against a key that is not there gets.
    fn no_key(self) -> Error {
        match self {
            Asked::ObjectLen => Error::new(Code::Invalid, "Path does not exist or not an object"),
            _ => no_key(),
        }
    }

    /// The answer for one match, which is a count for three of them and the
    /// keys themselves for the fourth.
    ///
    /// A document written through this file never has its keys interned, since
    /// interning is a collection's table and there is no collection here, so
    /// [`Value::key_at`] always answers. The fallback is there because the type
    /// allows the other case rather than because anything reaches it.
    fn one(self, v: &Value<'_>, out: &mut Out) {
        match self {
            Asked::TextLen => out.int(v.text_bytes().unwrap_or_default().len() as i64),
            Asked::ArrayLen | Asked::ObjectLen => out.int(v.len() as i64),
            Asked::ObjectKeys => {
                out.array(v.len());
                for i in 0..v.len() {
                    out.bulk(v.key_at(i).unwrap_or_default());
                }
            }
        }
    }
}

/// `JSON.ARRLEN`, `JSON.OBJLEN`, `JSON.STRLEN` and `JSON.OBJKEYS`.
fn sized(db: &mut Keyspace, args: Args<'_>, out: &mut Out, asked: Asked) -> Result<()> {
    let key = args.get(1);
    let raw = args.opt(2).unwrap_or(ROOT);
    let path = Path::parse(raw)?;
    let body = match doc(db, key, out)? {
        Doc::Wrong => return Ok(()),
        // A legacy path against a missing key is a nil and a JSONPath against
        // one is an error, which is the opposite way round from everywhere else
        // in this file and is what the module does.
        Doc::Gone if path.legacy() => {
            out.nil();
            return Ok(());
        }
        Doc::Gone => return Err(asked.no_key()),
        Doc::Here(body) => body,
    };
    let root = readable(&body.doc)?;
    let mut hits = Vec::new();
    path.select(&root, &mut hits);
    if path.legacy() {
        let Some(v) = hits.first() else {
            if asked.quiet() {
                out.nil();
                return Ok(());
            }
            return Err(missing());
        };
        if v.kind() != asked.kind() {
            return Err(asked.wrong());
        }
        asked.one(v, out);
        return Ok(());
    }
    out.array(hits.len());
    for v in &hits {
        if v.kind() == asked.kind() {
            asked.one(v, out);
        } else {
            out.nil();
        }
    }
    Ok(())
}

/// What every array command says when a legacy path did not name an array,
/// whether because it matched nothing or because it matched something else.
///
/// One line for both, which is the same shape as `JSON.TOGGLE`'s and for the
/// same reason: a legacy path answers at most one value, so there is nothing
/// for the message to say about which of the two happened.
fn not_an_array() -> Error {
    Error::new(Code::Invalid, "Path does not exist or not an array")
}

/// The arrays a path matched, with their offsets, and nothing else.
///
/// Every array command needs the same three things and treats a match that is
/// not an array the same way, so the walk is written once. The answer is one
/// entry per match, `None` for a match that is not an array, which is the hole
/// a JSONPath reply shows as a nil.
fn arrays<'r>(root: &Value<'r>, path: &Path<'_>) -> Result<Vec<Option<(usize, Value<'r>)>>> {
    let mut hits = Vec::new();
    path.select(root, &mut hits);
    let mut out = Vec::with_capacity(hits.len());
    for v in hits {
        if v.kind() == Kind::Array {
            out.push(Some((offset(root, &v)?, v)));
        } else {
            out.push(None);
        }
    }
    Ok(out)
}

/// That a legacy path named at least one array, or the error that says it did
/// not.
///
/// A legacy path can match more than one value, `.a[*]` being the short way to
/// get there, and a write applies to every match it can rather than to the
/// first one. So this refuses only when there was nothing it could do at all: a
/// path that matched a string and then two arrays appends to the two arrays and
/// answers, and a path that matched two strings is the error. Checked on 8.10.1
/// on `{"a":["x",[1],[1,2]]}`, which answers three.
fn any_array(found: &[Option<(usize, Value<'_>)>]) -> Result<()> {
    if found.iter().any(Option::is_some) {
        return Ok(());
    }
    Err(not_an_array())
}

/// Which of the matches a legacy path answers with.
///
/// A legacy path answers one value and a write touches every match, so there is
/// a choice here, and RedisJSON does not make the same one twice. The readers
/// answer the first match. The writes answer whichever match they wrote last,
/// and they do not all walk the matches the same way round: `JSON.ARRAPPEND`,
/// the number family, `JSON.STRAPPEND` and `JSON.TOGGLE` walk forwards and so
/// answer the last, while `JSON.ARRINSERT`, `JSON.ARRTRIM` and `JSON.ARRPOP`
/// walk backwards and so answer the first. It reads like an accident of how
/// each one was written, and it is what a client sees, so it is copied rather
/// than tidied up. Read off 8.10.1 with three arrays of one, two and three
/// elements, which tells the two apart in one command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pick {
    First,
    Last,
}

impl Pick {
    /// The answer a legacy path gets, out of one answer per match.
    ///
    /// A `None` is a match this command did not touch because it was the wrong
    /// kind of value, and those are skipped rather than counted, which is why a
    /// path that matched a string and then an array answers about the array.
    fn of<T>(self, all: &[Option<T>]) -> Option<&T> {
        let mut kept = all.iter().flatten();
        match self {
            Pick::First => kept.next(),
            Pick::Last => kept.next_back(),
        }
    }
}

/// Answer with the new length of every array the command touched, in whichever
/// of the two shapes the path asked for.
fn lengths(path: &Path<'_>, lens: &[Option<usize>], pick: Pick, out: &mut Out) {
    if path.legacy() {
        match pick.of(lens) {
            Some(n) => out.int(*n as i64),
            None => out.nil(),
        }
        return;
    }
    out.array(lens.len());
    for n in lens {
        match n {
            Some(n) => out.int(*n as i64),
            None => out.nil(),
        }
    }
}

/// The values a command takes from `from` onwards, encoded.
///
/// Parsed before the key is touched, so a value that is not JSON changes
/// nothing, which is the rule `JSON.SET` follows too.
fn values(args: Args<'_>, from: usize) -> Result<Vec<Vec<u8>>> {
    (from..args.len())
        .map(|i| yo_doc::from_json(args.get(i)))
        .collect()
}

/// `JSON.ARRAPPEND key path value [value ...]`.
fn arrappend(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let (key, raw) = (args.get(1), args.get(2));
    if args.len() < 4 {
        return Err(args::wrong_arity("json.arrappend"));
    }
    let path = Path::parse(raw)?;
    let added = match values(args, 3) {
        Ok(added) => added,
        Err(e) => {
            unprefixed(&e, out);
            return Ok(());
        }
    };
    let body = match doc_mut(db, key, out)? {
        Doc::Wrong => return Ok(()),
        Doc::Gone => return Err(no_key()),
        Doc::Here(body) => body,
    };
    let put: Vec<&[u8]> = added.iter().map(Vec::as_slice).collect();
    let (after, lens) = {
        let root = readable(&body.doc)?;
        let found = arrays(&root, &path)?;
        if path.legacy() {
            any_array(&found)?;
        }
        let mut at = Vec::new();
        let mut lens = Vec::with_capacity(found.len());
        for f in &found {
            match f {
                Some((off, v)) => {
                    at.push((
                        *off,
                        Edit::Splice {
                            at: v.len(),
                            take: 0,
                            put: &put,
                        },
                    ));
                    lens.push(Some(v.len() + put.len()));
                }
                None => lens.push(None),
            }
        }
        (edit(&root, &at)?, lens)
    };
    body.doc = after;
    lengths(&path, &lens, Pick::Last, out);
    Ok(())
}

/// `JSON.ARRINSERT key path index value [value ...]`.
fn arrinsert(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let (key, raw) = (args.get(1), args.get(2));
    if args.len() < 5 {
        return Err(args::wrong_arity("json.arrinsert"));
    }
    let path = Path::parse(raw)?;
    let want = args.int(3)?;
    let added = match values(args, 4) {
        Ok(added) => added,
        Err(e) => {
            unprefixed(&e, out);
            return Ok(());
        }
    };
    let body = match doc_mut(db, key, out)? {
        Doc::Wrong => return Ok(()),
        Doc::Gone => return Err(no_key()),
        Doc::Here(body) => body,
    };
    let put: Vec<&[u8]> = added.iter().map(Vec::as_slice).collect();
    let (after, lens) = {
        let root = readable(&body.doc)?;
        let found = arrays(&root, &path)?;
        if path.legacy() {
            any_array(&found)?;
        }
        let mut at = Vec::new();
        let mut lens = Vec::with_capacity(found.len());
        for f in &found {
            match f {
                Some((off, v)) => {
                    // An index the array does not reach is refused rather than
                    // clamped, which is what makes this different from every
                    // other index here. The end itself is allowed, so an insert
                    // at the length is an append.
                    let Some(i) = place(want, v.len()) else {
                        return Err(Error::new(Code::Invalid, "index out of bounds"));
                    };
                    at.push((
                        *off,
                        Edit::Splice {
                            at: i,
                            take: 0,
                            put: &put,
                        },
                    ));
                    lens.push(Some(v.len() + put.len()));
                }
                None => lens.push(None),
            }
        }
        (edit(&root, &at)?, lens)
    };
    body.doc = after;
    lengths(&path, &lens, Pick::First, out);
    Ok(())
}

/// Where an insert index lands in an array of `len`, or nothing if it does not
/// land in it at all.
///
/// Negative counts back from the end, so `-1` is before the last element and
/// `-len` is the front. The end is a place and one past it is not.
fn place(want: i64, len: usize) -> Option<usize> {
    let at = if want < 0 { len as i64 + want } else { want };
    if at < 0 || at > len as i64 {
        return None;
    }
    Some(at as usize)
}

/// `JSON.ARRTRIM key path start stop`.
fn arrtrim(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let (key, raw) = (args.get(1), args.get(2));
    if args.len() != 5 {
        return Err(args::wrong_arity("json.arrtrim"));
    }
    let path = Path::parse(raw)?;
    let (from, to) = (args.int(3)?, args.int(4)?);
    let body = match doc_mut(db, key, out)? {
        Doc::Wrong => return Ok(()),
        Doc::Gone => return Err(no_key()),
        Doc::Here(body) => body,
    };
    let (after, lens) = {
        let root = readable(&body.doc)?;
        let found = arrays(&root, &path)?;
        if path.legacy() {
            any_array(&found)?;
        }
        // The kept elements are the bytes already in the document, spliced back
        // over the whole array. Held out here because the edits below borrow
        // them and the borrow has to outlive the loop that builds them.
        let mut kept: Vec<Vec<&[u8]>> = Vec::with_capacity(found.len());
        for f in &found {
            let mut keep = Vec::new();
            if let Some((_, v)) = f {
                let (start, stop) = span(from, to, v.len());
                for i in start..stop {
                    if let Some(e) = v.at(i).and_then(|e| e.as_bytes()) {
                        keep.push(e);
                    }
                }
            }
            kept.push(keep);
        }
        let mut at = Vec::new();
        let mut lens = Vec::with_capacity(found.len());
        for (f, keep) in found.iter().zip(&kept) {
            match f {
                Some((off, v)) => {
                    at.push((
                        *off,
                        Edit::Splice {
                            at: 0,
                            take: v.len(),
                            put: keep,
                        },
                    ));
                    lens.push(Some(keep.len()));
                }
                None => lens.push(None),
            }
        }
        (edit(&root, &at)?, lens)
    };
    body.doc = after;
    lengths(&path, &lens, Pick::First, out);
    Ok(())
}

/// The half open run `JSON.ARRTRIM` keeps out of an array of `len`.
///
/// Both ends are inclusive on the wire and both count back from the end when
/// negative, and both clamp rather than refuse. A start past the end or a stop
/// before the start leaves nothing, which is an empty array and not an error.
fn span(from: i64, to: i64, len: usize) -> (usize, usize) {
    let n = len as i64;
    let start = if from < 0 {
        (n + from).max(0)
    } else {
        from.min(n)
    };
    let stop = if to < 0 {
        (n + to).max(0)
    } else {
        to.min(n - 1)
    };
    if start > stop || len == 0 {
        return (0, 0);
    }
    (start as usize, stop as usize + 1)
}

/// `JSON.ARRPOP key [path [index]]`.
fn arrpop(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let key = args.get(1);
    let raw = args.opt(2).unwrap_or(ROOT);
    if args.len() > 4 {
        return Err(args::wrong_arity("json.arrpop"));
    }
    let path = Path::parse(raw)?;
    let want = if args.len() == 4 { args.int(3)? } else { -1 };
    let body = match doc_mut(db, key, out)? {
        Doc::Wrong => return Ok(()),
        Doc::Gone => return Err(no_key()),
        Doc::Here(body) => body,
    };
    let (after, gone) = {
        let root = readable(&body.doc)?;
        let found = arrays(&root, &path)?;
        if path.legacy() {
            any_array(&found)?;
        }
        let mut at = Vec::new();
        // Two levels of nothing, and they are not the same nothing. The outer
        // one is a match this command did not touch at all because it was not
        // an array, which [`Pick`] skips over. The inner one is an array it did
        // look at and found nothing in, which is an answer and is the answer a
        // legacy path gets if that array came first.
        let mut gone: Vec<Option<Option<Vec<u8>>>> = Vec::with_capacity(found.len());
        for f in &found {
            match f {
                // An empty array pops nothing and is not an error, on either
                // syntax, which is the one case where a legacy path that named
                // a real array still answers nil.
                Some((_, v)) if v.is_empty() => gone.push(Some(None)),
                Some((off, v)) => {
                    let i = reach(want, v.len());
                    let mut text = Vec::new();
                    match v.at(i) {
                        Some(e) => e.write_json(&mut text)?,
                        None => return Err(missing()),
                    }
                    at.push((
                        *off,
                        Edit::Splice {
                            at: i,
                            take: 1,
                            put: &[],
                        },
                    ));
                    gone.push(Some(Some(text)));
                }
                None => gone.push(None),
            }
        }
        (edit(&root, &at)?, gone)
    };
    body.doc = after;
    if path.legacy() {
        match Pick::First.of(&gone) {
            Some(Some(text)) => out.bulk(text),
            _ => out.nil(),
        }
        return Ok(());
    }
    out.array(gone.len());
    for g in &gone {
        match g {
            Some(Some(text)) => out.bulk(text),
            _ => out.nil(),
        }
    }
    Ok(())
}

/// Which element of an array of `len` an index reaches, clamped to one that is
/// there.
///
/// Unlike the insert index this never refuses. A pop past the end takes the
/// last element and a pop before the front takes the first, and `len` is never
/// zero here because an empty array is handled before this is called.
fn reach(want: i64, len: usize) -> usize {
    let n = len as i64;
    let at = if want < 0 { n + want } else { want };
    at.clamp(0, n - 1) as usize
}

/// `JSON.ARRINDEX key path value [start [stop]]`.
///
/// The one array command whose errors do not match the others. A legacy path
/// that matched nothing is `Path does not exist` and one that matched something
/// that is not an array is a `WRONGTYPE`, where every other command here says
/// `Path does not exist or not an array` for both.
fn arrindex(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let (key, raw, text) = (args.get(1), args.get(2), args.get(3));
    if args.len() < 4 || args.len() > 6 {
        return Err(args::wrong_arity("json.arrindex"));
    }
    let path = Path::parse(raw)?;
    let looking = yo_doc::from_json(text)?;
    let from = if args.len() > 4 { args.int(4)? } else { 0 };
    // Zero is the end of the array rather than the front of it, so the default
    // searches everything and a stop that was left off means the same thing as
    // one that was given as zero.
    let to = if args.len() > 5 { args.int(5)? } else { 0 };
    let body = match doc(db, key, out)? {
        Doc::Wrong => return Ok(()),
        Doc::Gone => return Err(missing()),
        Doc::Here(body) => body,
    };
    let root = readable(&body.doc)?;
    let looking = readable(&looking)?;
    let mut hits = Vec::new();
    path.select(&root, &mut hits);
    if path.legacy() {
        let Some(v) = hits.first() else {
            return Err(missing());
        };
        if v.kind() != Kind::Array {
            return Err(Error::new(
                Code::WrongType,
                "wrong type of path value - expected array",
            ));
        }
        out.int(seek(v, &looking, from, to));
        return Ok(());
    }
    out.array(hits.len());
    for v in &hits {
        if v.kind() == Kind::Array {
            out.int(seek(v, &looking, from, to));
        } else {
            out.nil();
        }
    }
    Ok(())
}

/// Where `looking` first sits in `v` between `from` and `to`, or `-1`.
///
/// `to` is exclusive, which is not what the rest of Redis does with a stop and
/// is what RedisJSON does here, and zero means the end rather than the front.
///
/// `from` counts back from the end when it is negative and is then clamped to
/// the last element rather than to one past it, so a start of five into an array
/// of four still looks at the fourth. That reads like a mistake and it is what
/// RedisJSON does, checked on `[1,2,3,1]` where a start of four, five or minus
/// one all answer three. An empty array answers minus one whatever the start is,
/// which falls out of the stop being zero.
fn seek(v: &Value<'_>, looking: &Value<'_>, from: i64, to: i64) -> i64 {
    let n = v.len() as i64;
    let start = if from < 0 { n + from } else { from }.clamp(0, (n - 1).max(0));
    let stop = if to == 0 {
        n
    } else if to < 0 {
        (n + to).max(0)
    } else {
        to.min(n)
    };
    let mut i = start;
    while i < stop {
        if let Some(e) = v.at(i as usize)
            && same(&e, looking)
        {
            return i;
        }
        i += 1;
    }
    -1
}

/// Whether two values are the same value.
///
/// Structural rather than a comparison of the encoded bytes, because an object
/// inside a stored document may hold its keys as intern table ids where the one
/// parsed off the wire holds them as bytes, and those two encode differently
/// while meaning the same thing.
fn same(a: &Value<'_>, b: &Value<'_>) -> bool {
    if a.kind() != b.kind() {
        return false;
    }
    match a.kind() {
        Kind::Null => true,
        Kind::Bool => a.as_bool() == b.as_bool(),
        Kind::Int => a.as_int() == b.as_int(),
        Kind::Float => a.as_float() == b.as_float(),
        Kind::Text => a.text_bytes() == b.text_bytes(),
        Kind::Array => {
            a.len() == b.len()
                && (0..a.len()).all(|i| match (a.at(i), b.at(i)) {
                    (Some(x), Some(y)) => same(&x, &y),
                    _ => false,
                })
        }
        Kind::Object => {
            a.len() == b.len()
                && (0..a.len()).all(|i| {
                    let key = a.key_at(i);
                    match (key, key.and_then(|k| b.get(k)), a.at(i)) {
                        (Some(_), Some(y), Some(x)) => same(&x, &y),
                        _ => false,
                    }
                })
        }
    }
}

// ------------------------------------------------- the numbers and the strings

/// Which of `JSON.NUMINCRBY`, `JSON.NUMMULTBY` and `JSON.NUMPOWBY` is being
/// run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arith {
    Add,
    Mul,
    Pow,
}

/// A JSON number, which is an integer until something makes it a double.
///
/// This is the whole reason the number family is not one line. JSON has one
/// number type and every implementation that cares about round tripping keeps
/// two, so `7 + 2` has to answer `9` and `7 + 2.0` has to answer `9.0`, and the
/// document has to hold what the reply said.
#[derive(Debug, Clone, Copy)]
enum Num {
    Int(i64),
    Float(f64),
}

impl Num {
    /// The number a value holds, if it holds one.
    fn of(v: &Value<'_>) -> Option<Num> {
        match v.kind() {
            Kind::Int => v.as_int().map(Num::Int),
            Kind::Float => v.as_float().map(Num::Float),
            _ => None,
        }
    }

    /// The same number as a double, which is what mixed arithmetic works in.
    fn as_f64(self) -> f64 {
        match self {
            Num::Int(i) => i as f64,
            Num::Float(f) => f,
        }
    }

    /// The number as a document value.
    fn encode(self) -> Result<Vec<u8>> {
        let mut b = Builder::new();
        match self {
            Num::Int(i) => b.int(i)?,
            Num::Float(f) => b.float(f)?,
        }
        Ok(b.finish()?.to_vec())
    }
}

impl Arith {
    /// `a` combined with `b`, in integers when both of them are integers.
    ///
    /// Two integers are checked and overflow is refused rather than promoted to
    /// a double, so `JSON.NUMINCRBY` on the largest integer there is answers an
    /// error and leaves the document alone. A power with a negative exponent
    /// falls into the same error even though nothing overflowed, because the
    /// exponent has to be a count and there is no integer answer to two to the
    /// minus one. Both of those are RedisJSON's, read off 8.10.1.
    ///
    /// Anything with a double in it is done in doubles, and a result that is not
    /// finite is refused, which covers a product that overflowed and the square
    /// root of a negative number in one test.
    fn apply(self, a: Num, b: Num) -> Result<Num> {
        if let (Num::Int(x), Num::Int(y)) = (a, b) {
            let done = match self {
                Arith::Add => x.checked_add(y),
                Arith::Mul => x.checked_mul(y),
                Arith::Pow => u32::try_from(y).ok().and_then(|y| x.checked_pow(y)),
            };
            return done.map(Num::Int).ok_or_else(overflowed);
        }
        let (x, y) = (a.as_f64(), b.as_f64());
        let done = match self {
            Arith::Add => x + y,
            Arith::Mul => x * y,
            Arith::Pow => x.powf(y),
        };
        if !done.is_finite() {
            return Err(not_a_number());
        }
        Ok(Num::Float(done))
    }
}

/// `JSON.NUMINCRBY`, `JSON.NUMMULTBY` and `JSON.NUMPOWBY`, which are one
/// command with three operators.
///
/// The reply is a bulk string holding JSON text rather than a number, on both
/// syntaxes: a legacy path answers the new value as text and a JSONPath answers
/// a JSON array of the new values with a `null` for every match that was not a
/// number. A client that wants the number back has to parse it, which is odd
/// and is what every RedisJSON client already does.
fn arith(db: &mut Keyspace, args: Args<'_>, out: &mut Out, how: Arith) -> Result<()> {
    let (key, raw, operand) = (args.get(1), args.get(2), args.get(3));
    let path = Path::parse(raw)?;
    let body = match doc_mut(db, key, out)? {
        Doc::Wrong => return Ok(()),
        Doc::Gone => return Err(no_key()),
        Doc::Here(body) => body,
    };
    // The new values, encoded, held out here because the edits below borrow
    // them. One entry per match, `None` for a match that was not a number.
    let mut made: Vec<Option<Vec<u8>>> = Vec::new();
    let after = {
        let root = readable(&body.doc)?;
        let mut hits = Vec::new();
        path.select(&root, &mut hits);
        let was: Vec<Option<Num>> = hits.iter().map(Num::of).collect();
        // The operand is looked at only once a match turns out to be a number.
        // `JSON.NUMINCRBY key $.a_string "x"` answers `[null]` on the path and
        // never says anything about the `"x"`, which is RedisJSON's order and
        // matters because the two answers are nothing alike.
        let holder;
        let by = if was.iter().any(Option::is_some) {
            // Parsed as JSON rather than as a number, so ` 3 ` is a three and
            // `1 2` is a parse error rather than a one.
            holder = yo_doc::from_json(operand)?;
            match Num::of(&readable(&holder)?) {
                Some(by) => Some(by),
                None => {
                    // Valid JSON that is not a number, which is `true` or `"3"`
                    // or a whole object. Unprefixed, like the other two lines
                    // this file writes itself.
                    out.error(b"bad input number");
                    return Ok(());
                }
            }
        } else if path.legacy() {
            return Err(no_number());
        } else {
            None
        };
        for w in &was {
            match (w, by) {
                (Some(w), Some(by)) => made.push(Some(how.apply(*w, by)?.encode()?)),
                _ => made.push(None),
            }
        }
        let mut at = Vec::new();
        for (v, m) in hits.iter().zip(&made) {
            if let Some(bytes) = m {
                at.push((offset(&root, v)?, Edit::Set(bytes)));
            }
        }
        edit(&root, &at)?
    };
    body.doc = after;

    // The text is written off the encoded value rather than off the number, so
    // what the client reads back is the same bytes `JSON.GET` will answer with.
    let mut text = Vec::new();
    if path.legacy() {
        if let Some(bytes) = Pick::Last.of(&made) {
            readable(bytes)?.write_json(&mut text)?;
        }
        out.bulk(&text);
        return Ok(());
    }
    text.push(b'[');
    for (i, m) in made.iter().enumerate() {
        if i > 0 {
            text.push(b',');
        }
        match m {
            Some(bytes) => readable(bytes)?.write_json(&mut text)?,
            None => text.extend_from_slice(b"null"),
        }
    }
    text.push(b']');
    out.bulk(&text);
    Ok(())
}

/// `JSON.STRAPPEND key [path] value`.
///
/// The path is optional and sits in the middle, which no other command here
/// does, so the shape has to be read off the count: three arguments means the
/// value is the last one and the path is the root. Anything past the value is
/// ignored rather than refused, which is RedisJSON's and is why the arity is
/// open ended.
fn strappend(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let key = args.get(1);
    let (raw, text) = if args.len() == 3 {
        (ROOT, args.get(2))
    } else {
        (args.get(2), args.get(3))
    };
    let path = Path::parse(raw)?;
    let body = match doc_mut(db, key, out)? {
        Doc::Wrong => return Ok(()),
        Doc::Gone => return Err(no_key()),
        Doc::Here(body) => body,
    };
    let mut made: Vec<Option<Vec<u8>>> = Vec::new();
    let after = {
        let root = readable(&body.doc)?;
        let mut hits = Vec::new();
        path.select(&root, &mut hits);
        let was: Vec<Option<&[u8]>> = hits.iter().map(Value::text_bytes).collect();
        // The value is looked at only once a match turns out to be a string,
        // the same order the number family follows.
        let holder;
        let more = if was.iter().any(Option::is_some) {
            holder = yo_doc::from_json(text)?;
            // The value is JSON and has to be a JSON string, so a client
            // appends `"a"` and not `a`. A value that is a number is a
            // `WRONGTYPE` about a path value even though it was the value and
            // not the path that was wrong, which reads like the check was
            // written in the wrong place and is what the module answers.
            match readable(&holder)?.text_bytes() {
                Some(more) => Some(more),
                None => {
                    return Err(Error::new(
                        Code::WrongType,
                        "wrong type of path value - expected string",
                    ));
                }
            }
        } else if path.legacy() {
            return Err(not_a_string());
        } else {
            None
        };
        for w in &was {
            match (w, more) {
                (Some(was), Some(more)) => {
                    let mut joined = Vec::with_capacity(was.len() + more.len());
                    joined.extend_from_slice(was);
                    joined.extend_from_slice(more);
                    let mut b = Builder::new();
                    b.text_bytes(&joined)?;
                    made.push(Some(b.finish()?.to_vec()));
                }
                _ => made.push(None),
            }
        }
        let mut at = Vec::new();
        for (v, m) in hits.iter().zip(&made) {
            if let Some(bytes) = m {
                at.push((offset(&root, v)?, Edit::Set(bytes)));
            }
        }
        edit(&root, &at)?
    };
    body.doc = after;

    // The length is in bytes and not in characters, so appending one `é` to a
    // two byte string answers four.
    let lens: Vec<Option<usize>> = made
        .iter()
        .map(|m| {
            m.as_ref()
                .and_then(|b| Value::new(b))
                .and_then(|v| v.text_bytes())
                .map(<[u8]>::len)
        })
        .collect();
    lengths(&path, &lens, Pick::Last, out);
    Ok(())
}

// ------------------------------------------------------- the two odd commands

/// `JSON.RESP key [path]`, the document as RESP types rather than as JSON text.
///
/// An object is an array whose first element is the simple string `{` followed
/// by its members flattened into key, value, key, value. An array is an array
/// whose first element is the simple string `[`. An integer is an integer, a
/// double is a bulk string holding its JSON text, a string is a bulk string, a
/// boolean is the simple string `true` or `false` and a null is a nil. The
/// reason it exists is that a client can walk the answer without a JSON parser,
/// and the reason the two containers carry a marker element is that an empty
/// array and an empty object would otherwise both be an empty array.
fn resp(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let key = args.get(1);
    // Anything past the path is ignored, the same as `JSON.DEL`.
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
        let Some(v) = hits.first() else {
            return Err(missing());
        };
        return shape(v, out);
    }
    out.array(hits.len());
    for v in &hits {
        shape(v, out)?;
    }
    Ok(())
}

/// One value, as the RESP `JSON.RESP` answers with.
fn shape(v: &Value<'_>, out: &mut Out) -> Result<()> {
    match v.kind() {
        Kind::Null => out.nil(),
        Kind::Bool => out.simple(if v.as_bool() == Some(true) {
            b"true"
        } else {
            b"false"
        }),
        Kind::Int => out.int(v.as_int().ok_or_else(damaged)?),
        Kind::Float => {
            // A double goes out as its JSON text and not as a RESP double, so
            // the reply is the same on both protocols and a client that reads
            // it gets the same digits `JSON.GET` would have given it.
            let mut text = Vec::new();
            v.write_json(&mut text)?;
            out.bulk(&text);
        }
        Kind::Text => out.bulk(v.text_bytes().ok_or_else(damaged)?),
        Kind::Array => {
            out.array(v.len() + 1);
            out.simple(b"[");
            for e in v.iter() {
                shape(&e, out)?;
            }
        }
        Kind::Object => {
            out.array(v.len() * 2 + 1);
            out.simple(b"{");
            for i in 0..v.len() {
                out.bulk(v.key_at(i).ok_or_else(damaged)?);
                shape(&v.at(i).ok_or_else(damaged)?, out)?;
            }
        }
    }
    Ok(())
}

/// `JSON.DEBUG MEMORY key [path]` and `JSON.DEBUG HELP`.
///
/// The byte counts are this encoding's and not RedisJSON's, which is D-42: the
/// two servers do not store a document the same way, so there is no arrangement
/// under which the same document would answer the same number twice.
fn debug(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    if args::is(args.get(1), b"help") {
        // Two lines, spaced the way the module spaces them.
        out.array(2);
        out.bulk(b"MEMORY <key> [path] - reports memory usage");
        out.bulk(b"HELP                - this message");
        return Ok(());
    }
    if !args::is(args.get(1), b"memory") {
        return Err(Error::new(
            Code::Invalid,
            "unknown subcommand - try `JSON.DEBUG HELP`",
        ));
    }
    if args.len() < 3 {
        return Err(args::wrong_arity("json.debug"));
    }
    let key = args.get(2);
    let raw = args.opt(3).unwrap_or(ROOT);
    let path = Path::parse(raw)?;
    let body = match doc(db, key, out)? {
        Doc::Wrong => return Ok(()),
        Doc::Gone => {
            // A key that is not there is a zero on a legacy path and an empty
            // set on a JSONPath, which is the one reader here that does not
            // answer a nil for it.
            if path.legacy() {
                out.int(0);
            } else {
                out.array(0);
            }
            return Ok(());
        }
        Doc::Here(body) => body,
    };
    let root = readable(&body.doc)?;
    let mut hits = Vec::new();
    path.select(&root, &mut hits);
    if path.legacy() {
        let Some(v) = hits.first() else {
            return Err(missing());
        };
        out.int(i64::try_from(v.encoded_len().ok_or_else(damaged)?).unwrap_or(i64::MAX));
        return Ok(());
    }
    out.array(hits.len());
    for v in &hits {
        out.int(i64::try_from(v.encoded_len().ok_or_else(damaged)?).unwrap_or(i64::MAX));
    }
    Ok(())
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
    Value::new(doc).ok_or_else(damaged)
}

/// What a walk that ran into bytes it could not read gets.
///
/// Only reachable if a stored document is damaged, since every document here is
/// built by this server. It is an error rather than a panic because a client
/// asking for a key it has no way of knowing is broken should get a line back
/// and not a dropped connection.
fn damaged() -> Error {
    Error::new(Code::Invalid, "the document stored here is damaged")
}

/// Send an error with no `ERR` in front of it, which is how RedisJSON answers a
/// value that is not JSON on `JSON.SET`, `JSON.ARRAPPEND`, `JSON.ARRINSERT`,
/// `JSON.MERGE` and `JSON.MSET`.
///
/// It does prefix the same error on `JSON.ARRINDEX`, `JSON.STRAPPEND` and the
/// number family, so this is not a rule about parse errors, it is a rule about
/// which command the client sent. The wording is ours either way and is D-37.
fn unprefixed(e: &Error, out: &mut Out) {
    out.error_line(b"", e.message().as_bytes());
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

/// What a legacy number command gets when no match was a number.
///
/// The wording is RedisJSON's, typo and all, and it is one line for a path that
/// matched nothing and for one that matched a string.
fn no_number() -> Error {
    Error::new(
        Code::Invalid,
        "Path does not exist or does not contains a number",
    )
}

/// What a legacy `JSON.STRAPPEND` gets when no match was a string.
fn not_a_string() -> Error {
    Error::new(Code::Invalid, "Path does not exist or not a string")
}

/// What arithmetic that ran off the end of an integer gets.
fn overflowed() -> Error {
    Error::new(Code::Invalid, "numeric overflow")
}

/// What arithmetic that left the real numbers gets.
fn not_a_number() -> Error {
    Error::new(Code::Invalid, "result is not a number")
}

/// What the commands that need a document get when there is not one.
fn no_key() -> Error {
    Error::new(
        Code::Invalid,
        "could not perform this operation on a key that doesn't exist",
    )
}
