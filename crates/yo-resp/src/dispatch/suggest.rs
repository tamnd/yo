//! `FT.SUG*`, the suggestion dictionaries the search module puts in the
//! keyspace.
//!
//! The trie is [`yo_search::suggest::Suggestions`] and this is the wire in
//! front of it, the same split the sketch families have. Four commands: one to
//! put a suggestion in, one to take it out, one to count them and one to ask
//! for the best few under a prefix.
//!
//! These are the only search commands that name a key. Everything else in the
//! `FT.*` family works on the index registry, which no client can see with
//! `TYPE`, so those arms are handed the registry and these are handed a
//! database. The type a client sees here is `trietype0`, which is the module's
//! own name for it, and a `DEL` removes the dictionary the same as any key.
//!
//! # Errors
//!
//! Five sentences and none of them are written the same way. Two carry the
//! `SEARCH_PARSE_ARGS` code the newer search errors use, one carries `ERR`,
//! and two carry nothing at all. That is what the module sends and clients
//! match on the whole line, so they are copied rather than tidied into a
//! family.
//!
//! # The order the checks happen in
//!
//! `FT.SUGADD` reads its keywords before it converts the score, so an add with
//! both a bad score and an unknown word complains about the word. It also takes
//! between four and seven words and nothing longer, so a second `PAYLOAD` is a
//! wrong arity rather than an unknown argument. `FT.SUGGET` reads its options
//! left to right and lets them repeat, so a good `MAX` followed by a bad one is
//! an error and the last one given is the one that counts.
//!
//! # Where the two protocols disagree
//!
//! One place. A score in a `WITHSCORES` reply is a double on RESP3 and a bulk
//! string on RESP2. The reply itself is a flat array on both, which is not what
//! the two list replies in `super::search` do, and a missing payload is a null
//! in the middle of that array rather than an empty string.

use yo_common::num::parse_f64;
use yo_common::{Code, Error, Result};
use yo_kv::{Db, Foreign, Keyspace};
use yo_search::suggest::Suggestions;

use super::args::{self, Args};
use super::table::Spec;
use crate::reply::Out;

/// How many suggestions `FT.SUGGET` answers with when it is not told.
const DEFAULT_MAX: u64 = 5;
/// The largest `MAX` the module takes, which is what a `u32` holds. A real
/// server accepts every number up to this one and then crashes trying to make
/// room for the answer, so the number is copied and the crash is not.
const MAX_MAX: u64 = u32::MAX as u64;
/// The longest an `FT.SUGADD` can be: a key, a term, a score, `INCR` and a
/// `PAYLOAD` with its value.
const LONGEST_ADD: usize = 7;

/// A score that is not a number, or is one the conversion refused.
const BAD_SCORE: &str = "invalid score";
/// A word after the score that is not `INCR` or `PAYLOAD`. The module writes
/// this one with no code in front of it and with the word in backticks.
const UNKNOWN_ADD: &[u8] = b"Unknown argument `";
/// A `PAYLOAD` with nothing behind it.
const NO_PAYLOAD: &[u8] = b"Invalid payload: Expected an argument, but none provided";
/// A word in `FT.SUGGET` that is not one of the four options.
const UNKNOWN_GET: &[u8] = b"SEARCH_PARSE_ARGS Unrecognized argument: ";
/// A `MAX` outside one to four billion.
const MAX_RANGE: &[u8] = b"SEARCH_PARSE_ARGS MAX: Value is outside acceptable bounds";
/// A `MAX` the conversion could not read at all, or read as less than one
/// without it being a whole number to begin with.
const MAX_KIND: &[u8] = b"SEARCH_PARSE_ARGS MAX: Could not convert argument to expected type";
/// A `MAX` with nothing behind it.
const MAX_MISSING: &[u8] = b"SEARCH_PARSE_ARGS MAX: Expected an argument, but none provided";

/// What a key holding anything else gets.
const WRONG_KIND: &str = "Operation against a key holding the wrong kind of value";

/// A dictionary under a key.
#[derive(Debug)]
pub(super) struct SugBody {
    /// The trie. Everything the four commands do is a call on this.
    s: Suggestions,
}

impl Foreign for SugBody {
    fn type_name(&self) -> &'static str {
        // The module's own name for the type, trailing zero and all. It is the
        // version number of the trie encoding and it has never moved.
        "trietype0"
    }

    fn encoding(&self) -> &'static str {
        "raw"
    }

    fn memory_bytes(&self) -> usize {
        self.s.memory_bytes()
    }

    fn is_empty(&self) -> bool {
        // Unlike the sketches, this one really does go when it empties: taking
        // the last suggestion out with `FT.SUGDEL` leaves no key behind.
        self.s.is_empty()
    }
}

pub(super) fn execute(db: &Db, spec: &Spec, args: Args<'_>, out: &mut Out) -> Result<()> {
    // Every one of them names one dictionary and names it first, so the stripe
    // is found once and held for the whole command.
    let mut held = db.hold(args.get(1));
    let db = &mut *held;
    match spec.name {
        "FT.SUGADD" => add(db, args, out),
        "FT.SUGGET" => get(db, args, out),
        "FT.SUGDEL" => del(db, args, out),
        "FT.SUGLEN" => len(db, args, out),
        other => unreachable!("{other} is not a suggestion command"),
    }
}

/// `FT.SUGADD key string score [INCR] [PAYLOAD payload]`.
///
/// The reply is how many suggestions the dictionary holds afterwards, which is
/// neither what was added nor whether anything changed. An empty string is
/// taken and not stored, so it answers the length it already had, and it still
/// makes the key: a dictionary holding nothing at all is a real key that
/// `EXISTS` sees and `TYPE` names.
fn add(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    if args.len() > LONGEST_ADD {
        return Err(args::wrong_arity("FT.SUGADD"));
    }
    let mut incr = false;
    let mut payload = None;
    let mut at = 4;
    while at < args.len() {
        let word = args.get(at);
        if word.eq_ignore_ascii_case(b"incr") {
            incr = true;
            at += 1;
        } else if word.eq_ignore_ascii_case(b"payload") {
            let Some(bytes) = args.opt(at + 1) else {
                out.error(NO_PAYLOAD);
                return Ok(());
            };
            // An empty payload is the same as not having asked for one, so it
            // leaves whatever the term already had rather than clearing it.
            if !bytes.is_empty() {
                payload = Some(bytes);
            }
            at += 2;
        } else {
            // The word as the client typed it, in the backticks the module put
            // round it.
            let mut line = UNKNOWN_ADD.to_vec();
            line.extend_from_slice(word);
            line.push(b'`');
            out.error(&line);
            return Ok(());
        }
    }
    // After the keywords, which is the order the module reads them in and is
    // why an add with two things wrong complains about the second one.
    let Some(score) = double(args.get(3)) else {
        return Err(Error::new(Code::Invalid, BAD_SCORE));
    };
    let key = args.get(1);
    // Asked and answered before the dictionary is made, so a key holding
    // something else is refused rather than overwritten.
    if mine(db.foreign(key)?)?.is_none() {
        db.put_foreign(
            key,
            Box::new(SugBody {
                s: Suggestions::new(),
            }),
        );
    }
    let body = mine_mut(db.foreign_mut(key)?.expect("there or just made"))?;
    body.s.add(args.get(2), score, incr, payload);
    out.uint(body.s.len() as u64);
    // No reaping here even though the dictionary can be empty, because an add
    // of an empty term really does leave a key behind holding nothing. Only
    // `FT.SUGDEL` clears one away.
    Ok(())
}

/// `FT.SUGGET key prefix [FUZZY] [MAX n] [WITHSCORES] [WITHPAYLOADS]`.
///
/// A missing key answers an empty array rather than complaining, so a client
/// asking for completions on a dictionary nobody has filled in yet gets
/// nothing and carries on.
fn get(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let mut fuzzy = false;
    let mut scores = false;
    let mut payloads = false;
    let mut max = DEFAULT_MAX;
    let mut at = 3;
    while at < args.len() {
        let word = args.get(at);
        at += 1;
        if word.eq_ignore_ascii_case(b"fuzzy") {
            fuzzy = true;
        } else if word.eq_ignore_ascii_case(b"withscores") {
            scores = true;
        } else if word.eq_ignore_ascii_case(b"withpayloads") {
            payloads = true;
        } else if word.eq_ignore_ascii_case(b"max") {
            let Some(count) = args.opt(at) else {
                out.error(MAX_MISSING);
                return Ok(());
            };
            at += 1;
            match whole(count) {
                Ok(count) => max = count,
                Err(line) => {
                    out.error(line);
                    return Ok(());
                }
            }
        } else {
            let mut line = UNKNOWN_GET.to_vec();
            line.extend_from_slice(word);
            out.error(&line);
            return Ok(());
        }
    }
    let Some(body) = mine(db.foreign(args.get(1))?)? else {
        out.array(0);
        return Ok(());
    };
    // A flat array on both protocols, with the extras interleaved rather than
    // grouped, which is what a client reading three at a time expects.
    let hits = body.s.best(args.get(2), fuzzy, max as usize);
    let each = 1 + usize::from(scores) + usize::from(payloads);
    out.array(hits.len() * each);
    for hit in hits {
        out.bulk(hit.term);
        if scores {
            out.double(hit.score);
        }
        if payloads {
            match hit.payload {
                Some(bytes) => out.bulk(bytes),
                None => out.nil(),
            }
        }
    }
    Ok(())
}

/// `FT.SUGDEL key string`, which answers whether the term was there.
///
/// The spelling has to match byte for byte even though a lookup does not care
/// about case, so deleting `ABC` leaves `abc` where it is.
fn del(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let key = args.get(1);
    let gone = match mine_mut_opt(db.foreign_mut(key)?)? {
        Some(body) => body.s.remove(args.get(2)),
        None => false,
    };
    // An integer on both protocols and not a boolean, which is what the module
    // sends even on RESP3 where it had the choice.
    out.uint(u64::from(gone));
    // The last suggestion out takes the key with it.
    db.reap_foreign(key);
    Ok(())
}

/// `FT.SUGLEN key`, which is zero for a key that is not there.
fn len(db: &mut Keyspace, args: Args<'_>, out: &mut Out) -> Result<()> {
    let held = mine(db.foreign(args.get(1))?)?;
    out.uint(held.map_or(0, |body| body.s.len() as u64));
    Ok(())
}

/// A double, with the two ranges the conversion refuses.
///
/// A number too large or too small for a double is an error rather than an
/// infinity or a zero, because the module looks at `errno` after the
/// conversion, so `1e400` and `1e-400` are both refused. A client that spells
/// out `inf` is not asking for a range it missed and gets one.
///
/// Both a score and a `MAX` are read through here, which is why `0x10` is
/// sixteen in either of them.
fn double(arg: &[u8]) -> Option<f64> {
    let n = parse_f64(arg)?;
    let text = core::str::from_utf8(arg).ok()?;
    let word = text.trim_start_matches(['+', '-']);
    if n.is_infinite()
        && !word.eq_ignore_ascii_case("inf")
        && !word.eq_ignore_ascii_case("infinity")
    {
        return None;
    }
    if n == 0.0 && text.bytes().any(|b| (b'1'..=b'9').contains(&b)) {
        return None;
    }
    Some(n)
}

/// A `MAX`, or the sentence the module answers a bad one with.
///
/// It gets two goes at the number, which is the only reason a count the module
/// does not like is sometimes one sentence and sometimes the other. First it is
/// read as a whole number the strict way Redis reads one, where a leading zero,
/// a plus sign or a point all fail; a number that reads is range checked, so
/// `0` and `-1` are out of bounds. Anything that did not read is then taken as
/// a double and cut towards zero, and there a value under one is a number it
/// could not convert rather than one out of range. That is why `0` is out of
/// bounds and `0.0`, `00` and `-0` are not, which is a difference nobody would
/// invent and every one of them was read off a real server.
fn whole(arg: &[u8]) -> core::result::Result<u64, &'static [u8]> {
    if let Some(n) = yo_common::num::parse_i64(arg) {
        return u64::try_from(n)
            .ok()
            .filter(|n| (1..=MAX_MAX).contains(n))
            .ok_or(MAX_RANGE);
    }
    let n = double(arg).ok_or(MAX_KIND)?.trunc();
    if n < 1.0 {
        return Err(MAX_KIND);
    }
    #[allow(clippy::cast_precision_loss)]
    if n > MAX_MAX as f64 {
        return Err(MAX_RANGE);
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Ok(n as u64)
}

/// The dictionary a body holds, or WRONGTYPE if it is another module's.
fn mine(body: Option<&dyn Foreign>) -> Result<Option<&SugBody>> {
    match body {
        Some(body) => match body.downcast_ref::<SugBody>() {
            Some(body) => Ok(Some(body)),
            None => Err(Error::new(Code::WrongType, WRONG_KIND)),
        },
        None => Ok(None),
    }
}

/// The same, with a mutable borrow and a key that has to be there.
fn mine_mut(body: &mut dyn Foreign) -> Result<&mut SugBody> {
    body.downcast_mut::<SugBody>()
        .ok_or_else(|| Error::new(Code::WrongType, WRONG_KIND))
}

/// The same again, where a missing key is not an error.
fn mine_mut_opt(body: Option<&mut dyn Foreign>) -> Result<Option<&mut SugBody>> {
    match body {
        Some(body) => mine_mut(body).map(Some),
        None => Ok(None),
    }
}
