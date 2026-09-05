//! `WITHCURSOR`, and the `FT.CURSOR` command that reads the rest of a reply.
//!
//! A cursor is a reply that came back in pieces. The command that asked for one
//! answers a two element array of the first piece and a number, and
//! `FT.CURSOR READ` hands back the next piece under that number until the
//! number comes back nought. Both `FT.SEARCH` and `FT.AGGREGATE` take
//! `WITHCURSOR`, and a chunk is written in the protocol of the connection that
//! read it rather than the one that made it, so a RESP2 client can read a
//! cursor a RESP3 client opened and gets an array where the other got a map.
//!
//! A read has to name the index the cursor was made on and a delete does not,
//! which is measured on a real server rather than reasoned about: reading a
//! cursor through the name of another index says the cursor is not there, and
//! deleting it through the name of another index takes it away and says `OK`.
//! Both check that the name is an index at all first.
//!
//! # The rows are made once and kept
//!
//! A real server's cursor is a paused pipeline. The chunk a read answers is
//! pulled out of the index at the moment of the read, so a document rewritten
//! between two reads can be missed altogether or answered twice. This one runs
//! the whole query when the cursor is made and holds on to the rows it made, so
//! a chunk cannot change under the client's feet. That is divergence D-74, and
//! what it costs is the memory of one whole answer per open cursor, which the
//! limit of a hundred and twenty eight cursors per index bounds.
//!
//! # The count at the front is worked out per chunk
//!
//! The number in front of the rows means the same thing it means without a
//! cursor: not how many rows answered but how far the reply had got when the
//! number went on the wire. A pipeline that had to see the whole answer before
//! any of it could be written reports the real total on its first chunk and
//! nought on every chunk after it, since by then the number has already gone. A
//! pipeline that did not reports what the chunk itself reached, and that moves
//! with the protocol and with whether anything read a field, exactly as it does
//! without a cursor. [`Cursor::counted`] has the whole rule and where each line
//! of it was measured.
//!
//! One case parts company with the reply that has no cursor in it: a loader
//! starting at the top of the answer reports the real total in one go and
//! reports the chunk under a cursor, because a chunk is bounded and so there is
//! no point at which the whole answer had been read.

use std::collections::HashMap;

use yo_common::{Result, parse_i64};
use yo_search::expr::Value;

use super::super::Server;
use super::super::args::{self, Args};
use super::aggregate::writes;
use super::{Built, Fail, MISSING, Pairs, Rolled, Row, Shows, found, rolls};
use crate::reply::Out;

/// The most cursors one index may hold at a time, which `FT.INFO` reports as
/// the capacity beside how many are open.
pub(super) const LIMIT: usize = 128;

/// How long a cursor nobody reads from is kept, in milliseconds.
///
/// `CURSOR_MAX_IDLE` on a real server, and five minutes there and here.
const IDLE: u64 = 300_000;

/// How many rows a chunk holds when the client did not say.
///
/// Measured rather than read off the documentation: an index of fifteen hundred
/// documents answers a thousand rows and then five hundred.
const CHUNK: usize = 1000;

const OVER: &str = "SEARCH_LIMIT_OVER INDEX_CURSOR_LIMIT of 128 has been reached for an index";
/// What a read of an id that is not there says. It has no code word in front of
/// it, which is a real server's line and not a slip here.
const NOT_FOUND: &str = "Cursor not found, id: ";
const BAD_ID: &str = "Bad cursor ID";
const NO_CURSOR: &str = "Cursor does not exist";
const BAD_COUNT: &str = "Bad value for COUNT: `";
const QUOTE: &str = "`";

/// The most a `COUNT` or a `MAXIDLE` may be, which is what a thirty two bit
/// unsigned number holds. A real server takes `COUNT 4294967295` and refuses
/// the number after it as outside acceptable bounds.
pub(super) const MOST: i64 = u32::MAX as i64;

/// What a `WITHCURSOR` asked for.
#[derive(Clone, Copy)]
pub(super) struct Asks {
    /// How many rows a chunk holds.
    pub(super) count: usize,
    /// How long the cursor is kept between reads, in milliseconds.
    pub(super) idle: u64,
}

impl Default for Asks {
    fn default() -> Asks {
        Asks {
            count: CHUNK,
            idle: IDLE,
        }
    }
}

/// The rows a cursor is handing out, in the shape whoever writes them takes.
///
/// Three shapes because there are three writers: a search row is a key with the
/// fields that were read off it, an aggregation that ran no pipeline step is a
/// key with the properties that were read off it, and a pipeline row is a list
/// of values under a list of names with the document it came from, if it still
/// has one, beside it.
///
/// Everything in here is owned. A reply that goes out in one piece can point at
/// the documents it read, because they are still alive when it is written. A
/// cursor is written long after the command that made it returned.
pub(super) enum Made {
    Found(Vec<(Row, Option<Pairs>)>),
    Rolled(Vec<(Row, Pairs)>),
    Piped {
        names: Vec<Box<[u8]>>,
        rows: Vec<(Option<Row>, Vec<Value>)>,
        sorted: Option<usize>,
        warning: Option<Vec<u8>>,
    },
}

impl Made {
    /// How many rows there are in all.
    fn len(&self) -> usize {
        match self {
            Made::Found(rows) => rows.len(),
            Made::Rolled(rows) => rows.len(),
            Made::Piped { rows, .. } => rows.len(),
        }
    }
}

/// A whole answer, and everything the chunks of it need to know about it.
pub(super) struct Kept {
    pub(super) made: Made,
    /// The walk the query made, one entry per document, true where the document
    /// came out the far end of everything that throws rows away and false where
    /// a `FILTER` took it.
    ///
    /// This is what the number at the front of a chunk is worked out from, so it
    /// is only filled in when that number is not simply the total. The window is
    /// not in here: a document the window stepped over is still a document the
    /// walk reached, and the chunks apply the offset themselves.
    pub(super) walk: Vec<bool>,
    pub(super) shows: Shows,
    /// The number of rows that answered, which is what the first chunk reports
    /// when the whole answer had to exist before any of it could be written.
    pub(super) total: usize,
    /// Whether the count at the front is the real total rather than how far the
    /// chunk reached.
    pub(super) whole: bool,
    /// Whether anything on a row is read off a key, which is what makes the
    /// pipeline pull documents in fills rather than one at a time.
    pub(super) loader: bool,
    /// Where the window starts, which the walk has to step over before it
    /// reaches the first row anybody sees.
    pub(super) offset: usize,
    /// How wide the window is, or every row when nobody said. A window narrower
    /// than a chunk bounds how many documents one fill pulls.
    pub(super) window: usize,
}

/// One open cursor.
struct Cursor {
    kept: Kept,
    /// Which row the next chunk starts at.
    at: usize,
    /// How many rows a chunk holds, which a `COUNT` on a read replaces.
    count: usize,
    idle: u64,
    /// When it was last read, on the server clock.
    touched: u64,
    /// Whether a chunk has gone out yet.
    first: bool,
    /// Documents pulled and not yet used, which is the loader's buffer. It
    /// carries across chunks, because what a cursor holds is a pipeline that
    /// stopped in the middle rather than one that started again.
    buffer: usize,
    /// How far along the walk the pipeline has got.
    pulled: usize,
    /// How much of the window offset the walk has still to step over.
    skip: usize,
    /// How many rows have gone out in all, which the window bounds.
    given: usize,
}

impl Cursor {
    /// The number at the front of the chunk about to be written.
    ///
    /// It is not how many rows the chunk holds. It is how far the walk had got
    /// when the number went on the wire, less whatever a `FILTER` had thrown
    /// away by then, and that is worked out here by walking the chunk the way
    /// the pipeline walks it.
    ///
    /// Three things decide it, and all three were measured rather than reasoned
    /// about:
    ///
    /// * A settled answer reports the real total on its first chunk and nought
    ///   on every one after it, because by then the number has already gone. A
    ///   search is always settled, and so is a pipeline that grouped or sorted,
    ///   one asked for no rows at all, and one whose scorer had to see every
    ///   score before it could write any of them.
    /// * A pipeline that reads fields pulls documents a fill at a time rather
    ///   than one at a time, so a chunk of two over an index of seven reports
    ///   two even where it hands back one row. One that reads nothing pulls one
    ///   at a time. [`Cursor::fill`] has how wide a fill is.
    /// * RESP2 writes the number as soon as the first row of the chunk exists
    ///   and RESP3 writes it once the whole chunk does, so the same query
    ///   answers a different number on the two protocols.
    fn counted(&mut self, deep: bool) -> usize {
        if self.kept.whole {
            return match self.first {
                true => self.kept.total,
                false => 0,
            };
        }
        let mut walked = 0;
        let mut dropped = 0;
        let mut delivered = 0;
        let mut header = None;
        while delivered < self.count && self.given < self.kept.window {
            if self.buffer == 0 {
                if self.pulled >= self.kept.walk.len() {
                    break;
                }
                let fill = self.fill(delivered);
                self.buffer = fill;
                walked += fill;
            }
            self.buffer -= 1;
            let alive = self.kept.walk[self.pulled];
            self.pulled += 1;
            // A document a `FILTER` threw away comes off the number. One the
            // window stepped over does not, because the walk still went
            // through it.
            if !alive {
                dropped += 1;
                continue;
            }
            if self.skip > 0 {
                self.skip -= 1;
                continue;
            }
            delivered += 1;
            self.given += 1;
            if !deep && header.is_none() {
                header = Some(walked.saturating_sub(dropped));
            }
        }
        header.unwrap_or(walked.saturating_sub(dropped))
    }

    /// How many documents the pipeline pulls in one go.
    ///
    /// A pipeline with nothing reading a key is a stream and pulls one document
    /// at a time. One that reads keys reads them in bulk, and what it asks for
    /// is the rest of the chunk plus whatever of the window offset it still has
    /// to step over, which is why `LIMIT 2 2 WITHCURSOR COUNT 3` reports four:
    /// two stepped over and two asked for.
    ///
    /// A window with an offset on it holds the pull back to its width as well.
    /// A window without one does not, because `LIMIT 0 n` is a cap on what gets
    /// written rather than a step of the pipeline, so `LIMIT 0 1 WITHCURSOR
    /// COUNT 5` walks five documents to write one and says five.
    fn fill(&self, delivered: usize) -> usize {
        let left = self.kept.walk.len().saturating_sub(self.pulled);
        if !self.kept.loader {
            return 1;
        }
        let mut want = self.count.saturating_sub(delivered);
        if self.kept.offset > 0 {
            want = want.min(self.kept.window);
        }
        self.skip.saturating_add(want).min(left).max(1)
    }

    /// Writes the next chunk of rows, steps over them, and answers whether there
    /// is another chunk to come.
    ///
    /// A chunk that filled exactly keeps the cursor alive and a short one closes
    /// it, which is measured: seven rows at a chunk of seven answer a full chunk
    /// and then an empty one, and the same seven rows at a chunk of eight answer
    /// once and close.
    fn chunk(&mut self, out: &mut Out) -> bool {
        let from = self.at;
        let to = (from + self.count).min(self.kept.made.len());
        let count = self.counted(out.proto().is_resp3());
        let shows = self.kept.shows;
        let window = from..to;
        match &self.kept.made {
            Made::Found(rows) => {
                let built: Vec<Built<'_>> = rows[window]
                    .iter()
                    .map(|(row, fields)| (row, fields.as_ref().map(|fields| pairs(fields))))
                    .collect();
                found(count, &built, shows, out);
            }
            Made::Rolled(rows) => {
                let built: Vec<Rolled<'_>> = rows[window]
                    .iter()
                    .map(|(row, props)| (row, pairs(props)))
                    .collect();
                rolls(count, &built, shows, out);
            }
            Made::Piped {
                names,
                rows,
                sorted,
                warning,
            } => {
                let shown: Vec<(Option<&Row>, &Vec<Value>)> = rows[window]
                    .iter()
                    .map(|(row, values)| (row.as_ref(), values))
                    .collect();
                writes(
                    count,
                    names,
                    &shown,
                    *sorted,
                    shows,
                    warning.as_deref(),
                    out,
                );
            }
        }
        self.at = to;
        self.first = false;
        to - from == self.count
    }
}

/// Borrows a row's pairs, which is the shape every writer here takes.
fn pairs(held: &Pairs) -> Vec<(&[u8], &[u8])> {
    held.iter()
        .map(|(name, value)| (&**name, &**value))
        .collect()
}

/// Every cursor on the server, by the number the client reads it under.
///
/// One table for the server and not one per connection, which is measured: a
/// cursor one connection opened is read by another, and a cursor outlives the
/// connection that made it.
#[derive(Default)]
pub(in crate::dispatch) struct Cursors {
    held: HashMap<u64, Held>,
    /// Where the next number comes from.
    seed: u64,
}

/// One cursor and the index it was made on.
struct Held {
    index: Box<[u8]>,
    cursor: Cursor,
}

impl Cursors {
    /// Takes away every cursor nobody has read from for longer than it asked to
    /// be kept.
    fn sweep(&mut self, now: u64) {
        self.held
            .retain(|_, held| now.saturating_sub(held.cursor.touched) <= held.cursor.idle);
    }

    /// How many are open on one index.
    fn on(&self, index: &[u8]) -> usize {
        self.held.iter().filter(|(_, h)| *h.index == *index).count()
    }

    /// How many are open in all, which `FT.INFO` reports twice: once as the
    /// number that exist and once as the number nobody is reading, and here
    /// those are the same number because a read runs to the end before the next
    /// command starts.
    pub(super) fn total(&self) -> usize {
        self.held.len()
    }

    /// A number that is not in use, which looks like the thirty two bit numbers
    /// a real server hands out.
    fn mint(&mut self, now: u64) -> u64 {
        if self.seed == 0 {
            self.seed = now | 1;
        }
        loop {
            self.seed ^= self.seed << 13;
            self.seed ^= self.seed >> 7;
            self.seed ^= self.seed << 17;
            let id = self.seed & 0xffff_ffff;
            if id != 0 && !self.held.contains_key(&id) {
                return id;
            }
        }
    }
}

/// What `FT.INFO` reports: how many cursors are open on the whole server and how
/// many of those are on this index. The ones that have been left alone too long
/// are dropped first, so the numbers match what a read would find.
pub(super) fn stats(server: &Server, index: &[u8]) -> (u64, u64) {
    let now = server.clock.now_ms();
    let mut cursors = server.cursors.lock();
    cursors.sweep(now);
    (cursors.total() as u64, cursors.on(index) as u64)
}

/// Whether one more cursor may be opened on an index.
///
/// Asked before the reply is written, because a real server reserves the cursor
/// before it runs the pipeline: a `WITHCURSOR COUNT 1000` over seven rows, which
/// would hand everything back and close, is refused all the same when the index
/// is already holding its hundred and twenty eight.
pub(super) fn room<'a>(server: &Server, index: &[u8]) -> core::result::Result<(), Fail<'a>> {
    match server.cursors.lock().on(index) < LIMIT {
        true => Ok(()),
        false => Err(Fail::plain(OVER)),
    }
}

/// Answers the first chunk of a cursor and keeps the rest of it.
pub(super) fn open(server: &Server, index: &[u8], kept: Kept, asks: Asks, out: &mut Out) {
    let now = server.clock.now_ms();
    let mut cursor = Cursor {
        skip: kept.offset,
        kept,
        at: 0,
        count: asks.count,
        idle: asks.idle,
        touched: now,
        first: true,
        buffer: 0,
        pulled: 0,
        given: 0,
    };
    out.array(2);
    // The two element array goes out whatever happens next, because a cursor
    // that hands everything over at once still answers in the shape a cursor
    // answers in, with a nought where the number would be.
    if !cursor.chunk(out) {
        out.int(0);
        return;
    }
    let mut cursors = server.cursors.lock();
    cursors.sweep(now);
    let id = cursors.mint(now);
    cursors.held.insert(
        id,
        Held {
            index: index.into(),
            cursor,
        },
    );
    out.int(id as i64);
}

/// `FT.CURSOR READ|DEL|GC index id`.
///
/// There is no `HELP` here, which is worth saying because nearly every other
/// container command in this build has one: `FT.CURSOR HELP` on a real server
/// answers that `HELP` is not a subcommand, and the line it answers with names
/// `FT.CURSOR HELP` as the thing to try.
pub(in crate::dispatch) fn execute(server: &Server, args: Args<'_>, out: &mut Out) -> Result<()> {
    let sub = args.get(1);
    if args::is(sub, b"READ") {
        return read(server, args, out);
    }
    if args::is(sub, b"DEL") {
        return del(server, args, out);
    }
    if args::is(sub, b"GC") {
        return gc(server, args, out);
    }
    Err(args::unknown_subcommand(sub, "FT.CURSOR"))
}

/// `FT.CURSOR READ index id [COUNT n]`.
///
/// A `COUNT` here replaces the chunk size the cursor was made with rather than
/// applying to this read alone, and a `COUNT 0` or a `COUNT` with nothing after
/// it leaves it as it was. Anything else after the number is dropped without a
/// word, which is a real server's doing and not a corner cut here: a `FOO` where
/// the `COUNT` would go is read straight past.
fn read(server: &Server, args: Args<'_>, out: &mut Out) -> Result<()> {
    if args.len() < 4 {
        return Err(args::wrong_arity_sub("FT.CURSOR", "READ"));
    }
    let Some(id) = names(server, args, out)? else {
        return Ok(());
    };
    let mut count = None;
    if let Some(word) = args.opt(4)
        && args::is(word, b"COUNT")
        && let Some(value) = args.opt(5)
    {
        let Some(asked) = parse_i64(value) else {
            out.error_about(BAD_COUNT.as_bytes(), value, QUOTE.as_bytes());
            return Ok(());
        };
        if asked > 0 {
            count = usize::try_from(asked).ok();
        }
    }
    let name = args.get(2);
    let now = server.clock.now_ms();
    let mut cursors = server.cursors.lock();
    cursors.sweep(now);
    // A read has to come through the index the cursor was made on. A delete does
    // not, which is why the two look the cursor up differently.
    let key = u64::try_from(id).unwrap_or(0);
    let Some(held) = cursors
        .held
        .get_mut(&key)
        .filter(|held| *held.index == *name)
    else {
        return missing(id, out);
    };
    if let Some(count) = count {
        held.cursor.count = count;
    }
    held.cursor.touched = now;
    out.array(2);
    match held.cursor.chunk(out) {
        true => out.int(id),
        false => {
            cursors.held.remove(&key);
            out.int(0);
        }
    }
    Ok(())
}

/// `FT.CURSOR DEL index id`.
///
/// The index name has to be an index and then has nothing more to do with it: a
/// cursor made on one index is deleted through the name of another, which is
/// measured and is not what a read does.
fn del(server: &Server, args: Args<'_>, out: &mut Out) -> Result<()> {
    if args.len() < 4 {
        return Err(args::wrong_arity_sub("FT.CURSOR", "DEL"));
    }
    let Some(id) = names(server, args, out)? else {
        return Ok(());
    };
    let key = u64::try_from(id).unwrap_or(0);
    match server.cursors.lock().held.remove(&key) {
        Some(_) => out.ok(),
        None => out.error(NO_CURSOR.as_bytes()),
    }
    Ok(())
}

/// `FT.CURSOR GC index n`.
///
/// The number is how many to look at and a real server answers nought whatever
/// it is, because the collection it does is not the one the client asked for.
/// The sweep here is the one every read does anyway.
fn gc(server: &Server, args: Args<'_>, out: &mut Out) -> Result<()> {
    if args.len() < 4 {
        return Err(args::wrong_arity_sub("FT.CURSOR", "GC"));
    }
    let name = args.get(2);
    if server.search.lock().named(name).is_none() {
        Fail::naming(MISSING, name).write(out);
        return Ok(());
    }
    let now = server.clock.now_ms();
    server.cursors.lock().sweep(now);
    out.int(0);
    Ok(())
}

/// The index name and the cursor number in front of a read or a delete.
///
/// The name is checked first and the number second, which is a real server's
/// order: `FT.CURSOR READ nope notanumber` says the index is not there. A number
/// that will not fit answers about the number, and a negative one is a number
/// that fits and is simply not a cursor anybody has.
fn names(server: &Server, args: Args<'_>, out: &mut Out) -> Result<Option<i64>> {
    let name = args.get(2);
    if server.search.lock().named(name).is_none() {
        Fail::naming(MISSING, name).write(out);
        return Ok(None);
    }
    let Some(id) = parse_i64(args.get(3)) else {
        out.error(BAD_ID.as_bytes());
        return Ok(None);
    };
    Ok(Some(id))
}

/// `Cursor not found, id: 12345`, which is the one error line here that carries
/// a number rather than a word the client sent.
fn missing(id: i64, out: &mut Out) -> Result<()> {
    let mut line = Vec::with_capacity(NOT_FOUND.len() + 20);
    line.extend_from_slice(NOT_FOUND.as_bytes());
    line.extend_from_slice(id.to_string().as_bytes());
    out.error(&line);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cursor over a walk, with nothing on the other end of it. Every number
    /// in these tests was read off Redis 8.10.1 over the seven document corpus
    /// the module header describes.
    fn over(walk: Vec<bool>, loader: bool, offset: usize, window: usize, count: usize) -> Cursor {
        Cursor {
            kept: Kept {
                made: Made::Rolled(Vec::new()),
                walk,
                shows: Shows::default(),
                total: 7,
                whole: false,
                loader,
                offset,
                window,
            },
            at: 0,
            count,
            idle: IDLE,
            touched: 0,
            first: true,
            buffer: 0,
            pulled: 0,
            skip: offset,
            given: 0,
        }
    }

    /// The number at the front of each of the first few chunks.
    fn heads(mut cursor: Cursor, deep: bool, chunks: usize) -> Vec<usize> {
        (0..chunks)
            .map(|_| {
                let count = cursor.counted(deep);
                cursor.first = false;
                count
            })
            .collect()
    }

    #[test]
    fn a_pipeline_that_reads_keys_pulls_a_chunk_of_them_at_a_time() {
        let cursor = over(vec![true; 7], true, 0, usize::MAX, 2);
        assert_eq!(heads(cursor, false, 4), vec![2, 2, 2, 1]);
    }

    #[test]
    fn a_pipeline_that_reads_nothing_pulls_one_document_at_a_time() {
        let cursor = over(vec![true; 7], false, 0, usize::MAX, 2);
        assert_eq!(heads(cursor, false, 4), vec![1, 1, 1, 1]);
    }

    #[test]
    fn a_window_with_an_offset_holds_the_pull_back_and_one_without_does_not() {
        // `LIMIT 0 3` is a cap on what gets written, so a chunk of two still
        // pulls two to write one and says two.
        assert_eq!(
            heads(over(vec![true; 7], true, 0, 3, 2), false, 2),
            vec![2, 2]
        );
        // `LIMIT 1 3` is a step, so the pull is the chunk plus the one document
        // the step has still to get past.
        assert_eq!(
            heads(over(vec![true; 7], true, 1, 3, 2), false, 2),
            vec![3, 2]
        );
        // Two stepped over and two asked for, and the chunk of three never
        // comes into it.
        assert_eq!(heads(over(vec![true; 7], true, 2, 2, 3), false, 1), vec![4]);
    }

    #[test]
    fn a_filter_takes_its_rows_off_the_count_and_the_protocol_says_when() {
        let walk = vec![false, true, true, false, true, true, false];
        // RESP2 writes the number as soon as the first row of the chunk exists
        // and RESP3 writes it once the whole chunk does.
        let shallow = over(walk.clone(), true, 0, usize::MAX, 2);
        assert_eq!(heads(shallow, false, 3), vec![1, 1, 0]);
        let deep = over(walk, true, 0, usize::MAX, 2);
        assert_eq!(heads(deep, true, 3), vec![2, 2, 0]);
    }

    #[test]
    fn a_settled_answer_says_the_total_once_and_nought_after_that() {
        let mut cursor = over(vec![true; 7], true, 0, usize::MAX, 2);
        cursor.kept.whole = true;
        assert_eq!(heads(cursor, false, 3), vec![7, 0, 0]);
    }

    #[test]
    fn two_numbers_out_of_the_mint_are_not_the_same_number() {
        let mut cursors = Cursors::default();
        let one = cursors.mint(12);
        let two = cursors.mint(12);
        assert_ne!(one, two);
        // Thirty two bits wide and never nought, which is the number that says
        // a cursor is finished.
        assert!(one > 0 && one <= u64::from(u32::MAX));
        assert!(two > 0 && two <= u64::from(u32::MAX));
    }

    #[test]
    fn a_cursor_nobody_has_read_from_for_long_enough_is_swept_away() {
        let mut cursors = Cursors::default();
        cursors.held.insert(
            7,
            Held {
                index: b"i".as_slice().into(),
                cursor: Cursor {
                    kept: Kept {
                        made: Made::Rolled(Vec::new()),
                        walk: Vec::new(),
                        shows: Shows::default(),
                        total: 0,
                        whole: true,
                        loader: false,
                        offset: 0,
                        window: 0,
                    },
                    at: 0,
                    count: 1,
                    idle: 100,
                    touched: 1_000,
                    first: true,
                    buffer: 0,
                    pulled: 0,
                    skip: 0,
                    given: 0,
                },
            },
        );
        cursors.sweep(1_100);
        assert_eq!(cursors.total(), 1);
        cursors.sweep(1_101);
        assert_eq!(cursors.total(), 0);
    }
}
