//! A whole dataset written out as one RDB file.
//!
//! [`crate::rdb`] writes one value with no key and no frame around it, because
//! that is what fits in a bulk string and it is all `DUMP` and `MIGRATE` need. A
//! file is the other shape of the same bytes: a header, then every key in every
//! database with its name and its deadline in front of it, then an end marker
//! and a checksum over the lot.
//!
//! ```text
//! REDIS0012                      the magic and the version, nine ASCII bytes
//! FA <str> <str>                 an aux field, repeated, provenance only
//! FE <len>                       from here on, this database
//! FB <len> <len>                 how many keys and how many of them expire
//! FC <8 bytes LE>                this key's deadline, only if it has one
//! <type> <str key> <the value>   the value in exactly the shape DUMP writes
//! FF <8 bytes LE>                the end, and crc64 over everything above it
//! ```
//!
//! # Why this exists at all
//!
//! Three things want it and they are the same thing three times. `BACKUP` needs
//! a consistent copy of the dataset in a file. `yodb dump --format rdb` is G16
//! in spec `07` and has to produce bytes a real `redis-server` will start on.
//! `PSYNC` sends a replica the whole dataset before it sends the stream of
//! changes, and what it sends is this. So the writer is one piece of work and
//! not three, and it is here rather than in the server because the keyspace is
//! the thing that knows how to walk itself.
//!
//! # Version twelve and not the newest
//!
//! Same rule the payload writer follows and the same reason. What goes in the
//! header is a promise about how old a server can be and still read the file,
//! and every type this writes has been loadable since twelve. A newer number
//! buys nothing and locks out every server below it.
//!
//! # What it costs
//!
//! One key at a time, and the value is copied twice on the way: once because
//! [`Keyspace::export`] hands back an owned record, and once as it is written.
//! The peak extra memory is therefore the largest single value and not the
//! dataset, which is the number that matters, and the copy is worth calling out
//! because the honest fix is a borrowing walk and that is a bigger change to the
//! keyspace than this file should make. Key names are taken a batch at a time
//! into one buffer that is reused, so a database of ten million keys does not
//! build a list of ten million names to walk.

use yo_common::crc::crc64;

use crate::keys::Record;
use crate::keyspace::Keyspace;
use crate::rdb;
use crate::value::Kind;
use yo_index::Cursor as KeyCursor;

/// The nine bytes at the front, magic and version together.
///
/// Written as one constant rather than assembled from [`rdb::VERSION`] because
/// the file version and the payload version are free to move apart, and a header
/// that silently followed a payload bump would be a compatibility change nobody
/// asked for.
const HEADER: &[u8] = b"REDIS0012";

/// An aux field: two strings, a name and a value, that a loader may ignore.
const OP_AUX: u8 = 0xFA;
/// How many keys are in this database and how many of them carry a deadline.
const OP_RESIZEDB: u8 = 0xFB;
/// The deadline of the key that follows, in milliseconds since the epoch.
const OP_EXPIRETIME_MS: u8 = 0xFC;
/// Everything after this belongs to the database whose number follows.
const OP_SELECTDB: u8 = 0xFE;
/// The end of the file, followed by the checksum.
const OP_EOF: u8 = 0xFF;

/// How many key names to take out of the index before going back for the values.
///
/// The walk borrows the index and the export needs it back, so the two cannot be
/// interleaved and the names have to be copied somewhere in between. A batch
/// rather than the whole database keeps that buffer at a few kilobytes however
/// many keys there are.
const BATCH: usize = 256;

/// A file being written.
///
/// Built in order: [`Snapshot::new`], then any [`Snapshot::aux`] fields, then one
/// [`Snapshot::database`] per database that has anything in it, then
/// [`Snapshot::finish`]. Nothing enforces that order beyond a debug assertion on
/// the aux fields, which are the one part a loader reads positionally.
pub struct Snapshot {
    /// The file so far.
    out: Vec<u8>,
    /// The checksum of everything up to `taken`.
    crc: u64,
    /// How much of `out` the checksum has already been over.
    ///
    /// The checksum is kept up to date as the file grows rather than taken over
    /// the whole buffer at the end. It costs the same, and it is the seam a
    /// diskless `PSYNC` needs: the bytes between `taken` and the end are exactly
    /// the ones that could be handed to a socket and dropped.
    taken: usize,
    /// Whether a database has been written, so aux fields can say they are late.
    started: bool,
    /// Key names for the batch being written, back to back.
    names: Vec<u8>,
    /// Where each name in `names` starts and ends.
    bounds: Vec<(usize, usize)>,
    /// Keys that were passed over because their type has no RDB shape.
    skipped: usize,
}

impl Default for Snapshot {
    fn default() -> Snapshot {
        Snapshot::new()
    }
}

impl Snapshot {
    /// Start a file, with the header written and nothing else.
    #[must_use]
    pub fn new() -> Snapshot {
        let mut snap = Snapshot {
            out: Vec::new(),
            crc: 0,
            taken: 0,
            started: false,
            names: Vec::new(),
            bounds: Vec::new(),
            skipped: 0,
        };
        snap.out.extend_from_slice(HEADER);
        // Ours and not a Redis field name, so a real server logs it and carries
        // on. It is here because a file that will be handed to a support case
        // should say what wrote it.
        snap.aux(b"yo-ver", env!("CARGO_PKG_VERSION").as_bytes());
        snap
    }

    /// Add an aux field, which has to happen before the first database.
    ///
    /// Redis writes `redis-ver`, `redis-bits`, `ctime` and `used-mem` here and
    /// reads none of them back except to log. The two that do change behaviour
    /// are `aof-base`, which tells the loader this file is the base of an append
    /// only file rather than a plain dump, and `repl-id` with `repl-offset`,
    /// which is how a replica knows where it is. Neither is written here: the
    /// caller that needs one knows the value and this does not.
    pub fn aux(&mut self, name: &[u8], value: &[u8]) {
        debug_assert!(!self.started, "an aux field after the first database");
        self.out.push(OP_AUX);
        rdb::put_str(&mut self.out, name);
        rdb::put_str(&mut self.out, value);
        self.absorb();
    }

    /// Write every key in one database, under the number it answers to.
    ///
    /// A database with nothing in it is skipped entirely rather than written as
    /// an empty selector, which is what Redis does and is why a file from a
    /// server with one key in database nine has one selector in it.
    ///
    /// Keys past their deadline are dropped on the way through by the walk, the
    /// same as they are for `SCAN`, so a snapshot does not carry dead keys into
    /// the file and then into whatever loads it.
    pub fn database(&mut self, index: usize, db: &mut Keyspace) {
        if db.is_empty() {
            return;
        }
        self.started = true;
        self.out.push(OP_SELECTDB);
        rdb::put_len(&mut self.out, index as u64);
        // Both counts are hints. Redis presizes its two tables from them and
        // does not check them against what arrives, which is just as well: a key
        // holding something with no RDB shape is not written and the count here
        // was taken before anybody knew that.
        self.out.push(OP_RESIZEDB);
        rdb::put_len(&mut self.out, db.len() as u64);
        rdb::put_len(&mut self.out, db.expires() as u64);
        self.absorb();

        let mut at = KeyCursor::START;
        loop {
            self.names.clear();
            self.bounds.clear();
            let names = &mut self.names;
            let bounds = &mut self.bounds;
            at = db.scan(at, BATCH, None, |key| {
                let from = names.len();
                names.extend_from_slice(key);
                bounds.push((from, names.len()));
            });
            for i in 0..self.bounds.len() {
                let (from, to) = self.bounds[i];
                // The borrow checker will not let the name be held across the
                // export, since one borrows this and the other takes it mutably,
                // so the name is copied into the entry writer instead. It is a
                // key name and it is a few bytes.
                let key = self.names[from..to].to_vec();
                let Some(rec) = db.export(&key) else {
                    // Gone between the walk and here, which nothing can do while
                    // this holds the database, or read back off the store and
                    // failed. Either way there is no value to write.
                    continue;
                };
                self.entry(&key, &rec);
            }
            if at.is_end() {
                break;
            }
        }
    }

    /// Close the file: the end marker and the checksum over everything.
    #[must_use]
    pub fn finish(mut self) -> Vec<u8> {
        self.out.push(OP_EOF);
        self.absorb();
        self.out.extend_from_slice(&self.crc.to_le_bytes());
        self.out
    }

    /// How many keys were passed over because their type has no RDB shape.
    ///
    /// A graph or a vector index lives in an engine above this crate and there
    /// is no Redis type byte for either, so a file cannot carry them and a
    /// caller that has them should say so rather than let a backup quietly come
    /// back smaller than it went in.
    #[must_use]
    pub const fn skipped(&self) -> usize {
        self.skipped
    }

    /// One key: the deadline if it has one, then the type, name and value.
    fn entry(&mut self, key: &[u8], rec: &Record) {
        // Asked before a byte is written, because the alternative is finding out
        // halfway through an entry and having to unwind it.
        if matches!(rec.kind(), Kind::Foreign | Kind::Array) {
            self.skipped += 1;
            return;
        }
        if let Some(at) = rec.expire_at() {
            self.out.push(OP_EXPIRETIME_MS);
            self.out.extend_from_slice(&at.to_le_bytes());
        }
        let wrote = rdb::object(rec, Some(key), &mut self.out);
        debug_assert!(wrote, "the kinds without a shape were refused above");
        self.absorb();
    }

    /// Run the checksum up to the end of what has been written.
    fn absorb(&mut self) {
        self.crc = crc64(self.crc, &self.out[self.taken..]);
        self.taken = self.out.len();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::Clock;
    use crate::lists::End;
    use crate::rdb::FOOTER;
    use crate::streams::{Add, Trim};
    use crate::ttl::Cond;
    use crate::zsets::ZAdd;

    /// The reader the tests check the writer with.
    ///
    /// Deliberately a separate parser rather than the one in [`crate::rdb`]:
    /// checking a writer with its own reader agrees mostly that the two are
    /// consistent, and what is wanted here is that the bytes are the ones the
    /// format says. It knows only the frame, not the values, which is the part
    /// [`crate::rdb`] already has its own tests for.
    struct Parse<'a> {
        buf: &'a [u8],
        at: usize,
    }

    impl<'a> Parse<'a> {
        fn byte(&mut self) -> u8 {
            let b = self.buf[self.at];
            self.at += 1;
            b
        }

        fn len(&mut self) -> u64 {
            let first = self.byte();
            match first >> 6 {
                0 => u64::from(first & 0x3f),
                1 => (u64::from(first & 0x3f) << 8) | u64::from(self.byte()),
                _ => panic!("the tests do not write a length that big"),
            }
        }

        fn str(&mut self) -> Vec<u8> {
            let n = self.len() as usize;
            let s = self.buf[self.at..self.at + n].to_vec();
            self.at += n;
            s
        }
    }

    /// Everything a test wants to know about a file, without the values.
    #[derive(Default, PartialEq, Eq, Debug)]
    struct Seen {
        aux: Vec<(Vec<u8>, Vec<u8>)>,
        dbs: Vec<u64>,
        /// Key, deadline, and the payload with a `DUMP` footer put back on it.
        keys: Vec<(Vec<u8>, Option<u64>, Vec<u8>)>,
    }

    /// Walk a file, checking the frame and pulling the entries out of it.
    ///
    /// The value of each entry is found by handing the rest of the file to the
    /// payload reader, which stops where the value stops, so this does not need
    /// to know how long a value is. That is also the check that matters: a
    /// writer that got a length wrong anywhere would leave the next type byte
    /// somewhere this does not find it.
    fn walk(file: &[u8]) -> Seen {
        assert_eq!(&file[..9], HEADER, "the header");
        let crc = u64::from_le_bytes(file[file.len() - 8..].try_into().unwrap());
        assert_eq!(crc, crc64(0, &file[..file.len() - 8]), "the checksum");

        let mut seen = Seen::default();
        let mut p = Parse {
            buf: file,
            at: HEADER.len(),
        };
        let mut expire = None;
        loop {
            match p.byte() {
                OP_AUX => {
                    let name = p.str();
                    let value = p.str();
                    seen.aux.push((name, value));
                }
                OP_SELECTDB => seen.dbs.push(p.len()),
                OP_RESIZEDB => {
                    p.len();
                    p.len();
                }
                OP_EXPIRETIME_MS => {
                    let at = u64::from_le_bytes(p.buf[p.at..p.at + 8].try_into().unwrap());
                    p.at += 8;
                    expire = Some(at);
                }
                OP_EOF => {
                    assert_eq!(p.at + 8, file.len(), "the end is where the file ends");
                    return seen;
                }
                ty => {
                    // Back up over the type byte, take the key, then rebuild the
                    // payload a `DUMP` would have produced for the same value so
                    // that the test can compare the two directly.
                    let key = p.str();
                    let mut payload = vec![ty];
                    let rest = &p.buf[p.at..];
                    let taken = crate::rdb::measure(ty, rest).expect("a value we wrote");
                    payload.extend_from_slice(&rest[..taken]);
                    p.at += taken;
                    seen.keys.push((key, expire.take(), payload));
                }
            }
        }
    }

    fn db() -> Keyspace {
        Keyspace::with_clock(Clock::fixed(1_000))
    }

    #[test]
    fn an_empty_server_is_a_header_an_aux_field_and_an_end() {
        let mut snap = Snapshot::new();
        snap.database(0, &mut db());
        let file = snap.finish();
        let seen = walk(&file);
        assert!(seen.dbs.is_empty(), "no database has anything in it");
        assert!(seen.keys.is_empty());
        assert_eq!(seen.aux.len(), 1, "the one we write");
    }

    #[test]
    fn every_type_goes_out_as_the_payload_dump_would_have_written() {
        let mut d = db();
        d.set_plain(b"str", b"hello").unwrap();
        d.push(b"list", End::Right, [b"a".as_slice(), b"b"].into_iter())
            .unwrap();
        d.sadd(b"set", [b"x".as_slice(), b"y"].into_iter()).unwrap();
        d.sadd(b"ints", [b"1".as_slice(), b"2"].into_iter())
            .unwrap();
        d.zadd(
            b"zset",
            [(1.5, b"m".as_slice())].into_iter(),
            ZAdd::default(),
        )
        .unwrap();
        d.hset(b"hash", [(b"f".as_slice(), b"v".as_slice())].into_iter())
            .unwrap();
        d.xadd(
            b"stream",
            Add::Auto,
            &[(b"f".as_slice(), b"v".as_slice())],
            Trim::None,
            true,
            1_000,
        )
        .unwrap();

        let mut snap = Snapshot::new();
        snap.database(0, &mut d);
        let file = snap.finish();
        let seen = walk(&file);

        assert_eq!(seen.dbs, vec![0]);
        assert_eq!(seen.keys.len(), 7, "one entry per key");
        for (key, expire, payload) in &seen.keys {
            assert_eq!(*expire, None, "nothing was given a deadline");
            // The same bytes `DUMP` would have handed a client, which is the
            // whole claim: a file is the payloads with a frame around them.
            assert_eq!(
                Some(payload.clone()),
                d.dump(key).map(|p| p[..p.len() - FOOTER].to_vec()),
                "{}",
                String::from_utf8_lossy(key)
            );
        }
    }

    #[test]
    fn a_deadline_travels_with_the_key_and_a_dead_key_does_not() {
        let mut d = db();
        for key in [&b"alive"[..], b"soon", b"gone"] {
            d.set_plain(key, b"1").unwrap();
        }
        d.expire(b"soon", 9_000, Cond::Always);
        d.expire(b"gone", 500, Cond::Always);

        let mut snap = Snapshot::new();
        snap.database(0, &mut d);
        let seen = walk(&snap.finish());

        let mut found: Vec<(Vec<u8>, Option<u64>)> = seen
            .keys
            .iter()
            .map(|(k, at, _)| (k.clone(), *at))
            .collect();
        found.sort();
        assert_eq!(
            found,
            vec![(b"alive".to_vec(), None), (b"soon".to_vec(), Some(9_000))],
            "the key whose deadline has gone is not in the file"
        );
    }

    #[test]
    fn each_database_is_selected_once_and_the_empty_ones_are_not_there() {
        let mut zero = db();
        let mut empty = db();
        let mut nine = db();
        zero.set_plain(b"a", b"1").unwrap();
        nine.set_plain(b"b", b"2").unwrap();

        let mut snap = Snapshot::new();
        snap.database(0, &mut zero);
        snap.database(4, &mut empty);
        snap.database(9, &mut nine);
        let seen = walk(&snap.finish());
        assert_eq!(seen.dbs, vec![0, 9], "the empty one in the middle is gone");
        assert_eq!(seen.keys.len(), 2);
    }

    #[test]
    fn more_keys_than_one_batch_all_arrive_once() {
        let mut d = db();
        let many = BATCH * 3 + 7;
        for i in 0..many {
            d.set_plain(format!("k{i}").as_bytes(), b"v").unwrap();
        }
        let mut snap = Snapshot::new();
        snap.database(0, &mut d);
        let seen = walk(&snap.finish());

        let mut names: Vec<&Vec<u8>> = seen.keys.iter().map(|(k, _, _)| k).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), many, "every key, and none of them twice");
    }

    /// A sparse array has no Redis type byte, so it cannot go in a file and the
    /// snapshot has to say so rather than write something a loader would choke
    /// on halfway through the entry before it.
    #[test]
    fn a_type_with_no_rdb_shape_is_counted_and_left_out() {
        let mut d = db();
        d.set_plain(b"ordinary", b"1").unwrap();
        d.arset(b"sparse", 7, [&b"a"[..]].into_iter()).unwrap();

        let mut snap = Snapshot::new();
        snap.database(0, &mut d);
        assert_eq!(snap.skipped(), 1);
        let seen = walk(&snap.finish());
        assert_eq!(seen.keys.len(), 1);
        assert_eq!(seen.keys[0].0, b"ordinary");
    }

    #[test]
    fn an_aux_field_the_caller_adds_is_in_the_file() {
        let mut snap = Snapshot::new();
        snap.aux(b"redis-ver", b"8.8.0");
        let seen = walk(&snap.finish());
        assert!(
            seen.aux
                .contains(&(b"redis-ver".to_vec(), b"8.8.0".to_vec())),
            "{:?}",
            seen.aux
        );
    }
}
