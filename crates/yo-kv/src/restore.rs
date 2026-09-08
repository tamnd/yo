//! A whole RDB file read back into values.
//!
//! [`crate::snapshot`] writes one, this reads one, and the two are the same
//! bytes from opposite ends. [`crate::rdb`] already had the half of this that
//! reads a single value out of a `RESTORE` payload, which is the hard half and
//! is where every type lives. What is here is the frame around those values: the
//! nine byte header, the opcodes that say which database a key is in and when it
//! dies, the end marker and the checksum over the lot.
//!
//! ```text
//! REDIS0012                      the magic and the version, nine ASCII bytes
//! FA <str> <str>                 an aux field, repeated
//! F5 <str>                       a function library, code and all
//! FE <len>                       from here on, this database
//! FB <len> <len>                 how many keys and how many of them expire
//! FC <8 bytes LE>                this key's deadline, only if it has one
//! <type> <str key> <the value>   the value in exactly the shape DUMP writes
//! FF <8 bytes LE>                the end, and crc64 over everything above it
//! ```
//!
//! # What wants it
//!
//! Four things, and none of them can be built without it. Starting on a file
//! left behind by the last run is the obvious one and is the difference between
//! a server that persists and a server that only writes. `DEBUG RELOAD` is a
//! save followed by a load, and a large part of the Redis test suite calls it
//! after almost every case to prove that what is in memory survives the trip. A
//! replica taking `PSYNC` is handed one of these before it is handed the stream
//! of changes. And `yodb restore` is the other end of `yodb dump --format rdb`,
//! which is G16 in spec `07`.
//!
//! # Reading further back than we write
//!
//! The header says twelve on the way out and this takes anything up to fifteen
//! on the way in, which is the same asymmetry [`crate::rdb::VERSION`] and
//! [`crate::rdb::READS_UP_TO`] have for a payload and it is there for the same
//! reason. What a version promises is how old a reader can be, so writing a low
//! one makes a file useful to more servers, and refusing a high one is the only
//! safe answer to a file whose type bytes may not mean what they used to.
//!
//! # A pull rather than a push
//!
//! The load hands back one item at a time rather than filling a keyspace in,
//! because the caller is the only one that knows where a key should land. A
//! server puts it in the database the file names, a `DEBUG RELOAD` puts it back
//! where it came from, and a tool printing a file puts it nowhere at all. It
//! also means the borrow of the keyspace is the caller's to arrange, which
//! matters because building a value needs the four thresholds out of it and
//! importing one needs it mutably.
//!
//! # What it costs
//!
//! One value at a time, built straight out of the bytes it was written from, so
//! the peak extra memory is the largest single value and not the file. Nothing
//! is copied on the way that was not going to be copied anyway: a key name comes
//! back borrowed from the file when it was written plain, and owned only when it
//! was written compressed or as an integer.

use std::borrow::Cow;
use std::fmt;

use yo_common::crc::crc64;

use crate::keys::Record;
use crate::rdb::{
    self, Bad, Limits, OP_AUX, OP_EOF, OP_EXPIRETIME, OP_EXPIRETIME_MS, OP_FREQ,
    OP_FUNCTION_PRE_GA, OP_FUNCTION2, OP_IDLE, OP_MODULE_AUX, OP_RESIZEDB, OP_SELECTDB,
    OP_SLOT_INFO, Reader,
};

/// The five letters every file starts with, before the four version digits.
const MAGIC: &[u8] = b"REDIS";

/// The header: the magic and the version together.
const HEADER: usize = 9;

/// The checksum on the end, which is eight bytes and no version.
///
/// A payload carries ten because it has to say which version wrote it. A file
/// says that in the header instead, so the only thing on the end is the sum.
const FOOTER: usize = 8;

/// Why a file was not read.
///
/// Six of them rather than one, because the answers are genuinely different and
/// the person reading the log is trying to work out whether the file is from the
/// future, from a fork, damaged, or truncated by a full disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fault {
    /// The first five bytes are not `REDIS`, so this is not an RDB file at all.
    Magic,
    /// The version in the header is newer than this server reads.
    Version(u16),
    /// The checksum on the end does not match the bytes in front of it.
    Checksum,
    /// The file ends in the middle of something.
    Truncated,
    /// An opcode that cannot be skipped without knowing what wrote it.
    Opcode(u8),
    /// A value that did not parse, or a type this server has no shape for.
    Value,
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Fault::Magic => f.write_str("the file does not start with REDIS"),
            Fault::Version(v) => write!(
                f,
                "the file is version {v} and this server reads up to {}",
                rdb::READS_UP_TO
            ),
            Fault::Checksum => f.write_str("the checksum does not match, so the file is damaged"),
            Fault::Truncated => f.write_str("the file ends in the middle of something"),
            Fault::Opcode(op) => write!(
                f,
                "the file holds opcode {op}, which needs whatever wrote it"
            ),
            Fault::Value => f.write_str("a value in the file is not one this server can hold"),
        }
    }
}

/// One thing out of a file.
///
/// Three variants and not one per opcode, because most of the opcodes are
/// bookkeeping the reader deals with itself. A `SELECTDB` is answered by putting
/// the number on every key after it rather than by handing the caller a marker
/// it would have to remember, and a resize hint, an idle time and a frequency
/// are read and dropped.
///
/// The key arm is much the largest, because a record holds a whole value inline
/// rather than behind a pointer, and that is deliberate. Boxing it to even the
/// arms up would buy a smaller move and cost an allocation for every key in the
/// file, which is the wrong way round: the move happens once per key either way
/// and the allocation would be new work on the one path this is built for.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Item<'a> {
    /// A name and a value the file carries for whoever wants them.
    ///
    /// Redis writes `redis-ver`, `redis-bits`, `ctime` and `used-mem` and reads
    /// none of them back except to log. The two that mean something are
    /// `aof-base`, which says the file is the base of an append only file, and
    /// `repl-id` with `repl-offset`, which is where a replica is. Which of those
    /// matter is the caller's question, so all of them come out.
    Aux {
        /// What the field is called.
        name: Cow<'a, [u8]>,
        /// What it says.
        value: Cow<'a, [u8]>,
    },
    /// A function library, the whole code including its shebang.
    Library(Cow<'a, [u8]>),
    /// A key, the database it was in, and everything under it.
    Key {
        /// Which database the file put it in.
        db: usize,
        /// The name.
        key: Cow<'a, [u8]>,
        /// The value and its deadline.
        record: Record,
    },
}

/// A file being read, one item at a time.
///
/// Built by [`Load::open`], which checks the header and the checksum, and then
/// walked as an iterator. An error ends the walk: the frame is a chain where
/// each item says how long it is, so once one of them has not made sense there
/// is no way to find where the next one starts.
pub struct Load<'a> {
    r: Reader<'a>,
    limits: Limits<'a>,
    now: u64,
    version: u16,
    /// The database every key belongs to until a selector says otherwise.
    db: usize,
    /// The deadline waiting for the key that comes next, if one was written.
    expire_at: Option<u64>,
    /// Keys that were read and thrown away because their deadline had gone.
    expired: usize,
    /// Whether the walk is over, either at the end marker or at an error.
    done: bool,
}

impl<'a> Load<'a> {
    /// Check a file and get ready to walk it.
    ///
    /// Everything cheap that can be wrong is answered here: the magic, the
    /// version and the checksum. The checksum is over the whole file, so this
    /// touches every byte before the first value comes out, which is the right
    /// trade for a load that is about to build the whole dataset anyway and is
    /// the only order in which a damaged file is refused rather than half
    /// loaded.
    ///
    /// A checksum of zero means the file was written by a server that had been
    /// told not to spend the time on one, and is accepted, the same way a
    /// payload with a zero checksum is.
    ///
    /// # Errors
    ///
    /// [`Fault::Magic`], [`Fault::Version`], [`Fault::Checksum`] or
    /// [`Fault::Truncated`] for a file too short to hold a header and a footer.
    pub fn open(file: &'a [u8], limits: Limits<'a>, now: u64) -> Result<Load<'a>, Fault> {
        if file.len() < HEADER + FOOTER {
            return Err(Fault::Truncated);
        }
        if &file[..MAGIC.len()] != MAGIC {
            return Err(Fault::Magic);
        }
        let mut version = 0u16;
        for &d in &file[MAGIC.len()..HEADER] {
            if !d.is_ascii_digit() {
                return Err(Fault::Magic);
            }
            version = version * 10 + u16::from(d - b'0');
        }
        if version > rdb::READS_UP_TO {
            return Err(Fault::Version(version));
        }
        let body = &file[HEADER..file.len() - FOOTER];
        let stored = u64::from_le_bytes(
            file[file.len() - FOOTER..]
                .try_into()
                .expect("eight bytes of checksum"),
        );
        if stored != 0 && stored != crc64(0, &file[..file.len() - FOOTER]) && !rdb::skipping() {
            return Err(Fault::Checksum);
        }
        Ok(Load {
            r: Reader::new(body),
            limits,
            now,
            version,
            db: 0,
            expire_at: None,
            expired: 0,
            done: false,
        })
    }

    /// The version out of the header.
    #[must_use]
    pub const fn version(&self) -> u16 {
        self.version
    }

    /// How many keys were read and thrown away because they had already died.
    ///
    /// Redis reports this as `rdb_last_load_keys_expired` and drops the keys,
    /// which is what a master does. A replica keeps them and waits to be told,
    /// because on a replica the master owns the decision, and the day the
    /// replication path lands that becomes a choice this takes rather than one
    /// it makes.
    #[must_use]
    pub const fn expired(&self) -> usize {
        self.expired
    }

    /// The next thing in the file, or nothing when the end marker is reached.
    fn step(&mut self) -> Result<Option<Item<'a>>, Fault> {
        loop {
            if self.done {
                return Ok(None);
            }
            let op = self.r.byte().map_err(|_| Fault::Truncated)?;
            match op {
                OP_EOF => {
                    self.done = true;
                    // The end marker has to be the end. Anything after it is a
                    // file that was appended to or two files stuck together, and
                    // reading on would mean trusting bytes the checksum did not
                    // cover in the way the writer meant.
                    if !self.r.done() {
                        return Err(Fault::Truncated);
                    }
                    return Ok(None);
                }
                OP_SELECTDB => self.db = self.r.len().map_err(|_| Fault::Truncated)?,
                OP_RESIZEDB => {
                    // Both counts are hints Redis presizes its tables from. This
                    // grows as it goes, so they are read to get past them.
                    self.r.len().map_err(|_| Fault::Truncated)?;
                    self.r.len().map_err(|_| Fault::Truncated)?;
                }
                OP_EXPIRETIME_MS => {
                    let b = self.r.take(8).map_err(|_| Fault::Truncated)?;
                    self.expire_at = Some(u64::from_le_bytes(b.try_into().expect("eight bytes")));
                }
                OP_EXPIRETIME => {
                    // The form every file used before 2.6 and no writer has used
                    // since. It is four bytes of whole seconds, and it is read
                    // because a file that old is exactly the file somebody is
                    // trying to get their data out of.
                    let b = self.r.take(4).map_err(|_| Fault::Truncated)?;
                    let secs = u32::from_le_bytes(b.try_into().expect("four bytes"));
                    self.expire_at = Some(u64::from(secs) * 1_000);
                }
                OP_IDLE => {
                    // How long the key had been idle and how often it was used.
                    // Both are for an eviction policy to weigh, and a key loaded
                    // here starts fresh, so they are read and dropped.
                    self.r.num().map_err(|_| Fault::Truncated)?;
                }
                OP_FREQ => {
                    self.r.byte().map_err(|_| Fault::Truncated)?;
                }
                OP_AUX => {
                    let name = self.r.str().map_err(|_| Fault::Truncated)?;
                    let value = self.r.str().map_err(|_| Fault::Truncated)?;
                    return Ok(Some(Item::Aux { name, value }));
                }
                OP_FUNCTION2 => {
                    let code = self.r.str().map_err(|_| Fault::Truncated)?;
                    return Ok(Some(Item::Library(code)));
                }
                OP_SLOT_INFO => {
                    // Which slot the keys that follow are in and how many there
                    // are. A cluster presizes from it and a server that is not
                    // one reads past it, which is what makes a file written by a
                    // cluster node loadable by anything.
                    for _ in 0..3 {
                        self.r.len().map_err(|_| Fault::Truncated)?;
                    }
                }
                // Both of these are somebody else's bytes. A pre GA library is
                // in a format no server has ever converted, and a module's own
                // data has no length in front of it, so the only thing that can
                // read past it is the module that wrote it. Neither can be
                // skipped, and guessing would mean reading the rest of the file
                // out of the middle of a value.
                OP_FUNCTION_PRE_GA | OP_MODULE_AUX => return Err(Fault::Opcode(op)),
                kind => {
                    let key = self.r.str().map_err(|_| Fault::Truncated)?;
                    let expire_at = self.expire_at.take();
                    let body = rdb::read_object(&mut self.r, kind, self.limits, self.now).map_err(
                        |e| match e {
                            Bad::Footer => Fault::Checksum,
                            Bad::Format => Fault::Value,
                        },
                    )?;
                    // A key whose deadline has gone is read and dropped rather
                    // than skipped, because the only way to find where the next
                    // one starts is to read this one to its end.
                    if expire_at.is_some_and(|at| at <= self.now) {
                        self.expired += 1;
                        continue;
                    }
                    return Ok(Some(Item::Key {
                        db: self.db,
                        key,
                        record: Record::new(body, expire_at),
                    }));
                }
            }
        }
    }
}

impl<'a> Iterator for Load<'a> {
    type Item = Result<Item<'a>, Fault>;

    fn next(&mut self) -> Option<Result<Item<'a>, Fault>> {
        match self.step() {
            Ok(None) => None,
            Ok(Some(item)) => Some(Ok(item)),
            Err(fault) => {
                // One error and the walk is over. Every item says how long it
                // is, so a reader that has lost its place cannot find it again.
                self.done = true;
                Some(Err(fault))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::Clock;
    use crate::db::Db;
    use crate::hash;
    use crate::list;
    use crate::lists::End;
    use crate::set;
    use crate::snapshot::Snapshot;
    use crate::streams::{Add, Trim};
    use crate::ttl::Ask;
    use crate::ttl::Cond;
    use crate::value::Kind;
    use crate::zset;
    use crate::zsets::ZAdd;

    /// The four thresholds a test loads under, which are the defaults.
    fn bands() -> rdb::Bands {
        rdb::Bands {
            set: set::Limits::DEFAULT,
            hash: hash::Limits::DEFAULT,
            list: list::Limits::default(),
            zset: zset::Limits::DEFAULT,
        }
    }

    fn db() -> Db {
        Db::with_clock(Clock::fixed(1_000), 1)
    }

    /// Every type, so that a round trip covers every reader.
    fn filled() -> Db {
        let mut d = db();
        d.at(b"str").set_plain(b"str", b"hello").unwrap();
        // Long enough and dull enough to be compressed on the way out, which
        // puts the LZF reader in the path of the round trip.
        d.at(b"long")
            .set_plain(b"long", &b"ab".repeat(500))
            .unwrap();
        d.at(b"list")
            .push(b"list", End::Right, [b"a".as_slice(), b"b"].into_iter())
            .unwrap();
        d.at(b"set")
            .sadd(b"set", [b"x".as_slice(), b"y"].into_iter())
            .unwrap();
        d.at(b"ints")
            .sadd(b"ints", [b"1".as_slice(), b"2"].into_iter())
            .unwrap();
        d.at(b"zset")
            .zadd(
                b"zset",
                [(1.5, b"m".as_slice())].into_iter(),
                ZAdd::default(),
            )
            .unwrap();
        d.at(b"hash")
            .hset(b"hash", [(b"f".as_slice(), b"v".as_slice())].into_iter())
            .unwrap();
        d.at(b"stream")
            .xadd(
                b"stream",
                Add::Auto,
                &[(b"f".as_slice(), b"v".as_slice())],
                Trim::None,
                true,
                1_000,
            )
            .unwrap();
        d
    }

    /// Load a file into a fresh database per selector, panicking on a fault.
    fn into_dbs(file: &[u8], now: u64) -> Vec<(usize, Db)> {
        let bands = bands();
        let mut out: Vec<(usize, Db)> = Vec::new();
        for item in Load::open(file, bands.limits(), now).unwrap() {
            if let Item::Key { db: n, key, record } = item.unwrap() {
                if !out.iter().any(|(i, _)| *i == n) {
                    out.push((n, db()));
                }
                let at = out.iter_mut().find(|(i, _)| *i == n).unwrap();
                at.1.at(&key).import(&key, record);
            }
        }
        out
    }

    #[test]
    fn every_type_comes_back_the_way_it_went_in() {
        let mut before = filled();
        let mut snap = Snapshot::new();
        snap.database(0, &before);
        let file = snap.finish();

        let mut after = into_dbs(&file, 1_000);
        assert_eq!(after.len(), 1, "one database");
        let (index, ref mut back) = after[0];
        assert_eq!(index, 0);

        let keys: Vec<&[u8]> = vec![
            b"str", b"long", b"list", b"set", b"ints", b"zset", b"hash", b"stream",
        ];
        for key in keys {
            // The payload is the whole value, so two values with the same
            // payload are the same value. That is a stronger check than walking
            // the members and it is the same one the snapshot tests use.
            assert_eq!(
                back.at(key).dump(key),
                before.at(key).dump(key),
                "{}",
                String::from_utf8_lossy(key)
            );
        }
    }

    #[test]
    fn a_key_lands_in_the_database_the_file_put_it_in() {
        let mut zero = db();
        let mut nine = db();
        zero.at(b"a").set_plain(b"a", b"1").unwrap();
        nine.at(b"b").set_plain(b"b", b"2").unwrap();
        let mut snap = Snapshot::new();
        snap.database(0, &zero);
        snap.database(9, &nine);

        let mut after = into_dbs(&snap.finish(), 1_000);
        assert_eq!(
            after.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![0, 9]
        );
        assert!(after[0].1.at(b"a").exists(b"a"));
        assert!(after[1].1.at(b"b").exists(b"b"));
    }

    #[test]
    fn a_deadline_travels_with_the_key_and_a_dead_one_is_dropped() {
        let mut d = db();
        for key in [&b"alive"[..], b"soon", b"gone"] {
            d.at(key).set_plain(key, b"1").unwrap();
        }
        d.at(b"soon").expire(b"soon", 9_000, Cond::Always);
        d.at(b"gone").expire(b"gone", 5_000, Cond::Always);
        let mut snap = Snapshot::new();
        snap.database(0, &d);
        let file = snap.finish();

        // Read at a moment after one of the two deadlines but not the other.
        let bands = bands();
        let mut load = Load::open(&file, bands.limits(), 7_000).unwrap();
        let mut found: Vec<(Vec<u8>, Option<u64>)> = Vec::new();
        for item in &mut load {
            if let Item::Key { key, record, .. } = item.unwrap() {
                found.push((key.into_owned(), record.expire_at()));
            }
        }
        found.sort();
        assert_eq!(
            found,
            vec![(b"alive".to_vec(), None), (b"soon".to_vec(), Some(9_000))],
        );
        assert_eq!(load.expired(), 1, "the one that had gone was counted");
    }

    #[test]
    fn the_aux_fields_come_out_in_the_order_they_were_written() {
        let mut snap = Snapshot::new();
        snap.aux(b"redis-ver", b"8.8.0");
        snap.aux(b"aof-base", b"0");
        let file = snap.finish();

        let bands = bands();
        let seen: Vec<(Vec<u8>, Vec<u8>)> = Load::open(&file, bands.limits(), 0)
            .unwrap()
            .map(|item| match item.unwrap() {
                Item::Aux { name, value } => (name.into_owned(), value.into_owned()),
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(seen.len(), 3, "ours and the two the test wrote");
        assert_eq!(seen[0].0, b"yo-ver");
        assert_eq!(seen[1], (b"redis-ver".to_vec(), b"8.8.0".to_vec()));
        assert_eq!(seen[2], (b"aof-base".to_vec(), b"0".to_vec()));
    }

    #[test]
    fn a_file_with_more_keys_than_one_batch_comes_back_whole() {
        let mut d = db();
        let many = 1_000;
        for i in 0..many {
            let key = format!("k{i}");
            d.at(key.as_bytes())
                .set_plain(key.as_bytes(), b"v")
                .unwrap();
        }
        let mut snap = Snapshot::new();
        snap.database(0, &d);

        let after = into_dbs(&snap.finish(), 1_000);
        assert_eq!(after[0].1.len(), many);
    }

    /// A file written with the header this server writes is not the only file
    /// this server reads, and the version in the header is the thing that says
    /// so. Fourteen is a real Redis header and fifteen is the newest one read.
    #[test]
    fn a_newer_header_is_read_up_to_the_version_and_refused_past_it() {
        let mut snap = Snapshot::new();
        snap.database(0, &filled());
        let file = snap.finish();
        let bands = bands();

        // The checksum covers the header, so the version cannot be changed
        // without the sum being taken again.
        let stamped = |version: &[u8]| {
            let mut edited = file.clone();
            edited[5..9].copy_from_slice(version);
            let end = edited.len() - FOOTER;
            let crc = crc64(0, &edited[..end]);
            edited[end..].copy_from_slice(&crc.to_le_bytes());
            edited
        };

        for version in [&b"0006"[..], b"0011", b"0014", b"0015"] {
            let edited = stamped(version);
            let load = Load::open(&edited, bands.limits(), 1_000).unwrap();
            assert_eq!(
                load.filter(|item| matches!(item, Ok(Item::Key { .. })))
                    .count(),
                8,
                "{version:?}"
            );
        }

        assert_eq!(
            Load::open(&stamped(b"0016"), bands.limits(), 1_000).err(),
            Some(Fault::Version(16)),
            "the version is looked at before the checksum, which is right"
        );
    }

    #[test]
    fn a_file_that_is_not_one_is_turned_down_before_anything_is_built() {
        let bands = bands();
        let short = b"REDIS0012";
        assert_eq!(
            Load::open(short, bands.limits(), 0).err(),
            Some(Fault::Truncated),
            "a header with no room for a checksum"
        );
        let wrong = b"NOTAREDISFILE0000000";
        assert_eq!(
            Load::open(wrong, bands.limits(), 0).err(),
            Some(Fault::Magic)
        );
        let letters = b"REDISabcd0000000000";
        assert_eq!(
            Load::open(letters, bands.limits(), 0).err(),
            Some(Fault::Magic),
            "a version that is not four digits"
        );
    }

    #[test]
    fn a_damaged_file_is_refused_rather_than_half_loaded() {
        let mut snap = Snapshot::new();
        snap.database(0, &filled());
        let mut file = snap.finish();
        // One bit in the middle of a value, which is exactly what a checksum is
        // for and exactly what a reader would otherwise not notice until it had
        // already built half the dataset.
        let middle = file.len() / 2;
        file[middle] ^= 0x01;

        let bands = bands();
        assert_eq!(
            Load::open(&file, bands.limits(), 1_000).err(),
            Some(Fault::Checksum)
        );
    }

    #[test]
    fn a_file_cut_short_is_a_fault_and_not_a_shorter_dataset() {
        let mut snap = Snapshot::new();
        snap.database(0, &filled());
        let file = snap.finish();
        // Cut off the end marker and the checksum, then put a checksum back on
        // so that the truncation is the only thing wrong with it.
        let mut cut = file[..file.len() - 40].to_vec();
        let crc = crc64(0, &cut);
        cut.extend_from_slice(&crc.to_le_bytes());

        let bands = bands();
        let load = Load::open(&cut, bands.limits(), 1_000).unwrap();
        let faults: Vec<Fault> = load.filter_map(|item| item.err()).collect();
        assert_eq!(faults.len(), 1, "the walk stops at the first fault");
        assert!(
            matches!(faults[0], Fault::Truncated | Fault::Value),
            "{:?}",
            faults[0]
        );
    }

    /// The opcodes a file can carry that this server has nothing to do with, and
    /// the two it cannot get past.
    #[test]
    fn the_opcodes_that_are_bookkeeping_are_read_and_the_rest_are_refused() {
        let mut snap = Snapshot::new();
        snap.database(0, &filled());
        let good = snap.finish();

        // A file with an idle time, a frequency and a slot header in front of
        // the first key, which is what a cluster node with an eviction policy
        // writes and what nothing here has any use for.
        let mut edited = Vec::new();
        edited.extend_from_slice(&good[..HEADER]);
        edited.push(OP_IDLE);
        edited.push(40); // a six bit length
        edited.push(OP_FREQ);
        edited.push(255);
        edited.push(OP_SLOT_INFO);
        edited.extend_from_slice(&[7, 3, 0]);
        edited.extend_from_slice(&good[HEADER..good.len() - FOOTER]);
        let crc = crc64(0, &edited);
        edited.extend_from_slice(&crc.to_le_bytes());

        let bands = bands();
        let keys = Load::open(&edited, bands.limits(), 1_000)
            .unwrap()
            .filter(|item| matches!(item, Ok(Item::Key { .. })))
            .count();
        assert_eq!(keys, 8, "every key still arrived");

        for op in [OP_FUNCTION_PRE_GA, OP_MODULE_AUX] {
            let mut edited = Vec::new();
            edited.extend_from_slice(&good[..HEADER]);
            edited.push(op);
            edited.extend_from_slice(&good[HEADER..good.len() - FOOTER]);
            let crc = crc64(0, &edited);
            edited.extend_from_slice(&crc.to_le_bytes());
            let faults: Vec<Fault> = Load::open(&edited, bands.limits(), 1_000)
                .unwrap()
                .filter_map(|item| item.err())
                .collect();
            assert_eq!(faults, vec![Fault::Opcode(op)]);
        }
    }

    /// A file with no checksum, which is what a server writes when it has been
    /// told the time is better spent elsewhere.
    #[test]
    fn a_file_that_says_it_has_no_checksum_is_read_anyway() {
        let mut snap = Snapshot::new();
        snap.database(0, &filled());
        let mut file = snap.finish();
        let end = file.len() - FOOTER;
        file[end..].fill(0);

        let bands = bands();
        let keys = Load::open(&file, bands.limits(), 1_000)
            .unwrap()
            .filter(|item| matches!(item, Ok(Item::Key { .. })))
            .count();
        assert_eq!(keys, 8);
    }

    /// A file a real `redis-server` 8.10.1 wrote, ten keys and every type.
    ///
    /// Frozen here as hex because the whole point of it is bytes this project
    /// did not write. Every other test in this file checks the reader against
    /// the writer next door, which proves the two agree and proves nothing at
    /// all about whether either of them agrees with Redis. This one carries a
    /// header version this server never writes, a quicklist, three listpack
    /// collections, an intset, a hash with a field deadline in the newest shape
    /// and a stream with a group and a pending entry, every one of them in the
    /// encoding 8.10.1 picked for itself.
    ///
    /// It was made with SET, PEXPIREAT, RPUSH, SADD, HSET, HEXPIREAT, ZADD,
    /// XADD, XGROUP and XREADGROUP against a server built from the 8.10.1
    /// source, followed by SAVE.
    const REAL: &str = concat!(
        "524544495330303135fa0972656469732d76657206382e31302e31fa0a72656469732d62",
        "697473c040fa056374696d65c248a99f6afa08757365642d6d656dc2d0941f00fa08616f",
        "662d62617365c000fe00fb0a010001730568656c6c6f1001680d0d000000020081660281",
        "7602ff1902687800d8c32cbb0300001d1d00000006008161020101f400d8c32cbb030000",
        "0981620202010001ff0b04696e74730e020000000300000001000200030012016c010210",
        "100000000300816102816202816302ff11017a0f0f0000000200816d0283312e3504ff14",
        "0273740d0d0000000200817802817902fffc00d8c32cbb030000000374746c017600016e",
        "c287d612001b01780110000000000000000100000000000000011d1d0000000a00010100",
        "01010181660200010201000100018176020401ff01010101010000010101670101010100",
        "0000000000000100000000000000017b44ad7fa0010000010101637b44ad7fa00100007b",
        "44ad7fa001000001000000000000000100000000000000010040644064000000ffa2531e",
        "bc6edefb1f",
    );

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn a_file_a_real_redis_wrote_comes_back_whole() {
        let file = unhex(REAL);
        let bands = bands();
        let mut load = Load::open(&file, bands.limits(), 1_000).unwrap();
        assert_eq!(load.version(), 15, "the header 8.10.1 writes");

        let mut d = db();
        let mut aux = Vec::new();
        let mut kinds: Vec<(Vec<u8>, Kind)> = Vec::new();
        let mut deadline = None;
        for item in &mut load {
            match item.unwrap() {
                Item::Aux { name, value } => aux.push((name.into_owned(), value.into_owned())),
                Item::Library(_) => panic!("the file holds no functions"),
                Item::Key { db: n, key, record } => {
                    assert_eq!(n, 0);
                    kinds.push((key.to_vec(), record.kind()));
                    if &*key == b"ttl" {
                        deadline = record.expire_at();
                    }
                    d.at(&key).import(&key, record);
                }
            }
        }
        assert_eq!(load.expired(), 0);

        assert!(
            aux.contains(&(b"redis-ver".to_vec(), b"8.10.1".to_vec())),
            "{aux:?}"
        );
        kinds.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            kinds,
            vec![
                (b"h".to_vec(), Kind::Hash),
                (b"hx".to_vec(), Kind::Hash),
                (b"ints".to_vec(), Kind::Set),
                (b"l".to_vec(), Kind::List),
                (b"n".to_vec(), Kind::String),
                (b"s".to_vec(), Kind::String),
                (b"st".to_vec(), Kind::Set),
                (b"ttl".to_vec(), Kind::String),
                (b"x".to_vec(), Kind::Stream),
                (b"z".to_vec(), Kind::Zset),
            ]
        );
        // The year 2100, which is what the file was told and is far enough away
        // that this test does not start failing on a Tuesday.
        assert_eq!(deadline, Some(4_102_444_800_000));

        let mut text = Vec::new();
        d.at(b"s").get(b"s").unwrap().unwrap().write_to(&mut text);
        assert_eq!(text, b"hello");
        text.clear();
        d.at(b"n").get(b"n").unwrap().unwrap().write_to(&mut text);
        assert_eq!(text, b"1234567", "an integer encoded string");
        let list: Vec<Vec<u8>> = d
            .at(b"l")
            .lrange(b"l", 0, -1)
            .unwrap()
            .map(|e| e.to_vec())
            .collect();
        assert_eq!(list, vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]);
        let mut members: Vec<Vec<u8>> = d
            .at(b"st")
            .smembers(b"st")
            .unwrap()
            .unwrap()
            .map(|m| m.to_vec())
            .collect();
        members.sort();
        assert_eq!(members, vec![b"x".to_vec(), b"y".to_vec()]);
        assert_eq!(d.at(b"ints").scard(b"ints").unwrap(), 3, "an intset");
        assert_eq!(
            d.at(b"h")
                .hget(b"h", b"f", |v| v.map(|t| t.to_vec()))
                .unwrap(),
            Some(b"v".to_vec())
        );
        assert_eq!(d.at(b"z").zscore(b"z", b"m").unwrap(), Some(1.5));

        // The field deadline is the part of the newest hash shape that is easy
        // to read past, so it is asked for by name: one field has one and the
        // other does not.
        let mut asked = Vec::new();
        d.at(b"hx")
            .httl(b"hx", [b"a".as_slice(), b"b"].into_iter(), |ask| {
                asked.push(ask)
            })
            .unwrap();
        assert_eq!(asked, vec![Ask::At(4_102_444_800_000), Ask::NoDeadline]);

        let stream = d.at(b"x").stream(b"x").unwrap().expect("a stream");
        assert_eq!(stream.len(), 1);
        assert_eq!(stream.groups().count(), 1, "the group came back with it");
    }
}
