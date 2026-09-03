//! The file a running server puts cold values in.
//!
//! `14` section 4.1 inverts what a memory limit means: a server with a file
//! under it answers memory pressure by moving values to the file rather than by
//! deleting them. The engine has had both halves of that for a while, the tier
//! that moves values and the limit that decides when, and what it has not had is
//! a file. Everything above here was written against `yo_kv::cold::Blocks`,
//! which is an append that hands back an address and a read that takes one, and
//! this is the twenty lines that make a real `.yo` file answer to it.
//!
//! # Why this lives in the tool
//!
//! Because it is the first place both sides are allowed in the same binary. The
//! engine crates do not depend on the storage crates on purpose, so that the
//! memory engine can be tested without a file and the file can be tested without
//! an engine, and the seam between them is a trait with three methods. A program
//! is what decides to put the two together, and `yodb` is the program.
//!
//! # One log per database
//!
//! A `.yo` file holds a log per shard, and each of those believes it owns its
//! tail, so two databases cannot share one. Sixteen logs open at once would be
//! sixteen resident windows for a server that almost certainly uses database
//! zero, so [`Store::source`] hands one out per database on demand and the
//! engine only asks when that database is actually under memory pressure. A
//! server that is given a file and never fills memory never writes to it.
//!
//! # What durability means here
//!
//! Nothing that is only in memory survives a restart today, so a value that was
//! demoted has no more to lose than one that was not. The store is written with
//! [`Durability::None`], which appends into the page in memory and lets the log
//! write pages out as they fill, and it is not asked to sync. That is not a
//! statement about what a `.yo` file is worth, it is a statement about what this
//! particular file holds: a copy of something that would have been lost anyway.
//! When the keyspace itself is durable this becomes a real durability setting
//! and the reasoning here goes with it.
//!
//! For the same reason the file has to be a new one. What is in a store from a
//! previous run is reachable only through index entries that died with that
//! process, so reopening one would inherit bytes nothing can name. The path is
//! created and never opened, and a path that is already there is an error rather
//! than something to quietly write over, because the one thing worse than
//! refusing to start is truncating a file somebody meant to keep.

use std::path::Path;

use yo_common::{Addr, Code, Error, Result, Space};
use yo_file::{CreateOptions, LogFile, Yo};
use yo_format::{RecordHeader, RecordKind};
use yo_kv::cold::Blocks;
use yo_record::{Durability, Log, LogConfig};

/// How many databases get a log of their own, which is how many there are.
///
/// One per `SELECT` slot, so that a server using more than database zero is not
/// a server where the second one cannot migrate.
const SHARDS: u32 = 16;

/// A `.yo` file, waiting to be asked for logs.
pub struct Store {
    yo: Yo,
}

impl Store {
    /// Make the file at `path`, with a log waiting for every database.
    ///
    /// # Errors
    ///
    /// Whatever the file says, which includes the path already existing.
    pub fn create(path: &Path) -> Result<Store> {
        let yo = Yo::create(
            path,
            &CreateOptions {
                shard_count: SHARDS,
                ..CreateOptions::default()
            },
        )?;
        Ok(Store { yo })
    }

    /// A closure the engine can ask for database `at`'s store.
    ///
    /// Answers `None` for a database this file has no shard for, and for a log
    /// that will not open, because a database that cannot migrate evicts and
    /// that is a worse answer rather than a broken one.
    pub fn source(mut self) -> impl FnMut(usize) -> Option<Box<dyn Blocks>> {
        move |at| {
            let sink = self.yo.log(u32::try_from(at).ok()?).ok()?;
            let cfg = LogConfig {
                shard: u32::try_from(at).ok()?,
                durability: Durability::None,
                ..LogConfig::default()
            };
            let log = Log::new(cfg, sink).ok()?;
            Some(Box::new(Chunks { log }))
        }
    }
}

/// One database's log, wearing the interface the tier asks for.
struct Chunks {
    log: Log<LogFile>,
}

impl Blocks for Chunks {
    fn put(&mut self, bytes: &[u8]) -> Result<Addr> {
        // The kind the format already has a name for. A chunk of a value that
        // left memory and a chunk of a collection that did are the same thing
        // to a reader walking the file.
        let h = RecordHeader::new(RecordKind::CollectionChunk);
        let at = self.log.append(&h, b"", bytes)?;
        Ok(Addr::new(Space::Log, at.addr))
    }

    fn get(&self, at: Addr) -> Result<&[u8]> {
        if at.space() != Some(Space::Log) {
            return Err(Error::new(Code::Invalid, "that address is not in the log"));
        }
        Ok(self.log.read(at.offset())?.value)
    }

    fn bytes(&self) -> u64 {
        // From the oldest address the log still holds to where the next append
        // goes, which is what it occupies. Not `durable_upto`, which is where
        // the writes have got to: this log is not asked to sync and its pages
        // are 32 MiB, so a store holding thirty megabytes of demoted values
        // would report nothing at all and a storage limit would never fire.
        self.log.tail().saturating_sub(self.log.begin())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yo_kv::{Keyspace, Str};

    /// A file that goes away with the test.
    struct Tmp(std::path::PathBuf);

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn tmp(name: &str) -> Tmp {
        let mut p = std::env::temp_dir();
        p.push(format!("yo-store-{name}-{}.yo", std::process::id()));
        let _ = std::fs::remove_file(&p);
        Tmp(p)
    }

    #[test]
    fn a_value_written_to_the_file_reads_back() {
        let path = tmp("roundtrip");
        let store = Store::create(&path.0).expect("a file");
        let mut source = store.source();
        let mut blocks = source(0).expect("database zero has a log");

        let val = b"the quick brown fox jumps over the lazy dog".repeat(4);
        let at = blocks.put(&val).expect("written");
        assert_eq!(blocks.get(at).expect("read back"), &val[..]);
        assert!(blocks.bytes() > 0, "the log knows it holds something");
    }

    #[test]
    fn every_database_gets_a_log_of_its_own() {
        let path = tmp("shards");
        let store = Store::create(&path.0).expect("a file");
        let mut source = store.source();
        for at in 0..SHARDS as usize {
            assert!(source(at).is_some(), "database {at} has no log");
        }
        assert!(
            source(SHARDS as usize).is_none(),
            "and there is not a seventeenth"
        );
    }

    #[test]
    fn a_keyspace_demotes_into_the_file_and_still_answers() {
        // The whole point, end to end. A database with one of these attached
        // moves its values into a real file and every key still answers with
        // what was stored in it.
        let path = tmp("demote");
        let store = Store::create(&path.0).expect("a file");
        let mut source = store.source();
        let mut k = Keyspace::new();
        k.attach(source(0).expect("database zero has a log"));

        let val = b"a value long enough to be worth moving out of memory".repeat(2);
        for i in 0..200u32 {
            k.set_plain(&i.to_le_bytes(), &val).expect("stored");
        }
        let moved = k.relieve(usize::MAX).expect("swept");
        assert!(moved > 0, "nothing was moved");
        assert!(k.store_bytes().expect("attached") > 0, "the file is empty");

        for i in 0..200u32 {
            assert_eq!(
                k.get(&i.to_le_bytes()).expect("read"),
                Some(Str::Bytes(&val)),
                "key {i} did not read back off the file"
            );
        }
        assert_eq!(k.len(), 200, "a sweep moves values and not keys");
    }

    #[test]
    fn a_path_that_is_already_there_is_refused() {
        let path = tmp("existing");
        let first = Store::create(&path.0);
        assert!(first.is_ok(), "{:?}", first.err());
        assert!(
            Store::create(&path.0).is_err(),
            "the second start wrote over the first one's file"
        );
    }
}
