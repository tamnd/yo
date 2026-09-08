//! Reading a whole RDB file back into the keyspace.
//!
//! The other direction from [`super::persist`], and the one that arrived second.
//! Writing a file is what makes the dataset readable by everything else in the
//! Redis world, and reading one is what makes everything else in the Redis world
//! readable here, which is the half a person migrating actually cares about.
//!
//! # Two callers, one walk
//!
//! `DEBUG RELOAD` writes the file and reads it straight back, so that a suite can
//! prove a value survives the round trip. `yodb serve --restore` reads a file
//! somebody else wrote before the port opens, so that a server starts up holding
//! a dataset that came off a real Redis. They want the same thing done to the
//! keyspace and they differ only in where the bytes came from and what gets
//! printed when it goes wrong, so the walk is here and the two of them bring
//! their own file and their own log line.
//!
//! # Why a file that goes wrong halfway leaves a mess
//!
//! Because that is what a real server does, and matching it is worth more than
//! improving on it. Everything cheap that can be wrong is answered before the
//! keyspace is touched at all: the magic, the version and the checksum are all
//! checked by [`Load::open`], which reads every byte of the file, so a damaged
//! file is refused with the dataset still standing. What is left after that is a
//! value that will not build, and the only way to be atomic about one of those
//! would be to build the whole dataset beside the live one and swap, which costs
//! twice the memory on the one path where memory is already the problem.

use yo_kv::restore::{Fault, Item, Load};

use super::{DATABASES, Server};

/// What a file put where.
///
/// Counts and not keys, because both callers report rather than inspect. What a
/// person wants to see after loading a file somebody handed them is how much of
/// it landed and whether any of it was dropped on the way, and the answer to the
/// second is what tells them the file is older than they thought.
#[derive(Debug, Default, Clone, Copy)]
pub struct Loaded {
    /// The version out of the header.
    pub version: u16,
    /// Keys that landed, per database.
    ///
    /// A fixed array because there are sixteen databases and there is no
    /// arrangement of a file that makes seventeen. Reported rather than summed
    /// because a file that put everything in database nine is a file somebody
    /// needs to know about before they wonder why `DBSIZE` says nought.
    pub keys: [usize; DATABASES],
    /// Keys the file carried that were already dead, so were read and dropped.
    pub expired: usize,
    /// Function libraries the file carried.
    pub libraries: usize,
}

impl Loaded {
    /// How many keys landed in all.
    #[must_use]
    pub fn total(&self) -> usize {
        self.keys.iter().sum()
    }
}

/// Why a load stopped.
///
/// Two arms, because there are two kinds of bad file. One is a file the reader
/// can say something specific about, which is [`Fault`], and the other is a file
/// that parsed perfectly and asked for a database this server does not have,
/// which is not the reader's business to refuse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refused {
    /// The reader would not go on, and said why.
    Fault(Fault),
    /// The file has a key in a database past the last one.
    Database(usize),
}

impl core::fmt::Display for Refused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Refused::Fault(fault) => write!(f, "{fault}"),
            Refused::Database(db) => write!(
                f,
                "the file has a key in database {db} and there are {DATABASES}"
            ),
        }
    }
}

impl Server {
    /// Build the dataset in `image` into this server.
    ///
    /// `flush` says whether what is already here goes first. Dropping it is what
    /// a reload wants, since the file it just wrote holds all of it, and keeping
    /// it is what `NOFLUSH` and a merge want.
    ///
    /// The flush takes the databases and leaves the search indexes alone, which
    /// is the one thing this does differently from `FLUSHALL`. An index is a
    /// schema and its own copy of what it has read, and on a reload every key it
    /// follows comes back under the same name with the same value, so dropping
    /// it would mean rebuilding it against a keyspace that already agrees with
    /// it. On a restore into an empty server there is no index to drop.
    ///
    /// Nothing here allocates on a command path by accident, so the caller wraps
    /// it in [`yo_alloc::allow`]: building a dataset is all allocation and that
    /// is what it is for.
    ///
    /// # Errors
    ///
    /// [`Refused`] for a file that will not parse or that wants a database this
    /// server has not got. The keyspace is untouched when the header, the
    /// version or the checksum is the problem, because those are answered before
    /// the flush, and half loaded when a value in the middle of the file is.
    pub fn load_image(&self, image: &[u8], flush: bool) -> Result<Loaded, Refused> {
        // Any stripe of any database carries the same four thresholds, and they
        // are copied out rather than borrowed because the keyspace they came off
        // is about to be written into.
        let bands = self.dbs[0].hold_stripe(0).bands();
        let mut load =
            Load::open(image, bands.limits(), self.clock.now_ms()).map_err(Refused::Fault)?;
        let mut done = Loaded {
            version: load.version(),
            ..Loaded::default()
        };
        if flush {
            for db in &self.dbs {
                db.clear();
            }
        }
        // By reference, because the count of keys the file dropped for having
        // died is on the reader and a walk that takes it cannot be asked after.
        for item in load.by_ref() {
            match item {
                Ok(Item::Key { db, key, record }) => {
                    let Some(into) = self.dbs.get(db) else {
                        return Err(Refused::Database(db));
                    };
                    into.hold(&key).import(&key, record);
                    done.keys[db] += 1;
                }
                Ok(Item::Library(_)) => done.libraries += 1,
                // The writer talking about itself. Redis writes `redis-ver`,
                // `redis-bits` and `ctime` and reads none of them back except to
                // log, and there is nothing here that would read them either.
                Ok(Item::Aux { .. }) => {}
                Err(fault) => return Err(Refused::Fault(fault)),
            }
        }
        done.expired = load.expired();
        self.persist.note_load(&done);
        Ok(done)
    }
}
