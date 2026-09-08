//! `yodb restore`, which reads a Redis RDB file.
//!
//! The other end of `yodb dump --format rdb`, which is G16 in spec `07`, and the
//! half a person migrating actually cares about. Somebody with a `dump.rdb` off a
//! real Redis wants two things out of it, in this order: to know whether yo can
//! read it at all, and then to have a server holding it. This subcommand answers
//! the first and `yodb serve --restore` does the second, and both of them go
//! through the same walk in `yo_resp`, so an answer here is an answer about the
//! thing that will actually run.
//!
//! # Why reading and reporting is a command of its own
//!
//! Because it is the only one of the two that is safe to run on a file you are
//! not sure about, and because it says more than `redis-check-rdb` does.
//! `redis-check-rdb` walks the frame and checks the checksum, which tells you the
//! file is not damaged. This builds every value in it into a real keyspace, which
//! tells you the stronger thing: that every type and every encoding in the file
//! is one this build has a shape for. A file that passes here is a file
//! `yodb serve --restore` will start on.
//!
//! Nothing is written. The keyspace it builds lives as long as the command and
//! goes away with it, and the file is opened for reading.

use std::path::Path;

use yo_resp::dispatch::{Loaded, Server};

/// What went wrong, and which kind of wrong it was.
///
/// Two arms because they get different exit codes, the same two `yodb check`
/// uses. A file that could not be opened is the same class of mistake as an
/// argument that did not make sense, and the caller usually mistyped a path. A
/// file that opened and would not load is the answer the command was asked for,
/// and it is a failure rather than an error in running it.
#[derive(Debug)]
pub enum Trouble {
    /// The file could not be read at all.
    Unreadable(String),
    /// The file was read and would not load.
    Refused(String),
}

impl core::fmt::Display for Trouble {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Trouble::Unreadable(s) | Trouble::Refused(s) => f.write_str(s),
        }
    }
}

/// Read the file into a keyspace of its own and say what was in it.
///
/// # Errors
///
/// [`Trouble`] for a file that could not be read or would not load, carrying the
/// sentence to print either way.
pub fn restore(path: &Path) -> Result<Loaded, Trouble> {
    let image =
        std::fs::read(path).map_err(|e| Trouble::Unreadable(format!("{}: {e}", path.display())))?;
    // A server with one stripe per database, because nothing is serving out of
    // this one and a stripe is only ever contention between threads.
    let server = Server::new();
    // The flush is asked for even though there is nothing to flush, because the
    // load this reports on has to be the load that would happen, and a restore
    // into a fresh server is a load onto an empty keyspace either way.
    //
    // Wrapped the way every other caller wraps it: building a dataset is all
    // allocation, so a run with `YO_ALLOC=abort` set has to be told that this one
    // is meant rather than take the process down on the first key.
    yo_alloc::allow(|| server.load_image(&image, true))
        .map_err(|refused| Trouble::Refused(format!("{}: {refused}", path.display())))
}

/// What the command prints when it worked.
///
/// Every database that got a key gets a line, and a file that put everything in
/// database zero gets one line, which is nearly every file. The point of listing
/// them is the file that did not: a dataset spread over four databases is
/// something the person restoring needs to know before they connect a client to
/// database zero and find it empty.
pub fn report(done: &Loaded, out: &mut String) {
    use core::fmt::Write as _;
    let _ = writeln!(out, "RDB version {}", done.version);
    for (db, &keys) in done.keys.iter().enumerate() {
        if keys > 0 {
            let _ = writeln!(out, "db{db}: {keys} key{}", plural(keys));
        }
    }
    let total = done.total();
    let _ = write!(out, "{total} key{} loaded", plural(total));
    if done.expired > 0 {
        // Not a fault. A file holds the deadline a key was written with, and a
        // file that sat on a disk for a week is a file most of whose deadlines
        // have gone by. Redis drops them on load too, and reports the count under
        // `rdb_last_load_keys_expired`.
        let _ = write!(out, ", {} already expired and dropped", done.expired);
    }
    if done.libraries > 0 {
        let _ = write!(
            out,
            ", {} function librar{}",
            done.libraries,
            if done.libraries == 1 { "y" } else { "ies" }
        );
    }
    out.push('\n');
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

#[cfg(test)]
mod tests {
    use super::{report, restore};
    use yo_resp::dispatch::Loaded;

    /// A file that is not there is the caller's mistake, and the sentence says
    /// which file rather than leaving them to guess which of their paths it was.
    #[test]
    fn a_file_that_is_not_there_says_which_one() {
        let missing = std::env::temp_dir().join("yo-restore-nothing-here.rdb");
        let _ = std::fs::remove_file(&missing);
        let e = restore(&missing).expect_err("there is no file").to_string();
        assert!(e.contains("yo-restore-nothing-here.rdb"), "{e}");
    }

    /// Not an RDB at all, which is the common way to get this wrong: a `.yo`
    /// file, a tarball, or a `dump.rdb` that is actually an append only file.
    #[test]
    fn a_file_that_is_not_an_rdb_says_so() {
        let path = std::env::temp_dir().join(format!("yo-restore-junk-{}.rdb", std::process::id()));
        std::fs::write(&path, b"this is not a Redis dump, not even a little bit").unwrap();
        let e = restore(&path).expect_err("not an RDB").to_string();
        let _ = std::fs::remove_file(&path);
        assert!(e.contains("does not start with REDIS"), "{e}");
    }

    /// One line per database that got anything, and the total on the end.
    #[test]
    fn the_report_names_every_database_that_got_a_key() {
        let mut done = Loaded {
            version: 12,
            ..Loaded::default()
        };
        done.keys[0] = 3;
        done.keys[9] = 1;
        let mut out = String::new();
        report(&done, &mut out);
        assert_eq!(
            out,
            "RDB version 12\ndb0: 3 keys\ndb9: 1 key\n4 keys loaded\n"
        );
    }

    /// The two counts that are not keys, on the end of the same line, because
    /// they are both about the file rather than about the keyspace it made.
    #[test]
    fn the_report_says_what_the_file_carried_and_the_keyspace_did_not() {
        let mut done = Loaded {
            version: 11,
            expired: 2,
            libraries: 1,
            ..Loaded::default()
        };
        done.keys[0] = 1;
        let mut out = String::new();
        report(&done, &mut out);
        assert_eq!(
            out,
            "RDB version 11\ndb0: 1 key\n1 key loaded, 2 already expired and dropped, 1 function library\n"
        );
    }

    /// An empty file is a file, and reporting nought keys is the answer rather
    /// than a complaint. A server that was flushed before it saved wrote one.
    #[test]
    fn a_file_with_nothing_in_it_reports_nothing_in_it() {
        let done = Loaded {
            version: 12,
            ..Loaded::default()
        };
        let mut out = String::new();
        report(&done, &mut out);
        assert_eq!(out, "RDB version 12\n0 keys loaded\n");
    }
}
