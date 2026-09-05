//! The seam between a key changing and the search indexes hearing about it.
//!
//! `yo-search` knows what to do with a key that changed and `yo-kv` knows that
//! one did, and neither can reach the other: a hash command is handed one
//! database and the registry lives on the server. So the dispatcher is where
//! the two meet, and this is that meeting.
//!
//! Two directions. [`changed`] is a key that has just been written, which is
//! read back out of the keyspace and handed to every index that follows it.
//! [`scan`] is the other way round, an index that has just been made walking
//! every key that was already there.
//!
//! # Why the key is read again
//!
//! A hash command knows the fields it touched and that is not enough. A
//! document is read from nothing every time, so what an index needs is the
//! whole of what is under the key now, and `HDEL` of one field would otherwise
//! hand over nothing at all. Reading it back is one more lookup on a stripe
//! that was warm a moment ago, and it only happens when an index actually
//! follows the key, so a server with no indexes on it never pays for this.
//!
//! # Why the fields are copied
//!
//! Reading a hash holds its stripe, and writing the registry cannot happen with
//! a stripe held: another connection would be waiting on a lock while a
//! document is tokenized. So the fields come out into one buffer and the lock
//! goes, which costs one copy of a document per write and buys back the
//! contention that would otherwise land on whichever stripe is busiest.
//!
//! # One lock at a time
//!
//! The registry is behind its own lock and so is every stripe, and nothing here
//! ever holds both. That is not tidiness, it is the only thing keeping the two
//! orders apart: a write takes the registry to ask whether the key matters and
//! then the stripe to read it, and the scan would otherwise take the stripe to
//! walk the keys and then the registry to ask about each one, which is the same
//! pair the other way round and is how a deadlock is built. So the scan lists
//! the names first and asks about them afterwards.

use yo_kv::Db;
use yo_kv::hash::Text;
use yo_search::Source;

use super::Server;

/// A hash lifted out of the keyspace so an index can be handed it.
///
/// One buffer with the ends beside it rather than a vector of vectors, so a
/// hash of forty fields is two allocations and not eighty. The fields and the
/// values alternate, the way they arrived.
#[derive(Debug, Default)]
struct Document {
    /// Every field name and value, one after another.
    bytes: Vec<u8>,
    /// Where each of them ends.
    ends: Vec<usize>,
}

impl Document {
    /// Adds one field name or one value.
    ///
    /// An integer is written out in digits, because that is what it was when a
    /// client sent it and what the client gets back. A hash stores `5` as a
    /// number to save the room, and an index that saw `Int(5)` and a client
    /// that sees `"5"` have to agree.
    fn push(&mut self, text: Text<'_>) {
        match text {
            Text::Str(s) => self.bytes.extend_from_slice(s),
            Text::Int(n) => yo_common::num::push_i64(&mut self.bytes, n),
        }
        self.ends.push(self.bytes.len());
    }

    /// The pairs, in the order they were added.
    fn pairs(&self) -> Vec<(&[u8], &[u8])> {
        let mut at = 0;
        let mut out = Vec::with_capacity(self.ends.len() / 2);
        let mut parts = self.ends.iter().map(|&end| {
            let part = &self.bytes[at..end];
            at = end;
            part
        });
        while let (Some(field), Some(value)) = (parts.next(), parts.next()) {
            out.push((field, value));
        }
        out
    }
}

/// Reads a hash back, or `None` when the key is not there or is not a hash.
///
/// A key of the wrong type is `None` and not an error. An index `ON HASH` walks
/// past a string sitting under its prefix without a word and without counting a
/// failure, which is measured against a real server and is the opposite of the
/// obvious guess.
fn read(db: &Db, key: &[u8]) -> Option<Document> {
    let mut doc = Document::default();
    let mut held = db.hold(key);
    let found = held.hgetall(key, |field, value| {
        doc.push(field);
        doc.push(value);
    });
    drop(held);
    match found {
        Ok(true) => Some(doc),
        _ => None,
    }
}

/// One key has changed, so every index that follows it reads it again.
///
/// Called after the command has already written its reply, because indexing is
/// not something a client can be told went wrong: a document that will not read
/// is counted in `FT.INFO` and the `HSET` that caused it still answers `OK`.
pub(super) fn changed(server: &Server, db: usize, key: &[u8]) {
    // Two questions before any work. The first is a look at an empty vector on
    // nearly every server there will ever be, and the second is a walk over a
    // handful of short prefixes.
    let follows = {
        let search = server.search.lock();
        if !search.watching() {
            return;
        }
        search.follows(Source::Hash, key)
    };
    if !follows {
        // It could still be a key an index used to hold, which is what a
        // `RENAME` out of a prefix leaves behind, so it is erased rather than
        // ignored.
        server.search.lock().went(key);
        return;
    }
    // The read happens with the registry let go, and the lock is taken again to
    // write what it found.
    let doc = read(&server.dbs[db], key);
    let mut search = server.search.lock();
    match doc {
        Some(doc) => search.wrote(Source::Hash, key, &doc.pairs()),
        None => search.went(key),
    }
}

/// A fresh index reads every key that was already there.
///
/// One database and not all of them, which is the odd half of a pair. An index
/// follows a key by name across every database once it is running, so a `HSET`
/// on database one reaches an index made on database zero. The scan does not:
/// it reads the database `FT.CREATE` was run on and no other, so the same key
/// on database one is invisible until something writes to it. Both halves are
/// measured against 8.10.1, and the asymmetry is what falls out of a real
/// server walking one keyspace while its notifications are server wide.
///
/// Every key is listed before any of them is asked about, for two reasons. The
/// walk holds a stripe and reading a key back wants the same stripe, and asking
/// the registry with a stripe held is the lock order a write does not use. So
/// the names come out first and the prefixes are matched afterwards, which
/// costs a list of the names in one database on a command nobody sends twice.
pub(super) fn scan(server: &Server, db: usize, name: &[u8]) {
    if !server.search.lock().scanning(name) {
        return;
    }
    let mut keys = Vec::new();
    server.dbs[db].keys(|key| keys.push(key.to_vec()));
    {
        let search = server.search.lock();
        keys.retain(|key| search.wants(name, Source::Hash, key));
    }
    for key in keys {
        if let Some(doc) = read(&server.dbs[db], &key) {
            server
                .search
                .lock()
                .filled(name, Source::Hash, &key, &doc.pairs());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pairs come back in the order they went in, and an integer value
    /// comes back in digits.
    #[test]
    fn a_document_hands_back_the_pairs_it_was_given() {
        let mut doc = Document::default();
        doc.push(Text::Str(b"t"));
        doc.push(Text::Str(b"alpha"));
        doc.push(Text::Str(b"n"));
        doc.push(Text::Int(-42));
        assert_eq!(
            doc.pairs(),
            vec![(&b"t"[..], &b"alpha"[..]), (&b"n"[..], &b"-42"[..])]
        );
    }

    /// An empty hash is no pairs rather than a panic, and so is a buffer with
    /// a field and no value, which nothing should ever build.
    #[test]
    fn a_document_with_nothing_in_it_is_no_pairs() {
        assert!(Document::default().pairs().is_empty());
        let mut odd = Document::default();
        odd.push(Text::Str(b"t"));
        assert!(odd.pairs().is_empty());
    }
}
