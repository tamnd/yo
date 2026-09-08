//! The fingerprint of a value, which is what `DEBUG DIGEST` compares.
//!
//! # What it is for
//!
//! Two servers holding the same data answer the same forty characters. That is
//! the whole purpose, and it is the reason the recipe below is copied to the
//! byte rather than improved on: a digest that is computed differently is not a
//! worse digest, it is a useless one, because the only thing anybody does with
//! the number is hold it next to another server's.
//!
//! Redis's own test suite leans on this the way it leans on `DEBUG RELOAD`.
//! After a replica catches up, after a reload, after an AOF rewrite, the check
//! is that the digest still matches, because comparing a million keys one at a
//! time over a socket is not a test anybody would run.
//!
//! # The two ways bytes go in
//!
//! [`xor`] hashes the bytes and exclusive ors the result into what is there.
//! Exclusive or does not care what order things arrive in, so this is how the
//! members of a set go in: a set is not ordered and a digest that changed when
//! the same members came back in a different order would report a difference
//! that is not one.
//!
//! [`mix`] does the same and then hashes the whole accumulator, so what comes
//! out depends on everything before it. That is how a list goes in, because a
//! list is ordered and two lists holding the same elements in different orders
//! really are different.
//!
//! The pair of them is what lets one recipe cover both, and the pattern for
//! anything that is a set of ordered things, which is a hash and a sorted set,
//! is to build the little digest of one pair with [`mix`] and then [`xor`] that
//! into the value's own.
//!
//! # What is not here
//!
//! The key name. `DEBUG DIGEST-VALUE` is about the value only, and the whole
//! dataset digest is the one that folds the name in, because two keys holding
//! the same value in the same database are a different dataset. So everything in
//! this module answers about a value and the caller adds the name.

use yo_common::dtoa;
use yo_common::num::{self, DIGITS_MAX, DOUBLE_MAX};
use yo_common::sha1;

use crate::array::Array;
use crate::hash::Hash;
use crate::list::List;
use crate::listpack::Entry;
use crate::set::Set;
use crate::stream::{Id, Stream};
use crate::zset::Zset;

/// Twenty bytes, which is a SHA-1 and is what everything here accumulates into.
pub type Digest = [u8; 20];

/// A value that has nothing in it yet.
pub const EMPTY: Digest = [0; 20];

/// The object type numbers a real server hashes in front of a value.
///
/// These are Redis's `OBJ_` constants and they are on the wire in the sense that
/// matters: a digest computed with the wrong number does not match. They are
/// also the reason a string and a one element list of the same bytes come out
/// different, which is the point of hashing them at all.
mod kinds {
    pub const STRING: u32 = 0;
    pub const LIST: u32 = 1;
    pub const SET: u32 = 2;
    pub const ZSET: u32 = 3;
    pub const HASH: u32 = 4;
    pub const MODULE: u32 = 5;
    pub const STREAM: u32 = 6;
    pub const ARRAY: u32 = 7;
}

/// Hash `bytes` and exclusive or the result in, which does not care about order.
pub fn xor(digest: &mut Digest, bytes: &[u8]) {
    for (into, byte) in digest.iter_mut().zip(sha1::digest(bytes)) {
        *into ^= byte;
    }
}

/// The same, and then hash the whole thing, which does care about order.
///
/// `digest = SHA1(digest xor SHA1(bytes))`, which is Redis's `mixDigest` and is
/// written this way rather than as a hash of the two run together. The
/// difference matters because it is the difference.
pub fn mix(digest: &mut Digest, bytes: &[u8]) {
    xor(digest, bytes);
    *digest = sha1::digest(digest);
}

/// The forty lowercase characters a client is answered.
#[must_use]
pub fn hex(digest: &Digest) -> [u8; 40] {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = [0u8; 40];
    for (pair, byte) in out.as_chunks_mut::<2>().0.iter_mut().zip(digest) {
        pair[0] = DIGITS[usize::from(byte >> 4)];
        pair[1] = DIGITS[usize::from(byte & 0xf)];
    }
    out
}

/// Fold a number in as the four big endian bytes a real server folds in.
///
/// Big endian because the reference calls `htonl` on it, and that is the whole
/// reason: on the machines anybody runs this on the number in memory is the
/// other way round, so writing it out the way it lies would give a digest that
/// matches nothing.
pub fn number(digest: &mut Digest, n: u32) {
    mix(digest, &n.to_be_bytes());
}

/// What a key with a deadline has folded in on top of its value.
///
/// A key that expires and the same key without a deadline are different, and
/// how long it has left is deliberately not part of it: two servers that agree
/// about a dataset disagree about the millisecond, and a digest that moved as
/// the clock ran would never match anything.
pub const EXPIRE: &[u8] = b"!!expire!!";

/// The same for a hash field that has one of its own.
const HASH_EXPIRE: &[u8] = b"!!hexpire!!";

/// One element as the bytes it reads as, which is what goes into the hash.
///
/// An integer element is written down as a number here and read back as text by
/// every command that answers one, and the text is what a real server hashes,
/// because on that side the element has already been turned back into a string
/// object by the time the digest sees it.
fn element(entry: &Entry<'_>, digest: &mut Digest, how: fn(&mut Digest, &[u8])) {
    match entry {
        Entry::Str(bytes) => how(digest, bytes),
        Entry::Int(n) => {
            let mut buf = [0u8; DIGITS_MAX];
            how(digest, num::i64_digits(&mut buf, *n));
        }
    }
}

/// A string, which is the whole value and nothing else.
pub fn string(digest: &mut Digest, bytes: &[u8]) {
    number(digest, kinds::STRING);
    mix(digest, bytes);
}

/// A list, front to back, where the order is the point.
pub fn list(digest: &mut Digest, list: &List) {
    number(digest, kinds::LIST);
    for entry in list.iter() {
        element(&entry, digest, mix);
    }
}

/// A set, where the order is not.
pub fn set(digest: &mut Digest, set: &Set) {
    number(digest, kinds::SET);
    for member in set.iter() {
        element(&member, digest, xor);
    }
}

/// A sorted set, which is a set of ordered pairs.
///
/// The member and the score are mixed into a digest of their own, in that order,
/// and that digest is exclusive ored in. So a member is bound to its score and
/// the members are still unordered between themselves, which is what a sorted
/// set is: `ZADD k 1 a 2 b` and `ZADD k 2 b 1 a` are the same value.
///
/// The score is written the way `fpconv_dtoa` writes one, which is the shortest
/// decimal that reads back as the same double, and it is deliberately not the
/// text `ZSCORE` answers. A reply goes through `d2string`, which spots a double
/// that is a whole number and prints it as an integer, so a score of two to the
/// sixtieth replies as its exact nineteen digits. The digest skips that step, so
/// the same score hashes as `1152921504606847000`, which is the shortest form
/// padded out. Both are the same double and only one of them is the number the
/// other server hashed.
pub fn zset(digest: &mut Digest, zset: &Zset) {
    number(digest, kinds::ZSET);
    zset.walk(0, zset.len(), false, |member, score| {
        let mut pair = EMPTY;
        element(&member, &mut pair, mix);
        let mut buf = [0u8; DOUBLE_MAX];
        let len = dtoa::dtoa(score, &mut buf);
        mix(&mut pair, &buf[..len]);
        xor(digest, &pair);
    });
}

/// A hash, on the same pattern, with a field that expires saying so.
pub fn hash(digest: &mut Digest, hash: &Hash) {
    number(digest, kinds::HASH);
    for i in 0..hash.len() {
        let Some((field, value)) = hash.at(i) else {
            continue;
        };
        let mut pair = EMPTY;
        element(&field, &mut pair, mix);
        element(&value, &mut pair, mix);
        // The word and not the deadline, for the reason [`EXPIRE`] gives.
        if hash.deadline_at(i).is_some() {
            xor(&mut pair, HASH_EXPIRE);
        }
        xor(digest, &pair);
    }
}

/// A stream, entry by entry, ordered throughout.
///
/// The ID goes in as `ms.seq` written out with a full stop between, then every
/// field and every value in the order they were added. What does not go in is
/// everything around the entries, which is the groups, the consumers, the
/// pending lists and the counters. That is the reference's choice rather than a
/// shortcut here, and it means two streams that hold the same entries under
/// different groups have the same digest.
pub fn stream(digest: &mut Digest, stream: &Stream) {
    number(digest, kinds::STREAM);
    stream.range(Id::MIN, Id::MAX, None, |at, fields| {
        // Two numbers and a full stop, on the stack, because a digest of a
        // large stream would otherwise allocate once per entry to write down
        // something that is never longer than forty one characters.
        let mut buf = [0u8; DIGITS_MAX * 2 + 1];
        let mut written = 0;
        for (part, sep) in [(at.ms, true), (at.seq, false)] {
            let mut digits = [0u8; DIGITS_MAX];
            let text = num::u64_digits(&mut digits, part);
            buf[written..written + text.len()].copy_from_slice(text);
            written += text.len();
            if sep {
                buf[written] = b'.';
                written += 1;
            }
        }
        mix(digest, &buf[..written]);
        for (field, value) in fields {
            element(&field, digest, mix);
            element(&value, digest, mix);
        }
        true
    });
}

/// A sparse array, every slot in order, with an empty one saying so.
///
/// The word for a hole is the reference's and it is a word rather than nothing
/// on purpose: an array of three values with a hole in the middle and an array
/// of the same three values packed together are different arrays.
pub fn array(digest: &mut Digest, array: &Array) {
    number(digest, kinds::ARRAY);
    for i in 0..array.len() {
        match array.get(i) {
            Some(value) => {
                let mut buf = [0u8; crate::array::ELEMENT_MAX];
                mix(digest, value.text(&mut buf));
            }
            None => mix(digest, b"(null)"),
        }
    }
}

/// A body this crate does not own, which is a graph or a vector index.
///
/// The type number and nothing after it. A real server does the same for a
/// module type that has no digest function registered, which is the same
/// position: something is under the key, the code doing the hashing cannot see
/// inside it, and saying so is better than pretending the key is not there.
pub fn foreign(digest: &mut Digest) {
    number(digest, kinds::MODULE);
}

#[cfg(test)]
mod tests {
    use super::{Digest, EMPTY, hex, mix, xor};

    /// The two ways in, and the difference between them, which is the one
    /// property everything else rests on.
    #[test]
    fn xoring_forgets_the_order_and_mixing_keeps_it() {
        let (mut ab, mut ba) = (EMPTY, EMPTY);
        xor(&mut ab, b"a");
        xor(&mut ab, b"b");
        xor(&mut ba, b"b");
        xor(&mut ba, b"a");
        assert_eq!(ab, ba);

        let (mut ab, mut ba) = (EMPTY, EMPTY);
        mix(&mut ab, b"a");
        mix(&mut ab, b"b");
        mix(&mut ba, b"b");
        mix(&mut ba, b"a");
        assert_ne!(ab, ba);
    }

    /// Exclusive or is its own undoing, which is what lets a set be walked in
    /// any order and what would let a member be taken back out.
    #[test]
    fn the_same_bytes_twice_leave_nothing_behind() {
        let mut d = EMPTY;
        xor(&mut d, b"member");
        assert_ne!(d, EMPTY);
        xor(&mut d, b"member");
        assert_eq!(d, EMPTY);
    }

    /// The forty characters, in the order a client reads them.
    #[test]
    fn the_hex_form_is_the_bytes_in_order() {
        let mut d: Digest = [0; 20];
        d[0] = 0x0a;
        d[19] = 0xff;
        let text = hex(&d);
        assert_eq!(&text[..2], b"0a");
        assert_eq!(&text[38..], b"ff");
    }
}
