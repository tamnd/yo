//! How a string value sits in a record, and what comes back out of one.
//!
//! The map underneath stores opaque bytes against a key. Everything a string
//! needs beyond those bytes, which is its encoding and its expiry deadline, is
//! carried in a one byte header in front of them:
//!
//! ```text
//! +--------+-------------------------+-------------------+-----------+
//! | meta   | expire at, u64 LE       | access, 24 bits   | payload   |
//! | u8     | 8 bytes, only if tagged | 3 bytes, tagged   | see below |
//! +--------+-------------------------+-------------------+-----------+
//! ```
//!
//! One byte, and eight more only for a key that has a deadline, which most keys
//! do not. The alternative is a second lookup into a side table for the TTL, and
//! a second lookup is a second cache miss on a path whose whole budget is one.
//!
//! # The access field
//!
//! Three bytes saying when the key was last read, or how often, depending on
//! which eviction policy is in force. It is [`crate::access::Access`] and the
//! reasoning about what goes in it lives there.
//!
//! It is behind a tag bit the same way the deadline is, but unlike the deadline
//! every record written now has one. The bit is there so that a record written
//! before the field existed still reads back correctly rather than to make the
//! field optional, and it sits after the deadline for the same reason: the
//! deadline stays at offset one and everything that reads one goes on working.
//!
//! Three bytes on every key is a real cost and it is worth being straight about
//! why it is paid unconditionally rather than only under a policy that reads it.
//! A key written under `noeviction` and then read under `allkeys-lru` has to be
//! rankable, and it cannot become rankable later without the record growing,
//! which means moving it, on what is usually a read. Paying three bytes always
//! is the version where switching policy at runtime does the obvious thing.
//!
//! The payload depends on the encoding. An `int` holds the eight bytes of the
//! integer and not its digits, which is what makes `INCR` a probe, an add and a
//! store with no arena traffic at all (`08` section 2). An `embstr` and a `raw`
//! hold the bytes as given. The difference between those two is the name
//! `OBJECT ENCODING` reports and nothing else, which is also true in Redis:
//! `embstr` there means the value was allocated next to its object header, a
//! distinction `yo` does not have because every value is already next to its
//! key.
//!
//! # The type tag
//!
//! The meta byte also says which type the key holds, in three bits that were
//! spare. That is what `TYPE` reads and it is what a command will read before it
//! decides whether it is looking at its own type or at somebody else's, and both
//! of those want the answer to come out of the byte the lookup already fetched
//! rather than out of a second structure. A string is zero, so nothing written
//! before the tag existed reads back as anything else.
//!
//! # The tier tag
//!
//! The last bit says whether the payload is the value or a place on the file
//! where the value is. That is `14` section 4.1's tier tag, and the reason it is
//! a bit here rather than a lookup somewhere else is the number it exists to
//! make possible: at most 1.05 device reads for every point read. A point read
//! that has to consult a second structure to find out whether the value is
//! resident has already spent the miss it was trying to avoid.
//!
//! A demoted record keeps its deadline, its access field, its type and its
//! encoding, and adds a length. So `TTL`, `TYPE`, `OBJECT ENCODING`, `STRLEN`,
//! `EXISTS` and the whole of eviction go on answering at memory speed on a key
//! whose value is on the device, and only the commands that actually want the
//! bytes pay for them. [`write_cold_record`] writes one and [`cold`] is the
//! branch that reads it.

use crate::access::Access;
use yo_common::Addr;
use yo_common::num::{parse_i64, push_i64};

/// The longest value Redis calls `embstr` rather than `raw`.
///
/// Clients and test suites read `OBJECT ENCODING` and assert on the boundary,
/// so it is 44 here because it is 44 there (`12` section 2).
pub const EMBSTR_MAX: usize = 44;

/// What `OBJECT ENCODING` calls a string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// The value is an integer, held as an integer.
    Int,
    /// A short string, at or under [`EMBSTR_MAX`] bytes.
    Embstr,
    /// Everything else.
    Raw,
}

impl Encoding {
    /// The string `OBJECT ENCODING` returns.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Encoding::Int => "int",
            Encoding::Embstr => "embstr",
            Encoding::Raw => "raw",
        }
    }

    /// The encoding Redis would choose for these bytes.
    ///
    /// Integer first, because `SET k 42` is int encoded in Redis whatever the
    /// length, then the `embstr` boundary. The integer test is Redis's own
    /// `string2ll`, which refuses a leading zero, a leading plus and `-0`, so
    /// `SET k 007` stays a three byte string and gives back `007`.
    #[inline]
    pub fn of(bytes: &[u8]) -> Encoding {
        if parse_i64(bytes).is_some() {
            Encoding::Int
        } else if bytes.len() <= EMBSTR_MAX {
            Encoding::Embstr
        } else {
            Encoding::Raw
        }
    }
}

/// What `TYPE` calls a key, and what the meta byte's tag holds.
///
/// The numbers are the same numbers `yo_format::ValueType` uses on disk, so that
/// saving a key is a copy of the tag rather than a translation of it. There is a
/// test at the bottom of this file holding the two in step, and it takes a dev
/// dependency on `yo-format` for no other reason.
///
/// The list is shorter than the on disk one because some of those are the same
/// thing in memory. A bitmap is a string and a HyperLogLog is a string, in Redis
/// as much as here, and `TYPE` on either answers `string`. The catalog draws
/// finer lines because a reader wants to know what a blob meant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// A string, and everything Redis stores as one.
    String = 0,
    /// A hash.
    Hash = 1,
    /// A set.
    Set = 2,
    /// A sorted set.
    Zset = 3,
    /// A list.
    List = 4,
    /// A stream.
    Stream = 5,
    /// A sparse array.
    Array = 6,
    /// A body this crate does not know the type of.
    ///
    /// The last of the eight patterns the tag can hold, and it is spent on an
    /// escape rather than on one more type. Everything below this crate is a
    /// primitive that the document, vector and graph engines are built out of,
    /// so those engines sit above it and cannot be named here without a cycle,
    /// and there are three of them and one pattern. An escape holds all three,
    /// and the body says what it is when asked.
    ///
    /// It is the one kind with no `yo_format::catalog::ValueType` beside it,
    /// because this version's writer does not save a foreign body and there is
    /// nothing on disk for the catalog to name. When one of them becomes
    /// saveable it gets its own number there, which does not have to be this
    /// one, since nothing in a file has to agree with a tag that only ever
    /// exists in memory.
    Foreign = 7,
}

impl Kind {
    /// The word `TYPE` replies with.
    ///
    /// A foreign body knows its own word and this does not, so the answer here
    /// is the one a caller sees when it has a kind and no body. Go through
    /// [`crate::Keyspace::type_name`] to get what the client should be told.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Kind::String => "string",
            Kind::Hash => "hash",
            Kind::Set => "set",
            Kind::Zset => "zset",
            Kind::List => "list",
            Kind::Stream => "stream",
            Kind::Array => "array",
            Kind::Foreign => "foreign",
        }
    }

    /// Whether the body lives in a slab rather than in the record.
    ///
    /// Everything but a string, which is its own record. The arms that walk a
    /// slot ask this rather than listing six kinds, so a new body kind is one
    /// arm here and not six lists to find.
    #[inline]
    pub const fn is_body(self) -> bool {
        !matches!(self, Kind::String)
    }

    /// The kind for a three bit tag.
    ///
    /// All eight patterns are spoken for now, and zero is `String`, so this is
    /// total and no longer has a fallback to explain.
    #[inline]
    const fn from_bits(bits: u8) -> Kind {
        match bits {
            KIND_HASH => Kind::Hash,
            KIND_SET => Kind::Set,
            KIND_ZSET => Kind::Zset,
            KIND_LIST => Kind::List,
            KIND_STREAM => Kind::Stream,
            KIND_ARRAY => Kind::Array,
            KIND_FOREIGN => Kind::Foreign,
            _ => Kind::String,
        }
    }
}

/// Bits 0 and 1 of the meta byte: which encoding.
const ENC_MASK: u8 = 0b0000_0011;
const ENC_INT: u8 = 0;
const ENC_EMBSTR: u8 = 1;
const ENC_RAW: u8 = 2;
/// Bit 2: whether eight bytes of deadline follow the meta byte.
const HAS_EXPIRY: u8 = 0b0000_0100;
/// Bit 6: whether three bytes of access data follow the deadline.
///
/// Everything this crate writes now sets it, and the bit exists so that a file
/// written before it did still opens. A record with the bit clear has no access
/// field and its payload starts where it always did, which is what makes reading
/// an older file a matter of asking rather than of knowing which version wrote
/// it.
///
/// It does not work in the other direction. A binary from before this bit reads
/// a record that sets it, ignores the bit it does not know about, and takes the
/// three access bytes for the front of the payload. There is no format version
/// in the file to refuse on, which is worth fixing and is not this change.
const HAS_ACCESS: u8 = 0b0100_0000;
/// Bit 7: whether the payload is a place on the file rather than a value.
///
/// This is the tier tag `14` section 4.1 needs, and the reason it is a bit in
/// the meta byte rather than a side table is the whole point of it. A point read
/// has already paid for the cache line the record starts in by the time it looks
/// at anything, so asking whether the value is resident is free, and asking any
/// other way is a second miss on a path whose budget is one. That is what makes
/// `1.05` device reads per point read a reachable number rather than `2.05`.
///
/// A cold record keeps its deadline and its access field in memory. Expiry has
/// to work on a key whose value is on the device without reading the device, and
/// so does eviction, or the policy would have to fault in the very keys it is
/// deciding to get rid of.
const COLD: u8 = 0b1000_0000;
/// Bits 3, 4 and 5: which type the key holds.
///
/// String is zero, so every record written before the tag existed reads back as
/// a string, which is what it was.
const KIND_MASK: u8 = 0b0011_1000;
const KIND_SHIFT: u32 = 3;
const KIND_HASH: u8 = 1;
const KIND_SET: u8 = 2;
const KIND_ZSET: u8 = 3;
const KIND_LIST: u8 = 4;
const KIND_STREAM: u8 = 5;
const KIND_ARRAY: u8 = 6;
const KIND_FOREIGN: u8 = 7;

/// Bytes of integer payload, which is a whole `i64` and never its digits.
const INT_LEN: usize = 8;

/// Bytes of access data, which is [`Access`] and is twenty four bits.
const ACCESS_LEN: usize = 3;

/// The meta byte in front of every stored string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Meta(u8);

impl Meta {
    /// Build the byte for a type, an encoding and the presence of a deadline.
    #[inline]
    pub const fn new(kind: Kind, enc: Encoding, has_expiry: bool) -> Meta {
        let bits = match enc {
            Encoding::Int => ENC_INT,
            Encoding::Embstr => ENC_EMBSTR,
            Encoding::Raw => ENC_RAW,
        };
        Meta(bits | ((kind as u8) << KIND_SHIFT) | if has_expiry { HAS_EXPIRY } else { 0 })
    }

    /// The byte for a string, which is what everything in `strings.rs` writes.
    #[inline]
    pub const fn string(enc: Encoding, has_expiry: bool) -> Meta {
        Meta::new(Kind::String, enc, has_expiry)
    }

    /// The byte for a value that lives in a slab, with the record holding a
    /// number that says where.
    ///
    /// The encoding bits are written as zero and mean nothing here. A set's
    /// encoding is which of the three representations it is in, and that is a
    /// property of the body and not of the record, so `OBJECT ENCODING` follows
    /// the number and asks. Keeping a copy of it in these two bits would want
    /// the record rewritten every time a set was promoted, for a command nobody
    /// calls in a loop, and two places to disagree about the same fact.
    #[inline]
    pub const fn slot(kind: Kind, has_expiry: bool) -> Meta {
        Meta::new(kind, Encoding::Int, has_expiry)
    }

    /// Read the byte back.
    ///
    /// An unknown encoding is impossible from our own writer, so the two spare
    /// bit patterns fall to `raw`, which is the reading that returns the bytes
    /// unchanged rather than reinterpreting them as something else.
    #[inline]
    pub const fn from_byte(b: u8) -> Meta {
        Meta(b)
    }

    /// The raw byte, as stored.
    #[inline]
    pub const fn byte(self) -> u8 {
        self.0
    }

    /// Which encoding this value carries.
    #[inline]
    pub const fn encoding(self) -> Encoding {
        match self.0 & ENC_MASK {
            ENC_INT => Encoding::Int,
            ENC_EMBSTR => Encoding::Embstr,
            _ => Encoding::Raw,
        }
    }

    /// Which type this key holds.
    #[inline]
    pub const fn kind(self) -> Kind {
        Kind::from_bits((self.0 & KIND_MASK) >> KIND_SHIFT)
    }

    /// Whether a deadline follows.
    #[inline]
    pub const fn has_expiry(self) -> bool {
        self.0 & HAS_EXPIRY != 0
    }

    /// Whether an access field follows the deadline.
    ///
    /// Everything this crate writes sets it, so in a running server it is always
    /// true and the reader that checks it is checking something that cannot
    /// happen. It is here anyway because a bit in the byte costs nothing and the
    /// alternative was a flag day: without it, the day the field arrived, every
    /// reader had to agree with every writer at exactly the same moment.
    ///
    /// It is not a file format concern. These records live in the arena and
    /// never reach a file, and the on disk record in `yo_format` has its own
    /// layout and its own versioning.
    #[inline]
    pub const fn has_access(self) -> bool {
        self.0 & HAS_ACCESS != 0
    }

    /// The same byte with the access field declared.
    #[inline]
    const fn with_access(self) -> Meta {
        Meta(self.0 | HAS_ACCESS)
    }

    /// Whether the payload is a place on the file rather than the value.
    ///
    /// The one question a point read asks before it decides whether it is going
    /// to touch a device, and the answer comes out of the byte the lookup has
    /// already fetched.
    #[inline]
    pub const fn is_cold(self) -> bool {
        self.0 & COLD != 0
    }

    /// The same byte with the value declared to be on the file.
    #[inline]
    pub const fn with_cold(self) -> Meta {
        Meta(self.0 | COLD)
    }

    /// Where the access field starts, counting from the meta byte.
    ///
    /// After the deadline rather than before it, which is what keeps a record
    /// written before this field existed readable: the deadline is still at
    /// offset one and everything that reads one can go on doing so.
    #[inline]
    const fn access_at(self) -> usize {
        if self.has_expiry() { 1 + 8 } else { 1 }
    }

    /// Where the payload starts, counting from the meta byte.
    #[inline]
    pub const fn payload_at(self) -> usize {
        self.access_at() + if self.has_access() { ACCESS_LEN } else { 0 }
    }
}

/// A stored value, read back out of a record.
///
/// The point of the two arms is that neither of them copies. An integer comes
/// back as an integer and is written into the reply buffer as digits at the
/// moment the reply is built, and a string comes back as a slice of the record
/// it lives in. Y18 asks for the reply to be constructed once in wire form, and
/// a `Vec<u8>` in the middle of that is the thing it is asking to remove.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Str<'a> {
    /// An int encoded value.
    Int(i64),
    /// Everything else, as it lies in the record.
    Bytes(&'a [u8]),
}

impl Str<'_> {
    /// How many bytes this value is as a string, which is what `STRLEN` returns.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Str::Int(n) => yo_common::num::i64_len(*n),
            Str::Bytes(b) => b.len(),
        }
    }

    /// Whether the value is the empty string.
    #[inline]
    pub fn is_empty(&self) -> bool {
        match self {
            // No integer writes as no digits.
            Str::Int(_) => false,
            Str::Bytes(b) => b.is_empty(),
        }
    }

    /// Append the string form to a buffer, which for an integer is its digits.
    #[inline]
    pub fn write_to(&self, out: &mut Vec<u8>) {
        match self {
            Str::Int(n) => push_i64(out, *n),
            Str::Bytes(b) => out.extend_from_slice(b),
        }
    }

    /// The string form, copied. For the reply path prefer [`Str::write_to`].
    pub fn to_vec(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(self.len());
        self.write_to(&mut v);
        v
    }

    /// The integer this value is, if it is one.
    ///
    /// A value can be an integer without being int encoded: `APPEND` and
    /// `SETRANGE` leave a `raw` string behind, and `INCR` on `"10"` built that
    /// way is 11 in Redis. So the bytes are parsed rather than the encoding
    /// being trusted.
    #[inline]
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Str::Int(n) => Some(*n),
            Str::Bytes(b) => parse_i64(b),
        }
    }

    /// The XXH3 of the value's string form, which is Redis's `DIGEST`.
    ///
    /// An int encoded value hashes its digits and not the eight bytes the
    /// record holds, because the digest a client compares against is the digest
    /// of what a client would have read.
    #[must_use]
    pub fn digest(&self) -> u64 {
        match self {
            Str::Bytes(b) => yo_common::xxh3::hash64(b),
            // Twenty bytes at the most, and the alternative is a second
            // formatter that writes into a stack buffer for a path nobody calls
            // in a loop.
            Str::Int(_) => yo_common::xxh3::hash64(&self.to_vec()),
        }
    }

    /// Whether this value's string form is exactly `want`.
    ///
    /// `IFEQ` compares against what the client would have read, so an int
    /// encoded 42 is equal to `"42"` and not to `"042"`. Doing that without
    /// materialising the digits is why the integer arm exists.
    #[inline]
    pub(crate) fn eq_bytes(&self, want: &[u8]) -> bool {
        match self {
            Str::Bytes(b) => *b == want,
            Str::Int(n) => parse_i64(want) == Some(*n),
        }
    }
}

/// How many bytes a record holding this value will occupy.
///
/// The access field is counted unconditionally, because every record this crate
/// writes now carries one. It is not a parameter for that reason: making it one
/// would put a flag through twenty six call sites to describe something none of
/// them gets to decide.
#[inline]
pub fn record_len(enc: Encoding, payload: usize, has_expiry: bool) -> usize {
    let head = (if has_expiry { 1 + 8 } else { 1 }) + ACCESS_LEN;
    head + if enc == Encoding::Int {
        INT_LEN
    } else {
        payload
    }
}

/// Write a whole record into `out`, which must be exactly [`record_len`] long.
///
/// `bytes` is the string as the caller gave it. When `enc` is [`Encoding::Int`]
/// the digits are not stored, the integer they parse to is, and the caller has
/// already established that they parse by choosing that encoding.
#[inline]
pub fn write_record(out: &mut [u8], enc: Encoding, bytes: &[u8], expire_at: Option<u64>) {
    out[0] = Meta::string(enc, expire_at.is_some()).with_access().byte();
    let mut at = 1;
    if let Some(ms) = expire_at {
        out[at..at + 8].copy_from_slice(&ms.to_le_bytes());
        at += 8;
    }
    at += write_blank_access(&mut out[at..]);
    match enc {
        Encoding::Int => {
            let n =
                parse_i64(bytes).expect("int encoding was chosen for bytes that are not an int");
            out[at..at + INT_LEN].copy_from_slice(&n.to_le_bytes());
        }
        _ => out[at..].copy_from_slice(bytes),
    }
}

/// Write a record whose value is an integer the caller already has.
#[inline]
pub fn write_int_record(out: &mut [u8], n: i64, expire_at: Option<u64>) {
    out[0] = Meta::string(Encoding::Int, expire_at.is_some())
        .with_access()
        .byte();
    let mut at = 1;
    if let Some(ms) = expire_at {
        out[at..at + 8].copy_from_slice(&ms.to_le_bytes());
        at += 8;
    }
    at += write_blank_access(&mut out[at..]);
    out[at..at + INT_LEN].copy_from_slice(&n.to_le_bytes());
}

/// Leave room for the access field and put nothing in it.
///
/// The writers here do not know the clock or which policy is in force, and a
/// record layout is the wrong place to learn either. The keyspace stamps the
/// field through [`set_access`] once the record is in, which is also where the
/// decision about whether to stamp at all belongs.
///
/// Zero is the most evictable value a key can hold under either reading, which
/// is the right way round for a default: a key that somehow never got stamped
/// goes first rather than never.
#[inline]
fn write_blank_access(out: &mut [u8]) -> usize {
    out[..ACCESS_LEN].fill(0);
    ACCESS_LEN
}

/// Bytes of slab number, which is how a record points at a body.
const SLOT_LEN: usize = 4;

/// How many bytes a record pointing at a slab slot occupies.
#[inline]
pub fn slot_record_len(has_expiry: bool) -> usize {
    (if has_expiry { 1 + 8 } else { 1 }) + ACCESS_LEN + SLOT_LEN
}

/// Write a record that points at `slot` in the slab for `kind`.
///
/// `out` must be exactly [`slot_record_len`] long.
#[inline]
pub fn write_slot_record(out: &mut [u8], kind: Kind, slot: u32, expire_at: Option<u64>) {
    out[0] = Meta::slot(kind, expire_at.is_some()).with_access().byte();
    let mut at = 1;
    if let Some(ms) = expire_at {
        out[at..at + 8].copy_from_slice(&ms.to_le_bytes());
        at += 8;
    }
    at += write_blank_access(&mut out[at..]);
    out[at..at + SLOT_LEN].copy_from_slice(&slot.to_le_bytes());
}

/// Bytes of file address and value length in a demoted record.
///
/// The length is here rather than only on the device because `STRLEN`, `TYPE`,
/// `OBJECT ENCODING`, `TTL` and `MEMORY USAGE` all have answers that do not need
/// the bytes, and a tiering layer that faults a value in to answer `STRLEN` is a
/// tiering layer that has given the game away. Four bytes caps a demoted value
/// at four gigabytes, which is above the proto-max-bulk-len a server will accept
/// in the first place.
const COLD_LEN: usize = 8 + 4;

/// How many bytes a record pointing at a demoted value occupies.
///
/// Twelve bytes of payload against however many the value was, which is the
/// whole reason demotion buys anything. A key with an eight byte value is not
/// worth demoting and a caller that demotes one will find its memory going up.
#[inline]
pub fn cold_record_len(has_expiry: bool) -> usize {
    (if has_expiry { 1 + 8 } else { 1 }) + ACCESS_LEN + COLD_LEN
}

/// Write a record saying the value is at `at` on the file and is `len` bytes.
///
/// `out` must be exactly [`cold_record_len`] long. `kind` and `enc` are carried
/// across from the record being replaced so that the questions which can be
/// answered without the device still can be.
#[inline]
pub fn write_cold_record(
    out: &mut [u8],
    kind: Kind,
    enc: Encoding,
    at: Addr,
    len: u32,
    expire_at: Option<u64>,
) {
    out[0] = Meta::new(kind, enc, expire_at.is_some())
        .with_access()
        .with_cold()
        .byte();
    let mut i = 1;
    if let Some(ms) = expire_at {
        out[i..i + 8].copy_from_slice(&ms.to_le_bytes());
        i += 8;
    }
    i += write_blank_access(&mut out[i..]);
    out[i..i + 8].copy_from_slice(&at.to_bits().to_le_bytes());
    out[i + 8..i + 12].copy_from_slice(&len.to_le_bytes());
}

/// Where a demoted value is and how big it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cold {
    /// Where on the file the value lives.
    pub at: Addr,
    /// How many bytes it is, so the reader can size its buffer in one go and
    /// the commands that only want the length never go and look.
    pub len: u32,
}

/// Where the value of a demoted record is, or `None` if the record is resident.
///
/// This is the branch every point read makes. `None` is the common answer and
/// it costs one test of a bit in a byte that has already been fetched.
#[inline]
#[must_use]
pub fn cold(rec: &[u8]) -> Option<Cold> {
    let m = Meta::from_byte(rec[0]);
    if !m.is_cold() {
        return None;
    }
    let i = m.payload_at();
    let mut a = [0u8; 8];
    a.copy_from_slice(&rec[i..i + 8]);
    let mut l = [0u8; 4];
    l.copy_from_slice(&rec[i + 8..i + 12]);
    Some(Cold {
        at: Addr::from_bits(u64::from_le_bytes(a)),
        len: u32::from_le_bytes(l),
    })
}

/// How long the value in a record is, without reading it.
///
/// The point of the tier tag seen from the command side: `STRLEN` on a demoted
/// key is the same cost as `STRLEN` on a resident one. `None` means the record
/// holds something whose length is not a string length, which is every
/// collection, and those keep their length in their body.
#[inline]
#[must_use]
pub fn str_len(rec: &[u8]) -> Option<usize> {
    let m = Meta::from_byte(rec[0]);
    if m.kind() != Kind::String {
        return None;
    }
    match cold(rec) {
        Some(c) => Some(c.len as usize),
        None => Some(read(rec).len()),
    }
}

/// What the access field in a record says, or `None` if it has no room for one.
///
/// `None` means the record predates the field, which is a key that was written
/// by an older build and read back out of a file. It is not an error and the
/// caller should treat it as a key it knows nothing about rather than as a key
/// that has never been touched.
#[inline]
#[must_use]
pub fn access(rec: &[u8]) -> Option<Access> {
    let m = Meta::from_byte(rec[0]);
    if !m.has_access() {
        return None;
    }
    let at = m.access_at();
    Some(Access::from_bits(u32::from_le_bytes([
        rec[at],
        rec[at + 1],
        rec[at + 2],
        0,
    ])))
}

/// Stamp the access field, in place, over whatever was there.
///
/// Returns false for a record with no room, which is the same older record
/// [`access`] answers `None` for. It is not worth growing one to make room: the
/// record would have to move, on a path that is usually a read, and the next
/// write to that key rewrites it with a field anyway.
#[inline]
pub fn set_access(rec: &mut [u8], a: Access) -> bool {
    let m = Meta::from_byte(rec[0]);
    if !m.has_access() {
        return false;
    }
    let at = m.access_at();
    rec[at..at + ACCESS_LEN].copy_from_slice(&a.bits().to_le_bytes()[..ACCESS_LEN]);
    true
}

/// The slab number in a record that has one.
///
/// # Panics
///
/// If the record is not one [`write_slot_record`] wrote, which is a caller that
/// did not read the kind first.
#[inline]
pub fn slot(rec: &[u8]) -> u32 {
    let at = Meta::from_byte(rec[0]).payload_at();
    let mut b = [0u8; SLOT_LEN];
    b.copy_from_slice(&rec[at..at + SLOT_LEN]);
    u32::from_le_bytes(b)
}

/// The type a record holds.
#[inline]
pub fn kind(rec: &[u8]) -> Kind {
    Meta::from_byte(rec[0]).kind()
}

/// The deadline in a record, if it has one.
#[inline]
pub fn expire_at(rec: &[u8]) -> Option<u64> {
    let m = Meta::from_byte(rec[0]);
    if !m.has_expiry() {
        return None;
    }
    let mut b = [0u8; 8];
    b.copy_from_slice(&rec[1..9]);
    Some(u64::from_le_bytes(b))
}

/// Whether a record carries a deadline at all, without reading it.
///
/// One byte where [`expire_at`] reads nine, which is the difference between a
/// question worth asking on every write and one that is not. The count of keys
/// with deadlines is kept up to date on every record written and every record
/// deleted, and all it ever needs is this bit.
#[inline]
pub fn has_expiry(rec: &[u8]) -> bool {
    Meta::from_byte(rec[0]).has_expiry()
}

/// Whether a record's deadline has passed at `now_ms`.
///
/// A deadline exactly equal to now has passed, which is Redis's reading: a key
/// set to expire at time T is gone at time T.
#[inline]
pub fn is_expired(rec: &[u8], now_ms: u64) -> bool {
    match expire_at(rec) {
        Some(at) => at <= now_ms,
        None => false,
    }
}

/// The value in a record.
///
/// # Panics
///
/// In a debug build, if the record is demoted. A demoted record's payload is a
/// file address and reading it as a value would hand back twelve bytes of
/// address as though they were the string, which is the kind of wrong that
/// looks like data corruption three layers away. Callers check [`cold`] first,
/// and that check is the branch they were going to make anyway.
#[inline]
pub fn read(rec: &[u8]) -> Str<'_> {
    let m = Meta::from_byte(rec[0]);
    debug_assert!(!m.is_cold(), "read on a record whose value is on the file");
    let at = m.payload_at();
    match m.encoding() {
        Encoding::Int => {
            let mut b = [0u8; INT_LEN];
            b.copy_from_slice(&rec[at..at + INT_LEN]);
            Str::Int(i64::from_le_bytes(b))
        }
        _ => Str::Bytes(&rec[at..]),
    }
}

/// The integer in an int encoded record, and where its bytes start.
///
/// Returns `None` for any other encoding. This is the read half of `INCR`'s
/// fast path, and the offset it hands back is what the write half stores into.
#[inline]
pub fn read_int_in_place(rec: &[u8]) -> Option<(i64, usize)> {
    let m = Meta::from_byte(rec[0]);
    if m.is_cold() || m.encoding() != Encoding::Int {
        return None;
    }
    let at = m.payload_at();
    let mut b = [0u8; INT_LEN];
    b.copy_from_slice(&rec[at..at + INT_LEN]);
    Some((i64::from_le_bytes(b), at))
}

/// The bytes of a raw record, to be written over where they lie.
///
/// `None` for the other two encodings, and that is not a missing case: a bitmap
/// write turns an int or an embstr into a raw string, which means moving it, so
/// there is nothing here for those two to write into. `OBJECT ENCODING` says
/// `raw` after a `SETBIT` on a value that was `int` a moment before, which is
/// the same rule seen from the outside.
#[inline]
pub fn raw_in_place(rec: &mut [u8]) -> Option<&mut [u8]> {
    let m = Meta::from_byte(rec[0]);
    if m.is_cold() || m.encoding() != Encoding::Raw {
        return None;
    }
    let at = m.payload_at();
    Some(&mut rec[at..])
}

/// Store `n` back over an int payload that starts at `at`.
#[inline]
pub fn write_int_in_place(rec: &mut [u8], at: usize, n: i64) {
    rec[at..at + INT_LEN].copy_from_slice(&n.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_follows_redis_boundaries() {
        assert_eq!(Encoding::of(b"0"), Encoding::Int);
        assert_eq!(Encoding::of(b"-1"), Encoding::Int);
        assert_eq!(Encoding::of(b"9223372036854775807"), Encoding::Int);
        // Past an i64, so it is text and not a number.
        assert_eq!(Encoding::of(b"9223372036854775808"), Encoding::Embstr);
        // The three shapes string2ll refuses, all of which must survive as text.
        assert_eq!(Encoding::of(b"007"), Encoding::Embstr);
        assert_eq!(Encoding::of(b"+1"), Encoding::Embstr);
        assert_eq!(Encoding::of(b"-0"), Encoding::Embstr);
        assert_eq!(Encoding::of(b""), Encoding::Embstr);
        assert_eq!(Encoding::of(&[b'x'; EMBSTR_MAX]), Encoding::Embstr);
        assert_eq!(Encoding::of(&[b'x'; EMBSTR_MAX + 1]), Encoding::Raw);
    }

    const KINDS: [Kind; 8] = [
        Kind::String,
        Kind::Hash,
        Kind::Set,
        Kind::Zset,
        Kind::List,
        Kind::Stream,
        Kind::Array,
        Kind::Foreign,
    ];

    /// The seven that a saved file has a number for, which is every kind but
    /// the escape.
    const SAVED: [Kind; 7] = [
        Kind::String,
        Kind::Hash,
        Kind::Set,
        Kind::Zset,
        Kind::List,
        Kind::Stream,
        Kind::Array,
    ];

    #[test]
    fn the_meta_byte_survives_a_round_trip() {
        for kind in KINDS {
            for enc in [Encoding::Int, Encoding::Embstr, Encoding::Raw] {
                for expiry in [false, true] {
                    let m = Meta::new(kind, enc, expiry);
                    let back = Meta::from_byte(m.byte());
                    assert_eq!(back.kind(), kind);
                    assert_eq!(back.encoding(), enc);
                    assert_eq!(back.has_expiry(), expiry);
                    assert_eq!(back.payload_at(), if expiry { 9 } else { 1 });
                }
            }
        }
    }

    #[test]
    fn the_three_fields_of_the_meta_byte_do_not_reach_into_each_other() {
        // Forty two combinations, all of which have to come out of one byte
        // with nothing borrowed from a neighbour. A tag that overlapped the
        // expiry bit would read the payload at the wrong offset, which is a
        // corrupt value rather than a wrong answer.
        let mut seen = std::collections::HashSet::new();
        for kind in KINDS {
            for enc in [Encoding::Int, Encoding::Embstr, Encoding::Raw] {
                for expiry in [false, true] {
                    assert!(
                        seen.insert(Meta::new(kind, enc, expiry).byte()),
                        "{kind:?} {enc:?} {expiry} collides with something else"
                    );
                }
            }
        }
        assert_eq!(seen.len(), KINDS.len() * 6);
    }

    #[test]
    fn a_record_written_before_the_tag_existed_is_a_string() {
        // Bits 3 to 7 were zero in every record M2 wrote, and zero is String.
        // This is the whole reason String is zero, so it is worth a test that
        // fails if somebody renumbers the enum alphabetically one day.
        assert_eq!(Kind::String as u8, 0);
        assert_eq!(Meta::from_byte(0b0000_0101).kind(), Kind::String);
        assert_eq!(Meta::from_byte(0b0000_0101).encoding(), Encoding::Embstr);
        assert!(Meta::from_byte(0b0000_0101).has_expiry());
    }

    #[test]
    fn the_tag_is_the_number_the_file_format_uses() {
        use yo_format::catalog::ValueType;
        // Not a translation table, an assertion that no translation is needed.
        // If these ever diverge, saving a key has to map between them, and the
        // mapping is the kind of thing that gets one arm wrong.
        assert_eq!(Kind::String as u8, ValueType::String as u8);
        assert_eq!(Kind::Hash as u8, ValueType::Hash as u8);
        assert_eq!(Kind::Set as u8, ValueType::Set as u8);
        assert_eq!(Kind::Zset as u8, ValueType::Zset as u8);
        assert_eq!(Kind::List as u8, ValueType::List as u8);
        assert_eq!(Kind::Stream as u8, ValueType::Stream as u8);
        assert_eq!(Kind::Array as u8, ValueType::Array as u8);
        // `Kind::Foreign` is deliberately not in this list. It is the escape
        // for a body that lives above this crate, nothing saves one yet, and so
        // there is no number on disk for it to have to match. Its tag is 7 and
        // the catalog's 7 is a bitmap, which is a string in memory and never
        // becomes a `Kind`, so the two cannot meet.
        // And the words agree, because both of them end up on a wire.
        for k in SAVED {
            let v = ValueType::from_u8(k as u8).expect("the catalog knows this one");
            assert_eq!(k.name(), v.redis_name(), "{k:?}");
        }
    }

    #[test]
    fn a_string_record_is_tagged_as_one() {
        for text in [&b"42"[..], b"hello", &[b'z'; 100]] {
            for expire in [None, Some(9_000u64)] {
                assert_eq!(kind(&record(text, expire)), Kind::String);
            }
        }
        let mut v = vec![0u8; record_len(Encoding::Int, 0, false)];
        write_int_record(&mut v, 7, None);
        assert_eq!(kind(&v), Kind::String);
    }

    fn record(bytes: &[u8], expire: Option<u64>) -> Vec<u8> {
        let enc = Encoding::of(bytes);
        let mut v = vec![0u8; record_len(enc, bytes.len(), expire.is_some())];
        write_record(&mut v, enc, bytes, expire);
        v
    }

    #[test]
    fn a_record_gives_back_what_went_into_it() {
        for text in [
            &b""[..],
            b"x",
            b"0",
            b"-1",
            b"42",
            b"007",
            b"-0",
            b"hello world",
            &[b'z'; 100],
        ] {
            for expire in [None, Some(1_234_567_890_123u64)] {
                let r = record(text, expire);
                assert_eq!(read(&r).to_vec(), text, "{text:?} at {expire:?}");
                assert_eq!(read(&r).len(), text.len(), "{text:?} length");
                assert_eq!(expire_at(&r), expire, "{text:?} deadline");
            }
        }
    }

    #[test]
    fn an_integer_costs_the_same_however_many_digits_it_has() {
        let small = record(b"1", None);
        let large = record(b"-9223372036854775808", None);
        assert_eq!(small.len(), large.len());
        assert_eq!(read(&large), Str::Int(i64::MIN));
        assert_eq!(read(&large).to_vec(), b"-9223372036854775808");
    }

    #[test]
    fn an_integer_is_incremented_where_it_lies() {
        let mut r = record(b"41", Some(99));
        let (n, at) = read_int_in_place(&r).expect("int encoded");
        assert_eq!(n, 41);
        write_int_in_place(&mut r, at, n + 1);
        assert_eq!(read(&r), Str::Int(42));
        // The deadline was in front of the payload and is still there.
        assert_eq!(expire_at(&r), Some(99));
    }

    #[test]
    fn a_string_is_not_read_as_an_integer_in_place() {
        let r = record(b"hello", None);
        assert!(read_int_in_place(&r).is_none());
    }

    #[test]
    fn a_deadline_that_is_now_has_passed() {
        let r = record(b"v", Some(100));
        assert!(!is_expired(&r, 99));
        assert!(is_expired(&r, 100));
        assert!(is_expired(&r, 101));
        let forever = record(b"v", None);
        assert!(!is_expired(&forever, u64::MAX));
    }

    #[test]
    fn a_value_that_is_text_can_still_be_a_number() {
        // What `APPEND` leaves behind, and what `INCR` has to accept.
        assert_eq!(Str::Bytes(b"10").as_int(), Some(10));
        assert_eq!(Str::Bytes(b"10x").as_int(), None);
        assert_eq!(Str::Int(-5).as_int(), Some(-5));
    }

    #[test]
    fn an_unknown_encoding_reads_as_raw_bytes() {
        // Nothing we write produces bit pattern three, but a record that has
        // been through a future writer might, and guessing `int` on it would
        // reinterpret eight bytes of somebody's string as a number.
        let m = Meta::from_byte(0b11);
        assert_eq!(m.encoding(), Encoding::Raw);
    }

    #[test]
    fn every_tag_pattern_is_spoken_for() {
        // All eight of them now, the last one being the escape a body this
        // crate cannot name is held under. There is no pattern left over to
        // read as a string, so the day a ninth kind is wanted the tag has to
        // grow a bit rather than borrow one.
        assert_eq!(Meta::from_byte(6 << 3).kind(), Kind::Array);
        assert_eq!(Meta::from_byte(7 << 3).kind(), Kind::Foreign);
    }

    #[test]
    fn the_bits_above_the_tag_do_not_disturb_it() {
        // Bit 6 is now the access flag and bit 7 is still free, and neither of
        // them may move the tag, so this is the check that the tag is three
        // bits and not five.
        assert_eq!(Meta::from_byte(0b1100_0000).kind(), Kind::String);
        assert_eq!(Meta::from_byte(0b1101_0000).kind(), Kind::Set);
        // And the flag itself reads off the byte rather than off the tag.
        assert!(Meta::from_byte(0b0100_0000).has_access());
        assert!(!Meta::from_byte(0b1011_1111).has_access());
    }

    /// Every record this crate writes has room for an access field, and the
    /// field starts empty.
    #[test]
    fn a_fresh_record_has_an_unstamped_access_field() {
        for expire in [None, Some(1_700_000_000_000)] {
            for (enc, bytes) in [
                (Encoding::Int, &b"42"[..]),
                (Encoding::Embstr, b"hello"),
                (Encoding::Raw, &[b'x'; 64][..]),
            ] {
                let mut rec = vec![0u8; record_len(enc, bytes.len(), expire.is_some())];
                write_record(&mut rec, enc, bytes, expire);
                let a = access(&rec).expect("a record we just wrote has the field");
                assert!(a.is_unset(), "{enc:?} came out stamped");
                assert_eq!(expire_at(&rec), expire, "{enc:?} lost its deadline");
                match enc {
                    Encoding::Int => assert_eq!(read(&rec), Str::Int(42)),
                    _ => assert_eq!(read(&rec), Str::Bytes(bytes)),
                }
            }
        }
    }

    /// The same for the other two writers, which is every record shape there is.
    #[test]
    fn slot_and_int_records_have_the_field_too() {
        for expire in [None, Some(9_000)] {
            let mut rec = vec![0u8; slot_record_len(expire.is_some())];
            write_slot_record(&mut rec, Kind::Set, 77, expire);
            assert!(access(&rec).expect("the field").is_unset());
            assert_eq!(slot(&rec), 77);
            assert_eq!(kind(&rec), Kind::Set);
            assert_eq!(expire_at(&rec), expire);

            let mut rec = vec![0u8; record_len(Encoding::Int, 0, expire.is_some())];
            write_int_record(&mut rec, -5, expire);
            assert!(access(&rec).expect("the field").is_unset());
            assert_eq!(read(&rec), Str::Int(-5));
            assert_eq!(expire_at(&rec), expire);
        }
    }

    /// Stamping the field does not disturb anything either side of it.
    ///
    /// It sits between the deadline and the payload and it is written in place
    /// on a path that is usually a read, so an off by one here would corrupt a
    /// value quietly rather than fail.
    #[test]
    fn stamping_the_field_leaves_the_deadline_and_the_payload_alone() {
        let deadline = 1_700_000_000_123u64;
        let body = b"the payload nobody should touch";
        let mut rec = vec![0u8; record_len(Encoding::Raw, body.len(), true)];
        write_record(&mut rec, Encoding::Raw, body, Some(deadline));

        for bits in [1u32, 0xff, 0x00ff_ffff, 0x0012_3456] {
            let a = Access::from_bits(bits);
            assert!(set_access(&mut rec, a));
            assert_eq!(access(&rec), Some(a), "{bits:#x} did not survive");
            assert_eq!(
                expire_at(&rec),
                Some(deadline),
                "{bits:#x} hit the deadline"
            );
            assert_eq!(read(&rec), Str::Bytes(body), "{bits:#x} hit the payload");
        }
    }

    /// A record written before the field existed still reads correctly, and
    /// refuses to be stamped rather than being stamped over its payload.
    ///
    /// This is the whole reason the field is behind a tag bit instead of just
    /// always being there. A file written by an older build has records with the
    /// bit clear, and their payload starts three bytes earlier.
    #[test]
    fn a_record_from_before_the_field_still_reads() {
        // Built by hand, the way the old writer did it: meta, deadline, payload,
        // and no access field.
        let body = b"older";
        let mut old = vec![Meta::string(Encoding::Raw, true).byte()];
        old.extend_from_slice(&7_000u64.to_le_bytes());
        old.extend_from_slice(body);

        assert!(!Meta::from_byte(old[0]).has_access());
        assert_eq!(access(&old), None, "there is no field to read");
        assert_eq!(expire_at(&old), Some(7_000));
        assert_eq!(read(&old), Str::Bytes(body));

        let before = old.clone();
        assert!(
            !set_access(&mut old, Access::from_bits(0xabcdef)),
            "it should refuse rather than write over the payload"
        );
        assert_eq!(old, before, "it wrote something anyway");
    }

    /// The field costs three bytes on every record and no more.
    #[test]
    fn the_field_costs_three_bytes() {
        assert_eq!(record_len(Encoding::Raw, 10, false), 1 + 3 + 10);
        assert_eq!(record_len(Encoding::Raw, 10, true), 1 + 8 + 3 + 10);
        assert_eq!(record_len(Encoding::Int, 10, false), 1 + 3 + 8);
        assert_eq!(slot_record_len(false), 1 + 3 + 4);
        assert_eq!(slot_record_len(true), 1 + 8 + 3 + 4);
    }

    /// A place on the file to demote a value to.
    fn somewhere() -> Addr {
        Addr::new(yo_common::Space::Log, 4096)
    }

    #[test]
    fn a_demoted_record_says_so_in_the_byte_the_lookup_already_read() {
        let mut rec = vec![0u8; cold_record_len(false)];
        write_cold_record(
            &mut rec,
            Kind::String,
            Encoding::Raw,
            somewhere(),
            900,
            None,
        );
        // The one branch a point read makes, and it is a bit in rec[0].
        let c = cold(&rec).expect("the record is demoted");
        assert_eq!(c.at, somewhere());
        assert_eq!(c.len, 900);
    }

    #[test]
    fn a_resident_record_is_not_demoted() {
        let mut rec = vec![0u8; record_len(Encoding::Raw, 3, false)];
        write_record(&mut rec, Encoding::Raw, b"abc", None);
        assert_eq!(cold(&rec), None);
    }

    #[test]
    fn a_demoted_record_answers_everything_that_is_not_the_bytes() {
        let deadline = 1_700_000_000_000u64;
        let mut rec = vec![0u8; cold_record_len(true)];
        write_cold_record(
            &mut rec,
            Kind::String,
            Encoding::Raw,
            somewhere(),
            77,
            Some(deadline),
        );
        // None of these is allowed to want the device.
        assert_eq!(kind(&rec), Kind::String);
        assert_eq!(expire_at(&rec), Some(deadline));
        assert_eq!(str_len(&rec), Some(77));
        assert_eq!(Meta::from_byte(rec[0]).encoding(), Encoding::Raw);
        assert!(access(&rec).is_some());
    }

    #[test]
    fn a_demoted_record_can_still_be_stamped_and_ranked() {
        // Eviction has to be able to score a key whose value is on the file,
        // or the policy would fault in the keys it is trying to get rid of.
        let mut rec = vec![0u8; cold_record_len(false)];
        write_cold_record(
            &mut rec,
            Kind::String,
            Encoding::Embstr,
            somewhere(),
            5,
            None,
        );
        let a = Access::from_bits(12345);
        assert!(set_access(&mut rec, a));
        assert_eq!(access(&rec), Some(a));
        // And the address survived being written around.
        assert_eq!(cold(&rec).map(|c| c.at), Some(somewhere()));
    }

    #[test]
    fn the_in_place_writers_refuse_a_demoted_record() {
        let mut rec = vec![0u8; cold_record_len(false)];
        write_cold_record(&mut rec, Kind::String, Encoding::Int, somewhere(), 2, None);
        // The payload of this record parses as an int if you do not look at the
        // tier tag first, which is exactly the bug the check is here to stop.
        assert_eq!(read_int_in_place(&rec), None);
        assert_eq!(raw_in_place(&mut rec), None);
    }

    #[test]
    fn demoting_a_long_value_is_what_buys_the_memory() {
        // Twelve bytes of payload whatever the value was. A short value is not
        // worth demoting and the arithmetic says so rather than a comment.
        let long = record_len(Encoding::Raw, 4096, false);
        assert!(
            cold_record_len(false) < long / 100,
            "{} against {long}",
            cold_record_len(false)
        );
        assert!(cold_record_len(false) > record_len(Encoding::Raw, 8, false));
    }

    #[test]
    fn str_len_on_a_resident_string_is_the_value_length() {
        let mut rec = vec![0u8; record_len(Encoding::Raw, 3, false)];
        write_record(&mut rec, Encoding::Raw, b"abc", None);
        assert_eq!(str_len(&rec), Some(3));

        let mut n = vec![0u8; record_len(Encoding::Int, 0, false)];
        write_int_record(&mut n, -1234, None);
        assert_eq!(str_len(&n), Some(5));
    }
}
