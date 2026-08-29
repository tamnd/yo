//! How a string value sits in a record, and what comes back out of one.
//!
//! The map underneath stores opaque bytes against a key. Everything a string
//! needs beyond those bytes, which is its encoding and its expiry deadline, is
//! carried in a one byte header in front of them:
//!
//! ```text
//! +--------+-------------------------+-----------+
//! | meta   | expire at, u64 LE       | payload   |
//! | u8     | 8 bytes, only if tagged | see below |
//! +--------+-------------------------+-----------+
//! ```
//!
//! One byte, and eight more only for a key that has a deadline, which most keys
//! do not. The alternative is a second lookup into a side table for the TTL, and
//! a second lookup is a second cache miss on a path whose whole budget is one.
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
}

impl Kind {
    /// The word `TYPE` replies with.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Kind::String => "string",
            Kind::Hash => "hash",
            Kind::Set => "set",
            Kind::Zset => "zset",
            Kind::List => "list",
            Kind::Stream => "stream",
        }
    }

    /// The kind for a three bit tag.
    ///
    /// Six of the eight patterns are spoken for. The other two cannot come out
    /// of our own writer, and they fall to `String` for the same reason an
    /// unknown encoding falls to `raw`: it is the reading that hands the bytes
    /// back rather than the one that reinterprets them.
    #[inline]
    const fn from_bits(bits: u8) -> Kind {
        match bits {
            KIND_HASH => Kind::Hash,
            KIND_SET => Kind::Set,
            KIND_ZSET => Kind::Zset,
            KIND_LIST => Kind::List,
            KIND_STREAM => Kind::Stream,
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
/// Bits 3, 4 and 5: which type the key holds. Bits 6 and 7 are still spare.
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

/// Bytes of integer payload, which is a whole `i64` and never its digits.
const INT_LEN: usize = 8;

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

    /// Where the payload starts, counting from the meta byte.
    #[inline]
    pub const fn payload_at(self) -> usize {
        if self.has_expiry() { 1 + 8 } else { 1 }
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
#[inline]
pub fn record_len(enc: Encoding, payload: usize, has_expiry: bool) -> usize {
    let head = if has_expiry { 1 + 8 } else { 1 };
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
    out[0] = Meta::string(enc, expire_at.is_some()).byte();
    let mut at = 1;
    if let Some(ms) = expire_at {
        out[at..at + 8].copy_from_slice(&ms.to_le_bytes());
        at += 8;
    }
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
    out[0] = Meta::string(Encoding::Int, expire_at.is_some()).byte();
    let mut at = 1;
    if let Some(ms) = expire_at {
        out[at..at + 8].copy_from_slice(&ms.to_le_bytes());
        at += 8;
    }
    out[at..at + INT_LEN].copy_from_slice(&n.to_le_bytes());
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
#[inline]
pub fn read(rec: &[u8]) -> Str<'_> {
    let m = Meta::from_byte(rec[0]);
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
    if m.encoding() != Encoding::Int {
        return None;
    }
    let at = m.payload_at();
    let mut b = [0u8; INT_LEN];
    b.copy_from_slice(&rec[at..at + INT_LEN]);
    Some((i64::from_le_bytes(b), at))
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

    const KINDS: [Kind; 6] = [
        Kind::String,
        Kind::Hash,
        Kind::Set,
        Kind::Zset,
        Kind::List,
        Kind::Stream,
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
        // Thirty six combinations, all of which have to come out of one byte
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
        assert_eq!(seen.len(), 36);
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
        // And the words agree, because both of them end up on a wire.
        for k in KINDS {
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
    fn an_unknown_type_tag_reads_as_a_string() {
        // Six of eight patterns are used, and the other two answer String for
        // the same reason: handing the bytes back is the harmless reading.
        assert_eq!(Meta::from_byte(6 << 3).kind(), Kind::String);
        assert_eq!(Meta::from_byte(7 << 3).kind(), Kind::String);
    }

    #[test]
    fn the_top_two_bits_are_still_free() {
        // Whatever lands in them next must not disturb the tag, so this is the
        // check that the tag is three bits and not five.
        assert_eq!(Meta::from_byte(0b1100_0000).kind(), Kind::String);
        assert_eq!(Meta::from_byte(0b1101_0000).kind(), Kind::Set);
    }
}
