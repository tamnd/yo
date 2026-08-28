//! The collection catalogue, which is what makes a `.yo` file self describing.
//!
//! `07` section 5. One entry per named collection, chained from the
//! superblock's `catalog_addr`.
//!
//! The entry stores the `schema` bytes in full and not just their hash. That
//! costs space in a structure there is one of per collection, and it buys two
//! things. A shape mismatch can print both shapes instead of two hex digests,
//! which is the difference between an error a user can act on and an error a
//! user files a bug about. And a tool that has never seen the writer's source
//! can still say what is in the file, which is the concrete form of the promise
//! that the format is readable without us.

use crate::{
    checksum_skipping, get_u8, get_u16, get_u32, get_u64, put_u8, put_u16, put_u32, put_u64,
};
use yo_common::{Code, Error, Result};

/// The fixed part of an entry. Name and schema follow, then the checksum.
pub const ENTRY_HEAD_LEN: usize = 64;

/// The checksum at the end.
pub const ENTRY_TRAILER_LEN: usize = 4;

/// The largest name, because `name_len` is a `u16`.
pub const MAX_NAME_LEN: usize = u16::MAX as usize;

/// Which of the four models a collection belongs to.
///
/// This is the axis the whole engine is organised along, so it is one byte at a
/// fixed offset rather than something inferred from the value type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Model {
    /// Redis data types.
    Kv = 0,
    /// Documents.
    Document = 1,
    /// Vectors.
    Vector = 2,
    /// Graph.
    Graph = 3,
}

impl Model {
    /// Every model, in order.
    pub const ALL: [Model; 4] = [Model::Kv, Model::Document, Model::Vector, Model::Graph];

    /// The model for a byte, or `None` if this version does not know it.
    #[must_use]
    pub const fn from_u8(b: u8) -> Option<Model> {
        match b {
            0 => Some(Model::Kv),
            1 => Some(Model::Document),
            2 => Some(Model::Vector),
            3 => Some(Model::Graph),
            _ => None,
        }
    }

    /// The byte.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

/// The Redis type of a key value collection.
///
/// Meaningful only when the model is [`Model::Kv`]. For the other three models
/// the byte is zero and this is not the question to ask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ValueType {
    /// A string.
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
    /// A typed array, which RESP has no name for and the embedded API does.
    Array = 6,
    /// A bitmap.
    Bitmap = 7,
    /// A HyperLogLog.
    Hll = 8,
    /// A geospatial index.
    Geo = 9,
}

impl ValueType {
    /// Every type, in order.
    pub const ALL: [ValueType; 10] = [
        ValueType::String,
        ValueType::Hash,
        ValueType::Set,
        ValueType::Zset,
        ValueType::List,
        ValueType::Stream,
        ValueType::Array,
        ValueType::Bitmap,
        ValueType::Hll,
        ValueType::Geo,
    ];

    /// The type for a byte, or `None` if this version does not know it.
    #[must_use]
    pub const fn from_u8(b: u8) -> Option<ValueType> {
        match b {
            0 => Some(ValueType::String),
            1 => Some(ValueType::Hash),
            2 => Some(ValueType::Set),
            3 => Some(ValueType::Zset),
            4 => Some(ValueType::List),
            5 => Some(ValueType::Stream),
            6 => Some(ValueType::Array),
            7 => Some(ValueType::Bitmap),
            8 => Some(ValueType::Hll),
            9 => Some(ValueType::Geo),
            _ => None,
        }
    }

    /// The byte.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// The name `TYPE` replies with, for the ones Redis has a name for.
    #[must_use]
    pub const fn redis_name(self) -> &'static str {
        match self {
            ValueType::String | ValueType::Bitmap | ValueType::Hll => "string",
            ValueType::Hash => "hash",
            ValueType::Set => "set",
            ValueType::Zset | ValueType::Geo => "zset",
            ValueType::List => "list",
            ValueType::Stream => "stream",
            ValueType::Array => "array",
        }
    }
}

/// How a collection is laid out, which is chosen by size and not by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Band {
    /// Small enough to live inside the index entry's neighbourhood.
    Inline = 0,
    /// One structure, one owner, in memory.
    Native = 1,
    /// Split across partitions so a large collection is not one hot object.
    Partitioned = 2,
    /// Spilled, and read back a chunk at a time.
    ChunkedCold = 3,
}

impl Band {
    /// Every band, in order.
    pub const ALL: [Band; 4] = [
        Band::Inline,
        Band::Native,
        Band::Partitioned,
        Band::ChunkedCold,
    ];

    /// The band for a byte, or `None` if this version does not know it.
    #[must_use]
    pub const fn from_u8(b: u8) -> Option<Band> {
        match b {
            0 => Some(Band::Inline),
            1 => Some(Band::Native),
            2 => Some(Band::Partitioned),
            3 => Some(Band::ChunkedCold),
            _ => None,
        }
    }

    /// The byte.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

/// One catalogue entry, decoded, borrowing its name and schema from the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatalogEntry<'a> {
    /// Raw model byte. Ask [`CatalogEntry::model`] what it means.
    pub model: u8,
    /// Raw value type byte, zero unless the model is `kv`.
    pub value_type: u8,
    /// Raw band byte.
    pub band: u8,
    /// Partition count as a base two exponent. Never 1: a two way split buys
    /// nothing and costs an indirection, so the ladder goes one, four, sixteen.
    pub p_exp: u8,
    /// The 128 bit shape hash, or all zeroes for a collection that RESP created
    /// and that therefore has no declared shape.
    pub shape_tag: [u8; 16],
    /// Where the collection's root lives.
    pub root_addr: u64,
    /// Elements, for `SCARD` and friends without walking anything.
    pub element_count: u64,
    /// Bytes the collection occupies.
    pub bytes: u64,
    /// Which logical database.
    pub db: u16,
    /// The next entry in the chain, or 0.
    pub next: u64,
    /// The collection's name.
    pub name: &'a [u8],
    /// The canonical shape description the `shape_tag` hashes, or empty.
    pub schema: &'a [u8],
}

/// The bytes an entry with this name and schema needs.
///
/// # Errors
///
/// [`Code::Invalid`] if the name is longer than [`MAX_NAME_LEN`].
pub fn entry_len(name_len: usize, schema_len: usize) -> Result<usize> {
    if name_len > MAX_NAME_LEN {
        return Err(Error::new(
            Code::Invalid,
            "a collection name is at most 65535 bytes",
        ));
    }
    Ok(ENTRY_HEAD_LEN + name_len + schema_len + ENTRY_TRAILER_LEN)
}

impl<'a> CatalogEntry<'a> {
    /// An entry for a named collection with no declared shape.
    #[must_use]
    pub const fn new(model: Model, name: &'a [u8]) -> CatalogEntry<'a> {
        CatalogEntry {
            model: model.as_u8(),
            value_type: 0,
            band: Band::Native.as_u8(),
            p_exp: 0,
            shape_tag: [0; 16],
            root_addr: 0,
            element_count: 0,
            bytes: 0,
            db: 0,
            next: 0,
            name,
            schema: &[],
        }
    }

    /// Writes the entry and its checksum, returning the length written.
    ///
    /// Unlike a log record this one writes its `len` up front, because a
    /// catalogue entry is not appended to a log that a reader is scanning. It
    /// is written to a free segment and then reached by a pointer that is
    /// published in the next superblock flip, so the entry is either reachable
    /// and whole or not reachable at all.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] for an oversized name or an illegal `p_exp`,
    /// [`Code::Full`] if `buf` is too small.
    pub fn encode(&self, buf: &mut [u8]) -> Result<usize> {
        if self.p_exp == 1 {
            return Err(Error::new(
                Code::Invalid,
                "a two way partition split is not a thing; p_exp is 0 or 2 and up",
            ));
        }
        let n = entry_len(self.name.len(), self.schema.len())?;
        if buf.len() < n {
            return Err(Error::new(Code::Full, "the catalogue entry does not fit")
                .with_detail(format!("need={n} have={}", buf.len())));
        }
        put_u32(buf, 0, n as u32);
        put_u8(buf, 4, self.model);
        put_u8(buf, 5, self.value_type);
        put_u8(buf, 6, self.band);
        put_u8(buf, 7, self.p_exp);
        buf[8..24].copy_from_slice(&self.shape_tag);
        put_u64(buf, 24, self.root_addr);
        put_u64(buf, 32, self.element_count);
        put_u64(buf, 40, self.bytes);
        put_u16(buf, 48, self.db);
        put_u16(buf, 50, self.name.len() as u16);
        put_u32(buf, 52, self.schema.len() as u32);
        put_u64(buf, 56, self.next);
        let name_end = ENTRY_HEAD_LEN + self.name.len();
        buf[ENTRY_HEAD_LEN..name_end].copy_from_slice(self.name);
        buf[name_end..name_end + self.schema.len()].copy_from_slice(self.schema);
        let crc = checksum_skipping(&buf[..n], n - ENTRY_TRAILER_LEN);
        put_u32(buf, n - ENTRY_TRAILER_LEN, crc);
        Ok(n)
    }

    /// Reads the entry at the front of `bytes`.
    ///
    /// # Errors
    ///
    /// [`Code::Corrupt`] if the length is impossible, if the entry runs past
    /// the end of `bytes`, if the name and schema do not fit inside it, or if
    /// the checksum fails.
    pub fn decode(bytes: &'a [u8]) -> Result<CatalogEntry<'a>> {
        if bytes.len() < ENTRY_HEAD_LEN + ENTRY_TRAILER_LEN {
            return Err(Error::new(Code::Corrupt, "shorter than a catalogue entry"));
        }
        let n = get_u32(bytes, 0) as usize;
        if n < ENTRY_HEAD_LEN + ENTRY_TRAILER_LEN || n > bytes.len() {
            return Err(
                Error::new(Code::Corrupt, "the catalogue entry length is impossible")
                    .with_detail(format!("len={n} available={}", bytes.len())),
            );
        }
        let want = get_u32(bytes, n - ENTRY_TRAILER_LEN);
        let got = checksum_skipping(&bytes[..n], n - ENTRY_TRAILER_LEN);
        if want != got {
            return Err(
                Error::new(Code::Corrupt, "catalogue entry checksum mismatch")
                    .with_detail(format!("stored={want:#010x} computed={got:#010x}")),
            );
        }

        let name_len = get_u16(bytes, 50) as usize;
        let schema_len = get_u32(bytes, 52) as usize;
        // Checked even though the checksum passed. A checksum says the bytes are
        // the bytes that were written; it does not say the writer was us. A file
        // from a buggy or hostile writer must not be able to point a slice past
        // the end of the entry.
        if ENTRY_HEAD_LEN + name_len + schema_len + ENTRY_TRAILER_LEN != n {
            return Err(
                Error::new(Code::Corrupt, "the name and schema do not fill the entry").with_detail(
                    format!("len={n} name_len={name_len} schema_len={schema_len}"),
                ),
            );
        }

        let p_exp = get_u8(bytes, 7);
        if p_exp == 1 {
            return Err(Error::new(Code::Corrupt, "p_exp of 1 is not a legal value"));
        }

        let mut shape_tag = [0u8; 16];
        shape_tag.copy_from_slice(&bytes[8..24]);
        let name_end = ENTRY_HEAD_LEN + name_len;

        Ok(CatalogEntry {
            model: get_u8(bytes, 4),
            value_type: get_u8(bytes, 5),
            band: get_u8(bytes, 6),
            p_exp,
            shape_tag,
            root_addr: get_u64(bytes, 24),
            element_count: get_u64(bytes, 32),
            bytes: get_u64(bytes, 40),
            db: get_u16(bytes, 48),
            next: get_u64(bytes, 56),
            name: &bytes[ENTRY_HEAD_LEN..name_end],
            schema: &bytes[name_end..name_end + schema_len],
        })
    }

    /// The model, if this version knows it.
    #[must_use]
    pub fn model(&self) -> Option<Model> {
        Model::from_u8(self.model)
    }

    /// The value type, if the model is `kv` and this version knows the byte.
    #[must_use]
    pub fn value_type(&self) -> Option<ValueType> {
        if self.model()? != Model::Kv {
            return None;
        }
        ValueType::from_u8(self.value_type)
    }

    /// The band, if this version knows it.
    #[must_use]
    pub fn band(&self) -> Option<Band> {
        Band::from_u8(self.band)
    }

    /// How many partitions the collection is split into.
    #[must_use]
    pub const fn partitions(&self) -> u32 {
        if self.p_exp == 0 {
            1
        } else {
            1u32 << self.p_exp
        }
    }

    /// Whether the collection was declared with a shape.
    ///
    /// An all zero tag means it was not, which is the case for anything a RESP
    /// client created. Shape checking has nothing to check against there and
    /// says so rather than inventing a shape.
    #[must_use]
    pub fn is_shape_tagged(&self) -> bool {
        self.shape_tag != [0u8; 16]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn an_entry<'a>(name: &'a [u8], schema: &'a [u8]) -> CatalogEntry<'a> {
        CatalogEntry {
            model: Model::Kv.as_u8(),
            value_type: ValueType::Zset.as_u8(),
            band: Band::Partitioned.as_u8(),
            p_exp: 4,
            shape_tag: [7; 16],
            root_addr: 1 << 20,
            element_count: 1234,
            bytes: 98765,
            db: 3,
            next: 1 << 30,
            name,
            schema,
        }
    }

    #[test]
    fn an_entry_round_trips() {
        let e = an_entry(b"leaderboard", b"zset<u64, f64>");
        let mut buf = [0u8; 256];
        let n = e.encode(&mut buf).unwrap();
        assert_eq!(n, ENTRY_HEAD_LEN + 11 + 14 + 4);
        assert_eq!(CatalogEntry::decode(&buf[..n]).unwrap(), e);
    }

    #[test]
    fn every_field_lands_where_the_specification_says() {
        let e = an_entry(b"name", b"schema");
        let mut buf = [0u8; 256];
        let n = e.encode(&mut buf).unwrap();
        assert_eq!(get_u32(&buf, 0) as usize, n);
        assert_eq!(get_u8(&buf, 4), 0, "kv is model 0");
        assert_eq!(get_u8(&buf, 5), 3, "zset is type 3");
        assert_eq!(get_u8(&buf, 6), 2, "partitioned is band 2");
        assert_eq!(get_u8(&buf, 7), 4);
        assert_eq!(&buf[8..24], &[7u8; 16]);
        assert_eq!(get_u64(&buf, 24), 1 << 20);
        assert_eq!(get_u64(&buf, 32), 1234);
        assert_eq!(get_u64(&buf, 40), 98765);
        assert_eq!(get_u16(&buf, 48), 3);
        assert_eq!(get_u16(&buf, 50), 4);
        assert_eq!(get_u32(&buf, 52), 6);
        assert_eq!(get_u64(&buf, 56), 1 << 30);
        assert_eq!(&buf[64..68], b"name");
        assert_eq!(&buf[68..74], b"schema");
    }

    #[test]
    fn the_schema_is_stored_whole_and_not_hashed() {
        // The point of `07` section 5: a mismatch can print both shapes, and a
        // tool that has never seen our source can read the file.
        let schema = b"document { id: u64, tags: [string], score: f32 }";
        let e = CatalogEntry {
            schema,
            ..an_entry(b"docs", schema)
        };
        let mut buf = [0u8; 256];
        let n = e.encode(&mut buf).unwrap();
        let back = CatalogEntry::decode(&buf[..n]).unwrap();
        assert_eq!(back.schema, schema);
        assert!(back.is_shape_tagged());
    }

    #[test]
    fn an_untagged_collection_is_one_a_resp_client_made() {
        let e = CatalogEntry::new(Model::Kv, b"made-by-SET");
        let mut buf = [0u8; 128];
        let n = e.encode(&mut buf).unwrap();
        let back = CatalogEntry::decode(&buf[..n]).unwrap();
        assert!(!back.is_shape_tagged());
        assert_eq!(back.schema, b"");
        assert_eq!(back.partitions(), 1);
    }

    #[test]
    fn a_flipped_bit_anywhere_in_an_entry_is_caught() {
        let e = an_entry(b"leaderboard", b"zset<u64, f64>");
        let mut good = [0u8; 256];
        let n = e.encode(&mut good).unwrap();
        for i in 0..n {
            let mut bad = good;
            bad[i] ^= 0x08;
            assert!(
                CatalogEntry::decode(&bad[..n]).is_err(),
                "byte {i} was not caught"
            );
        }
    }

    #[test]
    fn lengths_that_do_not_add_up_are_refused_even_with_a_good_checksum() {
        // The attack this stops: a name_len that reaches past the entry into
        // whatever the next segment holds. A checksum does not stop it because
        // the attacker computes the checksum too.
        let e = an_entry(b"name", b"schema");
        let mut buf = [0u8; 256];
        let n = e.encode(&mut buf).unwrap();
        put_u16(&mut buf, 50, 4000);
        let crc = checksum_skipping(&buf[..n], n - ENTRY_TRAILER_LEN);
        put_u32(&mut buf, n - ENTRY_TRAILER_LEN, crc);
        let err = CatalogEntry::decode(&buf[..n]).unwrap_err();
        assert_eq!(err.code(), Code::Corrupt);
        assert!(err.detail().unwrap().contains("name_len=4000"));
    }

    #[test]
    fn an_entry_that_claims_to_be_longer_than_its_buffer_is_refused() {
        let e = an_entry(b"n", b"");
        let mut buf = [0u8; 128];
        let n = e.encode(&mut buf).unwrap();
        put_u32(&mut buf, 0, 100_000);
        let err = CatalogEntry::decode(&buf[..n]).unwrap_err();
        assert_eq!(err.code(), Code::Corrupt);
        assert!(err.detail().unwrap().contains("len=100000"));
    }

    #[test]
    fn a_two_way_partition_split_is_not_a_thing() {
        // Y5. One partition or four, never two: a two way split pays a full
        // indirection to halve a collection, which is never the right trade.
        let e = CatalogEntry {
            p_exp: 1,
            ..an_entry(b"n", b"")
        };
        let mut buf = [0u8; 128];
        assert_eq!(e.encode(&mut buf).unwrap_err().code(), Code::Invalid);

        let ok = CatalogEntry {
            p_exp: 0,
            ..an_entry(b"n", b"")
        };
        let n = ok.encode(&mut buf).unwrap();
        put_u8(&mut buf, 7, 1);
        let crc = checksum_skipping(&buf[..n], n - ENTRY_TRAILER_LEN);
        put_u32(&mut buf, n - ENTRY_TRAILER_LEN, crc);
        assert_eq!(
            CatalogEntry::decode(&buf[..n]).unwrap_err().code(),
            Code::Corrupt
        );
    }

    #[test]
    fn partition_counts_are_powers_of_two_from_four_up() {
        for (p_exp, want) in [(0u8, 1u32), (2, 4), (3, 8), (4, 16), (8, 256)] {
            let e = CatalogEntry {
                p_exp,
                ..an_entry(b"n", b"")
            };
            assert_eq!(e.partitions(), want);
        }
    }

    #[test]
    fn a_chain_of_entries_walks() {
        let mut buf = [0u8; 1024];
        // Not offset zero. In a real file the catalogue lives past `DATA_START`
        // so address zero can mean "no next entry", and a test that put the
        // first entry at zero would be testing a layout that cannot happen.
        let mut at = 64usize;
        let mut offsets = Vec::new();
        for i in 0..5usize {
            let name = format!("collection{i}");
            let e = CatalogEntry {
                next: 0,
                ..CatalogEntry::new(Model::Document, name.as_bytes())
            };
            let n = e.encode(&mut buf[at..]).unwrap();
            offsets.push((at, n));
            at += n;
        }
        // Link them backwards, which is how the writer does it: an entry points
        // at the one written before it, and the superblock points at the last.
        for i in 1..offsets.len() {
            let (off, n) = offsets[i];
            let prev = offsets[i - 1].0 as u64;
            put_u64(&mut buf[off..], 56, prev);
            let crc = checksum_skipping(&buf[off..off + n], n - ENTRY_TRAILER_LEN);
            put_u32(&mut buf[off..], n - ENTRY_TRAILER_LEN, crc);
        }

        let mut seen = Vec::new();
        let mut cursor = offsets.last().unwrap().0;
        loop {
            let e = CatalogEntry::decode(&buf[cursor..]).unwrap();
            seen.push(String::from_utf8(e.name.to_vec()).unwrap());
            if e.next == 0 {
                break;
            }
            cursor = e.next as usize;
        }
        seen.reverse();
        assert_eq!(
            seen,
            (0..5).map(|i| format!("collection{i}")).collect::<Vec<_>>()
        );
    }

    #[test]
    fn unknown_bytes_are_questions_with_no_answer_rather_than_errors() {
        let e = CatalogEntry {
            model: 9,
            value_type: 200,
            band: 250,
            ..an_entry(b"future", b"")
        };
        let mut buf = [0u8; 128];
        let n = e.encode(&mut buf).unwrap();
        let back = CatalogEntry::decode(&buf[..n]).unwrap();
        assert_eq!(back.model(), None);
        assert_eq!(back.value_type(), None);
        assert_eq!(back.band(), None);
        assert_eq!(
            back.model, 9,
            "the raw byte survives so it can be copied on"
        );
    }

    #[test]
    fn a_value_type_only_means_something_for_the_kv_model() {
        let e = CatalogEntry {
            model: Model::Vector.as_u8(),
            value_type: ValueType::Hash.as_u8(),
            ..an_entry(b"embeddings", b"")
        };
        assert_eq!(e.value_type(), None, "a vector has no Redis type");
        assert_eq!(e.model(), Some(Model::Vector));
    }

    #[test]
    fn the_enums_round_trip_and_stop_where_the_specification_stops() {
        for m in Model::ALL {
            assert_eq!(Model::from_u8(m.as_u8()), Some(m));
        }
        assert_eq!(Model::from_u8(4), None);
        for t in ValueType::ALL {
            assert_eq!(ValueType::from_u8(t.as_u8()), Some(t));
        }
        assert_eq!(ValueType::from_u8(10), None);
        for b in Band::ALL {
            assert_eq!(Band::from_u8(b.as_u8()), Some(b));
        }
        assert_eq!(Band::from_u8(4), None);
    }

    #[test]
    fn type_replies_the_way_redis_replies() {
        // Bitmaps and HyperLogLogs are strings to a Redis client, and a geo
        // index is a zset. A client that switches on TYPE has to see what it
        // would see from Redis or the compatibility claim is not true.
        assert_eq!(ValueType::String.redis_name(), "string");
        assert_eq!(ValueType::Bitmap.redis_name(), "string");
        assert_eq!(ValueType::Hll.redis_name(), "string");
        assert_eq!(ValueType::Geo.redis_name(), "zset");
        assert_eq!(ValueType::Zset.redis_name(), "zset");
        assert_eq!(ValueType::Stream.redis_name(), "stream");
    }

    #[test]
    fn a_buffer_with_no_room_says_how_much_it_needed() {
        let e = an_entry(b"a long collection name", b"");
        let mut buf = [0u8; 32];
        let err = e.encode(&mut buf).unwrap_err();
        assert_eq!(err.code(), Code::Full);
        assert!(err.detail().unwrap().contains("have=32"));
    }

    #[test]
    fn a_short_buffer_is_an_error_and_not_a_panic() {
        assert_eq!(
            CatalogEntry::decode(&[0u8; 16]).unwrap_err().code(),
            Code::Corrupt
        );
    }
}
