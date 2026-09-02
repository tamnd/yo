//! A vector record goes through the log and comes back bit for bit.
//!
//! `yo-format` has the layout and its own tests, and the log has its own tests,
//! and neither of them says the two fit together. This does, because the thing
//! that would go wrong is at the seam: a vector's value is longer than a string
//! value ever is, records are eight byte aligned while a vector is four byte
//! aligned inside its own body, and the length a reader gets back has to be the
//! exact one rather than the padded one.
//!
//! It matters more than a normal round trip test because rerank is the step
//! that decides the final ordering of a search. A vector that comes back nearly
//! right is worse than one that fails to come back at all.

use yo_format::vector::{Element, VectorBody, vector_len};
use yo_format::{RecordHeader, RecordKind};
use yo_record::sink::MemorySink;
use yo_record::{Durability, Log, LogConfig};

/// `n` coordinates that are all different and none of them round numbers, so
/// that a byte swap or an off by one shows up as a value from somewhere else
/// rather than as a plausible looking number.
fn corpus(n: usize, seed: u64) -> Vec<f32> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 40) as f32 / 16777216.0 - 0.5
        })
        .collect()
}

#[test]
fn a_vector_record_comes_back_exactly() {
    let cfg = LogConfig {
        page_len: 65536,
        durability: Durability::None,
        ..LogConfig::default()
    };
    let mut log = Log::new(cfg, MemorySink::new()).unwrap();

    // Three dimensions that a real collection actually uses, and one odd
    // number, because every dimension here is a multiple of four coordinates
    // and a vector whose length is not is exactly where a padding assumption
    // would hide.
    let dims = [128usize, 768, 1536, 37];
    let mut wrote = Vec::new();
    for (i, dim) in dims.into_iter().enumerate() {
        let values = corpus(dim, i as u64 + 1);
        let mut buf = vec![0u8; vector_len(dim, Element::F32).unwrap()];
        VectorBody::encode(&values, &mut buf).unwrap();
        let key = format!("v:{i}");
        let put = log
            .append(&RecordHeader::new(RecordKind::Vector), key.as_bytes(), &buf)
            .unwrap();
        wrote.push((put.addr, values));
    }

    for (addr, values) in &wrote {
        let rec = log.read(*addr).unwrap();
        assert_eq!(rec.kind, RecordKind::Vector.as_u8());
        let body = VectorBody::decode(rec.value).unwrap();
        assert_eq!(body.dim(), values.len());
        let mut out = vec![0f32; body.dim()];
        body.read_into(&mut out).unwrap();
        assert_eq!(&out, values, "the vector at {addr} came back changed");
    }
}

#[test]
fn a_vector_record_is_the_length_it_says_it_is() {
    let cfg = LogConfig {
        page_len: 65536,
        durability: Durability::None,
        ..LogConfig::default()
    };
    let mut log = Log::new(cfg, MemorySink::new()).unwrap();

    // Every key length from zero to eight, so the vector body lands at every
    // offset modulo eight that it can. If anything in the chain rounded a
    // length up instead of storing the exact one, the value read back would be
    // longer than the body and the decode would still succeed, so this checks
    // the length rather than only the values.
    let values = corpus(65, 7);
    let mut buf = vec![0u8; vector_len(values.len(), Element::F32).unwrap()];
    VectorBody::encode(&values, &mut buf).unwrap();
    for klen in 0..9usize {
        let key = "k".repeat(klen);
        let put = log
            .append(&RecordHeader::new(RecordKind::Vector), key.as_bytes(), &buf)
            .unwrap();
        let rec = log.read(put.addr).unwrap();
        assert_eq!(
            rec.value.len(),
            buf.len(),
            "a {klen} byte key changed the value's length"
        );
        let mut out = vec![0f32; values.len()];
        VectorBody::decode(rec.value)
            .unwrap()
            .read_into(&mut out)
            .unwrap();
        assert_eq!(out, values);
    }
}
