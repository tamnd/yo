//! A document goes through the log and comes back readable.
//!
//! `yo-doc` has the encoding and its own tests, the log has its own tests, and
//! `yo_format::document` says a document record's value is the encoding byte for
//! byte. None of those three says the three fit together, and this does.
//!
//! The thing that would go wrong is at the seam, and it is a different seam from
//! the vector one. A vector record carries its own length, so a value that came
//! back one byte long would be caught by the header. A document's length comes
//! out of the document, so a record that hands back a padded or truncated value
//! is a document whose last element is short or whose trailing bytes are
//! somebody else's, and both of those are the kind of failure that reads as a
//! decoder bug rather than as a storage bug.
//!
//! Alignment is the other one. A container's offsets are relative to its own
//! header and its entry tables are on a four byte stride, and a record lands
//! wherever the key length put it, so every key length from nothing to eight is
//! here.

use yo_doc::{Builder, Docs, Value};
use yo_format::{DocumentBody, RecordHeader, RecordKind, ValueTag};
use yo_record::sink::MemorySink;
use yo_record::{Durability, Log, LogConfig};

fn log() -> Log<MemorySink> {
    let cfg = LogConfig {
        page_len: 65536,
        durability: Durability::None,
        ..LogConfig::default()
    };
    Log::new(cfg, MemorySink::new()).unwrap()
}

/// The bytes of a value built by `f`.
fn built(f: impl FnOnce(&mut Builder) -> yo_common::Result<()>) -> Vec<u8> {
    let mut b = Builder::new();
    f(&mut b).expect("built");
    b.finish().expect("finished").to_vec()
}

/// An order, which is the shape a document collection actually holds.
fn order(id: i64, status: &str, total: f64) -> Vec<u8> {
    built(|b| {
        b.begin_object()?;
        b.key(b"id")?;
        b.int(id)?;
        b.key(b"lines")?;
        b.begin_array()?;
        for n in 0..3 {
            b.begin_object()?;
            b.key(b"qty")?;
            b.int(n + 1)?;
            b.key(b"sku")?;
            b.text(&format!("sku-{n}"))?;
            b.end_object()?;
        }
        b.end_array()?;
        b.key(b"status")?;
        b.text(status)?;
        b.key(b"total")?;
        b.float(total)?;
        b.end_object()
    })
}

#[test]
fn a_document_record_comes_back_exactly() {
    let mut log = log();

    // An object, an array, a nested document and a bare scalar. The scalar is
    // here because a document collection is allowed to hold one and it is the
    // case where the record's value is four bytes plus a payload, which is
    // where an off by one in the length would show.
    let docs = [
        order(1, "open", 12.5),
        built(|b| {
            b.begin_array()?;
            for n in 0..64 {
                b.int(n)?;
            }
            b.end_array()
        }),
        built(|b| b.text("a wrench")),
        built(|b| b.int(41_920)),
        built(|b| b.null()),
        built(|b| {
            b.begin_object()?;
            b.end_object()
        }),
    ];

    let mut wrote = Vec::new();
    for (i, doc) in docs.iter().enumerate() {
        let key = format!("doc:{i}");
        let put = log
            .append(
                &RecordHeader::new(RecordKind::Document),
                key.as_bytes(),
                doc,
            )
            .unwrap();
        wrote.push((put.addr, doc));
    }

    for (addr, doc) in wrote {
        let rec = log.read(addr).unwrap();
        assert_eq!(rec.kind, RecordKind::Document.as_u8());
        assert_eq!(
            rec.value.len(),
            doc.len(),
            "the document at {addr} changed length"
        );
        assert_eq!(rec.value, &doc[..], "the document at {addr} changed");
        DocumentBody::decode(rec.value).expect("the record layer reads it");
        let v = Value::new(rec.value).expect("the reader reads it");
        assert!(v.validate(), "the document at {addr} came back malformed");
        assert_eq!(
            v.encoded_len(),
            Some(doc.len()),
            "the length in the document disagrees with the length of the record"
        );
    }
}

#[test]
fn a_document_reads_the_same_at_every_key_length() {
    let mut log = log();
    let doc = order(7, "shipped", 99.0);

    // Every key length from zero to eight, so the document lands at every
    // offset modulo eight a record can put it at. A container's entry table is
    // read on a four byte stride and its offsets are relative to its own
    // header, so if either of those had picked up an assumption about where the
    // record starts, one of these nine would read the wrong field.
    for klen in 0..9usize {
        let key = "k".repeat(klen);
        let put = log
            .append(
                &RecordHeader::new(RecordKind::Document),
                key.as_bytes(),
                &doc,
            )
            .unwrap();
        let rec = log.read(put.addr).unwrap();
        assert_eq!(
            rec.value.len(),
            doc.len(),
            "a {klen} byte key changed the document's length"
        );
        let v = Value::new(rec.value).expect("readable");
        assert!(
            v.validate(),
            "a {klen} byte key made the document malformed"
        );
        assert_eq!(v.get(b"status").and_then(|s| s.as_text()), Some("shipped"));
        assert_eq!(v.get(b"total").and_then(|t| t.as_float()), Some(99.0));
        let lines = v.get(b"lines").expect("the array is there");
        assert_eq!(lines.len(), 3);
        assert_eq!(
            lines
                .at(2)
                .and_then(|l| l.get(b"sku"))
                .and_then(|s| s.as_text()),
            Some("sku-2")
        );
    }
}

#[test]
fn an_interned_document_survives_and_says_it_is_interned() {
    // What a collection stores is not what the caller handed it: the keys are
    // ids from the collection's table. That form is what would actually be
    // written, so it is the form that has to go through the log, and the
    // interned bit has to still be set at the other end or a reader will look
    // for key bytes that are not there.
    let mut collection = Docs::new();
    let doc = order(3, "open", 4.25);
    assert!(collection.put_bytes(b"order:3", &doc).unwrap());
    let stored = collection.get(b"order:3").expect("stored").value();
    assert!(stored.is_interned());
    let bytes = stored
        .as_bytes()
        .expect("a container has its bytes")
        .to_vec();

    let mut log = log();
    let put = log
        .append(&RecordHeader::new(RecordKind::Document), b"order:3", &bytes)
        .unwrap();
    let rec = log.read(put.addr).unwrap();
    assert_eq!(rec.value, &bytes[..]);

    let body = DocumentBody::decode(rec.value).expect("decodes");
    assert_eq!(body.tag(), ValueTag::Container);
    assert!(body.is_interned());
    assert_eq!(body.count(), 4);

    let back = Value::new(rec.value).expect("readable");
    assert!(back.validate());
    assert!(back.is_interned());
    // The ids only mean something against the collection's table, which is why
    // the table is a collection chunk and not part of any record. Reading a
    // member by id needs nothing else, and that much works from the bytes
    // alone.
    let want = stored.key_id_at(0).expect("an interned key has an id");
    assert_eq!(back.key_id_at(0), Some(want));
    assert!(back.get_id(want).is_some());
}

#[test]
fn a_document_that_lost_its_tail_is_caught() {
    // The failure this whole file is about, forced: a value that came back
    // shorter than the document in it. `Value::validate` is what says so, and
    // the point of the assertion is that it says so rather than reading past
    // the end or handing back a document with a truncated last member.
    let doc = order(9, "open", 1.0);
    for cut in 1..8 {
        let short = &doc[..doc.len() - cut];
        let hurt = Value::new(short).is_none_or(|v| !v.validate());
        assert!(hurt, "a document missing {cut} bytes read as whole");
    }
}
