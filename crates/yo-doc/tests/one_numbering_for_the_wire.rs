//! The tags and flags this crate writes are the ones `yo-format` says are on
//! disk.
//!
//! Both crates carry the numbering, for the reason `yo_format::document` gives:
//! these are the bytes in a record, so they belong with the other frozen shapes,
//! and a reader that only wants to know whether a record holds an object or an
//! array should not have to pull in the document model to find out. Two copies
//! of a number is two numbers unless something holds them together, and this is
//! that something.
//!
//! It is written against the bytes a [`Builder`] actually produces rather than
//! against this crate's own enum, which is deliberate. An assertion that two
//! declarations agree is worth much less than one that says the writer puts the
//! byte on disk that the format says it does.

use yo_doc::{Builder, Value};
use yo_format::{DocumentBody, ValueTag, doc_flags};

/// The bytes of a value built by `f`.
fn built(f: impl FnOnce(&mut Builder) -> yo_common::Result<()>) -> Vec<u8> {
    let mut b = Builder::new();
    f(&mut b).expect("built");
    b.finish().expect("finished").to_vec()
}

#[test]
fn every_scalar_lands_on_the_tag_the_format_names() {
    let cases: Vec<(ValueTag, Vec<u8>)> = vec![
        (ValueTag::Null, built(|b| b.null())),
        (ValueTag::False, built(|b| b.bool(false))),
        (ValueTag::True, built(|b| b.bool(true))),
        (ValueTag::Int, built(|b| b.int(41_920))),
        (ValueTag::Float, built(|b| b.float(12.5))),
        (ValueTag::Text, built(|b| b.text("a wrench"))),
    ];
    for (want, bytes) in cases {
        let d = DocumentBody::decode(&bytes)
            .unwrap_or_else(|e| panic!("the format refused a {want:?} this crate wrote: {e}"));
        assert_eq!(d.tag(), want);
        assert_eq!(
            d.count(),
            bytes.len() - 4,
            "a {want:?} carries its payload length"
        );
    }
}

#[test]
fn an_object_and_an_array_are_told_apart_the_same_way_by_both() {
    let object = built(|b| {
        b.begin_object()?;
        b.key(b"status")?;
        b.text("open")?;
        b.end_object()
    });
    let d = DocumentBody::decode(&object).expect("an object decodes");
    assert_eq!(d.tag(), ValueTag::Container);
    assert!(!d.is_array());
    assert_eq!(d.count(), 1);
    assert_eq!(
        Value::new(&object).expect("readable").len(),
        d.count(),
        "the count the record layer reads is the count the reader reads"
    );

    let array = built(|b| {
        b.begin_array()?;
        b.int(1)?;
        b.int(2)?;
        b.int(3)?;
        b.end_array()
    });
    let d = DocumentBody::decode(&array).expect("an array decodes");
    assert!(d.is_array());
    assert_eq!(d.count(), 3);
}

#[test]
fn the_flags_this_crate_sets_are_the_flags_the_format_declares() {
    let object = built(|b| {
        b.begin_object()?;
        b.key(b"a")?;
        b.int(1)?;
        b.key(b"b")?;
        b.int(2)?;
        b.end_object()
    });
    let head = DocumentBody::decode(&object).expect("decodes").head();
    // Every object this version writes is sorted and carries offsets, and no
    // object a `Builder` writes is interned, since interning happens when a
    // document is put into a collection and not before.
    assert_ne!(
        head & doc_flags::SORTED,
        0,
        "objects are written in key order"
    );
    assert_ne!(head & doc_flags::OFFSETS, 0, "entry tables carry offsets");
    assert_eq!(head & doc_flags::ARRAY, 0);
    assert_eq!(head & doc_flags::INTERNED, 0);
    assert!(!Value::new(&object).expect("readable").is_interned());
}

#[test]
fn an_interned_object_says_so_to_both_of_them() {
    // Interning is what `Docs::put` does, so this goes through a collection
    // rather than through a builder.
    let doc = built(|b| {
        b.begin_object()?;
        b.key(b"customer")?;
        b.int(7)?;
        b.key(b"status")?;
        b.text("open")?;
        b.end_object()
    });
    let mut docs = yo_doc::Docs::new();
    assert!(docs.put_bytes(b"order:1", &doc).expect("stored"));
    let stored = docs.get(b"order:1").expect("stored").value();
    assert!(stored.is_interned(), "a collection interns what it stores");

    let bytes = stored.as_bytes().expect("a container has its bytes");
    let d = DocumentBody::decode(bytes).expect("the format reads what the collection wrote");
    assert!(d.is_interned());
    assert!(!d.is_array());
    assert_eq!(d.count(), 2);
}
