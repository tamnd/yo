//! Changing a document at the places a path matched.
//!
//! [`Path::select`](crate::Path::select) answers the values a path names and
//! [`Value::offset_in`] turns each of them into a byte offset inside the root,
//! so a write is a list of offsets and what to do at each one. That is what
//! [`edit()`] takes.
//!
//! ```
//! use yo_doc::{Edit, Path, Value, edit, from_json};
//!
//! let doc = from_json(br#"{"a": {"n": 1}, "b": {"n": 2}}"#)?;
//! let root = Value::new(&doc).expect("readable");
//!
//! // Every n in the document, set to 9.
//! let nine = from_json(b"9")?;
//! let mut hits = Vec::new();
//! Path::parse(b"$..n")?.select(&root, &mut hits);
//! let at: Vec<_> = hits
//!     .iter()
//!     .map(|v| (v.offset_in(&root).expect("from this document"), Edit::Set(&nine)))
//!     .collect();
//!
//! let after = edit(&root, &at)?;
//! let out = Value::new(&after).expect("readable");
//! assert_eq!(out.to_json()?, br#"{"a":{"n":9},"b":{"n":9}}"#);
//! # Ok::<(), yo_common::Error>(())
//! ```
//!
//! # A document is rebuilt and not patched
//!
//! Nothing here writes into the bytes it was given. A value's length lives in
//! its header and every container above it holds an offset table, so changing
//! one number from `1` to `1000000000000` moves the end of the document and
//! every offset between there and the root. Patching that in place is the same
//! work as rebuilding, with the difference that a rebuild cannot leave a
//! document half changed if it fails in the middle.
//!
//! What the rebuild does not do is re-encode. Only the containers on the way
//! down to a change are opened, and everything else goes through
//! [`Builder::embed`](crate::Builder::embed), which is a memcpy of bytes that
//! are already in the right form. So a `JSON.SET` two levels into a hundred
//! kilobyte document copies a hundred kilobytes and encodes four values, and
//! the cost follows the size of the document rather than the number of changes.
//!
//! # An edit inside a value that is going away is dropped
//!
//! `$..a` on `{"a": {"a": 1}}` matches twice and the outer match holds the
//! inner one. A `JSON.DEL` with that path is meant to leave nothing behind, and
//! removing the outer object already removed the inner one, so the inner edit
//! has nothing left to change and is quietly skipped. The alternative is
//! refusing a path that a real Redis accepts.
//!
//! Going away is the important word. A [`Edit::Set`] replaces everything below
//! it and a [`Edit::Splice`] replaces only the run it names, so an edit inside
//! an element the splice keeps still happens, and so does an edit inside a
//! member an [`Edit::Put`] leaves alone. `$..*` on `{"a": [[7], [7, 7]]}`
//! matches the outer array and both inner ones, and `JSON.ARRAPPEND` with that
//! path has to reach all three.
//!
//! An offset that no value in the document begins at is a different thing and
//! is refused, because the only way to hold one is a bug in whatever worked it
//! out, and a write that silently does nothing is the worst way to find out.

use yo_common::{Code, Error, Result};

use crate::build::Builder;
use crate::head::Kind;
use crate::read::Value;

/// What to do at one place in a document.
///
/// Every set of bytes here is an encoded value and not JSON text. The callers
/// are the `JSON.*` commands, which have parsed their argument with
/// [`Builder::json`](crate::Builder::json) by this point, and the ones that
/// compute rather than parse, like `JSON.NUMINCRBY`, never had text at all.
#[derive(Debug, Clone, Copy)]
pub enum Edit<'a> {
    /// Put these bytes in place of the value that is here.
    Set(&'a [u8]),
    /// Take the value that is here out of whatever holds it.
    ///
    /// Removing the root is refused, because a document with no value in it is
    /// not a document. `JSON.DEL $` deletes the key, which is the keyspace's
    /// business and not this file's.
    Remove,
    /// Put a member under this key into the object that is here.
    ///
    /// The key replaces the one already there if there is one, so this is both
    /// halves of what `JSON.SET` does at a path whose last step is a name.
    Put(&'a [u8], &'a [u8]),
    /// Replace `take` elements of the array that is here, from `at`, with `put`.
    ///
    /// `at` and `take` are clamped to the array, so an append is `at` at or past
    /// the end with `take` of zero, and a trim is `at` of zero with `take` of
    /// the whole length and the kept elements in `put`.
    Splice {
        /// Where the replaced run starts.
        at: usize,
        /// How many elements it covers.
        take: usize,
        /// What goes in their place.
        put: &'a [&'a [u8]],
    },
}

/// Apply every edit in `at` to `root` and answer the document that results.
///
/// The offsets are into `root` and come from [`Value::offset_in`]. They may be
/// in any order. Two edits at the same offset is a caller bug and the first one
/// wins, and an edit inside a value that another edit replaces or removes is
/// dropped, for the reason in the module doc.
pub fn edit(root: &Value<'_>, at: &[(usize, Edit<'_>)]) -> Result<Vec<u8>> {
    if at
        .iter()
        .any(|(off, e)| *off == 0 && matches!(e, Edit::Remove))
    {
        return Err(no_root());
    }
    let mut done = vec![false; at.len()];
    let mut b = Builder::new();
    write(root, root, at, &mut done, &mut b)?;
    if done.contains(&false) {
        return Err(stray());
    }
    Ok(b.finish()?.to_vec())
}

/// Write `v` into `b`, with whatever the edits say about it and about the
/// values inside it.
fn write(
    root: &Value<'_>,
    v: &Value<'_>,
    at: &[(usize, Edit<'_>)],
    done: &mut [bool],
    b: &mut Builder,
) -> Result<()> {
    let off = v.offset_in(root).ok_or_else(stray)?;
    let Some(what) = find(at, off) else {
        if !inside(root, v, at)? {
            return b.embed(v);
        }
        return match v.kind() {
            Kind::Object => object(root, v, at, done, b),
            Kind::Array => array(root, v, at, done, b),
            // A scalar has nothing inside it, so an offset that landed in the
            // middle of one is not the start of any value in this document.
            _ => Err(stray()),
        };
    };
    // Everything this value holds goes with it, whichever of the four this is,
    // so nothing below this point is looked at again and the edits down there
    // are accounted for here.
    mark(at, done, off, off + v.encoded_len().ok_or_else(unreadable)?);
    match what {
        Edit::Set(bytes) => {
            let new = Value::new(bytes)
                .ok_or_else(|| Error::new(Code::Invalid, "the value written is not readable"))?;
            b.embed(&new)
        }
        // Handled by whoever holds this value, which skips it rather than
        // asking for it to be written. Reaching it here means it is the root,
        // and that is refused above.
        Edit::Remove => Err(no_root()),
        Edit::Put(key, value) => put(root, v, key, value, at, done, b),
        Edit::Splice {
            at: from,
            take,
            put,
        } => splice(root, v, Run { from, take, put }, at, done, b),
    }
}

/// Rebuild an object, following the edits inside it.
fn object(
    root: &Value<'_>,
    v: &Value<'_>,
    at: &[(usize, Edit<'_>)],
    done: &mut [bool],
    b: &mut Builder,
) -> Result<()> {
    let interned = v.is_interned();
    if interned {
        b.begin_object_interned()?;
    } else {
        b.begin_object()?;
    }
    for i in 0..v.len() {
        let child = v.at(i).ok_or_else(unreadable)?;
        if skipped(root, &child, at, done)? {
            continue;
        }
        if interned {
            b.key_id(v.key_id_at(i).ok_or_else(unreadable)?)?;
        } else {
            b.key(v.key_at(i).ok_or_else(unreadable)?)?;
        }
        write(root, &child, at, done, b)?;
    }
    b.end_object()
}

/// Rebuild an array, following the edits inside it.
fn array(
    root: &Value<'_>,
    v: &Value<'_>,
    at: &[(usize, Edit<'_>)],
    done: &mut [bool],
    b: &mut Builder,
) -> Result<()> {
    b.begin_array()?;
    for i in 0..v.len() {
        let child = v.at(i).ok_or_else(unreadable)?;
        if skipped(root, &child, at, done)? {
            continue;
        }
        write(root, &child, at, done, b)?;
    }
    b.end_array()
}

/// Rebuild an object with one more member in it.
///
/// The members already there are written through [`write`] rather than embedded,
/// because none of them is going away and an edit inside one of them still has
/// to happen.
fn put(
    root: &Value<'_>,
    v: &Value<'_>,
    key: &[u8],
    value: &[u8],
    at: &[(usize, Edit<'_>)],
    done: &mut [bool],
    b: &mut Builder,
) -> Result<()> {
    if v.kind() != Kind::Object {
        return Err(Error::new(
            Code::Invalid,
            "a key can only be put into an object",
        ));
    }
    if v.is_interned() {
        return Err(Error::new(
            Code::Invalid,
            "an object whose keys are interned needs the collection's key table to take a new key",
        ));
    }
    let new = Value::new(value)
        .ok_or_else(|| Error::new(Code::Invalid, "the value put is not readable"))?;
    b.begin_object()?;
    for i in 0..v.len() {
        let child = v.at(i).ok_or_else(unreadable)?;
        if skipped(root, &child, at, done)? {
            continue;
        }
        b.key(v.key_at(i).ok_or_else(unreadable)?)?;
        write(root, &child, at, done, b)?;
    }
    // Last, so that a key already in the object is the one that loses. The
    // builder keeps the last of a repeated key, which is what every JSON parser
    // does and what `JSON.SET` on a field that is already there has to do.
    b.key(key)?;
    b.embed(&new)?;
    b.end_object()
}

/// A run of an array being replaced, which is one [`Edit::Splice`] unpacked.
struct Run<'a> {
    /// Where the replaced run starts.
    from: usize,
    /// How many elements it covers.
    take: usize,
    /// What goes in their place.
    put: &'a [&'a [u8]],
}

/// Rebuild an array with a run of it replaced.
///
/// The elements outside the run are written through [`write`], for the reason
/// [`put`] gives. The ones inside it are the ones going away, so an edit in
/// there is dropped, which the `mark` in [`write`] already accounted for.
fn splice(
    root: &Value<'_>,
    v: &Value<'_>,
    run: Run<'_>,
    at: &[(usize, Edit<'_>)],
    done: &mut [bool],
    b: &mut Builder,
) -> Result<()> {
    if v.kind() != Kind::Array {
        return Err(Error::new(
            Code::Invalid,
            "elements can only be spliced into an array",
        ));
    }
    let n = v.len();
    let from = run.from.min(n);
    let end = from.saturating_add(run.take).min(n);
    b.begin_array()?;
    let kept = |i: usize, b: &mut Builder, done: &mut [bool]| -> Result<()> {
        let child = v.at(i).ok_or_else(unreadable)?;
        if skipped(root, &child, at, done)? {
            return Ok(());
        }
        write(root, &child, at, done, b)
    };
    for i in 0..from {
        kept(i, b, done)?;
    }
    for bytes in run.put {
        let new = Value::new(bytes)
            .ok_or_else(|| Error::new(Code::Invalid, "an element written is not readable"))?;
        b.embed(&new)?;
    }
    for i in end..n {
        kept(i, b, done)?;
    }
    b.end_array()
}

/// The edit at exactly this offset, if there is one.
///
/// A linear scan, because the list is the matches of one path and the walk only
/// reaches values that hold an edit or are one, so the two lengths multiply over
/// a small number rather than over the document.
fn find<'e>(at: &[(usize, Edit<'e>)], off: usize) -> Option<Edit<'e>> {
    at.iter().find(|(o, _)| *o == off).map(|(_, e)| *e)
}

/// Account for the edit at `off` and for every edit inside the value there.
fn mark(at: &[(usize, Edit<'_>)], done: &mut [bool], off: usize, end: usize) {
    for (i, (o, _)) in at.iter().enumerate() {
        if *o == off || (*o > off && *o < end) {
            done[i] = true;
        }
    }
}

/// Whether any edit lands strictly inside `v`.
fn inside(root: &Value<'_>, v: &Value<'_>, at: &[(usize, Edit<'_>)]) -> Result<bool> {
    let off = v.offset_in(root).ok_or_else(stray)?;
    let end = off + v.encoded_len().ok_or_else(unreadable)?;
    Ok(at.iter().any(|(o, _)| *o > off && *o < end))
}

/// Whether this child is being taken out of its container.
fn skipped(
    root: &Value<'_>,
    child: &Value<'_>,
    at: &[(usize, Edit<'_>)],
    done: &mut [bool],
) -> Result<bool> {
    let off = child.offset_in(root).ok_or_else(stray)?;
    if !matches!(find(at, off), Some(Edit::Remove)) {
        return Ok(false);
    }
    mark(
        at,
        done,
        off,
        off + child.encoded_len().ok_or_else(unreadable)?,
    );
    Ok(true)
}

fn no_root() -> Error {
    Error::new(
        Code::Invalid,
        "the whole document cannot be removed, only the key it is under",
    )
}

fn stray() -> Error {
    Error::new(
        Code::Invalid,
        "an offset that no value in this document begins at",
    )
}

fn unreadable() -> Error {
    Error::new(Code::Corrupt, "the document being edited is not readable")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Path;
    use crate::text::from_json;

    /// Every place `path` names in `doc`, as offsets into it.
    fn hits(doc: &[u8], path: &[u8]) -> Vec<usize> {
        let root = Value::new(doc).expect("readable");
        let mut found = Vec::new();
        Path::parse(path)
            .expect("the path parses")
            .select(&root, &mut found);
        found
            .iter()
            .map(|v| v.offset_in(&root).expect("from this document"))
            .collect()
    }

    /// `text` with `what` done at every place `path` names, back as JSON text.
    fn changed(text: &str, path: &[u8], what: Edit<'_>) -> String {
        let doc = from_json(text.as_bytes()).expect("the text parses");
        let root = Value::new(&doc).expect("readable");
        let at: Vec<_> = hits(&doc, path)
            .into_iter()
            .map(|off| (off, what))
            .collect();
        let after = edit(&root, &at).expect("the edit applies");
        let out = Value::new(&after).expect("readable");
        assert!(out.validate(), "the document that came out is whole");
        String::from_utf8(out.to_json().expect("writable")).expect("UTF-8")
    }

    #[test]
    fn a_value_is_replaced_wherever_the_path_names_it() {
        let nine = from_json(b"9").expect("parses");
        assert_eq!(
            changed(r#"{"a":{"n":1},"b":{"n":2}}"#, b"$..n", Edit::Set(&nine)),
            r#"{"a":{"n":9},"b":{"n":9}}"#
        );
        assert_eq!(
            changed(r#"{"a":[1,2,3]}"#, b"$.a[*]", Edit::Set(&nine)),
            r#"{"a":[9,9,9]}"#
        );
        assert_eq!(changed("[1,2]", b"$", Edit::Set(&nine)), "9");
    }

    #[test]
    fn a_value_that_is_removed_leaves_no_hole() {
        assert_eq!(
            changed(r#"{"a":1,"bb":2,"cc":3}"#, b"$.bb", Edit::Remove),
            r#"{"a":1,"cc":3}"#
        );
        assert_eq!(changed("[1,2,3]", b"$[1]", Edit::Remove), "[1,3]");
        assert_eq!(changed("[1,2,3]", b"$[*]", Edit::Remove), "[]");
        assert_eq!(
            changed(r#"{"a":{"n":1},"b":{"n":2}}"#, b"$..n", Edit::Remove),
            r#"{"a":{},"b":{}}"#
        );
    }

    #[test]
    fn a_removal_inside_a_removal_is_dropped_rather_than_refused() {
        // `$..a` matches the outer object and the number inside it. Taking the
        // outer one away already took the inner one, and the inner edit has
        // nothing left to change.
        assert_eq!(
            changed(r#"{"a":{"a":1},"b":2}"#, b"$..a", Edit::Remove),
            r#"{"b":2}"#
        );
    }

    #[test]
    fn the_whole_document_cannot_be_removed() {
        let doc = from_json(br#"{"a":1}"#).expect("parses");
        let root = Value::new(&doc).expect("readable");
        let why = edit(&root, &[(0, Edit::Remove)]).unwrap_err();
        assert!(
            why.message().contains("only the key it is under"),
            "{}",
            why.message()
        );
    }

    #[test]
    fn a_key_put_into_an_object_lands_in_key_order() {
        let one = from_json(b"1").expect("parses");
        assert_eq!(
            changed(r#"{"aa":1,"cc":3}"#, b"$", Edit::Put(b"bb", &one)),
            r#"{"aa":1,"bb":1,"cc":3}"#
        );
        // A key already there is replaced and not repeated.
        assert_eq!(
            changed(r#"{"aa":1,"cc":3}"#, b"$", Edit::Put(b"cc", &one)),
            r#"{"aa":1,"cc":1}"#
        );
        // Under every object a wildcard names, which is what `JSON.SET` with a
        // path like `$.*.tag` has to do.
        assert_eq!(
            changed(
                r#"{"a":{"x":1},"b":{"x":2}}"#,
                b"$.*",
                Edit::Put(b"n", &one)
            ),
            r#"{"a":{"n":1,"x":1},"b":{"n":1,"x":2}}"#
        );
    }

    #[test]
    fn a_key_cannot_be_put_into_something_that_is_not_an_object() {
        let one = from_json(b"1").expect("parses");
        let doc = from_json(b"[1,2]").expect("parses");
        let root = Value::new(&doc).expect("readable");
        let why = edit(&root, &[(0, Edit::Put(b"a", &one))]).unwrap_err();
        assert!(
            why.message().contains("only be put into an object"),
            "{}",
            why.message()
        );
    }

    #[test]
    fn a_splice_covers_appending_inserting_popping_and_trimming() {
        let seven = from_json(b"7").expect("parses");
        let eight = from_json(b"8").expect("parses");
        let put: &[&[u8]] = &[&seven, &eight];

        let append = Edit::Splice {
            at: usize::MAX,
            take: 0,
            put,
        };
        assert_eq!(changed("[1,2]", b"$", append), "[1,2,7,8]");

        let insert = Edit::Splice {
            at: 1,
            take: 0,
            put,
        };
        assert_eq!(changed("[1,2]", b"$", insert), "[1,7,8,2]");

        let pop = Edit::Splice {
            at: 2,
            take: 1,
            put: &[],
        };
        assert_eq!(changed("[1,2,3]", b"$", pop), "[1,2]");

        // A trim is the kept run put back over the whole array.
        let kept: &[&[u8]] = &[&seven];
        let trim = Edit::Splice {
            at: 0,
            take: usize::MAX,
            put: kept,
        };
        assert_eq!(changed("[1,2,3]", b"$", trim), "[7]");

        // Every array a descent names, all at once.
        let each = Edit::Splice {
            at: 0,
            take: 0,
            put: kept,
        };
        assert_eq!(
            changed(r#"{"a":[1],"b":[2]}"#, b"$..a", each),
            r#"{"a":[7,1],"b":[2]}"#
        );
    }

    #[test]
    fn elements_cannot_be_spliced_into_something_that_is_not_an_array() {
        let doc = from_json(br#"{"a":1}"#).expect("parses");
        let root = Value::new(&doc).expect("readable");
        let what = Edit::Splice {
            at: 0,
            take: 0,
            put: &[],
        };
        let why = edit(&root, &[(0, what)]).unwrap_err();
        assert!(
            why.message().contains("only be spliced into an array"),
            "{}",
            why.message()
        );
    }

    #[test]
    fn everything_the_edit_did_not_name_comes_back_byte_identical() {
        let text = br#"{"keep":[{"deep":[1,2,{"here":true}]},"text",null],"n":1.5,"go":0}"#;
        let doc = from_json(text).expect("parses");
        let root = Value::new(&doc).expect("readable");
        let at: Vec<_> = hits(&doc, b"$.go")
            .into_iter()
            .map(|off| (off, Edit::Remove))
            .collect();
        let after = edit(&root, &at).expect("applies");

        let out = Value::new(&after).expect("readable");
        assert_eq!(
            out.get(b"keep").expect("still there").as_bytes(),
            root.get(b"keep").expect("was there").as_bytes(),
            "the part nobody touched is the same bytes"
        );
        assert!(out.get(b"go").is_none());
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn a_splice_or_a_put_still_carries_the_edits_inside_what_it_keeps() {
        let one = from_json(b"1").expect("parses");
        let put: &[&[u8]] = &[&one];
        // `$..a` names the outer array and the inner one it holds, and an
        // append with that path has to reach both.
        assert_eq!(
            changed(
                r#"{"a":[{"a":[7]}]}"#,
                b"$..a",
                Edit::Splice {
                    at: usize::MAX,
                    take: 0,
                    put,
                }
            ),
            r#"{"a":[{"a":[7,1]},1]}"#
        );
        // The same for a member going into an object that also holds one.
        assert_eq!(
            changed(
                r#"{"o":{"o":{}}}"#,
                b"$..o",
                Edit::Put(b"n", one.as_slice())
            ),
            r#"{"o":{"n":1,"o":{"n":1}}}"#
        );
        // What the splice takes out is going away, so an edit in there is
        // dropped the way one inside a removed value is.
        assert_eq!(
            changed(
                r#"{"a":[{"a":[7]}]}"#,
                b"$..a",
                Edit::Splice {
                    at: 0,
                    take: usize::MAX,
                    put: &[],
                }
            ),
            r#"{"a":[]}"#
        );
    }

    #[test]
    fn no_edits_at_all_is_the_document_it_was_given() {
        let doc = from_json(br#"{"a":[1,{"b":"c"}],"d":null}"#).expect("parses");
        let root = Value::new(&doc).expect("readable");
        assert_eq!(edit(&root, &[]).expect("applies"), doc);
    }

    #[test]
    fn an_offset_that_is_not_a_value_says_so() {
        let doc = from_json(br#"{"a":1}"#).expect("parses");
        let root = Value::new(&doc).expect("readable");
        let one = from_json(b"1").expect("parses");
        // Two bytes into the root's own header, which is inside the document
        // and is not the start of anything in it.
        let why = edit(&root, &[(2, Edit::Set(&one))]).unwrap_err();
        assert!(
            why.message()
                .contains("no value in this document begins at"),
            "{}",
            why.message()
        );
        // Past the end of it, which is the same mistake from the other side.
        let why = edit(&root, &[(doc.len() + 8, Edit::Set(&one))]).unwrap_err();
        assert!(
            why.message()
                .contains("no value in this document begins at"),
            "{}",
            why.message()
        );
    }
}
