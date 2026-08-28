//! What changed, and whether it is safe (`15` sections 3.2 and 5).
//!
//! A tag comparison answers "the same or not". That answer on its own is the
//! worst error message a database can produce, because the person reading it
//! knows something moved and nothing about what. This module is the other
//! half: it finds the first real difference between two descriptions, says it
//! in a sentence, and says whether it is additive or breaking.

use core::fmt;

use yo_common::{Code, Error, Result};

use crate::desc::Desc;
use crate::parse::{Type, parse};

/// The kinds of difference worth naming separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeKind {
    /// One type became a different kind of type, or one primitive another.
    TypeChanged,
    /// A struct or an enum kept its shape and changed its name.
    TypeRenamed,
    /// A struct gained a field.
    FieldAdded,
    /// A struct lost a field.
    FieldRemoved,
    /// A field kept its position and changed its name.
    FieldRenamed,
    /// The same fields came back in a different order.
    FieldsReordered,
    /// A field kept its name and changed its type.
    FieldTypeChanged,
    /// An enum gained a variant.
    VariantAdded,
    /// An enum lost a variant.
    VariantRemoved,
    /// A variant kept its position and changed its name.
    VariantRenamed,
    /// A vector changed its width or the way it is compared.
    VectorChanged,
}

impl ChangeKind {
    /// Whether a change of this kind can be read through (`15` section 5).
    ///
    /// Only growth is additive: a new field reads as its default and a new
    /// variant is simply never seen in old elements. Everything else either
    /// moves bytes or drops them, and both need `migrate`.
    ///
    /// A rename annotated `#[yo(was = "old")]` is additive, and it is additive
    /// because the annotation makes the description keep the old name, so it
    /// never reaches this function as a rename at all.
    #[must_use]
    pub const fn is_additive(self) -> bool {
        matches!(self, ChangeKind::FieldAdded | ChangeKind::VariantAdded)
    }
}

/// The first difference between two shapes.
///
/// First rather than all: a shape mismatch is usually one edit, and a list of
/// every consequence of that edit is harder to read than the edit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Change {
    kind: ChangeKind,
    /// Where the owner sits in the shape, empty at the top.
    path: String,
    /// The struct or enum this is about, empty when the type has no name.
    owner: String,
    /// The field or variant this is about, empty when it is about a type.
    subject: String,
    stored: String,
    opening: String,
    position: Option<usize>,
}

impl Change {
    /// What kind of difference this is.
    #[must_use]
    pub const fn kind(&self) -> ChangeKind {
        self.kind
    }

    /// The path to the containing type, empty at the top.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// The struct or enum the difference is in, empty when the type has no
    /// name of its own.
    #[must_use]
    pub fn owner(&self) -> &str {
        &self.owner
    }

    /// The field or variant the difference is about, empty when it is about a
    /// whole type.
    #[must_use]
    pub fn subject(&self) -> &str {
        &self.subject
    }

    /// The stored side, rendered.
    #[must_use]
    pub fn stored(&self) -> &str {
        &self.stored
    }

    /// The opening side, rendered.
    #[must_use]
    pub fn opening(&self) -> &str {
        &self.opening
    }

    /// The field or variant position, where the difference has one.
    #[must_use]
    pub const fn position(&self) -> Option<usize> {
        self.position
    }

    /// Whether this change can be read through without a migration.
    #[must_use]
    pub const fn is_additive(&self) -> bool {
        self.kind.is_additive()
    }
}

impl fmt::Display for Change {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.position.unwrap_or_default();
        match self.kind {
            ChangeKind::TypeChanged => {
                write!(
                    f,
                    "the type changed from {} to {}",
                    self.stored, self.opening
                )?;
            }
            ChangeKind::TypeRenamed => write!(
                f,
                "the {} was renamed from \"{}\" to \"{}\", and a name is part of the shape",
                self.subject, self.stored, self.opening
            )?,
            ChangeKind::FieldAdded => write!(
                f,
                "struct {} gained field \"{}\" at position {n}",
                self.owner, self.opening
            )?,
            ChangeKind::FieldRemoved => write!(
                f,
                "struct {} lost field \"{}\", which was at position {n}",
                self.owner, self.stored
            )?,
            ChangeKind::FieldRenamed => write!(
                f,
                "struct {} renamed field {n} from \"{}\" to \"{}\"",
                self.owner, self.stored, self.opening
            )?,
            ChangeKind::FieldsReordered => write!(
                f,
                "struct {} has the same fields in a different order, and field order is layout",
                self.owner
            )?,
            ChangeKind::FieldTypeChanged => write!(
                f,
                "field \"{}\" of struct {} changed type from {} to {}",
                self.subject, self.owner, self.stored, self.opening
            )?,
            ChangeKind::VariantAdded => write!(
                f,
                "enum {} gained variant \"{}\" at position {n}",
                self.owner, self.opening
            )?,
            ChangeKind::VariantRemoved => write!(
                f,
                "enum {} lost variant \"{}\", which was at position {n}",
                self.owner, self.stored
            )?,
            ChangeKind::VariantRenamed => write!(
                f,
                "enum {} renamed variant {n} from \"{}\" to \"{}\"",
                self.owner, self.stored, self.opening
            )?,
            ChangeKind::VectorChanged => write!(
                f,
                "the vector changed from {} to {}",
                self.stored, self.opening
            )?,
        }
        // A named type says which one it is, so the path would only repeat
        // itself. An unnamed one needs it to be findable at all.
        if self.owner.is_empty() && !self.path.is_empty() {
            write!(f, " (at {})", self.path)?;
        }
        Ok(())
    }
}

/// Who created a collection, as the catalogue records it.
///
/// "Who wrote this and with what" is the first question when a diff is
/// surprising, so the answer travels with the error rather than being one more
/// thing to go and look up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance {
    /// When the collection was created, however the catalogue spells a date.
    pub created: String,
    /// The SDK that created it, such as `yo-python`.
    pub sdk: String,
    /// That SDK's version.
    pub version: String,
}

/// The first difference between two parsed shapes, or `None` if they agree.
#[must_use]
pub fn compare(stored: &Type, opening: &Type) -> Option<Change> {
    walk(stored, opening, "")
}

/// The check a typed handle runs when it opens a collection.
///
/// An empty stored description means the collection was created over RESP3 and
/// has no shape, which is not an error: it is checked per element instead
/// (`15` section 3.3).
///
/// The compatibility list that makes an additive change open silently lives in
/// the catalogue and arrives with it. Until then an additive change is
/// reported like any other, and says that it is additive.
///
/// # Errors
///
/// [`Code::ShapeMismatch`], with both shapes rendered, the difference
/// underlined, and `change=additive` or `change=breaking` in the detail.
pub fn check(
    collection: &str,
    stored: &Desc,
    opening: &Desc,
    by: Option<&Provenance>,
) -> Result<()> {
    if stored.is_empty() || stored.as_bytes() == opening.as_bytes() {
        return Ok(());
    }
    Err(mismatch(collection, stored, opening, by))
}

/// Build the mismatch error for two descriptions that differ.
///
/// Public because the wire path (`09` section 6) and the migration tool report
/// the same thing from descriptions they already hold.
#[must_use]
pub fn mismatch(collection: &str, stored: &Desc, opening: &Desc, by: Option<&Provenance>) -> Error {
    let (left, right) = (parse(stored), parse(opening));
    let (Ok(left), Ok(right)) = (left, right) else {
        // One of them will not parse, which is a different problem and not one
        // to dress up as a shape diff.
        return Error::fmt(
            Code::Corrupt,
            format_args!(
                "collection \"{collection}\" has a stored shape this build cannot read: {}",
                stored.as_text()
            ),
        );
    };

    let change = compare(&left, &right);
    let (stored_line, opening_line) = (left.to_string(), right.to_string());
    let credit = by.map_or_else(String::new, |p| {
        format!(" (created {} by {} {})", p.created, p.sdk, p.version)
    });

    let mut message = format!(
        "collection \"{collection}\" was created with a different shape\n\n  stored{credit}:\n      {stored_line}\n  opening:\n      {opening_line}\n"
    );
    if let Some(line) = underline(&stored_line, &opening_line) {
        message.push_str(&line);
        message.push('\n');
    }
    let additive = change.as_ref().is_some_and(Change::is_additive);
    if let Some(change) = &change {
        message.push_str(&format!("  {change}\n"));
    }
    message.push_str(if additive {
        "\n  This is an additive change."
    } else {
        "\n  This is a breaking change, so it needs a migrate."
    });

    Error::new(Code::ShapeMismatch, message).with_detail(if additive {
        "change=additive"
    } else {
        "change=breaking"
    })
}

/// Tildes under the part of the opening line that is not in the stored line.
///
/// Character based rather than byte based, because a name can be any UTF-8 and
/// an underline that is off by a few columns under a non-ASCII field name is
/// worse than none.
fn underline(stored: &str, opening: &str) -> Option<String> {
    let left: Vec<char> = stored.chars().collect();
    let right: Vec<char> = opening.chars().collect();

    let prefix = left.iter().zip(&right).take_while(|(a, b)| a == b).count();
    if prefix == left.len() && prefix == right.len() {
        return None;
    }
    let suffix = left[prefix..]
        .iter()
        .rev()
        .zip(right[prefix..].iter().rev())
        .take_while(|(a, b)| a == b)
        .count();

    let width = right.len() - prefix - suffix;
    // Nothing was added, only removed, so mark where it used to be.
    let width = width.max(1);
    let mut line = String::with_capacity(6 + prefix + width);
    line.push_str("      ");
    for _ in 0..prefix {
        line.push(' ');
    }
    for _ in 0..width {
        line.push('~');
    }
    Some(line)
}

fn join(path: &str, segment: &str) -> String {
    if path.is_empty() {
        segment.to_owned()
    } else {
        format!("{path}.{segment}")
    }
}

fn walk(stored: &Type, opening: &Type, path: &str) -> Option<Change> {
    match (stored, opening) {
        (Type::Prim(a), Type::Prim(b)) if a == b => None,
        (Type::Optional(a), Type::Optional(b)) => walk(a, b, path),
        (Type::List(a), Type::List(b)) => walk(a, b, &join(path, "item")),
        (Type::Map(ak, av), Type::Map(bk, bv)) => {
            walk(ak, bk, &join(path, "key")).or_else(|| walk(av, bv, &join(path, "value")))
        }
        (Type::Ref(a), Type::Ref(b)) if a == b => None,
        (
            Type::Vector {
                dim: ad,
                metric: am,
            },
            Type::Vector {
                dim: bd,
                metric: bm,
            },
        ) => {
            if ad == bd && am == bm {
                None
            } else {
                Some(Change {
                    kind: ChangeKind::VectorChanged,
                    path: path.to_owned(),
                    owner: String::new(),
                    subject: String::new(),
                    stored: format!("{ad} {am}"),
                    opening: format!("{bd} {bm}"),
                    position: None,
                })
            }
        }
        (
            Type::Struct {
                name: an,
                fields: af,
            },
            Type::Struct {
                name: bn,
                fields: bf,
            },
        ) => {
            if an != bn {
                return Some(renamed("struct", an, bn, path));
            }
            fields(an, af, bf, path)
        }
        (
            Type::Enum {
                name: an,
                variants: av,
            },
            Type::Enum {
                name: bn,
                variants: bv,
            },
        ) => {
            if an != bn {
                return Some(renamed("enum", an, bn, path));
            }
            variants(an, av, bv, path)
        }
        (a, b) if a == b => None,
        (a, b) => Some(Change {
            kind: ChangeKind::TypeChanged,
            path: path.to_owned(),
            owner: String::new(),
            subject: String::new(),
            stored: rendered(a),
            opening: rendered(b),
            position: None,
        }),
    }
}

fn renamed(what: &str, stored: &str, opening: &str, path: &str) -> Change {
    Change {
        kind: ChangeKind::TypeRenamed,
        path: path.to_owned(),
        owner: String::new(),
        subject: what.to_owned(),
        stored: stored.to_owned(),
        opening: opening.to_owned(),
        position: None,
    }
}

/// A type as it appears inside a sentence, which is the same rendering the
/// shape lines use.
fn rendered(ty: &Type) -> String {
    match ty {
        Type::Struct { .. } => ty.name().unwrap_or_default().to_owned(),
        other => other.to_string(),
    }
}

fn fields(
    owner: &str,
    stored: &[(String, Type)],
    opening: &[(String, Type)],
    path: &str,
) -> Option<Change> {
    let names =
        |list: &[(String, Type)]| -> Vec<String> { list.iter().map(|(n, _)| n.clone()).collect() };
    let (a, b) = (names(stored), names(opening));

    for i in 0..a.len().max(b.len()) {
        match (a.get(i), b.get(i)) {
            (Some(x), Some(y)) if x == y => {
                let here = join(path, x);
                if let Some(change) = walk(&stored[i].1, &opening[i].1, &here) {
                    // A plain type swap right here reads better as a sentence
                    // about the field than as one about a path.
                    if change.kind == ChangeKind::TypeChanged && change.path == here {
                        return Some(Change {
                            kind: ChangeKind::FieldTypeChanged,
                            path: path.to_owned(),
                            owner: owner.to_owned(),
                            subject: x.clone(),
                            ..change
                        });
                    }
                    return Some(change);
                }
            }
            (s, o) => {
                return Some(list_change(
                    ChangeKind::FieldAdded,
                    ChangeKind::FieldRemoved,
                    ChangeKind::FieldRenamed,
                    ChangeKind::FieldsReordered,
                    owner,
                    path,
                    &a,
                    &b,
                    i,
                    s,
                    o,
                ));
            }
        }
    }
    None
}

fn variants(owner: &str, stored: &[String], opening: &[String], path: &str) -> Option<Change> {
    for i in 0..stored.len().max(opening.len()) {
        let (s, o) = (stored.get(i), opening.get(i));
        if s == o {
            continue;
        }
        return Some(list_change(
            ChangeKind::VariantAdded,
            ChangeKind::VariantRemoved,
            ChangeKind::VariantRenamed,
            ChangeKind::FieldsReordered,
            owner,
            path,
            stored,
            opening,
            i,
            s,
            o,
        ));
    }
    None
}

/// The one piece of reasoning both lists share: at the first position where
/// two name lists disagree, decide whether something was added, removed,
/// renamed or only moved.
#[expect(
    clippy::too_many_arguments,
    reason = "four kinds and two lists, all of which the caller has and this does not want to own"
)]
fn list_change(
    added: ChangeKind,
    removed: ChangeKind,
    renamed_kind: ChangeKind,
    reordered: ChangeKind,
    owner: &str,
    path: &str,
    stored: &[String],
    opening: &[String],
    at: usize,
    s: Option<&String>,
    o: Option<&String>,
) -> Change {
    let has = |list: &[String], name: &String| list.iter().any(|n| n == name);
    let kind = match (s, o) {
        (None, Some(_)) => added,
        (Some(_), None) => removed,
        (Some(s), Some(o)) => {
            let (kept, brought) = (has(opening, s), has(stored, o));
            match (kept, brought) {
                // The stored name is still there further along and the new one
                // is genuinely new, so something was inserted.
                (true, false) => added,
                (false, true) => removed,
                (true, true) => reordered,
                (false, false) => renamed_kind,
            }
        }
        (None, None) => unreachable!("the loop only reaches a position one of the lists has"),
    };
    Change {
        kind,
        path: path.to_owned(),
        owner: owner.to_owned(),
        subject: String::new(),
        stored: s.cloned().unwrap_or_default(),
        opening: o.cloned().unwrap_or_default(),
        position: Some(at),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desc::{Describe, Metric, Shape};

    fn ty(build: impl FnOnce(&mut Desc)) -> Type {
        parse(&desc(build)).expect("this description was just written")
    }

    fn desc(build: impl FnOnce(&mut Desc)) -> Desc {
        let mut d = Desc::new();
        build(&mut d);
        d
    }

    fn order(fields: &[(&str, Describe)]) -> Desc {
        let owned: Vec<(&str, Describe)> = fields.to_vec();
        desc(move |d| d.strukt("Order", &owned))
    }

    #[test]
    fn the_same_shape_has_no_change() {
        let a = ty(|d| d.strukt("P", &[("x", u64::describe)]));
        assert_eq!(compare(&a, &a), None);
    }

    #[test]
    fn a_new_field_is_additive_and_says_where() {
        let a = ty(|d| d.strukt("Order", &[("id", u64::describe)]));
        let b = ty(|d| {
            d.strukt("Order", &[("id", u64::describe), ("total", f64::describe)]);
        });
        let change = compare(&a, &b).expect("a field appeared");
        assert_eq!(change.kind(), ChangeKind::FieldAdded);
        assert_eq!(change.position(), Some(1));
        assert!(change.is_additive());
        assert_eq!(
            change.to_string(),
            "struct Order gained field \"total\" at position 1"
        );
    }

    /// Inserted in the middle rather than appended, which is the case a naive
    /// pairwise walk calls a rename.
    #[test]
    fn a_field_inserted_in_the_middle_is_still_an_addition() {
        let a = ty(|d| {
            d.strukt("Order", &[("id", u64::describe), ("total", f64::describe)]);
        });
        let b = ty(|d| {
            d.strukt(
                "Order",
                &[
                    ("id", u64::describe),
                    ("customer", u64::describe),
                    ("total", f64::describe),
                ],
            );
        });
        let change = compare(&a, &b).expect("a field appeared");
        assert_eq!(change.kind(), ChangeKind::FieldAdded);
        assert_eq!(change.position(), Some(1));
        assert_eq!(change.opening(), "customer");
    }

    #[test]
    fn a_lost_field_is_breaking() {
        let a = order(&[("id", u64::describe), ("note", String::describe)]);
        let b = order(&[("id", u64::describe)]);
        let change = compare(&parse(&a).unwrap(), &parse(&b).unwrap()).expect("a field went");
        assert_eq!(change.kind(), ChangeKind::FieldRemoved);
        assert!(!change.is_additive());
        assert_eq!(
            change.to_string(),
            "struct Order lost field \"note\", which was at position 1"
        );
    }

    #[test]
    fn a_reorder_is_named_as_a_reorder() {
        let a = order(&[("id", u64::describe), ("total", f64::describe)]);
        let b = order(&[("total", f64::describe), ("id", u64::describe)]);
        let change = compare(&parse(&a).unwrap(), &parse(&b).unwrap()).expect("they moved");
        assert_eq!(change.kind(), ChangeKind::FieldsReordered);
        assert!(change.to_string().contains("field order is layout"));
    }

    #[test]
    fn a_rename_is_a_rename_and_not_two_edits() {
        let a = order(&[("id", u64::describe)]);
        let b = order(&[("key", u64::describe)]);
        let change = compare(&parse(&a).unwrap(), &parse(&b).unwrap()).expect("it was renamed");
        assert_eq!(change.kind(), ChangeKind::FieldRenamed);
        assert_eq!(
            change.to_string(),
            "struct Order renamed field 0 from \"id\" to \"key\""
        );
    }

    #[test]
    fn a_widened_field_says_both_types() {
        let a = order(&[("id", u32::describe)]);
        let b = order(&[("id", u64::describe)]);
        let change = compare(&parse(&a).unwrap(), &parse(&b).unwrap()).expect("it widened");
        assert_eq!(change.kind(), ChangeKind::FieldTypeChanged);
        assert!(!change.is_additive());
        assert_eq!(
            change.to_string(),
            "field \"id\" of struct Order changed type from u32 to u64"
        );
    }

    #[test]
    fn a_container_swap_is_a_type_change_with_a_path() {
        let a = order(&[("tags", <Vec<String> as Shape>::describe)]);
        let b = order(&[("tags", <Option<String> as Shape>::describe)]);
        let change = compare(&parse(&a).unwrap(), &parse(&b).unwrap()).expect("it changed");
        assert_eq!(change.kind(), ChangeKind::FieldTypeChanged);
        assert_eq!(
            change.to_string(),
            "field \"tags\" of struct Order changed type from L str to O str"
        );
    }

    /// The difference is two levels down, so the sentence names the inner type
    /// and the path leads to it.
    #[test]
    fn a_difference_inside_a_list_is_found() {
        let a = ty(|d| d.list(|d: &mut Desc| d.prim(crate::desc::Prim::U32)));
        let b = ty(|d| d.list(|d: &mut Desc| d.prim(crate::desc::Prim::U64)));
        let change = compare(&a, &b).expect("the element changed");
        assert_eq!(change.kind(), ChangeKind::TypeChanged);
        assert_eq!(
            change.to_string(),
            "the type changed from u32 to u64 (at item)"
        );
    }

    #[test]
    fn a_renamed_struct_is_reported_as_a_rename() {
        let a = ty(|d| d.strukt("Order", &[("id", u64::describe)]));
        let b = ty(|d| d.strukt("Purchase", &[("id", u64::describe)]));
        let change = compare(&a, &b).expect("the name changed");
        assert_eq!(change.kind(), ChangeKind::TypeRenamed);
        assert!(change.to_string().starts_with("the struct was renamed"));
    }

    #[test]
    fn a_new_variant_is_additive() {
        let a = ty(|d| d.enumeration("Status", &["Open", "Paid", "Shipped"]));
        let b = ty(|d| d.enumeration("Status", &["Open", "Paid", "Shipped", "Cancelled"]));
        let change = compare(&a, &b).expect("a variant appeared");
        assert_eq!(change.kind(), ChangeKind::VariantAdded);
        assert!(change.is_additive());
        assert_eq!(
            change.to_string(),
            "enum Status gained variant \"Cancelled\" at position 3"
        );
    }

    #[test]
    fn a_lost_variant_is_breaking() {
        let a = ty(|d| d.enumeration("Status", &["Open", "Paid"]));
        let b = ty(|d| d.enumeration("Status", &["Open"]));
        let change = compare(&a, &b).expect("a variant went");
        assert_eq!(change.kind(), ChangeKind::VariantRemoved);
        assert!(!change.is_additive());
    }

    #[test]
    fn a_vector_that_changed_metric_says_both() {
        let a = ty(|d| d.vector(768, Metric::Cosine));
        let b = ty(|d| d.vector(768, Metric::L2));
        let change = compare(&a, &b).expect("the metric changed");
        assert_eq!(change.kind(), ChangeKind::VectorChanged);
        assert_eq!(
            change.to_string(),
            "the vector changed from 768 cosine to 768 l2"
        );
    }

    #[test]
    fn an_untyped_collection_opens_with_any_type() {
        let opening = order(&[("id", u64::describe)]);
        assert!(check("orders", &Desc::new(), &opening, None).is_ok());
    }

    #[test]
    fn the_same_shape_opens() {
        let a = order(&[("id", u64::describe)]);
        let b = order(&[("id", u64::describe)]);
        assert!(check("orders", &a, &b, None).is_ok());
    }

    /// The whole message, because the message is the feature. This is the
    /// example in `15` section 3.2 with the parts that need a catalogue left
    /// out.
    #[test]
    fn the_mismatch_message_shows_both_shapes_and_underlines_the_difference() {
        fn open_status(d: &mut Desc) {
            d.enumeration("Status", &["Open", "Paid", "Shipped"]);
        }
        fn cancelled_status(d: &mut Desc) {
            d.enumeration("Status", &["Open", "Paid", "Shipped", "Cancelled"]);
        }
        let stored = order(&[("id", u64::describe), ("status", open_status)]);
        let opening = order(&[("id", u64::describe), ("status", cancelled_status)]);
        let by = Provenance {
            created: "2026-08-01".into(),
            sdk: "yo-python".into(),
            version: "0.3.1".into(),
        };

        let e = check("orders", &stored, &opening, Some(&by)).expect_err("the shape moved");
        assert_eq!(e.code(), Code::ShapeMismatch);
        assert_eq!(e.detail(), Some("change=additive"));
        assert_eq!(
            e.message(),
            "collection \"orders\" was created with a different shape\n\
             \n\
             \x20 stored (created 2026-08-01 by yo-python 0.3.1):\n\
             \x20     Order { id: u64, status: E Status[Open,Paid,Shipped] }\n\
             \x20 opening:\n\
             \x20     Order { id: u64, status: E Status[Open,Paid,Shipped,Cancelled] }\n\
             \x20                                                        ~~~~~~~~~~\n\
             \x20 enum Status gained variant \"Cancelled\" at position 3\n\
             \n\
             \x20 This is an additive change."
        );
        assert!(
            e.to_string()
                .contains("https://yo.tamnd.dev/errors/shape-mismatch")
        );
    }

    #[test]
    fn a_breaking_message_says_so_and_says_migrate() {
        let stored = order(&[("id", u64::describe)]);
        let opening = order(&[("id", String::describe)]);
        let e = check("orders", &stored, &opening, None).expect_err("the shape moved");
        assert_eq!(e.detail(), Some("change=breaking"));
        assert!(e.message().contains("needs a migrate"), "{e}");
        assert!(e.message().contains("  stored:\n"), "{e}");
    }

    /// A removal has nothing to underline in the opening line, so the mark
    /// goes where the removed part used to start rather than nowhere.
    #[test]
    fn a_removal_still_gets_a_mark() {
        let line = underline("Order { id: u64, note: str }", "Order { id: u64 }").unwrap();
        assert_eq!(line.trim_start(), "~");
        assert_eq!(line.len(), 6 + "Order { id: u64".len() + 1);
    }

    #[test]
    fn identical_lines_have_no_underline() {
        assert_eq!(underline("Order { id: u64 }", "Order { id: u64 }"), None);
    }

    /// A description the file has and this build cannot read is a corruption,
    /// not a shape difference, and it says so with the right code.
    #[test]
    fn an_unreadable_stored_shape_is_corruption() {
        let e = check(
            "orders",
            &Desc::from_bytes(b"S\x05Ord".to_vec()),
            &order(&[]),
            None,
        )
        .expect_err("that is not a shape");
        assert_eq!(e.code(), Code::Corrupt);
        assert!(e.message().contains("cannot read"), "{e}");
    }
}
