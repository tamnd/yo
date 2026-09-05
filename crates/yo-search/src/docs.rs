//! The document table: which key an index has under which number, and what it
//! knows about each one besides its terms.
//!
//! ```
//! use yo_search::docs::Docs;
//!
//! let mut d = Docs::new();
//! let id = d.add(b"book:1", 1.0);
//! assert_eq!(id, 1);
//! assert_eq!(d.id(b"book:1"), Some(1));
//! assert_eq!(d.get(1).map(|doc| &*doc.key), Some(b"book:1".as_slice()));
//!
//! // Reading a key again does not reuse its number, it hands out a new one.
//! assert_eq!(d.add(b"book:1", 1.0), 2);
//! assert_eq!(d.get(1), None);
//! assert_eq!(d.last(), 2);
//! ```
//!
//! # A number is never given out twice
//!
//! A document that is written again is not found in the posting lists and
//! rewritten, it is given a fresh number and indexed as if it were new, and the
//! number it had before is left behind in every list it was in. So a number
//! names one reading of one key and nothing else, and the way to ask whether a
//! number still means anything is to look it up here and find nothing.
//!
//! That is what a real server does and it is measurable there. Editing a hash
//! moves its id, the id it had is still in a dump of the term it carried, and
//! asking that server which key the old id belongs to answers that the document
//! was removed. Doing it any other way would mean finding and rewriting every
//! list a document appears in, on every write, which is the cost the whole
//! design exists to avoid.
//!
//! The price is that the table keeps a hole where a number used to be, and a key
//! written a thousand times leaves a thousand holes. A real server collects them
//! later and so will this, and until then the holes are counted so nothing has
//! to guess at how much of the table is worth reading.

use std::collections::HashMap;

use crate::posts::Id;
use crate::sorted::Sorted;

/// What is known about one document besides which terms are in it.
#[derive(Debug, Clone, PartialEq)]
pub struct Doc {
    /// The key it was read from.
    pub key: Box<[u8]>,
    /// What the client said this document is worth, before any query.
    pub score: f64,
    /// The sum of every frequency in it, which is its length for scoring.
    ///
    /// Weighted, so a word in a field of weight three counts three times, and
    /// stems are not in it. That is the number a real server reports as
    /// `num_tokens`.
    pub tokens: u32,
    /// The largest frequency any one term in it has, reported as `max_freq`.
    pub top: u32,
    /// The value of the payload field, when the index names one.
    pub payload: Option<Box<[u8]>>,
    /// What it holds at each of the schema's sortable fields, in the order the
    /// schema declares them and with the fields nobody called sortable left
    /// out. Empty on an index that has no sortable field at all.
    sortable: Box<[Option<Sorted>]>,
}

impl Doc {
    /// A document with a key and a score and nothing counted yet.
    #[must_use]
    pub fn new(key: &[u8], score: f64) -> Doc {
        Doc {
            key: key.into(),
            score,
            tokens: 0,
            top: 0,
            payload: None,
            sortable: Box::default(),
        }
    }

    /// The value it keeps for one sortable field, by the slot the schema puts
    /// that field in.
    ///
    /// `None` when the document has nothing there, and also when the slot is
    /// past the end, which is what an `FT.ALTER` that adds a sortable field
    /// leaves behind on every document written before it.
    #[must_use]
    pub fn sorted(&self, slot: usize) -> Option<&Sorted> {
        self.sortable.get(slot)?.as_ref()
    }
}

/// Every document one index holds, by number and by key.
#[derive(Debug, Clone, Default)]
pub struct Docs {
    slots: Vec<Option<Doc>>,
    ids: HashMap<Box<[u8]>, Id>,
    live: usize,
}

impl Docs {
    /// A table with nothing in it.
    #[must_use]
    pub fn new() -> Docs {
        Docs::default()
    }

    /// Takes a key in and gives back the number it is indexed under.
    ///
    /// A key that is already here is dropped first, so its old number stops
    /// meaning anything and the new reading gets a number of its own. Numbers
    /// start at one, because zero is what a reply uses for no document.
    pub fn add(&mut self, key: &[u8], score: f64) -> Id {
        self.remove(key);
        self.slots.push(Some(Doc::new(key, score)));
        let id = self.slots.len() as Id;
        self.ids.insert(key.into(), id);
        self.live += 1;
        id
    }

    /// Drops a key, giving back the number it had.
    pub fn remove(&mut self, key: &[u8]) -> Option<Id> {
        let id = self.ids.remove(key)?;
        if let Some(slot) = self.slot_mut(id) {
            *slot = None;
            self.live -= 1;
        }
        Some(id)
    }

    /// Moves a document to another key, keeping the number it has.
    ///
    /// The one thing that writes a document without giving it a new number, and
    /// it is not an exception to the rule above so much as the rule not
    /// applying: nothing was read, the same reading of the same value is now
    /// under another name. That is what a real server does with a `RENAME`
    /// inside a prefix an index follows, and it is worth having because the
    /// alternative is reading a value nobody changed.
    ///
    /// Whatever the target had is dropped. A real server leaves it there, so
    /// two live numbers end up answering the same key and its document count is
    /// one too many, which is a leak rather than a behaviour and is D-64.
    ///
    /// `None` when there is nothing under `from`, and the target is left alone
    /// in that case, because a caller that hears no has a key to read instead.
    pub fn rename(&mut self, from: &[u8], to: &[u8]) -> Option<Id> {
        let id = self.ids.remove(from)?;
        if to != from {
            self.remove(to);
        }
        let Some(Some(doc)) = self.slot_mut(id) else {
            return None;
        };
        doc.key = to.into();
        self.ids.insert(to.into(), id);
        Some(id)
    }

    /// Counts a term's frequency towards a document's length and its largest.
    ///
    /// Called once per term as a document is indexed, which is what makes
    /// `tokens` the sum and `top` the maximum without either being worked out
    /// again afterwards.
    pub fn note(&mut self, id: Id, freq: u32) {
        if let Some(Some(doc)) = self.slot_mut(id) {
            doc.tokens = doc.tokens.saturating_add(freq);
            doc.top = doc.top.max(freq);
        }
    }

    /// Puts the payload on a document.
    pub fn carry(&mut self, id: Id, payload: &[u8]) {
        if let Some(Some(doc)) = self.slot_mut(id) {
            doc.payload = Some(payload.into());
        }
    }

    /// Puts the sortable fields' values on a document.
    pub fn store(&mut self, id: Id, values: Vec<Option<Sorted>>) {
        if let Some(Some(doc)) = self.slot_mut(id) {
            doc.sortable = values.into();
        }
    }

    /// The document under a number, or `None` when that number means nothing.
    #[must_use]
    pub fn get(&self, id: Id) -> Option<&Doc> {
        self.slots
            .get(usize::try_from(id).ok()?.checked_sub(1)?)?
            .as_ref()
    }

    /// The number a key is under.
    #[must_use]
    pub fn id(&self, key: &[u8]) -> Option<Id> {
        self.ids.get(key).copied()
    }

    /// The key a number belongs to.
    #[must_use]
    pub fn key(&self, id: Id) -> Option<&[u8]> {
        self.get(id).map(|doc| &*doc.key)
    }

    /// Whether a number was given out and no longer means anything.
    ///
    /// Which is not the same as never having been given out, and the two are
    /// told apart because a walk over a posting list wants to skip the first
    /// quietly and would rather hear about the second.
    #[must_use]
    pub fn gone(&self, id: Id) -> bool {
        let Ok(at) = usize::try_from(id) else {
            return false;
        };
        at >= 1 && at <= self.slots.len() && self.get(id).is_none()
    }

    /// The largest number given out, or zero when none has been.
    #[must_use]
    pub fn last(&self) -> Id {
        self.slots.len() as Id
    }

    /// How many documents are in the index.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.live
    }

    /// Whether the index holds no documents.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// How many numbers were given out and no longer mean anything.
    #[must_use]
    pub const fn holes(&self) -> usize {
        self.slots.len() - self.live
    }

    /// How long every document in the table is, added up.
    ///
    /// Worked out on the way rather than kept, because the two things that ask
    /// for it are a scorer setting up and a debug reply, and neither is on the
    /// path a write takes.
    #[must_use]
    pub fn tokens(&self) -> u64 {
        self.all().map(|(_, doc)| u64::from(doc.tokens)).sum()
    }

    /// The average length of a document, which is what BM25 divides by.
    ///
    /// Zero when there is nothing to average, because a scorer that meets an
    /// empty index should get a number it can divide by rather than a `None` it
    /// has to have an opinion about.
    #[must_use]
    pub fn average(&self) -> f64 {
        if self.live == 0 {
            return 0.0;
        }
        self.tokens() as f64 / self.live as f64
    }

    /// Every document with its number, in the order the numbers were given out.
    pub fn all(&self) -> impl Iterator<Item = (Id, &Doc)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(at, slot)| Some(((at + 1) as Id, slot.as_ref()?)))
    }

    /// The slot a number sits in, whether or not anything is in it.
    fn slot_mut(&mut self, id: Id) -> Option<&mut Option<Doc>> {
        self.slots
            .get_mut(usize::try_from(id).ok()?.checked_sub(1)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Numbers start at one and go up by one, and nothing hands out zero.
    #[test]
    fn numbers_start_at_one_and_climb() {
        let mut d = Docs::new();
        assert_eq!(d.add(b"a", 1.0), 1);
        assert_eq!(d.add(b"b", 1.0), 2);
        assert_eq!(d.add(b"c", 1.0), 3);
        assert_eq!(d.last(), 3);
        assert_eq!(d.len(), 3);
        assert_eq!(d.get(0), None);
        assert!(!d.gone(0));
    }

    /// Writing a key again gives it a new number and takes the old one out of
    /// service, which is what a real server does and what lets a posting list be
    /// written once and never gone back over.
    #[test]
    fn a_second_reading_is_a_new_number() {
        let mut d = Docs::new();
        assert_eq!(d.add(b"x:1", 1.0), 1);
        assert_eq!(d.add(b"x:2", 1.0), 2);
        assert_eq!(d.add(b"x:1", 1.0), 3);
        assert_eq!(d.id(b"x:1"), Some(3));
        assert_eq!(d.key(3), Some(b"x:1".as_slice()));
        assert_eq!(d.get(1), None);
        assert!(d.gone(1));
        assert!(!d.gone(2));
        assert!(!d.gone(4));
        assert_eq!(d.len(), 2);
        assert_eq!(d.last(), 3);
        assert_eq!(d.holes(), 1);
    }

    /// A key that goes leaves its number behind, and the number stays out of
    /// service rather than being handed to the next document.
    #[test]
    fn a_key_that_goes_leaves_a_hole() {
        let mut d = Docs::new();
        d.add(b"a", 1.0);
        d.add(b"b", 1.0);
        assert_eq!(d.remove(b"a"), Some(1));
        assert_eq!(d.remove(b"a"), None);
        assert_eq!(d.id(b"a"), None);
        assert!(d.gone(1));
        assert_eq!(d.add(b"c", 1.0), 3);
        assert_eq!(d.len(), 2);
        assert_eq!(d.holes(), 1);
    }

    /// A rename keeps the number the document had, which is the one write that
    /// does, and takes whatever the target had away with it.
    #[test]
    fn a_rename_keeps_the_number_and_drops_what_the_target_had() {
        let mut d = Docs::new();
        d.add(b"p:1", 1.0);
        d.add(b"p:2", 1.0);
        assert_eq!(d.rename(b"p:1", b"p:2"), Some(1));
        assert_eq!(d.id(b"p:2"), Some(1));
        assert_eq!(d.id(b"p:1"), None);
        assert_eq!(d.key(1), Some(b"p:2".as_slice()));
        assert_eq!(d.len(), 1);
        assert!(d.gone(2));
        // No number was handed out, so the next document still gets the third.
        assert_eq!(d.last(), 2);
        assert_eq!(d.add(b"p:3", 1.0), 3);
    }

    /// A key with nothing under it renames to nothing and leaves the target
    /// where it was, and a key renamed to itself is still there afterwards.
    #[test]
    fn a_rename_of_nothing_leaves_the_target_alone() {
        let mut d = Docs::new();
        d.add(b"p:2", 1.0);
        assert_eq!(d.rename(b"p:1", b"p:2"), None);
        assert_eq!(d.id(b"p:2"), Some(1));
        assert_eq!(d.rename(b"p:2", b"p:2"), Some(1));
        assert_eq!(d.id(b"p:2"), Some(1));
        assert_eq!(d.len(), 1);
    }

    /// A document's length is the sum of the frequencies put on it and its top
    /// is the largest of them, both counted as the terms arrive.
    #[test]
    fn a_length_is_added_up_as_the_terms_arrive() {
        let mut d = Docs::new();
        let id = d.add(b"a", 1.0);
        d.note(id, 2);
        d.note(id, 5);
        d.note(id, 1);
        let doc = d.get(id).expect("the document is there");
        assert_eq!(doc.tokens, 8);
        assert_eq!(doc.top, 5);
        // A number that means nothing takes no notice.
        d.note(99, 4);
        d.note(0, 4);
        assert_eq!(d.get(id).map(|doc| doc.tokens), Some(8));
    }

    /// The average length is what BM25 divides by, and an empty index answers
    /// with a number rather than with nothing.
    #[test]
    fn the_average_length_is_over_the_documents_that_are_left() {
        let mut d = Docs::new();
        assert_eq!(d.average(), 0.0);
        let a = d.add(b"a", 1.0);
        let b = d.add(b"b", 1.0);
        d.note(a, 2);
        d.note(b, 8);
        assert!((d.average() - 5.0).abs() < f64::EPSILON);
        d.remove(b"b");
        assert!((d.average() - 2.0).abs() < f64::EPSILON);
    }

    /// The score and the payload belong to the document and come back with it.
    #[test]
    fn a_document_carries_its_score_and_its_payload() {
        let mut d = Docs::new();
        let id = d.add(b"a", 0.5);
        assert_eq!(d.get(id).map(|doc| doc.score), Some(0.5));
        assert_eq!(d.get(id).and_then(|doc| doc.payload.clone()), None);
        d.carry(id, b"anything");
        assert_eq!(
            d.get(id).and_then(|doc| doc.payload.clone()).as_deref(),
            Some(b"anything".as_slice())
        );
    }

    /// The sortable values come back by slot, a slot the document has nothing
    /// in answers nothing, and so does a slot past the end, which is what a
    /// document written before the schema grew a sortable field has.
    #[test]
    fn a_document_carries_a_value_for_every_sortable_field() {
        let mut d = Docs::new();
        let id = d.add(b"a", 1.0);
        assert_eq!(d.get(id).and_then(|doc| doc.sorted(0)), None);
        d.store(id, vec![Some(Sorted::Number(2.5)), None]);
        let doc = d.get(id).expect("the document is there");
        assert_eq!(doc.sorted(0), Some(&Sorted::Number(2.5)));
        assert_eq!(doc.sorted(1), None);
        assert_eq!(doc.sorted(2), None);
    }

    /// Walking the table gives the documents that are left, in number order,
    /// and steps over the holes without mentioning them.
    #[test]
    fn a_walk_gives_what_is_left_in_order() {
        let mut d = Docs::new();
        for key in [b"a".as_slice(), b"b", b"c", b"d"] {
            d.add(key, 1.0);
        }
        d.remove(b"b");
        d.add(b"c", 1.0);
        let seen: Vec<_> = d.all().map(|(id, doc)| (id, doc.key.to_vec())).collect();
        assert_eq!(
            seen,
            [(1, b"a".to_vec()), (4, b"d".to_vec()), (5, b"c".to_vec())]
        );
        assert_eq!(d.len(), 3);
        assert_eq!(d.holes(), 2);
    }
}
