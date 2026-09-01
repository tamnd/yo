//! A hash, in whichever of the two representations currently fits it.
//!
//! A hash is a listpack of alternating fields and values, or an element table
//! with the values in a blob beside it. Which one is not a free choice: `OBJECT
//! ENCODING` has to answer `listpack` or `hashtable` at exactly the sizes a real
//! server answers them, so the rule here is `hash_max_listpack_entries` and
//! `hash_max_listpack_value` read off `t_hash.c` in the 8.10.1 tarball.
//!
//! ```text
//!   small, any bytes                    everything else
//! +---------------------------+   +---------------------------------+
//! | f | v | f | v | f | v ... |-->| element table + a value blob    |
//! | ~2 B a side, walked        |   | one probe, no cap               |
//! +---------------------------+   +---------------------------------+
//!   to 512 fields, 64 B a side
//! ```
//!
//! Promotion is one-way and upward, which is Y4. The set has three bands because
//! an all integer set has an intset to be; a hash has no equivalent, because
//! there is no representation that is cheaper for a hash whose fields happen to
//! be numbers.
//!
//! # Where the values live
//!
//! In the listpack they are simply the odd elements, which is why
//! [`Listpack::find`] takes a step: a field is at an even index and its value is
//! the next one along, and searching with a step of two never matches a value by
//! accident. `HSET h a b` followed by `HGET h b` finds nothing, which is right,
//! and a search with a step of one would have found the `b` that is a value.
//!
//! In the table band a row's payload is an offset into a [`Blob`] the hash owns,
//! and the value's length is written into the blob in front of the bytes rather
//! than carried next to the offset. That is `05` section 4.2's element per row: a
//! value is bytes in a shared stretch, not an allocation of its own, and
//! rewriting one appends and abandons rather than moving everything after it. The
//! abandoned bytes are counted and come back when they outnumber the live ones.
//!
//! Field names are interned by the element table, which is the point of the
//! split. `HSET h field v1` and then `HSET h field v2` writes four bytes of
//! offset and the new value, and touches the field's name not at all.
//!
//! What a field costs, then, is eight bytes of row, four of offset, one of value
//! length, about five of slot array, and the field name and the value themselves.
//! Twelve of those eighteen are the two arrays and the only way past them is to
//! stop interning field names, which would put the name back in front of every
//! value and pay for it again on every rewrite.
//!
//! # Field TTL
//!
//! A field can be given its own deadline, which is the `HEXPIRE` family, and the
//! two bands pay for it differently.
//!
//! The packed band grows a third element per field the first time any field of
//! that hash is given one, holding the deadline in unix milliseconds or a zero
//! for no deadline. That is what Redis does and it is why `OBJECT ENCODING`
//! grows a third answer, `listpackex`. Everything below stays single path
//! because the walk takes a step of two or of three rather than there being two
//! copies of the code, and a hash that never sees `HEXPIRE` never widens.
//!
//! The table band hands the job to [`Deadlines`], a side array indexed by row
//! position that allocates nothing until the first deadline. [`crate::ttl`] is
//! where the reasoning for that lives, including why it is indexed by the row
//! and not by a number in the row.
//!
//! Widening is one way in both bands, the same as promotion: a hash whose last
//! deadline has been taken off keeps the shape, because going back would mean
//! rewriting the whole thing to save a byte a field on a hash that has already
//! shown it uses deadlines.
//!
//! # Expiry is lazy here too
//!
//! A field past its deadline is still sitting in the structure until something
//! looks at it. [`Hash::reap`] is that look, it is called by the keyspace before
//! any hash command runs, and it is guarded by one comparison against the
//! earliest deadline in the hash, so a hash with no field TTL pays a load and a
//! branch and nothing else. Every read path below can therefore treat what it
//! finds as live, which is what keeps `HGET` the shape it was before any of this
//! landed.
//!
//! A write clears a field's deadline. `HSET` on a field that had one leaves it
//! with no deadline, which is Redis's rule since 7.4 and is the reason `HGETEX`
//! exists to read a field without disturbing it.

use yo_common::num::{self, parse_i64};

use crate::blob::Blob;
use crate::elem::Elements;
use crate::listpack::{self, Listpack};
use crate::scan::Cursor;
use crate::ttl::{Applied, Ask, Cond, Deadlines, decide};

/// No deadline, the same sentinel [`crate::ttl`] uses and for the same reason.
const NONE: u64 = u64::MAX;

/// A field name or a value, as it is stored.
///
/// Both sides of a pair are the same thing to a listpack, which stores something
/// that looks like an integer as an integer. `HSET h f 42` and `HSET h f 042`
/// hold different bytes and both answer with what went in, and the formatting
/// happens once, into the reply buffer, the way Y18 asks.
pub type Text<'a> = listpack::Entry<'a>;

/// Where the encoding changes over.
///
/// These are `hash-max-listpack-entries` and `hash-max-listpack-value`, runtime
/// configuration in Redis, so they are passed in rather than being constants.
/// The value limit applies to a field name and to a value alike, which is what
/// `hashTypeTryConversion` does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// At this many fields a hash stops being a listpack.
    pub max_listpack_entries: usize,
    /// A field or a value longer than this cannot go in a listpack.
    pub max_listpack_value: usize,
}

impl Limits {
    /// Redis's defaults: 512 and 64.
    ///
    /// The count is 512 and not the 128 everyone remembers, and everyone
    /// remembers 128 because that is what it was for years. Read off a running
    /// 8.10.1 with nothing in its config file rather than off the documentation,
    /// which is the only way to be sure of a number like this. It matters
    /// because a hash of two hundred fields answers `listpack` there, so it has
    /// to answer `listpack` here too.
    pub const DEFAULT: Limits = Limits {
        max_listpack_entries: 512,
        max_listpack_value: 64,
    };
}

impl Default for Limits {
    fn default() -> Limits {
        Limits::DEFAULT
    }
}

/// Which representation a hash is in, which is what `OBJECT ENCODING` reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// One packed blob of alternating fields and values, walked linearly.
    Listpack,
    /// The same blob widened to three elements a field, the third a deadline.
    ///
    /// Not a third band, which is the point: it is the packed band with a wider
    /// step, and a hash arrives here by being given a field deadline rather than
    /// by growing.
    ListpackEx,
    /// The element table, with the values in a blob beside it.
    Hashtable,
}

impl Encoding {
    /// The word `OBJECT ENCODING` replies with.
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Encoding::Listpack => "listpack",
            Encoding::ListpackEx => "listpackex",
            Encoding::Hashtable => "hashtable",
        }
    }
}

/// The packed band, at two elements a field or at three.
#[derive(Debug, Clone)]
struct Packed {
    lp: Listpack,
    /// Whether there is a deadline element after every value.
    ///
    /// One bool rather than two variants of a band, so that everything reached
    /// through [`Packed::step`] is written once and a hash without field TTL
    /// runs the same code a hash with it does.
    ex: bool,
    /// A lower bound on the earliest deadline here, or [`NONE`].
    ///
    /// Leans early for the reason [`Deadlines::soonest`] gives: it goes down
    /// when a deadline is set and does not go back up when one is taken off, so
    /// [`Hash::reap`] can walk for nothing but cannot sleep through an expiry.
    soonest: u64,
}

impl Packed {
    fn new() -> Packed {
        Packed {
            lp: Listpack::new(),
            ex: false,
            soonest: NONE,
        }
    }

    /// Two elements a field, or three once any field has a deadline.
    #[inline]
    const fn step(&self) -> usize {
        if self.ex { 3 } else { 2 }
    }

    #[inline]
    fn len(&self) -> usize {
        self.lp.len() / self.step()
    }

    /// Where `field`'s name is, which is also where its row starts.
    #[inline]
    fn find(&self, field: &[u8]) -> Option<usize> {
        self.lp.find(field, self.step())
    }

    /// The deadline on the row starting at `at`, if it has one.
    fn deadline(&self, at: usize) -> Option<u64> {
        if !self.ex {
            return None;
        }
        match self.lp.get(at + 2) {
            // A field with no deadline holds a zero rather than the slot being
            // left out, so the rows stay three wide and the step stays a
            // constant. Redis writes the same zero.
            Some(Text::Int(n)) => u64::try_from(n).ok().filter(|&at| at != 0),
            // A deadline goes in as digits and a listpack holds digits as a
            // number, so nothing else is a shape this band can be in.
            _ => None,
        }
    }

    /// Write a deadline, or a zero for none, onto the row starting at `at`.
    fn write_deadline(&mut self, at: usize, deadline: u64) {
        debug_assert!(self.ex, "widen before writing a deadline");
        let mut buf = [0u8; num::DIGITS_MAX];
        self.lp.replace(at + 2, num::u64_digits(&mut buf, deadline));
    }

    /// Store `value` against `field` and say whether the field is new.
    fn set(&mut self, field: &[u8], value: &[u8]) -> bool {
        match self.find(field) {
            Some(at) => {
                self.lp.replace(at + 1, value);
                if self.ex {
                    // A write clears the deadline. Redis's rule, and the reason
                    // HGETEX is a command rather than a flag on HGET.
                    self.write_deadline(at, 0);
                }
                false
            }
            None => {
                self.lp.push(field);
                self.lp.push(value);
                if self.ex {
                    self.lp.push(b"0");
                }
                true
            }
        }
    }

    /// Take the whole row starting at `at` out.
    #[inline]
    fn remove_at(&mut self, at: usize) -> bool {
        self.lp.delete(at, self.step())
    }

    /// Grow the third element, which is where `listpackex` starts.
    fn widen(&mut self) {
        if self.ex {
            return;
        }
        let mut fresh = Listpack::new();
        let mut pair = self.lp.iter();
        while let (Some(field), Some(value)) = (pair.next(), pair.next()) {
            push_text(&mut fresh, field);
            push_text(&mut fresh, value);
            fresh.push(b"0");
        }
        self.lp = fresh;
        self.ex = true;
    }

    /// The earliest deadline actually here, or [`NONE`].
    ///
    /// A walk, so only [`Hash::reap`] calls it, and only once it has walked the
    /// whole thing anyway and knows the bound it was carrying is stale.
    fn earliest(&self) -> u64 {
        let mut soonest = NONE;
        let mut at = 0;
        while at < self.lp.len() {
            if let Some(deadline) = self.deadline(at) {
                soonest = soonest.min(deadline);
            }
            at += self.step();
        }
        soonest
    }

    /// Drop every field whose deadline has passed, and say how many went.
    fn reap(&mut self, now: u64) -> usize {
        let mut gone = 0;
        let mut at = 0;
        while at < self.lp.len() {
            match self.deadline(at) {
                Some(deadline) if deadline <= now => {
                    self.remove_at(at);
                    gone += 1;
                }
                // Only step past a row that survived, because taking one out
                // moves the next row into this position.
                _ => at += self.step(),
            }
        }
        gone
    }
}

/// Put an entry back into a listpack, writing the digits of a number once.
///
/// The only caller is [`Packed::widen`], which is copying a listpack it already
/// holds, so a number that went in as a number comes back out as one and is
/// stored as one again.
fn push_text(lp: &mut Listpack, t: Text<'_>) {
    match t {
        Text::Str(s) => lp.push(s),
        Text::Int(n) => {
            let mut buf = [0u8; num::DIGITS_MAX];
            lp.push(num::i64_digits(&mut buf, n));
        }
    }
}

/// The native band: interned field names, values in a blob of their own.
#[derive(Debug, Clone)]
struct Table {
    fields: Elements<u32>,
    values: Blob,
    /// One slot per row once any field has a deadline, and nothing before then.
    ///
    /// It has to be told about every row this table gains or loses, in the same
    /// order, or the deadlines after a hole belong to the wrong fields. That is
    /// what the `inserted` and `removed` calls below are, and there is a test in
    /// [`crate::ttl`] that fails when one goes missing.
    ttl: Deadlines,
}

impl Table {
    fn new(hint: usize) -> Table {
        Table {
            fields: Elements::with_capacity(hint),
            // Sixteen bytes a value is a guess and being wrong about it costs a
            // realloc, which is what a guess is allowed to cost.
            values: Blob::with_capacity(hint.saturating_mul(16)),
            ttl: Deadlines::new(),
        }
    }

    #[inline]
    fn get(&self, field: &[u8]) -> Option<&[u8]> {
        self.fields.get(field).map(|&at| self.values.sized(at))
    }

    /// Store `value` against `field` and say whether the field is new.
    fn set(&mut self, field: &[u8], value: &[u8]) -> bool {
        let at = self.values.push_sized(value);
        if let Some(row) = self.fields.index_of(field) {
            let slot = self.fields.at_mut(row).expect("the probe found it");
            let old = std::mem::replace(slot, at);
            self.values.release_sized(old);
            // A write clears the deadline, the same as in the packed band.
            self.ttl.clear(row);
            self.settle();
            return false;
        }
        match self.fields.insert(field, at) {
            Ok(_) => {
                self.ttl.inserted();
                true
            }
            Err(_) => {
                // A field name over NAME_MAX or a table at MAX_ROWS. The value
                // bytes are already in the blob, so they are given back rather
                // than left as a leak nothing accounts for.
                self.values.release_sized(at);
                self.settle();
                false
            }
        }
    }

    fn remove(&mut self, field: &[u8]) -> bool {
        match self.fields.index_of(field) {
            Some(row) => {
                self.remove_at(row);
                true
            }
            None => false,
        }
    }

    /// Take the row at `row` out, keeping the deadlines lined up with it.
    ///
    /// The one place a row leaves this table, so that the swap remove and the
    /// deadline that has to follow it cannot drift apart in a later edit.
    fn remove_at(&mut self, row: usize) {
        let at = self
            .fields
            .remove_at(row)
            .expect("the caller found the row");
        self.values.release_sized(at);
        self.ttl.removed(row);
        self.settle();
    }

    /// Give the dead value bytes back once there are more of them than live.
    fn settle(&mut self) {
        if !self.values.worth_compacting() {
            return;
        }
        let fields = &mut self.fields;
        self.values.compact(|keep| {
            for at in fields.payloads_mut() {
                keep.moved_sized(at);
            }
        });
    }
}

/// The two representations.
#[derive(Debug, Clone)]
enum Body {
    Packed(Packed),
    Table(Table),
}

/// A hash of fields to values.
#[derive(Debug, Clone)]
pub struct Hash {
    body: Body,
}

impl Default for Hash {
    fn default() -> Hash {
        Hash::new()
    }
}

impl Hash {
    /// An empty hash, which starts as a listpack.
    #[must_use]
    pub fn new() -> Hash {
        Hash {
            body: Body::Packed(Packed::new()),
        }
    }

    /// An empty hash sized for what is about to go in it.
    ///
    /// `HSET k f1 v1 f2 v2 ...` with a thousand pairs builds a table once rather
    /// than converting on the way there. The hint is only a hint and being wrong
    /// costs a conversion and no correctness.
    #[must_use]
    pub fn with_hint(hint: usize, limits: &Limits) -> Hash {
        if hint <= limits.max_listpack_entries {
            Hash::new()
        } else {
            Hash {
                body: Body::Table(Table::new(hint)),
            }
        }
    }

    /// Which representation this is in.
    #[inline]
    #[must_use]
    pub const fn encoding(&self) -> Encoding {
        match &self.body {
            Body::Packed(p) if p.ex => Encoding::ListpackEx,
            Body::Packed(_) => Encoding::Listpack,
            Body::Table(_) => Encoding::Hashtable,
        }
    }

    /// The bytes behind a hash on the packed band, for `DUMP` to copy.
    ///
    /// Field and value alternate in here exactly as `HASH_LISTPACK` wants them.
    /// `None` on the table, and `None` on the wider band as well: the deadline
    /// column has its own type byte and its own header, and a hash that has been
    /// widened once keeps the third element per field even after every deadline
    /// has been taken off again, so the blob is only ever safe to copy when
    /// there is no deadline column at all.
    #[inline]
    pub(crate) fn packed_bytes(&self) -> Option<&[u8]> {
        match &self.body {
            Body::Packed(p) if !p.ex => Some(p.lp.as_bytes()),
            _ => None,
        }
    }

    /// How many fields. This is `HLEN`.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        match &self.body {
            Body::Packed(p) => p.len(),
            Body::Table(t) => t.fields.len(),
        }
    }

    /// Whether there are none.
    ///
    /// An empty hash does not exist in Redis, so the caller deletes the key when
    /// this turns true rather than storing an empty one.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// What is stored against `field`. This is `HGET`.
    ///
    /// A field past its deadline is still here until [`Hash::reap`] runs, and
    /// the keyspace runs it before any command, so what this finds is live.
    #[must_use]
    pub fn get(&self, field: &[u8]) -> Option<Text<'_>> {
        match &self.body {
            Body::Packed(p) => {
                let at = p.find(field)?;
                p.lp.get(at + 1)
            }
            Body::Table(t) => t.get(field).map(Text::Str),
        }
    }

    /// Whether `field` is here at all. This is `HEXISTS`.
    #[must_use]
    pub fn contains(&self, field: &[u8]) -> bool {
        match &self.body {
            Body::Packed(p) => p.find(field).is_some(),
            Body::Table(t) => t.fields.contains(field),
        }
    }

    /// How long the value against `field` is. This is `HSTRLEN`.
    ///
    /// A missing field is zero to Redis and `None` here, because the layer that
    /// knows it is answering `HSTRLEN` is the one that should decide that a
    /// missing field and an empty value give the same number.
    #[must_use]
    pub fn value_len(&self, field: &[u8]) -> Option<usize> {
        match &self.body {
            Body::Packed(_) => self.get(field).map(|v| v.byte_len()),
            Body::Table(t) => t.fields.get(field).map(|&at| t.values.sized_len(at)),
        }
    }

    /// The pair at `index`, in whatever order the representation holds them.
    ///
    /// Insertion order in both bands, and neither is a promise. `HRANDFIELD`
    /// needs positions and this is what gives it them, the same way `SPOP` uses
    /// the set's.
    #[must_use]
    pub fn at(&self, index: usize) -> Option<(Text<'_>, Text<'_>)> {
        match &self.body {
            Body::Packed(p) => {
                let at = index * p.step();
                let field = p.lp.get(at)?;
                let value = p.lp.get(at + 1)?;
                Some((field, value))
            }
            Body::Table(t) => {
                let (name, at) = t.fields.at(index)?;
                Some((Text::Str(name), Text::Str(t.values.sized(*at))))
            }
        }
    }

    /// The deadline on the field at `index`, if it has one.
    ///
    /// The positional twin of [`Hash::deadline`], which takes a field name.
    /// `DUMP` is what wants this: it is already walking by index and looking the
    /// name back up to ask about its deadline would mean formatting every
    /// integer field into digits just to hand them straight back.
    #[must_use]
    pub fn deadline_at(&self, index: usize) -> Option<u64> {
        match &self.body {
            Body::Packed(p) => p.deadline(index * p.step()),
            Body::Table(t) => t.ttl.get(index),
        }
    }

    /// Every field and its value, in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (Text<'_>, Text<'_>)> {
        (0..self.len()).map(|i| self.at(i).expect("index is under the length"))
    }

    /// Walk part of the hash and say where to resume. This is `HSCAN`.
    ///
    /// Only the table band walks in windows, for the reason [`crate::set::Set`]
    /// gives: a hundred and twenty eight fields is smaller than the arithmetic
    /// to split them up, and a hash that small cannot hold the loop long enough
    /// for splitting to buy anything. A listpack hands back everything and
    /// [`Cursor::END`], ignoring the cursor it was given, which is safe because
    /// promotion is one way.
    pub fn scan<F>(&self, cursor: Cursor, count: usize, mut f: F) -> Cursor
    where
        F: FnMut(Text<'_>, Text<'_>),
    {
        match &self.body {
            Body::Table(t) => {
                let values = &t.values;
                t.fields.scan(cursor, count, |name, at| {
                    f(Text::Str(name), Text::Str(values.sized(*at)));
                })
            }
            Body::Packed(_) => {
                for (field, value) in self.iter() {
                    f(field, value);
                }
                Cursor::END
            }
        }
    }

    /// Store `value` against `field`, promoting if it no longer fits.
    ///
    /// Answers whether the field is new, which is the number `HSET` reports.
    pub fn set(&mut self, field: &[u8], value: &[u8], limits: &Limits) -> bool {
        if let Body::Packed(p) = &mut self.body {
            // Redis checks both sides against the value limit before it writes,
            // in hashTypeTryConversion, so a pair too long for the band converts
            // the hash and is never briefly stored in a listpack that should not
            // hold it.
            if field.len() > limits.max_listpack_value || value.len() > limits.max_listpack_value {
                self.become_table(1);
            } else {
                let fresh = p.set(field, value);
                // Strictly greater, so the 128th field is still a listpack and
                // the 129th is not.
                if fresh && p.len() > limits.max_listpack_entries {
                    self.become_table(0);
                }
                return fresh;
            }
        }
        match &mut self.body {
            Body::Table(t) => t.set(field, value),
            Body::Packed(_) => unreachable!("the conversion above left a table"),
        }
    }

    /// Take `field` out. Answers whether it was there. This is `HDEL`.
    ///
    /// Never demotes, which is Y4's one-way rule and Redis's behaviour.
    pub fn remove(&mut self, field: &[u8]) -> bool {
        match &mut self.body {
            Body::Packed(p) => match p.find(field) {
                // The field, its value and its deadline go together and they are
                // adjacent, which is the whole reason a row is stored this way
                // round.
                Some(at) => p.remove_at(at),
                None => false,
            },
            Body::Table(t) => t.remove(field),
        }
    }

    /// The earliest deadline any field here has, or `None`.
    ///
    /// A bound and not the answer, which [`crate::ttl`] explains: it can be
    /// earlier than the truth and never later, so acting on it wastes a walk at
    /// worst and cannot miss an expiry. M5's active cycle is the other caller.
    #[inline]
    #[must_use]
    pub fn soonest_deadline(&self) -> Option<u64> {
        match &self.body {
            Body::Packed(p) if p.soonest == NONE => None,
            Body::Packed(p) => Some(p.soonest),
            Body::Table(t) => t.ttl.soonest(),
        }
    }

    /// Drop every field whose deadline has passed, and say how many went.
    ///
    /// The keyspace calls this before every hash command, so it has to be cheap
    /// on a hash that has no deadlines at all, and it is: one load and one
    /// comparison. Only a hash that has actually been given a deadline that has
    /// actually passed pays for the walk.
    ///
    /// The caller deletes the key when this empties the hash, the same way it
    /// does after an `HDEL` that takes the last field, because an empty hash is
    /// not a thing Redis stores.
    pub fn reap(&mut self, now: u64) -> usize {
        match self.soonest_deadline() {
            Some(soonest) if soonest <= now => {}
            _ => return 0,
        }
        match &mut self.body {
            Body::Packed(p) => {
                let gone = p.reap(now);
                // The bound has been leaning early and this walk is the one that
                // knows the truth, so it is the one that pays to fix it.
                p.soonest = p.earliest();
                gone
            }
            Body::Table(t) => {
                let mut gone = 0;
                let mut row = 0;
                while row < t.fields.len() {
                    if t.ttl.is_expired(row, now) {
                        // The last row moves into this one, so stay put and look
                        // at whatever landed here.
                        t.remove_at(row);
                        gone += 1;
                    } else {
                        row += 1;
                    }
                }
                t.ttl.refresh_soonest();
                gone
            }
        }
    }

    /// Put a deadline on `field`, in absolute unix milliseconds.
    ///
    /// This is the whole `HEXPIRE` family, which all turn their argument into an
    /// absolute millisecond before they get here. [`Applied::Deleted`] means the
    /// deadline had already passed and the field has been taken out, which is
    /// what makes `HEXPIRE key 0 FIELDS 1 f` a roundabout `HDEL`.
    ///
    /// The caller has already checked `at` against [`crate::ttl::MAX_AT`],
    /// because Redis rejects the whole command rather than failing field by
    /// field.
    pub fn expire(&mut self, field: &[u8], at: u64, cond: Cond, now: u64) -> Applied {
        match &mut self.body {
            Body::Packed(p) => {
                let Some(row) = p.find(field) else {
                    return Applied::Missing;
                };
                let applied = decide(p.deadline(row), at, cond, now);
                match applied {
                    Applied::Ok => {
                        // Widening moves every row, so the position has to be
                        // found again. It happens once in the life of a hash.
                        if !p.ex {
                            p.widen();
                        }
                        let row = p.find(field).expect("widening kept every field");
                        p.write_deadline(row, at);
                        p.soonest = p.soonest.min(at);
                    }
                    Applied::Deleted => {
                        p.remove_at(row);
                    }
                    Applied::Missing | Applied::NotMet => {}
                }
                applied
            }
            Body::Table(t) => {
                let Some(row) = t.fields.index_of(field) else {
                    return Applied::Missing;
                };
                let applied = t.ttl.set(row, at, cond, now);
                if applied == Applied::Deleted {
                    t.remove_at(row);
                }
                applied
            }
        }
    }

    /// What deadline `field` has. This is `HTTL` and its relatives.
    #[must_use]
    pub fn deadline(&self, field: &[u8]) -> Ask {
        match &self.body {
            Body::Packed(p) => match p.find(field) {
                None => Ask::Missing,
                Some(at) => match p.deadline(at) {
                    Some(at) => Ask::At(at),
                    None => Ask::NoDeadline,
                },
            },
            Body::Table(t) => match t.fields.index_of(field) {
                None => Ask::Missing,
                Some(row) => t.ttl.ask(row),
            },
        }
    }

    /// Take `field`'s deadline off. This is `HPERSIST`.
    ///
    /// [`Ask::NoDeadline`] means there was nothing to take off, which is the -1
    /// Redis replies, and [`Ask::At`] hands back what was there.
    pub fn persist(&mut self, field: &[u8]) -> Ask {
        match &mut self.body {
            Body::Packed(p) => {
                let Some(at) = p.find(field) else {
                    return Ask::Missing;
                };
                match p.deadline(at) {
                    Some(was) => {
                        p.write_deadline(at, 0);
                        Ask::At(was)
                    }
                    None => Ask::NoDeadline,
                }
            }
            Body::Table(t) => match t.fields.index_of(field) {
                None => Ask::Missing,
                Some(row) => t.ttl.clear(row),
            },
        }
    }

    /// How many fields carry a deadline.
    #[must_use]
    pub fn deadline_count(&self) -> usize {
        match &self.body {
            Body::Packed(p) if !p.ex => 0,
            Body::Packed(p) => (0..p.len())
                .filter(|i| p.deadline(i * p.step()).is_some())
                .count(),
            Body::Table(t) => t.ttl.len(),
        }
    }

    /// Bytes held by whichever representation this is.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        match &self.body {
            Body::Packed(p) => p.lp.byte_len(),
            Body::Table(t) => {
                t.fields.memory_bytes() + t.values.memory_bytes() + t.ttl.memory_bytes()
            }
        }
    }

    /// Value bytes no field points at any more.
    ///
    /// Reported rather than hidden, the same as the element table's dead name
    /// bytes, because a hash that has been rewritten holds them and `INFO
    /// memory` should be able to say so.
    #[must_use]
    pub fn dead_value_bytes(&self) -> usize {
        match &self.body {
            Body::Packed(_) => 0,
            Body::Table(t) => t.values.dead(),
        }
    }

    /// Move to the table band, with room for `extra` more fields than are here.
    fn become_table(&mut self, extra: usize) {
        let Body::Packed(p) = &self.body else {
            return;
        };
        let mut t = Table::new(p.len() + extra);
        for i in 0..p.len() {
            let at = i * p.step();
            let (Some(field), Some(value)) = (p.lp.get(at), p.lp.get(at + 1)) else {
                break;
            };
            // A listpack holds a field that looks like a number as a number, and
            // the table holds names as bytes, so this is where the digits get
            // written. Once, on promotion, and never again.
            let f = field.to_vec();
            let v = value.to_vec();
            t.set(&f, &v);
            // The deadline comes over with the field. Set through Deadlines
            // rather than written straight in, so the array gets allocated and
            // the bound gets moved exactly the way an HEXPIRE would do it.
            if let Some(deadline) = p.deadline(at) {
                let row = t.fields.index_of(&f).expect("just inserted");
                t.ttl.set(row, deadline, Cond::Always, 0);
            }
        }
        self.body = Body::Table(t);
    }
}

/// Whether these bytes would be stored as an integer, for a caller deciding
/// what `OBJECT ENCODING` or an RDB writer should say about them.
#[must_use]
#[inline]
pub fn stores_as_int(bytes: &[u8]) -> bool {
    parse_i64(bytes).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What a hash actually costs per field, which is the other half of M3's
    /// memory gate row and was an argument rather than a number until this was
    /// written.
    ///
    /// Run it with `cargo test -p yo-kv --release measure_bytes_per_field --
    /// --ignored --nocapture`. Ignored and printing for the same reasons the
    /// set and sorted set measurements next to it are.
    ///
    /// The gate is sixteen bytes a field, and the payload has to be named for
    /// that to mean anything, so this uses an eight byte field and an eight
    /// byte value. Sixteen bytes a field is then a hash that holds what it
    /// stores and nothing else, which nothing can reach, so the number to read
    /// is the overhead column and the gate is really thirty two total.
    #[test]
    #[ignore = "a measurement, run it by name"]
    fn measure_bytes_per_field() {
        let limits = Limits::DEFAULT;
        for n in [512usize, 1_000, 100_000, 1_000_000] {
            let mut h = Hash::new();
            let mut payload = 0usize;
            for i in 0..n {
                let f = format!("f{i:07}");
                let v = format!("v{i:07}");
                payload += f.len() + v.len();
                h.set(f.as_bytes(), v.as_bytes(), &limits);
            }
            let total = h.memory_bytes();
            let (fields, values) = match &h.body {
                Body::Table(t) => (t.fields.memory_bytes(), t.values.memory_bytes()),
                Body::Packed(_) => (0, 0),
            };
            let per = |b: usize| b as f64 / n as f64;
            println!(
                "n={n:<9} band={:<9} total={total:<10} payload={payload:<9} fields={:.2}/f values={:.2}/f per_field={:.2} over_per_field={:.2}",
                if fields == 0 { "listpack" } else { "table" },
                per(fields),
                per(values),
                per(total),
                (total as f64 - payload as f64) / n as f64
            );
        }
    }

    /// A hash that never leaves the listpack band.
    const SMALL: Limits = Limits::DEFAULT;
    /// A hash that promotes on the 129th field.
    ///
    /// The default used to be this and the promotion tests used to lean on it.
    /// They say the number themselves now, because a test of where the line is
    /// should not move when the default does.
    const AT_128: Limits = Limits {
        max_listpack_entries: 128,
        max_listpack_value: 64,
    };
    /// A hash that is a table from its second field.
    const AS_TABLE: Limits = Limits {
        max_listpack_entries: 1,
        max_listpack_value: 64,
    };

    fn text(t: Text<'_>) -> Vec<u8> {
        t.to_vec()
    }

    fn pairs(h: &Hash) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = h
            .iter()
            .map(|(f, v)| {
                (
                    String::from_utf8(text(f)).expect("utf8"),
                    String::from_utf8(text(v)).expect("utf8"),
                )
            })
            .collect();
        out.sort();
        out
    }

    #[test]
    fn a_field_written_comes_back() {
        for limits in [&SMALL, &AS_TABLE] {
            let mut h = Hash::new();
            assert!(h.set(b"a", b"1", limits), "the field is new");
            assert!(h.set(b"b", b"2", limits));
            assert!(!h.set(b"a", b"3", limits), "and now it is not");

            assert_eq!(h.len(), 2);
            assert_eq!(h.get(b"a").map(text), Some(b"3".to_vec()));
            assert_eq!(h.get(b"b").map(text), Some(b"2".to_vec()));
            assert_eq!(h.get(b"c"), None);
            assert!(h.contains(b"a") && !h.contains(b"c"));
        }
    }

    #[test]
    fn a_value_is_never_mistaken_for_a_field() {
        // The listpack band searches with a step of two, and this is the shape
        // that catches a step of one: b is a value and never a field.
        let mut h = Hash::new();
        h.set(b"a", b"b", &SMALL);
        assert_eq!(h.get(b"b"), None, "b is a value, not a field");
        assert!(!h.contains(b"b"));
        assert!(!h.remove(b"b"), "and it cannot be deleted as one");
        assert_eq!(h.len(), 1);

        assert!(h.set(b"b", b"c", &SMALL), "so writing b is a new field");
        assert_eq!(h.get(b"a").map(text), Some(b"b".to_vec()));
        assert_eq!(h.get(b"b").map(text), Some(b"c".to_vec()));
    }

    #[test]
    fn deleting_takes_the_value_with_the_field() {
        for limits in [&SMALL, &AS_TABLE] {
            let mut h = Hash::new();
            for (f, v) in [("a", "1"), ("b", "2"), ("c", "3")] {
                h.set(f.as_bytes(), v.as_bytes(), limits);
            }
            assert!(h.remove(b"b"));
            assert!(!h.remove(b"b"), "twice is once");

            assert_eq!(h.len(), 2);
            assert_eq!(
                pairs(&h),
                [
                    ("a".to_owned(), "1".to_owned()),
                    ("c".to_owned(), "3".to_owned())
                ],
                "and nothing shifted into the wrong pairing"
            );
        }
    }

    #[test]
    fn it_promotes_on_the_count_and_on_the_length() {
        let mut h = Hash::new();
        for i in 0..128u32 {
            h.set(format!("f{i}").as_bytes(), b"v", &AT_128);
        }
        assert_eq!(h.encoding(), Encoding::Listpack, "128 is still a listpack");
        h.set(b"one more", b"v", &AT_128);
        assert_eq!(h.encoding(), Encoding::Hashtable, "and 129 is not");
        assert_eq!(h.len(), 129);

        // Either side being too long converts on its own, at any count.
        let long = vec![b'x'; 65];
        let mut by_value = Hash::new();
        by_value.set(b"f", &long, &AT_128);
        assert_eq!(by_value.encoding(), Encoding::Hashtable);
        assert_eq!(by_value.get(b"f").map(text), Some(long.clone()));

        let mut by_field = Hash::new();
        by_field.set(&long, b"v", &AT_128);
        assert_eq!(by_field.encoding(), Encoding::Hashtable);
        assert_eq!(by_field.get(&long).map(text), Some(b"v".to_vec()));
    }

    #[test]
    fn promotion_carries_every_pair_over_intact() {
        let mut h = Hash::new();
        // Numbers, so the listpack holds them as integers and the promotion has
        // to write the digits out on the way to the table.
        for i in 0..128u32 {
            h.set(
                format!("{i}").as_bytes(),
                format!("{}", i * 2).as_bytes(),
                &AT_128,
            );
        }
        assert_eq!(h.encoding(), Encoding::Listpack);
        let before = pairs(&h);

        h.set(b"last", b"one", &AT_128);
        assert_eq!(h.encoding(), Encoding::Hashtable);

        let mut after = pairs(&h);
        after.retain(|(f, _)| f != "last");
        assert_eq!(after, before, "the pairs survived the conversion");
        for i in 0..128u32 {
            assert_eq!(
                h.get(format!("{i}").as_bytes()).map(text),
                Some(format!("{}", i * 2).into_bytes()),
                "field {i} is findable by its digits"
            );
        }
    }

    #[test]
    fn it_never_demotes() {
        let mut h = Hash::new();
        for i in 0..200u32 {
            h.set(format!("f{i}").as_bytes(), b"v", &AT_128);
        }
        assert_eq!(h.encoding(), Encoding::Hashtable);
        for i in 0..199u32 {
            h.remove(format!("f{i}").as_bytes());
        }
        assert_eq!(h.len(), 1);
        assert_eq!(
            h.encoding(),
            Encoding::Hashtable,
            "one field left and still a table"
        );
    }

    #[test]
    fn a_length_is_answered_without_writing_the_digits() {
        for limits in [&SMALL, &AS_TABLE] {
            let mut h = Hash::new();
            h.set(b"n", b"1234567", limits);
            h.set(b"s", b"hello", limits);
            h.set(b"e", b"", limits);

            assert_eq!(h.value_len(b"n"), Some(7));
            assert_eq!(h.value_len(b"s"), Some(5));
            assert_eq!(h.value_len(b"e"), Some(0));
            assert_eq!(h.value_len(b"missing"), None);
        }
    }

    #[test]
    fn a_rewritten_value_gives_its_bytes_back_eventually() {
        let mut h = Hash::with_hint(1000, &SMALL);
        assert_eq!(h.encoding(), Encoding::Hashtable);
        let big = vec![b'z'; 200];
        for _ in 0..200 {
            h.set(b"one", &big, &SMALL);
        }
        assert_eq!(h.len(), 1);
        assert_eq!(h.get(b"one").map(text), Some(big.clone()));
        assert!(
            h.dead_value_bytes() < 4096,
            "{} bytes left dead",
            h.dead_value_bytes()
        );
    }

    #[test]
    fn compacting_the_values_moves_every_field_to_the_right_bytes() {
        let mut h = Hash::with_hint(1000, &SMALL);
        // Each field's value is its own name repeated, so a reference that moved
        // to the wrong place is visible rather than merely wrong.
        let want: Vec<(Vec<u8>, Vec<u8>)> = (0..300u32)
            .map(|i| {
                let f = format!("field{i}").into_bytes();
                let v = f.repeat(20);
                (f, v)
            })
            .collect();
        for (f, v) in &want {
            h.set(f, v, &SMALL);
        }
        // Rewrite every one of them, which abandons the whole first copy and is
        // far over both the floor and the ratio.
        for (f, v) in &want {
            h.set(f, v, &SMALL);
        }
        for (f, v) in &want {
            assert_eq!(
                h.get(f).map(text).as_deref(),
                Some(&v[..]),
                "field moved wrongly"
            );
        }
        assert_eq!(h.len(), 300);
    }

    #[test]
    fn a_scan_walks_a_hash_of_any_size_exactly_once() {
        for hint in [0usize, 2000] {
            let mut h = Hash::with_hint(hint, &SMALL);
            for i in 0..100u32 {
                h.set(
                    format!("f{i}").as_bytes(),
                    format!("v{i}").as_bytes(),
                    &SMALL,
                );
            }
            let mut seen: Vec<(String, String)> = Vec::new();
            let mut cursor = Cursor::START;
            loop {
                cursor = h.scan(cursor, 7, |f, v| {
                    seen.push((
                        String::from_utf8(text(f)).expect("utf8"),
                        String::from_utf8(text(v)).expect("utf8"),
                    ));
                });
                if cursor.is_end() {
                    break;
                }
            }
            seen.sort();
            assert_eq!(seen.len(), 100, "at hint {hint}");
            assert_eq!(seen, pairs(&h), "at hint {hint}");
        }
    }

    #[test]
    fn a_draw_reaches_every_pair_and_pairs_them_right() {
        for limits in [&SMALL, &AS_TABLE] {
            let mut h = Hash::new();
            for i in 0..50u32 {
                h.set(
                    format!("f{i}").as_bytes(),
                    format!("v{i}").as_bytes(),
                    limits,
                );
            }
            for i in 0..h.len() {
                let (f, v) = h.at(i).expect("under the length");
                let f = String::from_utf8(text(f)).expect("utf8");
                let v = String::from_utf8(text(v)).expect("utf8");
                assert_eq!(v, f.replace('f', "v"), "row {i} paired wrongly");
            }
            assert_eq!(h.at(h.len()), None, "and there is nothing past the end");
        }
    }

    #[test]
    fn a_hint_that_is_wrong_costs_a_conversion_and_no_answers() {
        // Sized for a table and given three fields, which is a waste and not a
        // bug, and sized for a listpack and given two hundred, which converts.
        let mut big = Hash::with_hint(5000, &SMALL);
        big.set(b"a", b"1", &SMALL);
        assert_eq!(big.encoding(), Encoding::Hashtable);
        assert_eq!(big.get(b"a").map(text), Some(b"1".to_vec()));

        let mut small = Hash::with_hint(2, &AT_128);
        for i in 0..200u32 {
            small.set(format!("f{i}").as_bytes(), b"v", &AT_128);
        }
        assert_eq!(small.encoding(), Encoding::Hashtable);
        assert_eq!(small.len(), 200);
    }

    /// Filled with `n` fields under `limits`, `f0` through `f{n-1}`.
    fn filled(n: u32, limits: &Limits) -> Hash {
        let mut h = Hash::new();
        for i in 0..n {
            h.set(
                format!("f{i}").as_bytes(),
                format!("v{i}").as_bytes(),
                limits,
            );
        }
        h
    }

    #[test]
    fn the_packed_band_widens_the_first_time_a_field_is_given_a_deadline() {
        let mut h = filled(3, &SMALL);
        assert_eq!(h.encoding(), Encoding::Listpack);
        assert_eq!(h.deadline(b"f1"), Ask::NoDeadline);

        assert_eq!(h.expire(b"f1", 5000, Cond::Always, 0), Applied::Ok);
        assert_eq!(h.encoding(), Encoding::ListpackEx, "three wide now");

        // And everything that was there is still there, still paired up.
        assert_eq!(h.len(), 3);
        assert_eq!(
            pairs(&h),
            [
                ("f0".to_owned(), "v0".to_owned()),
                ("f1".to_owned(), "v1".to_owned()),
                ("f2".to_owned(), "v2".to_owned()),
            ]
        );
        assert_eq!(h.deadline(b"f1"), Ask::At(5000));
        assert_eq!(h.deadline(b"f0"), Ask::NoDeadline, "and only that one");
        assert_eq!(h.deadline(b"nope"), Ask::Missing);
        assert_eq!(h.deadline_count(), 1);
        assert_eq!(h.soonest_deadline(), Some(5000));
    }

    #[test]
    fn the_table_band_keeps_deadlines_beside_the_rows() {
        let mut h = filled(3, &AS_TABLE);
        assert_eq!(h.encoding(), Encoding::Hashtable);
        assert_eq!(h.expire(b"f1", 5000, Cond::Always, 0), Applied::Ok);
        assert_eq!(
            h.encoding(),
            Encoding::Hashtable,
            "the table has nothing to widen"
        );
        assert_eq!(h.deadline(b"f1"), Ask::At(5000));
        assert_eq!(h.deadline(b"f0"), Ask::NoDeadline);
        assert_eq!(h.deadline(b"nope"), Ask::Missing);
        assert_eq!(h.deadline_count(), 1);
        assert_eq!(h.soonest_deadline(), Some(5000));
    }

    #[test]
    fn a_field_is_reaped_only_once_its_moment_has_passed() {
        for limits in [&SMALL, &AS_TABLE] {
            let mut h = filled(3, limits);
            h.expire(b"f1", 1000, Cond::Always, 0);

            assert_eq!(h.reap(999), 0, "not yet");
            assert_eq!(h.len(), 3);
            assert!(h.contains(b"f1"), "and it is still readable until then");

            assert_eq!(h.reap(1000), 1, "the deadline itself has passed");
            assert_eq!(h.len(), 2);
            assert!(!h.contains(b"f1"));
            assert!(h.contains(b"f0") && h.contains(b"f2"), "and only that one");
            assert_eq!(h.reap(1000), 0, "twice takes nothing");
            assert_eq!(h.soonest_deadline(), None, "the bound is exact again");
        }
    }

    #[test]
    fn a_hash_with_no_deadlines_is_reaped_without_a_walk() {
        for limits in [&SMALL, &AS_TABLE] {
            let mut h = filled(50, limits);
            assert_eq!(h.soonest_deadline(), None);
            assert_eq!(h.reap(u64::MAX), 0);
            assert_eq!(h.len(), 50);
        }
    }

    #[test]
    fn a_write_clears_the_deadline_it_wrote_over() {
        for limits in [&SMALL, &AS_TABLE] {
            let mut h = filled(3, limits);
            h.expire(b"f1", 1000, Cond::Always, 0);
            assert_eq!(h.deadline(b"f1"), Ask::At(1000));

            assert!(!h.set(b"f1", b"fresh", limits), "not a new field");
            assert_eq!(
                h.deadline(b"f1"),
                Ask::NoDeadline,
                "and HSET took the deadline off"
            );
            assert_eq!(h.reap(u64::MAX), 0, "so nothing expires it");
            assert_eq!(h.get(b"f1").map(text), Some(b"fresh".to_vec()));
        }
    }

    #[test]
    fn a_deadline_already_past_deletes_the_field_instead_of_being_stored() {
        for limits in [&SMALL, &AS_TABLE] {
            let mut h = filled(3, limits);
            assert_eq!(h.expire(b"f1", 500, Cond::Always, 500), Applied::Deleted);
            assert!(!h.contains(b"f1"));
            assert_eq!(h.len(), 2);
            assert_eq!(h.deadline_count(), 0);
            assert_eq!(h.expire(b"gone", 9000, Cond::Always, 0), Applied::Missing);
        }
    }

    #[test]
    fn the_conditions_reach_both_bands_the_same_way() {
        for limits in [&SMALL, &AS_TABLE] {
            let mut h = filled(2, limits);
            assert_eq!(h.expire(b"f0", 1000, Cond::AlreadySet, 0), Applied::NotMet);
            assert_eq!(h.deadline(b"f0"), Ask::NoDeadline);
            assert_eq!(h.expire(b"f0", 1000, Cond::NotSet, 0), Applied::Ok);
            assert_eq!(h.expire(b"f0", 2000, Cond::NotSet, 0), Applied::NotMet);
            assert_eq!(h.expire(b"f0", 500, Cond::Greater, 0), Applied::NotMet);
            assert_eq!(h.expire(b"f0", 2000, Cond::Greater, 0), Applied::Ok);
            assert_eq!(h.deadline(b"f0"), Ask::At(2000));
            // The condition is checked before the past deadline is, so this is
            // a 0 and the field survives rather than being deleted.
            assert_eq!(h.expire(b"f0", 0, Cond::NotSet, 5), Applied::NotMet);
            assert!(h.contains(b"f0"));
        }
    }

    #[test]
    fn persisting_takes_the_deadline_off_and_says_what_was_there() {
        for limits in [&SMALL, &AS_TABLE] {
            let mut h = filled(3, limits);
            h.expire(b"f1", 1000, Cond::Always, 0);

            assert_eq!(h.persist(b"f1"), Ask::At(1000));
            assert_eq!(
                h.persist(b"f1"),
                Ask::NoDeadline,
                "twice is -1, not an error"
            );
            assert_eq!(h.persist(b"gone"), Ask::Missing);
            assert_eq!(h.deadline_count(), 0);
            assert_eq!(h.reap(u64::MAX), 0, "and it does not expire any more");
            assert_eq!(h.len(), 3);
        }
    }

    /// The one that would silently give a deadline to the wrong field.
    #[test]
    fn deadlines_follow_their_fields_through_a_removal() {
        for limits in [&SMALL, &AS_TABLE] {
            let mut h = filled(5, limits);
            // Each field's deadline is derived from its own name, so a deadline
            // that has drifted is visible rather than merely plausible.
            for i in 0..5u32 {
                assert_eq!(
                    h.expire(
                        format!("f{i}").as_bytes(),
                        1000 + u64::from(i),
                        Cond::Always,
                        0
                    ),
                    Applied::Ok
                );
            }
            // The table swap removes, so taking a middle field out moves the
            // last row into the hole.
            assert!(h.remove(b"f1"));

            assert_eq!(h.len(), 4);
            for i in [0u32, 2, 3, 4] {
                assert_eq!(
                    h.deadline(format!("f{i}").as_bytes()),
                    Ask::At(1000 + u64::from(i)),
                    "f{i} kept someone else's deadline"
                );
            }
            assert_eq!(h.deadline_count(), 4);
        }
    }

    #[test]
    fn a_deadline_comes_over_with_its_field_on_promotion() {
        let mut h = Hash::new();
        for i in 0..128u32 {
            h.set(format!("f{i}").as_bytes(), b"v", &AT_128);
        }
        h.expire(b"f7", 4000, Cond::Always, 0);
        h.expire(b"f9", 2000, Cond::Always, 0);
        assert_eq!(h.encoding(), Encoding::ListpackEx);

        h.set(b"one more", b"v", &AT_128);
        assert_eq!(h.encoding(), Encoding::Hashtable, "and now it is a table");

        assert_eq!(h.len(), 129);
        assert_eq!(h.deadline(b"f7"), Ask::At(4000));
        assert_eq!(h.deadline(b"f9"), Ask::At(2000));
        assert_eq!(h.deadline(b"f8"), Ask::NoDeadline);
        assert_eq!(h.deadline_count(), 2);
        assert_eq!(h.soonest_deadline(), Some(2000));

        assert_eq!(h.reap(3000), 1, "f9 and not f7");
        assert!(!h.contains(b"f9") && h.contains(b"f7"));
    }

    #[test]
    fn a_widened_hash_still_scans_and_draws_every_pair_once() {
        for hint in [0usize, 2000] {
            let mut h = Hash::with_hint(hint, &SMALL);
            for i in 0..100u32 {
                h.set(
                    format!("f{i}").as_bytes(),
                    format!("v{i}").as_bytes(),
                    &SMALL,
                );
            }
            h.expire(b"f42", 9000, Cond::Always, 0);

            let mut seen: Vec<(String, String)> = Vec::new();
            let mut cursor = Cursor::START;
            loop {
                cursor = h.scan(cursor, 7, |f, v| {
                    seen.push((
                        String::from_utf8(text(f)).expect("utf8"),
                        String::from_utf8(text(v)).expect("utf8"),
                    ));
                });
                if cursor.is_end() {
                    break;
                }
            }
            seen.sort();
            assert_eq!(seen.len(), 100, "at hint {hint}");
            assert_eq!(seen, pairs(&h), "at hint {hint}");

            // And the draw positions still pair a field with its own value.
            for i in 0..h.len() {
                let (f, v) = h.at(i).expect("under the length");
                let f = String::from_utf8(text(f)).expect("utf8");
                let v = String::from_utf8(text(v)).expect("utf8");
                assert_eq!(
                    v,
                    f.replace('f', "v"),
                    "row {i} paired wrongly at hint {hint}"
                );
            }
        }
    }

    /// Reaping takes every field that is due, including two in a row, which is
    /// where a walk that stepped past the row it just removed would go wrong.
    #[test]
    fn a_run_of_expired_fields_all_go_together() {
        for limits in [&SMALL, &AS_TABLE] {
            let mut h = filled(6, limits);
            for i in [1u32, 2, 3] {
                h.expire(format!("f{i}").as_bytes(), 100, Cond::Always, 0);
            }
            assert_eq!(h.reap(200), 3);
            assert_eq!(h.len(), 3);
            for i in [0u32, 4, 5] {
                assert!(h.contains(format!("f{i}").as_bytes()), "f{i} went too");
            }
        }
    }

    #[test]
    fn an_empty_hash_has_allocated_almost_nothing() {
        let h = Hash::new();
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
        assert_eq!(h.get(b"a"), None);
        assert!(h.memory_bytes() < 64, "{} bytes", h.memory_bytes());
    }
}
