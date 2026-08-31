//! A set, in whichever representation currently fits it.
//!
//! A set is one of an [`Intset`], a [`Listpack`], an [`Elements`] table or a
//! [`Parts`] band, and which one is not a choice this file gets to make freely.
//! `OBJECT ENCODING` has to answer `intset`, `listpack` or `hashtable` at exactly
//! the sizes a real server answers them, because clients and test suites read it
//! (`08` §1), so the promotion rules here are Redis's rules and they were read
//! off `t_set.c` in the 8.10.1 tarball rather than reasoned out from what each
//! structure is good at.
//!
//! ```text
//!  all integers        small, any bytes      everything else       large
//! +-------------+   +------------------+   +---------------+   +-----------+
//! | intset      |-->| listpack         |-->| element table |-->| partition |
//! | 2 B member  |   | ~2 B + payload   |   | one probe     |   | band      |
//! +-------------+   +------------------+   +---------------+   +-----------+
//!  to 512 members     to 128 members                            past 262144
//!                                          <------- both say `hashtable` ---->
//! ```
//!
//! The fourth step is the one Redis does not have, and it is invisible on
//! purpose. A set past 262,144 members becomes several element tables rather than
//! one, which is what makes the merges and the growth pauses bounded, and it
//! still answers `hashtable` because a client that gets a fourth word back from
//! `OBJECT ENCODING` is a client whose assertions break. What partitioning
//! changes is how a large set is stored, not what it is. See [`crate::parts`].
//!
//! Promotion is one-way and upward, which is Y4. A set that has been a hash
//! table does not go back to an intset when it shrinks, and neither does Redis's:
//! a set that demoted on the way down would rewrite itself on every second
//! operation for a workload that adds and removes across a threshold. The band
//! follows the same rule for the same reason, so a set hovering at the partition
//! threshold rehashes once rather than on every other `SREM`.
//!
//! # The rules, and the two that are not obvious
//!
//! Adding an integer to an intset keeps it an intset until it holds more than
//! `set-max-intset-entries`, and then it becomes a **hash table** and not a
//! listpack, because the intset ceiling is 512 and the listpack ceiling is 128
//! and something over the first is well over the second.
//!
//! Adding a non integer to an intset is the asymmetric one. It becomes a
//! listpack only if the intset is currently under the *listpack* ceiling of 128,
//! so an intset of 200 integers that receives one string goes straight to a hash
//! table and is never a listpack at all. Reading that off the source was worth
//! more than reasoning about it, because the natural implementation converts to
//! a listpack whenever the members would fit and gets a different encoding name
//! than a real server for a shape a test suite actually builds.
//!
//! # Members
//!
//! A member comes back as a [`Member`], which is [`listpack::Entry`] under
//! another name: either the bytes as they lie or an integer that has not been
//! formatted yet. All three representations can produce one without copying, and
//! the formatting happens once, into the reply buffer, at the moment the reply is
//! built. That is Y18, and it is the same reason [`crate::value::Str`] has two
//! arms.

use crate::elem::Elements;
use crate::intset::Intset;
use crate::listpack::{self, Listpack};
use crate::parts::{PARTITION_AT, Parts, parts_for};
use crate::scan::Cursor;
use yo_common::num::{DIGITS_MAX, i64_digits, i64_len, parse_i64};

/// A set member: bytes as they lie, or an integer not yet formatted.
pub type Member<'a> = listpack::Entry<'a>;

/// A member on its way to being asked about, in every form the three
/// representations want it in.
///
/// Set algebra walks one set and asks every other set the same question about
/// each member, and the three representations do not want the question in the
/// same shape. An intset wants a number, a listpack wants bytes and a number
/// because it holds both kinds, and an element table wants bytes and their
/// hash. Asking through [`Set::contains`] would redo all of that per question:
/// a parse for the intset, another parse inside the listpack, and a hash per
/// table. This does each once per member and then asks `k - 1` times.
///
/// The hash is computed whether or not any operand is a table, which is waste
/// when none is. It is waste worth taking, because the sets where it is wasted
/// are an intset or a listpack, which are capped at a few hundred members, and
/// an operation over sets that small is finished before the saving could have
/// been measured. The sets where the hash pays are the large ones, and those
/// are tables by definition.
#[derive(Debug, Clone, Copy)]
pub struct Needle<'a> {
    /// The member as bytes, which for an integer member is a caller's buffer.
    bytes: &'a [u8],
    /// The number it is, if it is one, under the same rule that decides whether
    /// a set stores it as one.
    int: Option<i64>,
    /// What an element table would key it under.
    hash: u64,
}

impl<'a> Needle<'a> {
    /// A needle from bytes, which is what a command line argument is.
    #[must_use]
    pub fn new(bytes: &'a [u8]) -> Needle<'a> {
        Needle {
            bytes,
            int: parse_i64(bytes),
            hash: Elements::<()>::hash_of(bytes),
        }
    }

    /// A needle from a member walked out of a set.
    ///
    /// `digits` is where an integer member's text goes, because an intset holds
    /// the number and the digits do not exist anywhere until somebody writes
    /// them. It is the caller's buffer rather than a field so that the needle
    /// stays a borrow and the buffer is written once per member rather than
    /// allocated once per member.
    ///
    /// A member that came out as bytes is still parsed, because the set being
    /// asked may be an intset and `SINTER ints strings` has to find the members
    /// they share. A member that came out as a number is not, which is the
    /// whole saving.
    #[must_use]
    pub fn of(member: Member<'a>, digits: &'a mut [u8; DIGITS_MAX]) -> Needle<'a> {
        match member {
            Member::Str(s) => Needle::new(s),
            Member::Int(n) => {
                let bytes = i64_digits(digits, n);
                Needle {
                    bytes,
                    int: Some(n),
                    hash: Elements::<()>::hash_of(bytes),
                }
            }
        }
    }

    /// The member as bytes, which is what a caller collecting an answer wants.
    #[must_use]
    pub const fn bytes(&self) -> &'a [u8] {
        self.bytes
    }
}

/// Where the encodings change over.
///
/// These are `set-max-intset-entries`, `set-max-listpack-entries` and
/// `set-max-listpack-value`, and they are runtime configuration in Redis, so
/// they are a value passed in here rather than three constants. The defaults are
/// Redis's defaults and a client that never touches `CONFIG SET` sees exactly
/// the encodings a real server would give it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Past this many members an all integer set stops being an intset.
    pub max_intset_entries: usize,
    /// At this many members a set stops being a listpack.
    pub max_listpack_entries: usize,
    /// A member longer than this cannot go in a listpack.
    pub max_listpack_value: usize,
}

impl Limits {
    /// Redis's defaults: 512, 128 and 64.
    pub const DEFAULT: Limits = Limits {
        max_intset_entries: 512,
        max_listpack_entries: 128,
        max_listpack_value: 64,
    };
}

impl Default for Limits {
    fn default() -> Limits {
        Limits::DEFAULT
    }
}

/// Which of the three a set is in, which is what `OBJECT ENCODING` reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// All members are integers and there are few enough of them.
    Intset,
    /// One packed blob, walked linearly.
    Listpack,
    /// The element table.
    Hashtable,
}

impl Encoding {
    /// The word `OBJECT ENCODING` replies with.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Encoding::Intset => "intset",
            Encoding::Listpack => "listpack",
            Encoding::Hashtable => "hashtable",
        }
    }
}

/// The four representations, of which `OBJECT ENCODING` can see three.
///
/// [`Body::Split`] is the partitioned band and it is deliberately invisible from
/// outside. Redis has three set encodings and a client that gets a fourth word
/// back from `OBJECT ENCODING` is a client whose assertions break, so a split set
/// answers `hashtable` like the table it was. What partitioning changes is how a
/// large set is stored and merged, not what it is.
#[derive(Debug, Clone)]
enum Body {
    Ints(Intset),
    Packed(Listpack),
    Table(Elements<()>),
    Split(Parts<()>),
}

/// A set of members.
#[derive(Debug, Clone)]
pub struct Set {
    body: Body,
}

impl Set {
    /// An empty set, which starts as an intset.
    ///
    /// This is what `SADD` on a missing key creates when it has no size hint,
    /// and the first member decides nothing: an intset that receives a string
    /// converts on the spot, and it costs a conversion of nothing.
    #[must_use]
    pub fn new() -> Set {
        Set {
            body: Body::Ints(Intset::new()),
        }
    }

    /// An empty set sized for what is about to go in it.
    ///
    /// Redis's `setTypeCreate`, which picks the representation from the first
    /// member and the count the caller expects, so that `SADD k a b c ...` with
    /// a thousand arguments builds a table once rather than converting twice on
    /// the way there. `hint` is only a hint and being wrong about it costs a
    /// conversion and no correctness.
    #[must_use]
    pub fn with_hint(first: &[u8], hint: usize, limits: &Limits) -> Set {
        if parse_i64(first).is_some() && hint <= limits.max_intset_entries {
            Set {
                body: Body::Ints(Intset::with_capacity(hint)),
            }
        } else if hint <= limits.max_listpack_entries {
            Set {
                body: Body::Packed(Listpack::new()),
            }
        } else if hint > PARTITION_AT {
            // A caller that says up front it is about to load a million members
            // should not build one table, fill it past the threshold and then
            // rehash the lot into partitions. The hint is only a hint, and being
            // wrong about it here costs a set with more partitions than it needs
            // rather than anything incorrect.
            Set {
                body: Body::Split(Parts::with_parts(parts_for(hint))),
            }
        } else {
            Set {
                body: Body::Table(Elements::with_capacity(hint)),
            }
        }
    }

    /// Which representation this is in.
    #[inline]
    #[must_use]
    pub const fn encoding(&self) -> Encoding {
        match self.body {
            Body::Ints(_) => Encoding::Intset,
            Body::Packed(_) => Encoding::Listpack,
            Body::Table(_) | Body::Split(_) => Encoding::Hashtable,
        }
    }

    /// How many members. This is `SCARD`.
    #[inline]
    pub fn len(&self) -> usize {
        match &self.body {
            Body::Ints(s) => s.len(),
            Body::Packed(lp) => lp.len(),
            Body::Table(t) => t.len(),
            Body::Split(p) => p.len(),
        }
    }

    /// Whether there are none.
    ///
    /// An empty set does not exist in Redis, so the caller deletes the key when
    /// this turns true rather than storing an empty one.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether `member` is in the set. This is `SISMEMBER`.
    #[must_use]
    pub fn contains(&self, member: &[u8]) -> bool {
        match &self.body {
            // A member that is not an integer cannot be in a set of integers,
            // and answering that costs a parse rather than a search.
            Body::Ints(s) => parse_i64(member).is_some_and(|v| s.contains(v)),
            Body::Packed(lp) => lp.find(member, 1).is_some(),
            Body::Table(t) => t.contains(member),
            Body::Split(p) => p.contains(member),
        }
    }

    /// The same question asked with the work already done. See [`Needle`].
    ///
    /// This is what set algebra probes with. Every arm is the arm
    /// [`Set::contains`] would have taken, with the parse and the hash lifted
    /// out of it, so the two cannot disagree about what a member is.
    #[must_use]
    #[inline]
    pub fn has(&self, needle: &Needle<'_>) -> bool {
        match &self.body {
            Body::Ints(s) => needle.int.is_some_and(|v| s.contains(v)),
            Body::Packed(lp) => lp.find_parsed(needle.bytes, needle.int, 1).is_some(),
            Body::Table(t) => t.contains_hashed(needle.hash, needle.bytes),
            Body::Split(p) => p.contains_hashed(needle.hash, needle.bytes),
        }
    }

    /// The member at `index`, in whatever order the representation holds them.
    ///
    /// Ascending for an intset, insertion order for the other two. Redis makes
    /// no promise about set order and neither does this, but a uniform draw
    /// needs positions and this is what gives it them (K9).
    #[must_use]
    pub fn at(&self, index: usize) -> Option<Member<'_>> {
        match &self.body {
            Body::Ints(s) => s.get(index).map(Member::Int),
            Body::Packed(lp) => lp.get(index),
            Body::Table(t) => t.at(index).map(|(name, _)| Member::Str(name)),
            Body::Split(p) => p.at(index).map(|(name, _)| Member::Str(name)),
        }
    }

    /// Every member.
    pub fn iter(&self) -> impl Iterator<Item = Member<'_>> {
        (0..self.len()).map(|i| self.at(i).expect("index is under the length"))
    }

    /// Walk part of the set and say where to resume. This is `SSCAN`.
    ///
    /// Only the table and the partitioned band walk in windows. An intset or a
    /// listpack hands back
    /// every member in one call and a cursor of [`Cursor::END`], ignoring the
    /// cursor it was given, which is what Redis does for the same two encodings
    /// and for the same reason: a hundred and twenty eight members is smaller
    /// than the reply header arithmetic to split them up, and a set that small
    /// cannot block the loop long enough for the split to be worth anything.
    ///
    /// Ignoring the cursor is safe rather than merely convenient, because
    /// promotion is one way. A set that gave a client a listpack cursor is not
    /// going to be a listpack again, so the only way to arrive at those two arms
    /// with a cursor from somewhere else is for the key to have been deleted and
    /// remade underneath the scan, and returning everything to that client
    /// returns a member twice at worst, which the guarantee allows.
    ///
    /// A table cursor arriving at the band is the one crossing that does happen,
    /// because a set can split part way through a client's scan. That is handled
    /// rather than ignored: a table cursor names one partition, and
    /// [`Cursor::rebase`] reads the widening and restarts the walk at the top of
    /// the new layout, so the client sees some members a second time and misses
    /// none. Repeats are what the `SCAN` guarantee gives up in exchange for
    /// surviving a resize, and a set only splits once.
    pub fn scan<F>(&self, cursor: Cursor, count: usize, mut f: F) -> Cursor
    where
        F: FnMut(Member<'_>),
    {
        match &self.body {
            Body::Table(t) => t.scan(cursor, count, |name, ()| f(Member::Str(name))),
            Body::Split(p) => p.scan(cursor, count, |name, ()| f(Member::Str(name))),
            _ => {
                for m in self.iter() {
                    f(m);
                }
                Cursor::END
            }
        }
    }

    /// Bytes held by whichever representation this is.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        match &self.body {
            Body::Ints(s) => s.memory_bytes(),
            Body::Packed(lp) => lp.byte_len(),
            Body::Table(t) => t.memory_bytes(),
            Body::Split(p) => p.memory_bytes(),
        }
    }

    /// Add `member`, promoting if it no longer fits. Answers whether it was new.
    ///
    /// This is `setTypeAdd`, arm for arm.
    pub fn add(&mut self, member: &[u8], limits: &Limits) -> bool {
        match &mut self.body {
            Body::Table(t) => {
                let new = t.insert(member, ()).is_ok_and(|old| old.is_none());
                // Checked after the insert rather than before, so the set that
                // splits is the one that has actually outgrown a table and not
                // the one that is about to.
                if t.len() > PARTITION_AT {
                    self.become_split();
                }
                return new;
            }
            Body::Split(p) => {
                let new = p.insert(member, ()).is_ok_and(|old| old.is_none());
                // Asked rather than decided, because growing is a rehash of the
                // whole set and the band leaves the timing to whoever knows
                // whether this is one write or the middle of a bulk load.
                if let Some(want) = p.wants_parts() {
                    p.grow_to(want);
                }
                return new;
            }
            Body::Packed(lp) => {
                if lp.find(member, 1).is_some() {
                    return false;
                }
                if lp.len() < limits.max_listpack_entries
                    && member.len() <= limits.max_listpack_value
                {
                    lp.push(member);
                    return true;
                }
                // Too many members, or one too long. It falls out to a table.
            }
            Body::Ints(s) => {
                if let Some(v) = parse_i64(member) {
                    if !s.add(v) {
                        return false;
                    }
                    // Strictly greater, so the 512th member is still an intset
                    // and the 513th is not. And it becomes a table rather than
                    // a listpack, because anything past 512 is past 128 too.
                    if s.len() > limits.max_intset_entries {
                        self.become_table(0);
                    }
                    return true;
                }
                // Not an integer, so it is certainly not in a set of integers
                // already. If the set is still small enough it becomes a
                // listpack, and otherwise it falls out to a table.
                if self.ints_fit_a_listpack(member, limits) {
                    self.become_listpack();
                    self.push_new(member);
                    return true;
                }
            }
        }
        self.become_table(1);
        self.push_new(member);
        true
    }

    /// Put in a member already known to be new and known to fit where it is.
    ///
    /// Only ever called on the far side of a promotion, where both of those are
    /// facts the promotion established and not things worth establishing twice.
    fn push_new(&mut self, member: &[u8]) {
        match &mut self.body {
            Body::Packed(lp) => lp.push(member),
            Body::Table(t) => {
                t.insert(member, ())
                    .expect("the table was sized for this one");
            }
            Body::Split(p) => {
                p.insert(member, ())
                    .expect("the band was sized for this one");
            }
            Body::Ints(_) => unreachable!("no promotion ever lands on an intset"),
        }
    }

    /// Remove `member`. Answers whether it was there.
    ///
    /// Never demotes, which is Y4's one-way rule and Redis's behaviour.
    pub fn remove(&mut self, member: &[u8]) -> bool {
        match &mut self.body {
            Body::Ints(s) => parse_i64(member).is_some_and(|v| s.remove(v)),
            Body::Packed(lp) => match lp.find(member, 1) {
                Some(at) => lp.delete(at, 1),
                None => false,
            },
            Body::Table(t) => t.remove(member).is_some(),
            Body::Split(p) => p.remove(member).is_some(),
        }
    }

    /// Take out the member at `index` and hand it back.
    ///
    /// This is what `SPOP` runs on top of. The table moves its last row into the
    /// hole rather than shifting, so the position of every other member is
    /// stable except for one; the other two shift. Neither is a promise a caller
    /// can lean on, and `SPOP` does not need one because it draws again from the
    /// new length each time.
    pub fn remove_at(&mut self, index: usize) -> Option<Vec<u8>> {
        match &mut self.body {
            Body::Ints(s) => {
                let v = s.get(index)?;
                s.remove(v);
                let mut out = Vec::with_capacity(i64_len(v));
                Member::Int(v).write_to(&mut out);
                Some(out)
            }
            Body::Packed(lp) => {
                let out = lp.get(index)?.to_vec();
                lp.delete(index, 1);
                Some(out)
            }
            Body::Table(t) => t.take_at(index).map(|(name, ())| name),
            Body::Split(p) => p.take_at(index).map(|(name, ())| name),
        }
    }

    /// Take out the member at `index` without building it into a `Vec` first.
    ///
    /// The same removal as [`Set::remove_at`] for a caller that has already read
    /// the member and does not need it handed back. That caller is `SPOP` on the
    /// wire, which reads with [`Set::at`], writes the bytes straight into the
    /// reply buffer, and only then calls this. It is an allocation a member
    /// saved on the one command in the set whose whole cost is the allocating.
    ///
    /// [`Set::remove_at`] stays for the embedded API, where the caller wants the
    /// bytes and has nowhere to put them.
    pub fn drop_at(&mut self, index: usize) -> bool {
        match &mut self.body {
            Body::Ints(s) => match s.get(index) {
                Some(v) => {
                    s.remove(v);
                    true
                }
                None => false,
            },
            Body::Packed(lp) => {
                if index >= lp.len() {
                    return false;
                }
                lp.delete(index, 1);
                true
            }
            Body::Table(t) => t.remove_at(index).is_some(),
            Body::Split(p) => p.remove_at(index).is_some(),
        }
    }

    /// Whether an intset plus one non integer member would still be a listpack.
    ///
    /// Three tests, and the first is the asymmetric one: the count is compared
    /// against the *listpack* ceiling and not the intset one, so an intset with
    /// two hundred members is already too big to become a listpack even though
    /// it is a perfectly legal intset. The other two are the new member's length
    /// and the longest existing member's length once it is written as digits,
    /// which only bites when `set-max-listpack-value` has been turned down,
    /// because no integer is more than twenty characters.
    fn ints_fit_a_listpack(&self, member: &[u8], limits: &Limits) -> bool {
        let Body::Ints(s) = &self.body else {
            return false;
        };
        let widest = s
            .min()
            .map(i64_len)
            .unwrap_or(0)
            .max(s.max().map(i64_len).unwrap_or(0));
        s.len() < limits.max_listpack_entries
            && member.len() <= limits.max_listpack_value
            && widest <= limits.max_listpack_value
    }

    /// Rewrite as a listpack, which only an intset ever does.
    fn become_listpack(&mut self) {
        let Body::Ints(s) = &self.body else {
            return;
        };
        let mut lp = Listpack::new();
        let mut buf = Vec::with_capacity(20);
        for v in s.iter() {
            buf.clear();
            Member::Int(v).write_to(&mut buf);
            lp.push(&buf);
        }
        self.body = Body::Packed(lp);
    }

    /// Rewrite as an element table, with room for `extra` more members.
    fn become_table(&mut self, extra: usize) {
        let mut t = Elements::with_capacity(self.len() + extra);
        let mut buf = Vec::with_capacity(20);
        for m in self.iter() {
            match m {
                Member::Str(b) => {
                    t.insert(b, ()).expect("room, and every member was unique");
                }
                Member::Int(v) => {
                    buf.clear();
                    Member::Int(v).write_to(&mut buf);
                    t.insert(&buf, ())
                        .expect("room, and every member was unique");
                }
            }
        }
        self.body = Body::Table(t);
    }

    /// Spread an element table over partitions.
    ///
    /// One way, like every other promotion here. A set that drops back under the
    /// threshold keeps its partitions, which is Y4's rule and Redis's behaviour
    /// for the encodings it does expose: the cost of a representation is paid
    /// when it is entered, and paying it again on the way back out turns one
    /// `SREM` at the boundary into a rehash of the whole set.
    fn become_split(&mut self) {
        if let Body::Table(t) = &self.body {
            let p = Parts::from_table(t, parts_for(t.len()));
            self.body = Body::Split(p);
        }
    }
}

impl Default for Set {
    fn default() -> Set {
        Set::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn of(members: &[&str]) -> Set {
        let mut s = Set::new();
        for m in members {
            assert!(s.add(m.as_bytes(), &Limits::DEFAULT), "{m} was new");
        }
        s
    }

    /// What a set actually costs per member, which is half of M3's memory gate
    /// row and was an argument rather than a number until this was written.
    ///
    /// Run it with `cargo test -p yo-kv --release measure_bytes_per_member --
    /// --ignored --nocapture`. Ignored because a million members is not
    /// something every `cargo test` should pay for, and it prints rather than
    /// asserts because the number it prints is the thing being reported.
    ///
    /// Two shapes, because the gate names one of them and the other is what
    /// most sets actually hold. Integers first, at every band an all integer
    /// set passes through, and then strings, which never see the intset at all.
    #[test]
    #[ignore = "a measurement, run it by name"]
    fn measure_bytes_per_member() {
        let limits = Limits::DEFAULT;
        for n in [512usize, 1_000, 100_000, 1_000_000] {
            let mut s = Set::new();
            for i in 0..n {
                s.add(i.to_string().as_bytes(), &limits);
            }
            println!(
                "int   n={n:<9} band={:<10} total={:<10} per_member={:.2}",
                band(&s),
                s.memory_bytes(),
                s.memory_bytes() as f64 / n as f64
            );
        }
        // Sixteen byte members, so the payload is a round number and the
        // overhead is whatever is above it.
        for n in [128usize, 1_000, 100_000, 1_000_000] {
            let mut s = Set::new();
            let mut payload = 0usize;
            for i in 0..n {
                let m = format!("member:{i:09}");
                payload += m.len();
                s.add(m.as_bytes(), &limits);
            }
            let total = s.memory_bytes();
            println!(
                "bytes n={n:<9} band={:<10} total={total:<10} payload={payload:<9} per_member={:.2} over_per_member={:.2}",
                band(&s),
                total as f64 / n as f64,
                (total as f64 - payload as f64) / n as f64
            );
        }
    }

    /// Which of the four a set is in, spelled out rather than through
    /// [`Set::encoding`], which folds the two table bands into one word because
    /// that is what `OBJECT ENCODING` has to say.
    fn band(s: &Set) -> &'static str {
        match &s.body {
            Body::Ints(_) => "intset",
            Body::Packed(_) => "listpack",
            Body::Table(_) => "table",
            Body::Split(_) => "split",
        }
    }

    /// Everything about the partitioned band that needs a real set past the real
    /// threshold, in one test, because building 262,145 members is the expensive
    /// part and there is no reason to pay for it four times.
    #[test]
    fn a_set_past_the_threshold_splits_without_the_client_being_able_to_tell() {
        let limits = Limits::DEFAULT;
        let mut s = Set::new();
        // One short of the threshold. Still one table: the check is strictly
        // greater, so the set that splits is the one that has outgrown a table
        // and not the one that is about to.
        for i in 0..PARTITION_AT {
            assert!(s.add(format!("m{i}").as_bytes(), &limits));
        }
        assert!(matches!(s.body, Body::Table(_)));
        assert_eq!(s.len(), PARTITION_AT);

        // The member that tips it over.
        assert!(s.add(b"tipping", &limits));
        assert!(matches!(s.body, Body::Split(_)), "it should have split");
        assert_eq!(s.len(), PARTITION_AT + 1);

        // And the client cannot tell. This is the whole point: Redis has three
        // set encodings and a fourth word here breaks every suite that reads it.
        assert_eq!(s.encoding(), Encoding::Hashtable);
        assert_eq!(s.encoding().name(), "hashtable");

        // Every member survived the rehash, asked both ways.
        assert!(s.contains(b"tipping"));
        assert!(s.contains(b"m0"));
        assert!(s.contains(b"m262143"));
        assert!(!s.contains(b"m262144"));
        assert!(s.has(&Needle::new(b"m1000")));
        assert!(!s.has(&Needle::new(b"nothing")));

        // A rewrite is still not an add.
        assert!(!s.add(b"m0", &limits));
        assert_eq!(s.len(), PARTITION_AT + 1);

        // Removing goes back through the same partition it went into, and the
        // band never demotes however far the set shrinks.
        assert!(s.remove(b"tipping"));
        assert!(!s.remove(b"tipping"));
        assert_eq!(s.len(), PARTITION_AT);
        assert!(matches!(s.body, Body::Split(_)), "promotion is one way");
        assert_eq!(s.encoding(), Encoding::Hashtable);

        // The draw `SPOP` and `SRANDMEMBER` run on reaches the last position,
        // which is the one that has to land in the highest non empty partition.
        let last = s.at(s.len() - 1).expect("inside the set").to_vec();
        assert!(s.contains(&last));
        assert!(s.at(s.len()).is_none());
        assert_eq!(s.remove_at(s.len() - 1), Some(last.clone()));
        assert!(!s.contains(&last));
        assert!(s.drop_at(0));
        assert_eq!(s.len(), PARTITION_AT - 2);

        // And a full scan sees every member exactly once.
        let mut seen = 0usize;
        let mut cursor = Cursor::START;
        let mut rounds = 0;
        loop {
            cursor = s.scan(cursor, 1_000, |_| seen += 1);
            rounds += 1;
            assert!(rounds < 100_000, "the scan is not finishing");
            if cursor.is_end() {
                break;
            }
        }
        assert_eq!(seen, s.len());
    }

    /// The crossing that actually happens in production: a client holding a
    /// cursor from before the split. It has to see every member that stayed, and
    /// repeats are what the `SCAN` guarantee gives up in exchange.
    #[test]
    fn a_scan_survives_the_set_splitting_underneath_it() {
        let limits = Limits::DEFAULT;
        let mut s = Set::new();
        for i in 0..PARTITION_AT {
            s.add(format!("m{i}").as_bytes(), &limits);
        }
        assert!(matches!(s.body, Body::Table(_)));

        let mut seen = Vec::new();
        let cursor = s.scan(Cursor::START, 5_000, |m| seen.push(m.to_vec()));
        assert!(!cursor.is_end(), "the scan should have stopped part way");

        s.add(b"tipping", &limits);
        assert!(matches!(s.body, Body::Split(_)));

        let mut cursor = cursor;
        let mut rounds = 0;
        loop {
            cursor = s.scan(cursor, 5_000, |m| seen.push(m.to_vec()));
            rounds += 1;
            assert!(rounds < 100_000, "the scan is not finishing");
            if cursor.is_end() {
                break;
            }
        }
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(
            seen.len(),
            PARTITION_AT + 1,
            "the split lost a member the client was entitled to"
        );
    }

    #[test]
    fn a_hint_past_the_threshold_builds_the_band_up_front() {
        let limits = Limits::DEFAULT;
        // A caller loading a million members should not fill one table, cross the
        // threshold and then rehash the lot.
        let s = Set::with_hint(b"first", 1_000_000, &limits);
        assert!(matches!(s.body, Body::Split(_)));
        assert_eq!(s.encoding(), Encoding::Hashtable);
        assert!(s.is_empty());

        // A hint at the threshold is still one table, matching the add path.
        let s = Set::with_hint(b"first", PARTITION_AT, &limits);
        assert!(matches!(s.body, Body::Table(_)));

        // And a hint is only a hint: the band takes members like anything else.
        let mut s = Set::with_hint(b"a", 1_000_000, &limits);
        assert!(s.add(b"a", &limits));
        assert!(!s.add(b"a", &limits));
        assert!(s.contains(b"a"));
        assert_eq!(s.len(), 1);
        assert_eq!(s.at(0).map(|m| m.to_vec()), Some(b"a".to_vec()));
    }

    fn members(s: &Set) -> Vec<String> {
        let mut v: Vec<String> = s
            .iter()
            .map(|m| String::from_utf8(m.to_vec()).expect("utf8 in these tests"))
            .collect();
        // Order is not part of the contract and the three representations do not
        // agree on it, so every assertion here is against a sorted list.
        v.sort();
        v
    }

    #[test]
    fn a_new_set_is_an_empty_intset() {
        let s = Set::new();
        assert_eq!(s.encoding(), Encoding::Intset);
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
        assert!(!s.contains(b"1"));
        assert_eq!(s.at(0), None);
    }

    #[test]
    fn integers_stay_an_intset_and_come_back_as_members() {
        let s = of(&["1", "2", "3"]);
        assert_eq!(s.encoding(), Encoding::Intset);
        assert_eq!(members(&s), ["1", "2", "3"]);
        assert!(s.contains(b"2"));
        assert!(!s.contains(b"4"));
        assert!(!s.contains(b"two"));
    }

    #[test]
    fn adding_the_same_member_twice_says_so_in_all_three() {
        let mut ints = of(&["1", "2"]);
        assert!(!ints.add(b"2", &Limits::DEFAULT));
        assert_eq!(ints.len(), 2);

        let mut packed = of(&["a", "b"]);
        assert_eq!(packed.encoding(), Encoding::Listpack);
        assert!(!packed.add(b"b", &Limits::DEFAULT));
        assert_eq!(packed.len(), 2);

        let mut table = of(&["a", "b"]);
        table.become_table(0);
        assert!(!table.add(b"b", &Limits::DEFAULT));
        assert_eq!(table.len(), 2);
    }

    #[test]
    fn a_string_turns_a_small_intset_into_a_listpack() {
        let mut s = of(&["1", "2", "3"]);
        assert!(s.add(b"hello", &Limits::DEFAULT));
        assert_eq!(s.encoding(), Encoding::Listpack);
        assert_eq!(members(&s), ["1", "2", "3", "hello"]);
        assert!(s.contains(b"1"), "the integers survived the rewrite");
        assert!(s.contains(b"hello"));
    }

    #[test]
    fn a_string_turns_a_big_intset_straight_into_a_table() {
        // The asymmetric rule, and the one a natural implementation gets wrong.
        // Two hundred integers is a legal intset and is already past the
        // listpack ceiling, so this never passes through the listpack at all.
        let mut s = Set::new();
        for i in 0..200 {
            s.add(i.to_string().as_bytes(), &Limits::DEFAULT);
        }
        assert_eq!(s.encoding(), Encoding::Intset);
        assert_eq!(s.len(), 200);

        assert!(s.add(b"hello", &Limits::DEFAULT));
        assert_eq!(s.encoding(), Encoding::Hashtable);
        assert_eq!(s.len(), 201);
        assert!(s.contains(b"199"));
        assert!(s.contains(b"hello"));
    }

    #[test]
    fn an_intset_holds_five_hundred_and_twelve_and_converts_at_the_next_one() {
        let mut s = Set::new();
        for i in 0..512 {
            s.add(i.to_string().as_bytes(), &Limits::DEFAULT);
        }
        assert_eq!(s.encoding(), Encoding::Intset, "512 is still an intset");
        assert_eq!(s.len(), 512);

        s.add(b"512", &Limits::DEFAULT);
        assert_eq!(s.encoding(), Encoding::Hashtable, "513 is not");
        assert_eq!(s.len(), 513);
        // And not a listpack on the way, because 513 is well past 128.
        for i in 0..513 {
            assert!(s.contains(i.to_string().as_bytes()), "{i} survived");
        }
    }

    #[test]
    fn a_listpack_converts_at_a_hundred_and_twenty_eight_members() {
        let mut s = of(&["x"]);
        assert_eq!(s.encoding(), Encoding::Listpack);
        for i in 0..127 {
            s.add(format!("m{i}").as_bytes(), &Limits::DEFAULT);
        }
        assert_eq!(s.len(), 128);
        assert_eq!(s.encoding(), Encoding::Listpack, "128 is still a listpack");

        s.add(b"one more", &Limits::DEFAULT);
        assert_eq!(s.len(), 129);
        assert_eq!(s.encoding(), Encoding::Hashtable);
        assert!(s.contains(b"x"));
        assert!(s.contains(b"m126"));
        assert!(s.contains(b"one more"));
    }

    #[test]
    fn a_long_member_converts_a_listpack_whatever_the_count() {
        let mut s = of(&["a"]);
        let long = vec![b'z'; 65];
        assert!(s.add(&long, &Limits::DEFAULT));
        assert_eq!(s.encoding(), Encoding::Hashtable, "65 is past 64");
        assert_eq!(s.len(), 2);
        assert!(s.contains(&long));

        // And exactly at the boundary it does not.
        let mut ok = of(&["a"]);
        ok.add(&[b'z'; 64], &Limits::DEFAULT);
        assert_eq!(ok.encoding(), Encoding::Listpack, "64 fits");
    }

    #[test]
    fn a_long_member_sends_an_intset_to_a_table_and_not_a_listpack() {
        let mut s = of(&["1", "2"]);
        assert!(s.add(&[b'z'; 65], &Limits::DEFAULT));
        assert_eq!(s.encoding(), Encoding::Hashtable);
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn the_limits_are_configuration_and_moving_them_moves_the_encodings() {
        let tight = Limits {
            max_intset_entries: 2,
            max_listpack_entries: 2,
            max_listpack_value: 3,
        };
        let mut s = Set::new();
        s.add(b"1", &tight);
        s.add(b"2", &tight);
        assert_eq!(s.encoding(), Encoding::Intset);
        s.add(b"3", &tight);
        assert_eq!(s.encoding(), Encoding::Hashtable, "three is past two");

        // And a member longer than three characters cannot go in a listpack.
        let mut t = Set::new();
        t.add(b"abc", &tight);
        assert_eq!(t.encoding(), Encoding::Listpack, "three characters fit");
        t.add(b"defg", &tight);
        assert_eq!(t.encoding(), Encoding::Hashtable, "four do not");
        assert!(t.contains(b"abc"));
        assert!(t.contains(b"defg"));

        // Including as the first member, which is a table from the off rather
        // than a listpack that converts on the next thing to arrive. Redis gets
        // to the same place by the other road: it creates the listpack, tries
        // the add, and converts before the reply.
        let mut u = Set::new();
        u.add(b"abcd", &tight);
        assert_eq!(u.encoding(), Encoding::Hashtable);
        assert!(u.contains(b"abcd"));
    }

    #[test]
    fn with_hint_picks_the_representation_up_front() {
        let d = &Limits::DEFAULT;
        assert_eq!(
            Set::with_hint(b"1", 10, d).encoding(),
            Encoding::Intset,
            "an integer and few enough of them"
        );
        assert_eq!(
            Set::with_hint(b"1", 1000, d).encoding(),
            Encoding::Hashtable,
            "an integer and too many"
        );
        assert_eq!(
            Set::with_hint(b"x", 10, d).encoding(),
            Encoding::Listpack,
            "not an integer and few enough"
        );
        assert_eq!(
            Set::with_hint(b"x", 1000, d).encoding(),
            Encoding::Hashtable,
            "not an integer and too many"
        );
    }

    #[test]
    fn removing_works_in_all_three_and_never_demotes() {
        let mut ints = of(&["1", "2", "3"]);
        assert!(ints.remove(b"2"));
        assert!(!ints.remove(b"2"));
        assert!(!ints.remove(b"nope"), "not an integer, so not a member");
        assert_eq!(members(&ints), ["1", "3"]);
        assert_eq!(ints.encoding(), Encoding::Intset);

        let mut packed = of(&["a", "b", "c"]);
        assert!(packed.remove(b"b"));
        assert!(!packed.remove(b"b"));
        assert_eq!(members(&packed), ["a", "c"]);
        assert_eq!(packed.encoding(), Encoding::Listpack);

        let mut table = of(&["a", "b", "c"]);
        table.become_table(0);
        assert!(table.remove(b"b"));
        assert!(!table.remove(b"b"));
        assert_eq!(members(&table), ["a", "c"]);
        assert_eq!(
            table.encoding(),
            Encoding::Hashtable,
            "down to two members and still a table"
        );
    }

    #[test]
    fn a_set_can_be_emptied_a_member_at_a_time() {
        for mut s in [of(&["1", "2", "3"]), of(&["a", "b", "c"])] {
            let all: Vec<Vec<u8>> = s.iter().map(|m| m.to_vec()).collect();
            for m in &all {
                assert!(s.remove(m));
            }
            assert!(s.is_empty());
            assert_eq!(s.at(0), None);
        }
    }

    #[test]
    fn removing_by_position_hands_the_member_back() {
        // What `SPOP` runs on. Drawing position zero every time has to empty the
        // set rather than run off the end or repeat a member, in all three.
        let mut table = of(&["a", "b", "c", "d"]);
        table.become_table(0);
        for mut s in [
            of(&["10", "20", "30", "40"]),
            of(&["a", "b", "c", "d"]),
            table,
        ] {
            let mut got = Vec::new();
            while !s.is_empty() {
                got.push(String::from_utf8(s.remove_at(0).expect("not empty")).expect("utf8"));
            }
            got.sort();
            assert_eq!(got.len(), 4, "four members and no repeats");
            assert_eq!(s.len(), 0);
            assert_eq!(s.remove_at(0), None);
        }
    }

    #[test]
    fn an_integer_member_is_the_same_member_however_it_is_written() {
        // An intset holds 42 as a number, so `SADD s 42` twice is one member.
        // `042` does not parse as an integer, so it is a different member and it
        // converts the set, which is what a real server does too.
        let mut s = of(&["42"]);
        assert!(!s.add(b"42", &Limits::DEFAULT));
        assert_eq!(s.len(), 1);
        assert!(s.add(b"042", &Limits::DEFAULT));
        assert_eq!(s.encoding(), Encoding::Listpack);
        assert_eq!(members(&s), ["042", "42"]);
        assert!(s.contains(b"42"));
        assert!(s.contains(b"042"));
    }

    #[test]
    fn the_small_bands_answer_a_scan_in_one_go() {
        for s in [of(&["1", "2", "3"]), of(&["a", "b", "c"])] {
            let mut seen = Vec::new();
            // A count of one, which the table band would honour and these two
            // do not, and a cursor from nowhere, which these two ignore.
            let next = s.scan(Cursor::at(1, 0, 99), 1, |m| seen.push(m.to_vec()));
            assert!(next.is_end(), "{:?} split a scan up", s.encoding());
            assert_eq!(seen.len(), 3);
        }
    }

    #[test]
    fn the_table_band_walks_a_scan_in_windows_and_misses_nothing() {
        let mut s = Set::new();
        for i in 0..300 {
            s.add(format!("m{i}").as_bytes(), &Limits::DEFAULT);
        }
        assert_eq!(s.encoding(), Encoding::Hashtable);

        let mut seen = Vec::new();
        let mut c = Cursor::START;
        let mut turns = 0;
        loop {
            c = s.scan(c, 7, |m| seen.push(m.to_vec()));
            turns += 1;
            assert!(turns < 100, "the scan did not finish");
            if c.is_end() {
                break;
            }
        }
        assert!(
            turns > 1,
            "a window of seven over three hundred took one turn"
        );
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 300, "every member came back at least once");
    }

    #[test]
    fn a_conversion_loses_no_member_at_any_of_the_three_boundaries() {
        // One walk over each path out of an intset and out of a listpack, each
        // checking every member is still findable after the rewrite rather than
        // only checking the count.
        let mut wide = Set::new();
        for v in [i64::MIN, -1, 0, 1, i64::MAX] {
            wide.add(v.to_string().as_bytes(), &Limits::DEFAULT);
        }
        wide.add(b"str", &Limits::DEFAULT);
        assert_eq!(wide.encoding(), Encoding::Listpack);
        for v in [i64::MIN, -1, 0, 1, i64::MAX] {
            assert!(wide.contains(v.to_string().as_bytes()), "{v} survived");
        }

        wide.add(&[b'q'; 100], &Limits::DEFAULT);
        assert_eq!(wide.encoding(), Encoding::Hashtable);
        for v in [i64::MIN, -1, 0, 1, i64::MAX] {
            assert!(
                wide.contains(v.to_string().as_bytes()),
                "{v} survived twice"
            );
        }
        assert!(wide.contains(b"str"));
        assert_eq!(wide.len(), 7);
    }
}
