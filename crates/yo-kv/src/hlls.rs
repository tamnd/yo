//! The HyperLogLog commands, which are string commands over a documented value.
//!
//! Everything about the sketch itself is in [`hll`]. This file is where a key
//! turns into one, where Redis's edges are kept and where the writes happen.
//!
//! There is no HyperLogLog type here for the same reason there is no bitmap
//! type: in Redis there is not one either. A sketch is an ordinary string, `GET`
//! hands the bytes to a client, `SET` takes them back, and `TYPE` says `string`.
//! What makes these commands different from the other string commands is that
//! they refuse a string that is not a sketch, with their own sentence, and that
//! is the whole of the type discipline.
//!
//! Three edges, all measured on 8.10.1 rather than reasoned about.
//!
//! `PFCOUNT` is a `readonly` command that writes. The estimate is expensive
//! enough that Redis caches it in the eight header bytes, so the first `PFCOUNT`
//! after a `PFADD` walks the registers and every one after it reads eight bytes,
//! and a client watching `GET` can see the header change under a read. We do the
//! same, in place, without marking the key dirty.
//!
//! `PFADD` with no elements still creates the key and still answers 1, and the
//! sketch it leaves behind has its cache marked stale even though nothing was
//! added to it. A second `PFADD` of an element already in the sketch answers 0
//! and leaves the bytes exactly as they were.
//!
//! `PFDEBUG GETREG` converts the sketch to dense and leaves it that way. It is
//! a debugging command and it is allowed to, but it means a client cannot look
//! at the registers of a sparse sketch without changing it.

use crate::db::Db;
use crate::hll::{self, Encoding};
use crate::keyspace::Keyspace;
use crate::strings::check_len;
use crate::value::{self, Str};
use yo_common::{Code, Error, Result};
use yo_index::RawMap;

/// What Redis says about a `PFDEBUG` aimed at a key that is not there.
const NO_SUCH_KEY: &str = "The specified key does not exist";
/// What Redis says about `PFDEBUG DECODE` on a dense sketch.
const NOT_SPARSE: &str = "HLL encoding is not sparse";

impl Keyspace {
    /// `PFADD key [element ...]`, answering whether anything changed.
    ///
    /// Creating the key counts as a change, so `PFADD fresh` with no elements
    /// answers 1 and `PFADD fresh` again answers 0. An element that lands in a
    /// register already holding at least as large a count is not a change
    /// either, and in that case the value is not rewritten at all.
    pub fn pfadd<'e, I>(&mut self, key: &[u8], eles: I) -> Result<bool>
    where
        I: Iterator<Item = &'e [u8]>,
    {
        self.reap(key);
        self.string_only(key)?;
        self.thaw(key)?;
        check_len(key, hll::DENSE)?;

        let mut buf = std::mem::take(&mut self.scratch);
        buf.clear();
        let (deadline, fresh) = match self.map.get(key) {
            Some(rec) => {
                value::read(rec).write_to(&mut buf);
                (value::expire_at(rec), false)
            }
            None => (None, true),
        };
        if fresh {
            hll::empty(&mut buf);
        }

        let outcome = 'work: {
            if let Err(e) = hll::check(&buf) {
                break 'work Err(e);
            }
            let mut changed = fresh;
            for ele in eles {
                let (index, count) = hll::place(ele);
                match hll::set(&mut buf, index, count) {
                    Some(hit) => changed |= hit,
                    None => break 'work Err(hll::corrupt()),
                }
            }
            Ok(changed)
        };

        // Only a change is written back, which is what makes a `PFADD` of an
        // element that is already in the sketch free. The invalidation is here
        // and not in the sketch because Redis puts it here too: a `PFADD` that
        // creates an empty sketch and adds nothing to it still marks the cache
        // stale on the way out.
        if matches!(outcome, Ok(true)) {
            hll::invalidate(&mut buf);
            self.store_raw(key, &buf, deadline);
        }
        self.scratch = buf;
        outcome
    }

    /// `PFCOUNT key [key ...]`.
    ///
    /// One key answers out of the header cache when it is good and fills it in
    /// when it is not. Several keys are merged into one set of registers first,
    /// and that answer is never cached, because there is nowhere to put it: the
    /// union of two sketches is not a key.
    ///
    /// A key that is not there counts as an empty sketch rather than an error,
    /// so `PFCOUNT missing` is 0 and a missing key among several is skipped.
    pub fn pfcount<'k, I>(&mut self, keys: I) -> Result<u64>
    where
        I: Iterator<Item = &'k [u8]> + Clone,
    {
        for key in keys.clone() {
            self.hll_ready(key)?;
        }
        let mut one = keys.clone();
        if let (Some(key), None) = (one.next(), one.next()) {
            return self.count_one(key);
        }

        // Sixteen kibibytes of registers on the stack, which is what Redis puts
        // there for the same job. It does not go in the scratch buffer because
        // the sketches being read are borrowed out of the map and the buffer is
        // where a write would want to build its own copy.
        let mut max = [0u8; hll::REGISTERS];
        for key in keys {
            self.merge_sketch(key, &mut max)?;
        }
        Ok(estimate(&max))
    }

    /// `PFMERGE dest [source ...]`.
    ///
    /// The destination is one of the sources, so a merge never loses what was
    /// already there, and `PFMERGE dest` with no sources at all is a no-op that
    /// still answers OK. A destination that is not there is created.
    ///
    /// The result stays sparse when every input was sparse and it fits, which is
    /// what a real server does: merging two hundred element sketches leaves a
    /// two hundred and seventy nine byte one, not a dense one.
    pub fn pfmerge<'k, I>(&mut self, dest: &'k [u8], srcs: I) -> Result<()>
    where
        I: Iterator<Item = &'k [u8]> + Clone,
    {
        self.hll_ready(dest)?;
        check_len(dest, hll::DENSE)?;
        for src in srcs.clone() {
            self.hll_ready(src)?;
        }

        // Every input is read before anything is written, the destination
        // included, because the write wants the database back and the sources
        // are borrowed out of it.
        let mut max = [0u8; hll::REGISTERS];
        let mut dense = false;
        for key in std::iter::once(dest).chain(srcs) {
            dense |= self.merge_sketch(key, &mut max)?;
        }
        self.pfmerge_into(dest, &max, dense)
    }

    /// The three checks every one of these makes before it reads a key.
    ///
    /// The thaw is the one worth a sentence. Every sketch a command names is
    /// read in one pass, all of them borrowed out of the map at once, so they
    /// all have to be in memory rather than in the one buffer a served fault
    /// uses. A sketch is at most twelve kibibytes and a client counting them is
    /// going to count them again, so bringing them back is what it wanted
    /// anyway.
    pub(crate) fn hll_ready(&mut self, key: &[u8]) -> Result<()> {
        self.reap(key);
        self.string_only(key)?;
        self.thaw(key)?;
        Ok(())
    }

    /// Fold the sketch under `key` into `max`, saying whether it was dense.
    ///
    /// A key that is not there is an empty sketch, which changes no register and
    /// is not dense. The key is expected to have been through
    /// [`Keyspace::hll_ready`] already.
    pub(crate) fn merge_sketch(&self, key: &[u8], max: &mut [u8; hll::REGISTERS]) -> Result<bool> {
        let Some(bytes) = self.sketch(key)? else {
            return Ok(false);
        };
        let enc = hll::check(bytes)?;
        if !hll::merge(max, bytes, enc) {
            return Err(hll::corrupt());
        }
        Ok(enc == Encoding::Dense)
    }

    /// The write half of a merge: registers in, a sketch under `dest` out.
    ///
    /// `dense` is whether any input was dense, which is not the same question as
    /// whether the registers need the room.
    pub(crate) fn pfmerge_into(
        &mut self,
        dest: &[u8],
        max: &[u8; hll::REGISTERS],
        dense: bool,
    ) -> Result<()> {
        let mut buf = std::mem::take(&mut self.scratch);
        buf.clear();
        let deadline = match self.map.get(dest) {
            Some(rec) => {
                value::read(rec).write_to(&mut buf);
                value::expire_at(rec)
            }
            None => {
                hll::empty(&mut buf);
                None
            }
        };

        let outcome = 'work: {
            // A dense input makes the result dense whatever the destination was,
            // since the registers are coming from something that already needed
            // the room. Everything else goes in one register at a time and turns
            // dense on its own if it has to.
            if dense && !hll::to_dense(&mut buf) {
                break 'work Err(hll::corrupt());
            }
            for (i, &val) in max.iter().enumerate() {
                if val != 0 && hll::set(&mut buf, i, val).is_none() {
                    break 'work Err(hll::corrupt());
                }
            }
            Ok(())
        };

        if outcome.is_ok() {
            hll::invalidate(&mut buf);
            self.store_raw(dest, &buf, deadline);
        }
        self.scratch = buf;
        outcome
    }

    /// `PFDEBUG GETREG key`, which converts the sketch to dense first.
    ///
    /// The conversion is Redis's and it is not a side effect worth hiding: the
    /// registers of a sparse sketch cannot be handed out one at a time without
    /// walking the opcodes for each, so the debugging command that wants all
    /// 16384 of them converts once and leaves it converted.
    pub fn pfgetreg(&mut self, key: &[u8], regs: &mut [u8; hll::REGISTERS]) -> Result<()> {
        self.pftodense(key)?;
        let bytes = self.sketch(key)?.ok_or_else(no_such_key)?;
        let body = &bytes[hll::HDR..];
        for (i, slot) in regs.iter_mut().enumerate() {
            *slot = hll::dense_get(body, i);
        }
        Ok(())
    }

    /// `PFDEBUG TODENSE key`, answering whether it had to convert anything.
    pub fn pftodense(&mut self, key: &[u8]) -> Result<bool> {
        self.reap(key);
        self.string_only(key)?;
        self.thaw(key)?;
        let bytes = self.sketch(key)?.ok_or_else(no_such_key)?;
        if hll::check(bytes)? == Encoding::Dense {
            return Ok(false);
        }

        let mut buf = std::mem::take(&mut self.scratch);
        buf.clear();
        let deadline = match self.map.get(key) {
            Some(rec) => {
                value::read(rec).write_to(&mut buf);
                value::expire_at(rec)
            }
            None => None,
        };
        let outcome = if hll::to_dense(&mut buf) {
            self.store_raw(key, &buf, deadline);
            Ok(true)
        } else {
            Err(hll::corrupt())
        };
        self.scratch = buf;
        outcome
    }

    /// `PFDEBUG ENCODING key`, which is `sparse` or `dense`.
    pub fn pfencoding(&mut self, key: &[u8]) -> Result<Encoding> {
        self.reap(key);
        self.string_only(key)?;
        self.warm(key)?;
        let bytes = self.sketch(key)?.ok_or_else(no_such_key)?;
        hll::check(bytes)
    }

    /// `PFDEBUG DECODE key`, handing the opcodes to `run` as one line of text.
    ///
    /// The text goes into the scratch buffer and is lent out rather than
    /// returned, the way [`Keyspace::bitfield_with`] lends its value out, so that
    /// a debugging command does not allocate on a shard thread.
    pub fn pfdecode<T>(&mut self, key: &[u8], run: impl FnOnce(&[u8]) -> T) -> Result<T> {
        self.reap(key);
        self.string_only(key)?;
        self.warm(key)?;

        // The buffer comes out of `self` before the sketch is read out of it, so
        // that the text is being written into something the map does not own and
        // the two borrows never meet.
        let mut buf = std::mem::take(&mut self.scratch);
        buf.clear();
        let outcome = 'work: {
            let bytes = match self.sketch(key) {
                Ok(Some(bytes)) => bytes,
                Ok(None) => break 'work Err(no_such_key()),
                Err(e) => break 'work Err(e),
            };
            match hll::check(bytes) {
                Ok(Encoding::Sparse) => hll::decode(bytes, &mut buf),
                Ok(Encoding::Dense) => break 'work Err(Error::new(Code::Invalid, NOT_SPARSE)),
                Err(e) => break 'work Err(e),
            }
            Ok(())
        };
        let out = outcome.map(|()| run(&buf));
        self.scratch = buf;
        out
    }

    /// The estimate for one key, out of the cache or into it.
    fn count_one(&mut self, key: &[u8]) -> Result<u64> {
        let Some(bytes) = self.sketch(key)? else {
            return Ok(0);
        };
        let enc = hll::check(bytes)?;
        if let Some(n) = hll::cached(bytes) {
            return Ok(n);
        }
        let n = hll::count(bytes, enc)?;

        // Written back in place, which is why `PFCOUNT` can be `readonly` and
        // still leave the key's bytes different from how it found them. A sketch
        // is always stored raw, so the fast path is the only path.
        let hash = RawMap::hash_of(key);
        if let Some(rec) = self.map.value_mut_hashed(hash, key)
            && let Some(val) = value::raw_in_place(rec)
        {
            hll::cache(val, n);
        }
        Ok(n)
    }

    /// The bytes of a key, if it holds one, refusing a string that is not one.
    ///
    /// A missing key answers `None`, which every one of these commands treats as
    /// an empty sketch. An int encoded string is refused rather than read as its
    /// digits, since four digits cannot be a sketch and the sentence a client
    /// wants is the one about a HyperLogLog.
    fn sketch(&self, key: &[u8]) -> Result<Option<&[u8]>> {
        match self.peek(key) {
            None => Ok(None),
            Some(Str::Bytes(b)) => Ok(Some(b)),
            Some(Str::Int(_)) => Err(hll::not_hll()),
        }
    }
}

impl Db {
    /// `PFCOUNT key [key ...]` over a database of any width.
    ///
    /// Every key on one stripe is that one stripe's `PFCOUNT`, which is every
    /// `PFCOUNT` on a database of one stripe and every single key one wherever
    /// that key is. That matters more here than it does for the other multi key
    /// commands: one key is the form that answers out of the header cache
    /// without touching a register, and it stays that form.
    ///
    /// Keys on several stripes are checked first, all of them, and then merged
    /// into one set of registers a stripe at a time. Sixteen kibibytes of
    /// registers is the only state the merge needs, so nothing is held across
    /// the stripes but that.
    pub fn pfcount<'k, I>(&mut self, keys: I) -> Result<u64>
    where
        I: Iterator<Item = &'k [u8]> + Clone,
    {
        if let Some(home) = self.one_stripe(keys.clone()) {
            return self.stripe_mut(home).pfcount(keys);
        }
        for key in keys.clone() {
            self.at(key).hll_ready(key)?;
        }
        let mut max = [0u8; hll::REGISTERS];
        for key in keys {
            self.at_ref(key).merge_sketch(key, &mut max)?;
        }
        Ok(estimate(&max))
    }

    /// `PFMERGE dest [source ...]` over a database of any width.
    ///
    /// One stripe is the old path. Otherwise the checks run in the order a
    /// single keyspace runs them, the destination first and then the sources,
    /// so the sentence a client gets for a bad key is the sentence it would have
    /// got, and then every input is read before the destination is written.
    pub fn pfmerge<'k, I>(&mut self, dest: &'k [u8], srcs: I) -> Result<()>
    where
        I: Iterator<Item = &'k [u8]> + Clone,
    {
        if let Some(home) = self.one_stripe(std::iter::once(dest).chain(srcs.clone())) {
            return self.stripe_mut(home).pfmerge(dest, srcs);
        }
        self.at(dest).hll_ready(dest)?;
        check_len(dest, hll::DENSE)?;
        for src in srcs.clone() {
            self.at(src).hll_ready(src)?;
        }

        let mut max = [0u8; hll::REGISTERS];
        let mut dense = false;
        for key in std::iter::once(dest).chain(srcs) {
            dense |= self.at_ref(key).merge_sketch(key, &mut max)?;
        }
        self.at(dest).pfmerge_into(dest, &max, dense)
    }
}

/// The estimate a merged set of registers gives.
fn estimate(max: &[u8; hll::REGISTERS]) -> u64 {
    let mut hist = [0u32; 64];
    for &val in max {
        hist[val as usize] += 1;
    }
    hll::estimate(&hist)
}

/// What `PFDEBUG` says about a key that is not there.
fn no_such_key() -> Error {
    Error::new(Code::NotFound, NO_SUCH_KEY)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keyspace::Keyspace;

    fn db() -> Keyspace {
        Keyspace::new()
    }

    /// The key or element list these commands take, out of a test's own names.
    fn names<'k>(list: &'k [&'k [u8]]) -> impl Iterator<Item = &'k [u8]> + Clone {
        list.iter().copied()
    }

    /// What `GET` would hand a client, which is the whole point of the format.
    fn bytes(db: &mut Keyspace, key: &[u8]) -> Vec<u8> {
        db.get(key).expect("a value").expect("bytes").to_vec()
    }

    /// The three element sketch a real 8.10.1 writes, byte for byte.
    #[test]
    fn a_sketch_a_client_reads_is_the_one_a_real_server_wrote() {
        let mut db = db();
        assert!(db.pfadd(b"h", names(&[b"a", b"b", b"c"])).expect("an add"));
        assert_eq!(
            bytes(&mut db, b"h"),
            b"HYLL\x01\0\0\0\0\0\0\0\0\0\0\x80\x60\xf3\x80\x50\xb1\x84\x4b\xfb\x80\x42\x5a"
        );
        assert_eq!(db.pfcount(names(&[b"h"])).expect("a count"), 3);
        // Which a client can see: a `PFCOUNT` writes its answer into the header.
        assert_eq!(bytes(&mut db, b"h")[8..16], [3, 0, 0, 0, 0, 0, 0, 0]);
    }

    /// An empty add creates the key and answers 1, and a second one answers 0.
    #[test]
    fn adding_nothing_still_creates_the_key() {
        let mut db = db();
        assert!(db.pfadd(b"h", names(&[])).expect("an add"));
        assert!(db.exists(b"h"));
        assert_eq!(db.strlen(b"h").expect("a length"), 18);
        assert!(!db.pfadd(b"h", names(&[])).expect("an add"));
        assert_eq!(db.pfcount(names(&[b"h"])).expect("a count"), 0);
    }

    /// An element already in the sketch changes neither the answer nor the bytes.
    #[test]
    fn adding_an_element_twice_leaves_the_value_alone() {
        let mut db = db();
        db.pfadd(b"h", names(&[b"a"])).expect("an add");
        let before = bytes(&mut db, b"h");
        assert!(!db.pfadd(b"h", names(&[b"a"])).expect("an add"));
        assert_eq!(bytes(&mut db, b"h"), before);
    }

    /// The numbers a real server gives for the same elements, at three sizes.
    #[test]
    fn the_count_is_the_number_a_real_server_gives() {
        for (n, want) in [(100usize, 100u64), (1000, 995), (10_000, 10_077)] {
            let mut db = db();
            for i in 0..n {
                let ele = format!("e:{i}");
                db.pfadd(b"h", names(&[ele.as_bytes()])).expect("an add");
            }
            assert_eq!(db.pfcount(names(&[b"h"])).expect("a count"), want, "{n}");
        }
    }

    /// The two sizes the milestone gate names.
    #[test]
    fn a_sketch_is_sparse_until_it_is_not() {
        let mut db = db();
        for i in 0..1000 {
            let ele = format!("e:{i}");
            db.pfadd(b"k1", names(&[ele.as_bytes()])).expect("an add");
        }
        assert_eq!(db.strlen(b"k1").expect("a length"), 1880);
        assert_eq!(db.pfencoding(b"k1").expect("an encoding"), Encoding::Sparse);
        const { assert!(1880 <= hll::SPARSE_MAX) };

        for i in 0..10_000 {
            let ele = format!("e:{i}");
            db.pfadd(b"k2", names(&[ele.as_bytes()])).expect("an add");
        }
        assert_eq!(db.strlen(b"k2").expect("a length"), 12304);
        assert_eq!(db.pfencoding(b"k2").expect("an encoding"), Encoding::Dense);
    }

    /// Counting several keys is counting their union, and never caches.
    ///
    /// The three numbers are a real server's for the same elements, and not one
    /// of them is the true count: 150 distinct elements are counted as 151 and
    /// 50 as 49. That is what a HyperLogLog is, and agreeing with Redis about
    /// which way it is wrong is the thing being tested.
    #[test]
    fn counting_several_keys_counts_their_union() {
        let mut db = db();
        for i in 0..200 {
            let ele = format!("e:{i}");
            let key: &[u8] = if i < 150 { b"a" } else { b"b" };
            db.pfadd(key, names(&[ele.as_bytes()])).expect("an add");
        }
        assert_eq!(db.pfcount(names(&[b"a"])).expect("a count"), 151);
        assert_eq!(db.pfcount(names(&[b"b"])).expect("a count"), 49);
        assert_eq!(db.pfcount(names(&[b"a", b"b"])).expect("a count"), 199);
        // A key that is not there is an empty sketch and not an error.
        assert_eq!(db.pfcount(names(&[b"gone"])).expect("a count"), 0);
        assert_eq!(db.pfcount(names(&[b"a", b"gone"])).expect("a count"), 151);
        assert!(!db.exists(b"gone"));
    }

    /// A merge keeps what the destination had and stays sparse when it can.
    #[test]
    fn a_merge_is_a_union_and_keeps_the_smaller_form() {
        let mut db = db();
        for i in 0..100 {
            let ele = format!("e:{i}");
            db.pfadd(b"s1", names(&[ele.as_bytes()])).expect("an add");
            db.pfadd(b"s2", names(&[ele.as_bytes()])).expect("an add");
        }
        db.pfmerge(b"m", names(&[b"s1", b"s2"])).expect("a merge");
        assert_eq!(db.strlen(b"m").expect("a length"), 279);
        assert_eq!(db.pfencoding(b"m").expect("an encoding"), Encoding::Sparse);
        assert_eq!(db.pfcount(names(&[b"m"])).expect("a count"), 100);

        // The destination is one of the sources, so nothing is ever lost.
        for i in 100..200 {
            let ele = format!("e:{i}");
            db.pfadd(b"s3", names(&[ele.as_bytes()])).expect("an add");
        }
        db.pfmerge(b"m", names(&[b"s3"])).expect("a merge");
        assert_eq!(db.pfcount(names(&[b"m"])).expect("a count"), 199);
        assert_eq!(db.strlen(b"m").expect("a length"), 499);

        // And a merge with no sources at all leaves it exactly as it was.
        let before = bytes(&mut db, b"m");
        db.pfmerge(b"m", names(&[])).expect("a merge");
        assert_eq!(bytes(&mut db, b"m")[..8], before[..8]);
        assert_eq!(db.pfcount(names(&[b"m"])).expect("a count"), 199);
    }

    /// A dense source makes the result dense, whatever the destination was.
    #[test]
    fn a_dense_source_makes_the_result_dense() {
        let mut db = db();
        for i in 0..100 {
            let ele = format!("e:{i}");
            db.pfadd(b"small", names(&[ele.as_bytes()]))
                .expect("an add");
        }
        for i in 0..20_000 {
            let ele = format!("e:{i}");
            db.pfadd(b"big", names(&[ele.as_bytes()])).expect("an add");
        }
        db.pfmerge(b"m", names(&[b"small", b"big"]))
            .expect("a merge");
        assert_eq!(db.strlen(b"m").expect("a length"), 12304);
        assert_eq!(db.pfencoding(b"m").expect("an encoding"), Encoding::Dense);
        assert_eq!(db.pfcount(names(&[b"m"])).expect("a count"), 20096);
    }

    /// Every one of these keeps whatever deadline the key had.
    #[test]
    fn a_write_keeps_the_deadline() {
        let mut db = db();
        let mut fresh = Vec::new();
        hll::empty(&mut fresh);
        db.setex(b"h", 100, &fresh).expect("a set");
        let had = db.expire_at(b"h").expect("a deadline");
        db.pfadd(b"h", names(&[b"a"])).expect("an add");
        assert_eq!(db.expire_at(b"h"), Some(had));
        db.pfmerge(b"h", names(&[])).expect("a merge");
        assert_eq!(db.expire_at(b"h"), Some(had));
        db.pftodense(b"h").expect("a conversion");
        assert_eq!(db.expire_at(b"h"), Some(had));
        assert_eq!(db.pfcount(names(&[b"h"])).expect("a count"), 1);
    }

    /// The debugging commands, including the one that changes what it looks at.
    #[test]
    fn the_debug_commands_say_what_a_real_server_says() {
        let mut db = db();
        db.pfadd(b"h", names(&[b"a", b"b", b"c"])).expect("an add");
        let text = db.pfdecode(b"h", <[u8]>::to_vec).expect("a decode");
        assert_eq!(text, b"Z:8436 v:1,1 Z:4274 v:2,1 Z:3068 v:1,1 Z:603");

        // Which is the three registers a real server has, and reading them
        // converts the sketch and leaves it converted.
        let mut regs = [0u8; hll::REGISTERS];
        db.pfgetreg(b"h", &mut regs).expect("the registers");
        assert_eq!(regs[8436], 1);
        assert_eq!(regs[12711], 2);
        assert_eq!(regs[15780], 1);
        assert_eq!(regs.iter().filter(|&&v| v != 0).count(), 3);
        assert_eq!(db.pfencoding(b"h").expect("an encoding"), Encoding::Dense);
        assert_eq!(db.strlen(b"h").expect("a length"), 12304);
        assert_eq!(db.pfcount(names(&[b"h"])).expect("a count"), 3);

        // A converted sketch cannot be decoded and does not convert twice.
        assert!(db.pfdecode(b"h", <[u8]>::to_vec).is_err());
        assert!(!db.pftodense(b"h").expect("a conversion"));
    }

    /// A string that is not a sketch, and a key that is not a string.
    #[test]
    fn a_string_that_is_not_a_sketch_is_refused() {
        let mut db = db();
        db.set_plain(b"plain", b"not a sketch at all")
            .expect("a set");
        assert!(db.pfadd(b"plain", names(&[b"a"])).is_err());
        assert!(db.pfcount(names(&[b"plain"])).is_err());
        assert!(db.pfmerge(b"plain", names(&[])).is_err());
        assert!(db.pfencoding(b"plain").is_err());
        // An int encoded string is refused too, rather than read as its digits.
        db.set_plain(b"n", b"12345").expect("a set");
        assert!(db.pfcount(names(&[b"n"])).is_err());

        // A key that is not there is not an error for the three real commands
        // and is one for every `PFDEBUG` form.
        assert!(db.pfencoding(b"gone").is_err());
        assert!(db.pftodense(b"gone").is_err());
        assert!(db.pfdecode(b"gone", <[u8]>::to_vec).is_err());
        let mut regs = [0u8; hll::REGISTERS];
        assert!(db.pfgetreg(b"gone", &mut regs).is_err());
    }

    /// A sketch whose opcodes do not add up says so rather than answering.
    #[test]
    fn a_corrupted_sketch_is_reported() {
        let mut db = db();
        db.pfadd(b"h", names(&[b"a", b"b", b"c"])).expect("an add");
        let mut short = bytes(&mut db, b"h");
        short.pop();
        db.set_plain(b"h", &short).expect("a set");
        let err = db.pfcount(names(&[b"h"])).expect_err("a complaint");
        assert_eq!(err.code(), Code::Corrupt);
        assert!(db.pfcount(names(&[b"h", b"h"])).is_err());
        assert!(db.pfmerge(b"m", names(&[b"h"])).is_err());
    }
}
