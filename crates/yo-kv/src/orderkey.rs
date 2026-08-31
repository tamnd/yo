//! Order keys for a list, so that an insert between two elements never has to
//! move a third one.
//!
//! A list element can be addressed two ways. By position, which is what a client
//! asks for, and by a sort key, which is what an index stores. Addressing by
//! position is what aki did and it is why `LINSERT` came out at 87 operations a
//! second against a rival's 6,671: putting an element at position 5 means every
//! element from 5 upwards gets a new number, and every one of those is a row
//! rewritten through the index. Addressing by sort key means the new element
//! gets a key between its two neighbours and nothing else is touched at all.
//!
//! The whole difficulty is in the word between. Two neighbours have to have room
//! between them, and they have to keep having room however many times the same
//! spot is hammered, because a work queue that always inserts at one priority is
//! an ordinary thing to build and not a pathology.
//!
//! # Why the keys are variable length
//!
//! `03` section C, Y19, settles this and this module is the port. The obvious
//! design is a fixed width key, a float or a fixed size integer, and a midpoint
//! between two neighbours. It is compact, it compares in one instruction, and it
//! runs out. A `f64` midpoint between two adjacent representable values has
//! nowhere left to go after about 52 halvings, and at that point the structure
//! has to stop in the middle of a command and renumber. That is a correctness
//! cliff sitting inside a common access pattern, which is the one thing a
//! storage engine is not allowed to have.
//!
//! A variable length key subdivides forever. It costs bytes, and the number that
//! matters is how many: this allocator buys eight inserts at one spot for each
//! byte a key grows by, so twenty thousand inserts between the same two
//! neighbours leaves a key 2,508 bytes long and still strictly ordered. That is
//! K14, and `a_hammer_at_one_spot_grows_by_a_byte_every_eight_inserts` measures
//! it rather than trusting it.
//!
//! Growth is real and it is bounded by [`ORDER_KEY_MAX`] rather than by nothing.
//! Past that [`between`] answers `None` and the caller has to renumber the run,
//! which is a bounded, local, offline job because the growth is local to the
//! spot being hammered. Nothing in the tree needs it yet at four kilobytes, so
//! the renumberer is a follow up and not part of this.
//!
//! # The invariant
//!
//! **No key this module produces ends in a `0x00` byte.**
//!
//! It carries the whole thing. If a key could end in `0x00` then a pair like
//! `[0x41]` and `[0x41, 0x00]` could turn up as neighbours, and there is no key
//! at all that sorts strictly between those two: anything above `[0x41]` starts
//! with `[0x41]` and a further byte, and every further byte is at or above
//! `0x00`. The pair is a dead end and the list wedges, which is exactly the
//! failure the variable length key was chosen to avoid.
//!
//! aki held the invariant for the keys its midpoint descent produced and argued
//! the end keys were safe because they were all the same width. That argument
//! has a hole in it, because an eight byte end key for sequence zero really does
//! end in `0x00`, and a one byte interior key sorting just under it really is
//! reachable. So [`end`] holds the invariant too, by spending its low byte on a
//! fixed `0x80` and encoding the sequence in the seven bytes above it. Fifty six
//! bits of sequence is a hundred years of pushing at ten million a second, and
//! the invariant becomes something the module guarantees rather than something
//! the caller has to keep true.
//!
//! # The two ways a list grows
//!
//! Pushing, which is the hot path, and inserting, which is not.
//!
//! A push takes the next sequence number from the end it is pushing to and
//! encodes it, which is [`Ends`] and [`end`]. No descent, no comparison, no
//! allocation. The one rule worth stating is that a pop **retires** its sequence
//! rather than handing it back: the cursors only ever move outward. That is what
//! makes it safe for a pop to drop its row outside the lock later on, because no
//! future push can ever aim at the key a pop is in the middle of removing.
//!
//! An insert calls [`between`], which is a base 256 midpoint descent: at each
//! byte it tries to fit a value strictly between the two keys' bytes, and where
//! they are equal or next to each other it copies and goes one byte deeper.
//!
//! # Nothing here allocates
//!
//! [`between`] writes into a buffer the caller owns and answers how much of it
//! it used, so an insert on a command path costs no more than the stack slot the
//! caller already had. That is Y7 and it is why the signature is shaped the way
//! it is rather than returning a `Vec<u8>`.

/// The longest key [`between`] will build before it gives up and asks to be
/// renumbered.
///
/// Four kilobytes is thirty two thousand inserts at one spot at the measured
/// eight per byte, which is well past the twenty thousand K14 asks for and well
/// under anything a key is compared against in bulk.
pub const ORDER_KEY_MAX: usize = 4096;

/// How wide the key for a pushed element is, always.
pub const END_LEN: usize = 8;

/// The lowest sequence [`end`] will encode.
pub const SEQ_MIN: i64 = -(1 << 55);

/// The highest sequence [`end`] will encode.
pub const SEQ_MAX: i64 = (1 << 55) - 1;

/// The byte a key gets when it has to be made longer without being made bigger
/// by much. Halfway up the range, so that the next insert on either side of it
/// still has somewhere to go.
const MID: u8 = 0x80;

/// The fifty six bits of an end key that carry the sequence.
const SEQ_MASK: u64 = (1u64 << 56) - 1;

/// The key for the element at sequence `seq`.
///
/// Big endian with the sign bit flipped, which is the encoding that makes a
/// signed comparison and a bytewise comparison agree, in the seven bytes above a
/// fixed `0x80`. Panics outside [`SEQ_MIN`]`..=`[`SEQ_MAX`], which is a
/// programming error rather than a state a list can reach: it is fifty six bits
/// of pushes to one end.
#[must_use]
pub fn end(seq: i64) -> [u8; END_LEN] {
    assert!(
        (SEQ_MIN..=SEQ_MAX).contains(&seq),
        "list order sequence out of range"
    );
    // The flip turns the two's complement order into the unsigned one and the
    // mask drops the sign extension above the fifty six bits that are being
    // encoded, which would otherwise be shifted straight back into the key. The
    // shift makes room for the terminal byte.
    let u = (((seq as u64) ^ (1u64 << 55)) & SEQ_MASK) << 8;
    let mut key = u.to_be_bytes();
    key[END_LEN - 1] = MID;
    key
}

/// The sequence [`end`] encoded, for a key that came from it.
///
/// Answers `None` for a key that did not, which is any key that is not eight
/// bytes long or does not end in the terminal byte. An interior key can be eight
/// bytes long, so the terminal check is not decoration.
#[must_use]
pub fn seq_of(key: &[u8]) -> Option<i64> {
    if key.len() != END_LEN || key[END_LEN - 1] != MID {
        return None;
    }
    let mut bytes = [0u8; END_LEN];
    bytes.copy_from_slice(key);
    bytes[END_LEN - 1] = 0;
    let v = (u64::from_be_bytes(bytes) >> 8) ^ (1u64 << 55);
    // Back up into the sign bit and down again, which is how a fifty six bit
    // two's complement number is widened to sixty four.
    Some(((v << 8) as i64) >> 8)
}

/// The two sequences a list's header keeps, one per end.
///
/// A pop does not appear here on purpose. The cursors move outward on a push and
/// never come back, so a sequence is used once for the life of the list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ends {
    /// The next sequence a front push will take, counting down.
    front: i64,
    /// The next sequence a back push will take, counting up.
    back: i64,
}

impl Default for Ends {
    fn default() -> Ends {
        Ends::new()
    }
}

impl Ends {
    /// An empty list, with both ends starting from the middle of the range.
    #[must_use]
    pub fn new() -> Ends {
        Ends { front: -1, back: 0 }
    }

    /// The key for an element pushed onto the front, and the cursor moved down.
    ///
    /// `None` once the front has been pushed to fifty six bits of times, which
    /// is a list that cannot exist rather than a case with a recovery.
    pub fn push_front(&mut self) -> Option<[u8; END_LEN]> {
        if self.front < SEQ_MIN {
            return None;
        }
        let key = end(self.front);
        self.front -= 1;
        Some(key)
    }

    /// The key for an element pushed onto the back, and the cursor moved up.
    pub fn push_back(&mut self) -> Option<[u8; END_LEN]> {
        if self.back > SEQ_MAX {
            return None;
        }
        let key = end(self.back);
        self.back += 1;
        Some(key)
    }
}

/// Where the descent has got to.
///
/// It starts out pinned to both bounds and comes loose from one of them as soon
/// as a byte is written that is strictly inside. Once it is loose from a bound
/// that bound stops mattering, which is what lets the rest of the descent be a
/// walk down one key rather than two.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Descent {
    /// Equal to both bounds so far.
    Both,
    /// Already above `lo`, so only `hi` is still in the way.
    Upper,
    /// Already below `hi`, so only `lo` is still in the way.
    Lower,
}

/// A key that sorts strictly after `lo` and strictly before `hi`, written into
/// `out`, with how many bytes of it were used.
///
/// `None` when there is no such key, which happens three ways: `lo` is not
/// strictly below `hi`, the answer would be longer than `out` or longer than
/// [`ORDER_KEY_MAX`], or the two bounds are a dead end of the shape the
/// invariant exists to prevent. The last one is unreachable for keys this module
/// produced and is answered rather than asserted, because `between` is also the
/// thing an importer runs over keys it did not produce.
///
/// The answer never ends in `0x00`, so it can be a bound for the next call.
pub fn between(lo: &[u8], hi: &[u8], out: &mut [u8]) -> Option<usize> {
    if lo >= hi {
        return None;
    }
    let cap = out.len().min(ORDER_KEY_MAX);
    let mut mode = Descent::Both;
    let mut i = 0;
    let len = loop {
        if i >= cap {
            return None;
        }
        match mode {
            Descent::Both => {
                // `hi` cannot have run out here. It agrees with `lo` on every
                // byte so far, so a `hi` that ended here would be a prefix of
                // `lo` and would sort at or below it.
                let h = *hi.get(i)?;
                match lo.get(i) {
                    // `lo` ran out, so anything written from here on is already
                    // above it and only `hi` is left to stay under.
                    None => mode = Descent::Upper,
                    Some(&l) if h - l >= 2 => {
                        // Room for a byte strictly between the two, which ends
                        // the descent. It is at least `l + 1` and so never zero.
                        out[i] = l + (h - l) / 2;
                        break i + 1;
                    }
                    Some(&l) if h - l == 1 => {
                        // Copying the lower byte puts us under `hi` for good.
                        out[i] = l;
                        i += 1;
                        mode = Descent::Lower;
                    }
                    Some(&l) => {
                        out[i] = l;
                        i += 1;
                    }
                }
            }
            Descent::Upper => {
                // Under `hi` is all that is left. A byte that ended here would
                // be the dead end the invariant rules out, so answer rather than
                // loop.
                let h = *hi.get(i)?;
                if h == 0 {
                    out[i] = 0;
                    i += 1;
                } else {
                    out[i] = h / 2;
                    break i + 1;
                }
            }
            Descent::Lower => {
                // Above `lo` is all that is left, and `lo` running out is the
                // easy case rather than the hard one: one byte past its end is
                // already past it.
                match lo.get(i) {
                    Some(&0xff) => {
                        out[i] = 0xff;
                        i += 1;
                    }
                    Some(&l) => {
                        out[i] = l + 1 + (0xff - l) / 2;
                        break i + 1;
                    }
                    None => {
                        out[i] = MID;
                        break i + 1;
                    }
                }
            }
        }
    };
    // The one place a zero terminal byte can come out of the descent is halving
    // a `hi` byte of one. Another byte fixes it, and it stays under `hi` because
    // the key already differs from `hi` at an earlier byte rather than being a
    // prefix of it.
    if out[len - 1] == 0 {
        if len >= cap {
            return None;
        }
        out[len] = MID;
        return Some(len + 1);
    }
    Some(len)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A key long enough for anything these tests ask for.
    fn buf() -> [u8; ORDER_KEY_MAX] {
        [0u8; ORDER_KEY_MAX]
    }

    #[test]
    fn an_end_key_sorts_the_way_the_sequence_does() {
        let seqs = [
            SEQ_MIN,
            SEQ_MIN + 1,
            -1_000_000,
            -256,
            -1,
            0,
            1,
            255,
            256,
            1_000_000,
            SEQ_MAX - 1,
            SEQ_MAX,
        ];
        for pair in seqs.windows(2) {
            let (a, b) = (end(pair[0]), end(pair[1]));
            assert!(a < b, "{:?} should sort under {:?}", pair[0], pair[1]);
        }
        for seq in seqs {
            let key = end(seq);
            assert_eq!(seq_of(&key), Some(seq));
            assert_ne!(key[END_LEN - 1], 0, "an end key must not end in a zero");
        }
    }

    #[test]
    fn a_key_that_did_not_come_from_end_is_not_read_as_one() {
        assert_eq!(seq_of(b""), None);
        assert_eq!(seq_of(b"1234567"), None);
        assert_eq!(seq_of(b"123456789"), None);
        // Eight bytes and the wrong terminal, which an interior key can be.
        assert_eq!(seq_of(&[0x80, 0, 0, 0, 0, 0, 0, 0x40]), None);
    }

    #[test]
    fn the_ends_only_ever_move_outward() {
        let mut ends = Ends::new();
        let a = ends.push_back().unwrap();
        let b = ends.push_back().unwrap();
        let x = ends.push_front().unwrap();
        let y = ends.push_front().unwrap();
        assert!(y < x && x < a && a < b);
        // A pop is not a method here, so the next push after one lands past
        // everything that has ever been pushed rather than on top of it.
        let c = ends.push_back().unwrap();
        assert!(b < c);
    }

    #[test]
    fn a_key_between_two_lands_between_them() {
        let mut out = buf();
        let cases: &[(&[u8], &[u8])] = &[
            (b"a", b"c"),
            (b"a", b"b"),
            (b"aa", b"ab"),
            (b"a", b"aa"),
            (&[0x00], &[0xff]),
            (&[0x41], &[0x41, 0x01, 0x80]),
            (&[0xff, 0xff, 0x80], &[0xff, 0xff, 0x81]),
            (&end(0), &end(1)),
            (&end(-1), &end(0)),
            (&end(SEQ_MIN), &end(SEQ_MAX)),
        ];
        for &(lo, hi) in cases {
            let n = between(lo, hi, &mut out).expect("there is room between these");
            let key = &out[..n];
            assert!(lo < key, "{key:?} should sort above {lo:?}");
            assert!(key < hi, "{key:?} should sort under {hi:?}");
            assert_ne!(key[n - 1], 0, "{key:?} must not end in a zero");
        }
    }

    #[test]
    fn there_is_nothing_between_a_key_and_itself_or_a_pair_the_wrong_way_round() {
        let mut out = buf();
        assert_eq!(between(b"a", b"a", &mut out), None);
        assert_eq!(between(b"b", b"a", &mut out), None);
        assert_eq!(between(b"aa", b"a", &mut out), None);
        // The dead end the invariant exists to keep out of a live list. It is
        // answered rather than looped on.
        assert_eq!(between(b"a", b"a\x00", &mut out), None);
        assert_eq!(between(b"a", b"a\x00\x00", &mut out), None);
    }

    #[test]
    fn a_key_that_will_not_fit_is_refused_rather_than_truncated() {
        let mut small = [0u8; 2];
        assert_eq!(between(&end(0), &end(1), &mut small), None);
        let mut one = [0u8; 1];
        assert_eq!(between(b"a", b"c", &mut one), Some(1));
        assert_eq!(one[0], b'b');
    }

    /// Every ordered pair of short strings over a small alphabet, which is the
    /// only way to be sure the three ways the descent can end are all right.
    #[test]
    fn every_pair_that_honours_the_invariant_has_a_key_between_it() {
        let alphabet = [0x00u8, 0x01, 0x02, 0x7f, 0x80, 0xfe, 0xff];
        let mut keys: Vec<Vec<u8>> = Vec::new();
        for len in 1..=3 {
            let mut key = vec![0u8; len];
            let mut counter = 0usize;
            let total = alphabet.len().pow(u32::try_from(len).unwrap());
            while counter < total {
                let mut n = counter;
                for byte in key.iter_mut().take(len) {
                    *byte = alphabet[n % alphabet.len()];
                    n /= alphabet.len();
                }
                // The invariant is what the descent is allowed to assume, so a
                // pair that breaks it is not a pair a list can present.
                if key[len - 1] != 0 {
                    keys.push(key.clone());
                }
                counter += 1;
            }
        }
        keys.sort();
        let mut out = buf();
        let mut pairs = 0;
        for (i, lo) in keys.iter().enumerate() {
            for hi in &keys[i + 1..] {
                let n = between(lo, hi, &mut out).expect("the invariant leaves room");
                let key = &out[..n];
                assert!(lo.as_slice() < key, "{key:?} above {lo:?}");
                assert!(key < hi.as_slice(), "{key:?} under {hi:?}");
                assert_ne!(key[n - 1], 0);
                pairs += 1;
            }
        }
        assert_eq!(pairs, 58_311, "every ordered pair of the 342 legal keys");
    }

    /// K14, measured. Twenty thousand inserts at one spot, every one of them
    /// between the fixed left neighbour and the key the last insert produced,
    /// which is the adversary that a fixed precision key dies to at about 52.
    #[test]
    fn a_hammer_at_one_spot_grows_by_a_byte_every_eight_inserts() {
        const N: usize = 20_000;
        let lo = end(0);
        let mut hi = end(1).to_vec();
        let mut out = buf();
        let mut deepest = 0;
        for i in 0..N {
            let n = between(&lo, &hi, &mut out)
                .unwrap_or_else(|| panic!("wedged after {i} inserts, which is what scheme A does"));
            let key = &out[..n];
            assert!(lo.as_slice() < key && key < hi.as_slice());
            assert_ne!(key[n - 1], 0);
            deepest = deepest.max(n);
            hi.clear();
            hi.extend_from_slice(key);
        }
        // Eight halvings to a byte, on top of the eight the end key started at.
        let grown = deepest - END_LEN;
        let per_byte = N as f64 / grown as f64;
        assert!(
            per_byte >= 8.0,
            "{per_byte} inserts per byte, K14 asks for 8.0"
        );
        assert_eq!(deepest, 2508, "aki's number, to the byte");
    }

    /// The same hammer from the other side, because the descent takes a
    /// different branch going up than it does going down.
    #[test]
    fn a_hammer_under_the_upper_neighbour_grows_at_the_same_rate() {
        const N: usize = 20_000;
        let hi = end(1);
        let mut lo = end(0).to_vec();
        let mut out = buf();
        let mut deepest = 0;
        for _ in 0..N {
            let n = between(&lo, &hi, &mut out).expect("a variable key does not wedge");
            let key = &out[..n];
            assert!(lo.as_slice() < key && key < hi.as_slice());
            assert_ne!(key[n - 1], 0);
            deepest = deepest.max(n);
            lo.clear();
            lo.extend_from_slice(key);
        }
        let per_byte = N as f64 / (deepest - END_LEN) as f64;
        assert!(
            per_byte >= 8.0,
            "{per_byte} inserts per byte, K14 asks for 8.0"
        );
    }

    /// The faithful walk: start from two real end keys and only ever insert
    /// between real neighbours, so every bound is a key the allocator itself
    /// produced. Nothing here is allowed to wedge and the order is checked whole
    /// rather than pairwise.
    #[test]
    fn a_list_built_only_out_of_its_own_keys_stays_ordered() {
        let mut keys: Vec<Vec<u8>> = vec![end(0).to_vec(), end(1).to_vec()];
        let mut out = buf();
        // A cheap deterministic spread, so the inserts land all over the list
        // rather than at one spot.
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        for _ in 0..5_000 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let at = (seed % (keys.len() as u64 - 1)) as usize;
            let n = between(&keys[at], &keys[at + 1], &mut out).expect("no wedge");
            keys.insert(at + 1, out[..n].to_vec());
        }
        assert!(keys.windows(2).all(|w| w[0] < w[1]), "the list is ordered");
        assert!(keys.iter().all(|k| *k.last().unwrap() != 0), "invariant");
        assert_eq!(keys.len(), 5_002);
    }
}
