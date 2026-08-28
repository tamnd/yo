//! XXH3, the 64 bit form, with the default secret and no seed.
//!
//! This is here for one reason: Redis 8.4's `DIGEST` returns the XXH3 hash of a
//! string value as sixteen hex characters, and `SET ... IFDEQ`, `SET ... IFDNE`
//! and `DELEX ... IFDEQ` compare against exactly that number. A client computes
//! the digest of the value it read, sends the digest instead of the value, and
//! saves a round trip of the whole string on a compare and swap. That only
//! works if our number is bit for bit the same number, so this is the real
//! algorithm and not a hash that happens to be fast.
//!
//! Only the seedless default secret path exists, because that is the only one
//! Redis uses and a seed argument nobody passes is a branch on a hot path and
//! four more code paths to get wrong. The secret below is the 192 byte
//! `kSecret` from the reference implementation, which is a constant of the
//! algorithm and not a choice this crate is making.
//!
//! The tests check every length class against digests taken from a real Redis
//! 8.8, which is the only check worth having: an implementation that is
//! self consistent and wrong would pass anything else.

/// The 192 byte default secret, `kSecret` in the reference implementation.
///
/// Every length class reads a different window of it, and the long path reads
/// all of it, so a single wrong byte shows up in one class and not the others.
const SECRET: [u8; 192] = [
    0xb8, 0xfe, 0x6c, 0x39, 0x23, 0xa4, 0x4b, 0xbe, 0x7c, 0x01, 0x81, 0x2c, 0xf7, 0x21, 0xad, 0x1c,
    0xde, 0xd4, 0x6d, 0xe9, 0x83, 0x90, 0x97, 0xdb, 0x72, 0x40, 0xa4, 0xa4, 0xb7, 0xb3, 0x67, 0x1f,
    0xcb, 0x79, 0xe6, 0x4e, 0xcc, 0xc0, 0xe5, 0x78, 0x82, 0x5a, 0xd0, 0x7d, 0xcc, 0xff, 0x72, 0x21,
    0xb8, 0x08, 0x46, 0x74, 0xf7, 0x43, 0x24, 0x8e, 0xe0, 0x35, 0x90, 0xe6, 0x81, 0x3a, 0x26, 0x4c,
    0x3c, 0x28, 0x52, 0xbb, 0x91, 0xc3, 0x00, 0xcb, 0x88, 0xd0, 0x65, 0x8b, 0x1b, 0x53, 0x2e, 0xa3,
    0x71, 0x64, 0x48, 0x97, 0xa2, 0x0d, 0xf9, 0x4e, 0x38, 0x19, 0xef, 0x46, 0xa9, 0xde, 0xac, 0xd8,
    0xa8, 0xfa, 0x76, 0x3f, 0xe3, 0x9c, 0x34, 0x3f, 0xf9, 0xdc, 0xbb, 0xc7, 0xc7, 0x0b, 0x4f, 0x1d,
    0x8a, 0x51, 0xe0, 0x4b, 0xcd, 0xb4, 0x59, 0x31, 0xc8, 0x9f, 0x7e, 0xc9, 0xd9, 0x78, 0x73, 0x64,
    0xea, 0xc5, 0xac, 0x83, 0x34, 0xd3, 0xeb, 0xc3, 0xc5, 0x81, 0xa0, 0xff, 0xfa, 0x13, 0x63, 0xeb,
    0x17, 0x0d, 0xdd, 0x51, 0xb7, 0xf0, 0xda, 0x49, 0xd3, 0x16, 0x55, 0x26, 0x29, 0xd4, 0x68, 0x9e,
    0x2b, 0x16, 0xbe, 0x58, 0x7d, 0x47, 0xa1, 0xfc, 0x8f, 0xf8, 0xb8, 0xd1, 0x7a, 0xd0, 0x31, 0xce,
    0x45, 0xcb, 0x3a, 0x8f, 0x95, 0x16, 0x04, 0x28, 0xaf, 0xd7, 0xfb, 0xca, 0xbb, 0x4b, 0x40, 0x7e,
];

const PRIME32_1: u64 = 0x9E37_79B1;
const PRIME32_2: u64 = 0x85EB_CA77;
const PRIME32_3: u64 = 0xC2B2_AE3D;
const PRIME64_1: u64 = 0x9E37_79B1_85EB_CA87;
const PRIME64_2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const PRIME64_3: u64 = 0x1656_67B1_9E37_79F9;
const PRIME64_4: u64 = 0x85EB_CA77_C2B2_AE63;
const PRIME64_5: u64 = 0x27D4_EB2F_1656_67C5;
const PRIME_MX1: u64 = 0x1656_6791_9E37_79F9;
const PRIME_MX2: u64 = 0x9FB2_1C65_1E98_DF25;

/// One stripe, in bytes. The long path reads the input in these.
const STRIPE: usize = 64;
/// How many stripes go by before the accumulator is scrambled.
const STRIPES_PER_BLOCK: usize = (SECRET.len() - STRIPE) / 8;
/// How much input one block covers.
const BLOCK: usize = STRIPE * STRIPES_PER_BLOCK;

/// The XXH3 64 bit hash of `input`, with the default secret and no seed.
///
/// This is Redis's `DIGEST`, and formatting the result as sixteen lower case
/// hex characters is the reply.
#[must_use]
pub fn hash64(input: &[u8]) -> u64 {
    match input.len() {
        0..=16 => short(input),
        17..=128 => medium(input),
        129..=240 => long_ish(input),
        _ => long(input),
    }
}

#[inline]
fn rd64(at: usize, from: &[u8]) -> u64 {
    u64::from_le_bytes(from[at..at + 8].try_into().expect("eight bytes"))
}

#[inline]
fn rd32(at: usize, from: &[u8]) -> u64 {
    u64::from(u32::from_le_bytes(
        from[at..at + 4].try_into().expect("four bytes"),
    ))
}

/// The low half of a 64 by 64 multiply folded onto the high half.
#[inline]
fn fold(a: u64, b: u64) -> u64 {
    let p = u128::from(a) * u128::from(b);
    (p as u64) ^ ((p >> 64) as u64)
}

#[inline]
fn avalanche(mut h: u64) -> u64 {
    h ^= h >> 37;
    h = h.wrapping_mul(PRIME_MX1);
    h ^= h >> 32;
    h
}

/// XXH64's finaliser, which the two shortest classes still use.
#[inline]
fn avalanche64(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(PRIME64_2);
    h ^= h >> 29;
    h = h.wrapping_mul(PRIME64_3);
    h ^= h >> 32;
    h
}

#[inline]
fn rrmxmx(mut h: u64, len: u64) -> u64 {
    h ^= h.rotate_left(49) ^ h.rotate_left(24);
    h = h.wrapping_mul(PRIME_MX2);
    h ^= (h >> 35).wrapping_add(len);
    h = h.wrapping_mul(PRIME_MX2);
    h ^ (h >> 28)
}

/// Sixteen bytes of input against sixteen bytes of secret.
#[inline]
fn mix16(input: &[u8], at: usize, secret_at: usize) -> u64 {
    fold(
        rd64(at, input) ^ rd64(secret_at, &SECRET),
        rd64(at + 8, input) ^ rd64(secret_at + 8, &SECRET),
    )
}

/// Zero to sixteen bytes, which is four sub cases and no loop.
fn short(input: &[u8]) -> u64 {
    let len = input.len();
    if len > 8 {
        let lo = rd64(0, input) ^ (rd64(24, &SECRET) ^ rd64(32, &SECRET));
        let hi = rd64(len - 8, input) ^ (rd64(40, &SECRET) ^ rd64(48, &SECRET));
        let acc = (len as u64)
            .wrapping_add(lo.swap_bytes())
            .wrapping_add(hi)
            .wrapping_add(fold(lo, hi));
        return avalanche(acc);
    }
    if len >= 4 {
        let one = rd32(0, input);
        let two = rd32(len - 4, input);
        let bitflip = rd64(8, &SECRET) ^ rd64(16, &SECRET);
        let keyed = (two.wrapping_add(one << 32)) ^ bitflip;
        return rrmxmx(keyed, len as u64);
    }
    if len > 0 {
        let c1 = u64::from(input[0]);
        let c2 = u64::from(input[len >> 1]);
        let c3 = u64::from(input[len - 1]);
        let combined = (c1 << 16) | (c2 << 24) | c3 | ((len as u64) << 8);
        return avalanche64(combined ^ (rd32(0, &SECRET) ^ rd32(4, &SECRET)));
    }
    avalanche64(rd64(56, &SECRET) ^ rd64(64, &SECRET))
}

/// Seventeen to a hundred and twenty eight bytes, as up to eight overlapping
/// sixteen byte mixes chosen by size.
fn medium(input: &[u8]) -> u64 {
    let len = input.len();
    let mut acc = (len as u64).wrapping_mul(PRIME64_1);
    if len > 32 {
        if len > 64 {
            if len > 96 {
                acc = acc.wrapping_add(mix16(input, 48, 96));
                acc = acc.wrapping_add(mix16(input, len - 64, 112));
            }
            acc = acc.wrapping_add(mix16(input, 32, 64));
            acc = acc.wrapping_add(mix16(input, len - 48, 80));
        }
        acc = acc.wrapping_add(mix16(input, 16, 32));
        acc = acc.wrapping_add(mix16(input, len - 32, 48));
    }
    acc = acc.wrapping_add(mix16(input, 0, 0));
    acc = acc.wrapping_add(mix16(input, len - 16, 16));
    avalanche(acc)
}

/// A hundred and twenty nine to two hundred and forty bytes, where the mixes
/// become a loop and the secret window shifts by three bytes partway through.
fn long_ish(input: &[u8]) -> u64 {
    /// Where the second run of mixes starts reading the secret.
    const START_OFFSET: usize = 3;
    /// How far back from the minimum secret size the last mix reads.
    const LAST_OFFSET: usize = 17;
    /// The shortest secret the algorithm defines, which fixes the last window.
    const SECRET_SIZE_MIN: usize = 136;

    let len = input.len();
    let rounds = len / 16;
    let mut acc = (len as u64).wrapping_mul(PRIME64_1);
    for i in 0..8 {
        acc = acc.wrapping_add(mix16(input, 16 * i, 16 * i));
    }
    acc = avalanche(acc);
    for i in 8..rounds {
        acc = acc.wrapping_add(mix16(input, 16 * i, 16 * (i - 8) + START_OFFSET));
    }
    acc = acc.wrapping_add(mix16(input, len - 16, SECRET_SIZE_MIN - LAST_OFFSET));
    avalanche(acc)
}

/// One stripe into the eight accumulators.
#[inline]
fn accumulate(acc: &mut [u64; 8], input: &[u8], at: usize, secret_at: usize) {
    for i in 0..8 {
        let data = rd64(at + 8 * i, input);
        let key = data ^ rd64(secret_at + 8 * i, &SECRET);
        // The lane swap is what stops the accumulators from being eight
        // independent hashes of eight independent byte positions.
        acc[i ^ 1] = acc[i ^ 1].wrapping_add(data);
        acc[i] = acc[i].wrapping_add((key & 0xFFFF_FFFF).wrapping_mul(key >> 32));
    }
}

#[inline]
fn scramble(acc: &mut [u64; 8], secret_at: usize) {
    for (i, lane) in acc.iter_mut().enumerate() {
        let mut a = *lane;
        a ^= a >> 47;
        a ^= rd64(secret_at + 8 * i, &SECRET);
        *lane = a.wrapping_mul(PRIME32_1);
    }
}

fn merge(acc: &[u64; 8], secret_at: usize, start: u64) -> u64 {
    let mut out = start;
    for i in 0..4 {
        out = out.wrapping_add(fold(
            acc[2 * i] ^ rd64(secret_at + 16 * i, &SECRET),
            acc[2 * i + 1] ^ rd64(secret_at + 16 * i + 8, &SECRET),
        ));
    }
    avalanche(out)
}

/// Anything over two hundred and forty bytes, which is the streaming shape:
/// stripes into eight accumulators, a scramble every block, a final stripe that
/// overlaps whatever came before it, and a merge.
fn long(input: &[u8]) -> u64 {
    /// How far back from the end of the secret the last stripe reads.
    const LAST_ACC_START: usize = 7;
    /// Where the merge starts reading the secret.
    const MERGE_START: usize = 11;

    let len = input.len();
    let mut acc: [u64; 8] = [
        PRIME32_3, PRIME64_1, PRIME64_2, PRIME64_3, PRIME64_4, PRIME32_2, PRIME64_5, PRIME32_1,
    ];

    let blocks = (len - 1) / BLOCK;
    for b in 0..blocks {
        for s in 0..STRIPES_PER_BLOCK {
            accumulate(&mut acc, input, b * BLOCK + s * STRIPE, s * 8);
        }
        scramble(&mut acc, SECRET.len() - STRIPE);
    }

    // The tail of the last block, one stripe at a time, and then a stripe that
    // ends exactly at the end of the input. The two overlap for any length that
    // is not a multiple of the stripe, which is how the last few bytes get in
    // without a separate case for them.
    let stripes = ((len - 1) - BLOCK * blocks) / STRIPE;
    for s in 0..stripes {
        accumulate(&mut acc, input, blocks * BLOCK + s * STRIPE, s * 8);
    }
    accumulate(
        &mut acc,
        input,
        len - STRIPE,
        SECRET.len() - STRIPE - LAST_ACC_START,
    );

    merge(&acc, MERGE_START, (len as u64).wrapping_mul(PRIME64_1))
}

/// The sixteen lower case hex characters `DIGEST` replies with.
#[must_use]
pub fn hex(h: u64) -> [u8; 16] {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = [0u8; 16];
    for i in 0..16 {
        out[i] = DIGITS[((h >> (60 - 4 * i)) & 0xF) as usize];
    }
    out
}

/// The number behind sixteen hex characters, or `None` if that is not what it
/// is. `IFDEQ` takes the digest as text and refuses anything else.
#[must_use]
pub fn from_hex(text: &[u8]) -> Option<u64> {
    if text.len() != 16 {
        return None;
    }
    let mut h = 0u64;
    for &c in text {
        let d = match c {
            b'0'..=b'9' => u64::from(c - b'0'),
            b'a'..=b'f' => u64::from(c - b'a') + 10,
            b'A'..=b'F' => u64::from(c - b'A') + 10,
            _ => return None,
        };
        h = (h << 4) | d;
    }
    Some(h)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pattern the vectors below were taken over. Printable so the digests
    /// could be read out of a real server with `redis-cli` and nothing in the
    /// path could mangle a byte.
    fn pattern(len: usize) -> Vec<u8> {
        (0..len).map(|i| b'a' + ((i * 7 + 3) % 26) as u8).collect()
    }

    /// Digests of `pattern(len)` taken from Redis 8.8.0 with `DIGEST`, one from
    /// each length class and both sides of every boundary the implementation
    /// branches on.
    const VECTORS: &[(usize, u64)] = &[
        (0, 0x2d06800538d394c2),
        (1, 0x45f80274c9c7a7ca),
        (2, 0x0d220ff6d21aee49),
        (3, 0x3679b9ef946fc67d),
        (4, 0x9bb3172edc5e431f),
        (8, 0x3d346db850dc7593),
        (9, 0x38720337ee6918e1),
        (16, 0xe4424a9587784314),
        (17, 0xeb519a6261a45ed8),
        (32, 0x08e21e594ed849fe),
        (33, 0xf5e885748922e83c),
        (64, 0x3c3785a8778323e1),
        (65, 0x68b30f4d04ce84e7),
        (96, 0xf3a093621d11da29),
        (97, 0xddefa75f5368e693),
        (128, 0xf7fb4f534d8b6943),
        (129, 0x681f780fc27cd0f1),
        (144, 0x744ed62306351fa1),
        (240, 0x4208dfe93651a8d1),
        (241, 0x2d280426727fcbe2),
        (1024, 0xa73a2f96a9281a8f),
        (1025, 0xccf7ae021a00f73b),
        (2048, 0x48788f65cefc872b),
        (10000, 0x4af46bb79f3f8572),
    ];

    #[test]
    fn every_length_class_matches_a_real_redis() {
        for &(len, want) in VECTORS {
            let got = hash64(&pattern(len));
            assert_eq!(got, want, "length {len}");
        }
    }

    #[test]
    fn the_empty_string_has_the_documented_hash() {
        assert_eq!(hash64(b""), 0x2d06800538d394c2);
    }

    #[test]
    fn hello_has_the_documented_hash() {
        assert_eq!(hash64(b"hello"), 0x9555e8555c62dcfd);
        assert_eq!(&hex(hash64(b"hello")), b"9555e8555c62dcfd");
    }

    #[test]
    fn hex_and_back_round_trip() {
        for h in [0, 1, u64::MAX, 0x9555e8555c62dcfd, 0x0123456789abcdef] {
            assert_eq!(from_hex(&hex(h)), Some(h));
        }
        assert_eq!(from_hex(b"short"), None);
        assert_eq!(from_hex(b"0123456789abcdeg"), None);
        assert_eq!(from_hex(b"0123456789abcdef0"), None);
    }
}
