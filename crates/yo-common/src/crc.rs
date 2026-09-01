//! CRC16 for slot placement, CRC32C for integrity, CRC64 for the file format.
//!
//! Three different polynomials for three different jobs. CRC16 is Redis's XMODEM
//! variant and it exists here because `slot = crc16(key) & 0x3FFF` is how a key
//! reaches a shard (`04` section 1). Getting it wrong does not corrupt anything,
//! it just makes us incompatible with every Redis cluster client, so the hash
//! tag rules are implemented here too.
//!
//! CRC32C is Castagnoli, the same polynomial SSE4.2 and the ARM CRC extension
//! implement in hardware, and it is what guards pages and superblocks (`07`).
//!
//! CRC64 is the Jones polynomial and it is here for one reason only: it is the
//! eight bytes on the end of an RDB payload, so `DUMP` cannot produce something
//! a real Redis will accept and `RESTORE` cannot reject a corrupt payload
//! without it. Nothing else in the engine uses it, and nothing else should,
//! because CRC32C has hardware behind it and this does not.

// ---------------------------------------------------------------------------
// CRC16 / XMODEM, polynomial 0x1021, initial value 0, not reflected.
// ---------------------------------------------------------------------------

const fn crc16_table() -> [u16; 256] {
    let mut table = [0u16; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = (i as u16) << 8;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 0x8000 != 0 {
                (crc << 1) ^ 0x1021
            } else {
                crc << 1
            };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

static CRC16_TABLE: [u16; 256] = crc16_table();

/// Redis's CRC16, the XMODEM variant.
#[inline]
pub fn crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0;
    for &b in data {
        let idx = (((crc >> 8) ^ b as u16) & 0xff) as usize;
        crc = (crc << 8) ^ CRC16_TABLE[idx];
    }
    crc
}

/// The number of slots, which is Redis's 16384 and is not configurable.
pub const SLOT_COUNT: u16 = 16384;

/// The cluster slot a key belongs to, hash tags included.
///
/// If the key contains `{` followed by a non empty run and then `}`, only the
/// run between them is hashed. That rule is what lets a caller force two keys
/// onto one shard, and multi key commands depend on it, so it belongs next to
/// the CRC rather than in the command layer.
#[inline]
pub fn slot_of(key: &[u8]) -> u16 {
    crc16(hash_tag(key)) & (SLOT_COUNT - 1)
}

/// The part of a key that decides its slot.
///
/// Returns the whole key unless there is a `{...}` with something inside it.
#[inline]
pub fn hash_tag(key: &[u8]) -> &[u8] {
    let Some(open) = key.iter().position(|&b| b == b'{') else {
        return key;
    };
    let rest = &key[open + 1..];
    let Some(close) = rest.iter().position(|&b| b == b'}') else {
        return key;
    };
    if close == 0 {
        // `{}` is empty, so the whole key is used. This matches Redis.
        return key;
    }
    &rest[..close]
}

// ---------------------------------------------------------------------------
// CRC32C / Castagnoli, polynomial 0x1EDC6F41, reflected as 0x82F63B78.
// ---------------------------------------------------------------------------

const fn crc32c_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0x82F6_3B78
            } else {
                crc >> 1
            };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

static CRC32C_TABLE: [u32; 256] = crc32c_table();

#[inline]
fn crc32c_software(mut crc: u32, data: &[u8]) -> u32 {
    crc = !crc;
    for &b in data {
        crc = (crc >> 8) ^ CRC32C_TABLE[((crc ^ b as u32) & 0xff) as usize];
    }
    !crc
}

/// CRC32C over `data`, continuing from `crc`. Start with 0.
#[inline]
pub fn crc32c(crc: u32, data: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.2") {
            // SAFETY: guarded by the runtime feature check immediately above.
            return unsafe { crc32c_sse42(crc, data) };
        }
    }
    crc32c_software(crc, data)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn crc32c_sse42(crc: u32, data: &[u8]) -> u32 {
    use core::arch::x86_64::{_mm_crc32_u8, _mm_crc32_u64};

    let mut c = !crc;
    // `as_chunks` rather than `chunks_exact`, because the chunk size is a
    // constant and this way the length is one too. The compiler stops emitting
    // the bounds check that the eight byte load does not need.
    let (words, rest) = data.as_chunks::<8>();
    for chunk in words {
        // No unsafe block. These intrinsics are safe to call from a function
        // that carries the matching `#[target_feature]`, and wrapping them
        // anyway is an unused_unsafe warning on x86, which CI treats as an
        // error. The unsafety is at the call site in `crc32c`, where the
        // runtime feature check lives.
        c = _mm_crc32_u64(c as u64, u64::from_le_bytes(*chunk)) as u32;
    }
    for &b in rest {
        c = _mm_crc32_u8(c, b);
    }
    !c
}

// ---------------------------------------------------------------------------
// CRC64 / Jones, polynomial 0xad93d23594c935a9, reflected as 0x95ac9329ac4bc9b5.
// ---------------------------------------------------------------------------

const fn crc64_table() -> [u64; 256] {
    let mut table = [0u64; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut crc = i as u64;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0x95AC_9329_AC4B_C9B5
            } else {
                crc >> 1
            };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

static CRC64_TABLE: [u64; 256] = crc64_table();

/// CRC64 over `data`, continuing from `crc`. Start with 0.
///
/// The variant Redis puts on the end of an RDB payload. Published descriptions
/// of crc-64-jones give it an initial value of all ones, and Redis's own source
/// comment says so too, but the function Redis actually calls starts from the
/// value handed in and that value is zero. Copy the code, not the comment, or
/// every payload we produce fails somebody else's checksum.
#[inline]
pub fn crc64(crc: u64, data: &[u8]) -> u64 {
    let mut c = crc;
    for &b in data {
        c = (c >> 8) ^ CRC64_TABLE[((c ^ b as u64) & 0xff) as usize];
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The slot values Redis documents for its own examples. If these move, we
    /// are no longer wire compatible with cluster clients.
    #[test]
    fn redis_slot_examples() {
        assert_eq!(crc16(b"123456789"), 0x31C3);
        assert_eq!(slot_of(b"foo"), 12182);
        assert_eq!(slot_of(b"bar"), 5061);
    }

    #[test]
    fn hash_tags_pick_the_inner_run() {
        assert_eq!(hash_tag(b"{user1000}.following"), b"user1000");
        assert_eq!(hash_tag(b"foo{}{bar}"), b"foo{}{bar}");
        assert_eq!(hash_tag(b"foo{{bar}}zap"), b"{bar");
        assert_eq!(hash_tag(b"foo{bar}{zap}"), b"bar");
        assert_eq!(hash_tag(b"nothing"), b"nothing");
    }

    #[test]
    fn tagged_keys_land_on_one_slot() {
        assert_eq!(
            slot_of(b"{user1000}.following"),
            slot_of(b"{user1000}.followers")
        );
    }

    #[test]
    fn crc32c_reference() {
        // The check value every CRC32C implementation publishes.
        assert_eq!(crc32c(0, b"123456789"), 0xE306_9283);
        assert_eq!(crc32c(0, b""), 0);
    }

    /// The hardware and software paths must not disagree, because one machine
    /// writing a page and another verifying it is the normal case.
    #[test]
    fn crc32c_hardware_matches_software() {
        let buf: Vec<u8> = (0..1000u32).map(|i| (i * 7 % 251) as u8).collect();
        for n in [0usize, 1, 7, 8, 9, 15, 16, 63, 64, 65, 999, 1000] {
            assert_eq!(
                crc32c(0, &buf[..n]),
                crc32c_software(0, &buf[..n]),
                "length {n} disagrees between the two paths"
            );
        }
    }

    #[test]
    fn crc32c_is_resumable() {
        let buf: Vec<u8> = (0..256u32).map(|i| i as u8).collect();
        let one_shot = crc32c(0, &buf);
        let split = crc32c(crc32c(0, &buf[..100]), &buf[100..]);
        assert_eq!(one_shot, split);
    }

    /// The value Redis prints from its own self test in `crc64.c`. This is the
    /// whole reason the function is here, so if it moves nothing we produce is
    /// worth sending anywhere.
    #[test]
    fn crc64_matches_the_redis_self_test() {
        assert_eq!(crc64(0, b"123456789"), 0xe9c6_d914_c4b8_d9ca);
        assert_eq!(crc64(0, b""), 0);
    }

    #[test]
    fn crc64_is_resumable() {
        let buf: Vec<u8> = (0..256u32).map(|i| i as u8).collect();
        assert_eq!(crc64(0, &buf), crc64(crc64(0, &buf[..100]), &buf[100..]));
    }
}
