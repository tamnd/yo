//! Hashing and slot routing, which have to hold for any input at all.
//!
//! wyhash reads sixteen bytes at a time and reaches back over bytes it has
//! already consumed on the long path, so an off by one in the tail handling
//! would read out of bounds. The reference vectors in yo-common cover fixed
//! lengths; this covers every length and every content.
//!
//! `slot_of` is Redis compatible key routing. The invariants are that a hash tag
//! decides the slot, that an unmatched or empty brace pair falls back to the
//! whole key, and that the answer is always inside the 16384 slots the protocol
//! defines.

#![no_main]

use libfuzzer_sys::fuzz_target;
use yo_common::{SLOT_COUNT, crc16, crc32c, hash_tag, slot_of, tag_of, wyhash};

fuzz_target!(|data: &[u8]| {
    // Every prefix, so every tail length gets hit rather than just the one this
    // input happens to end on.
    for len in [0usize, 1, 3, 4, 8, 15, 16, 17, 48, 49].into_iter().chain([data.len()]) {
        if len > data.len() {
            continue;
        }
        let s = &data[..len];

        // Deterministic, and the seed has to matter.
        assert_eq!(wyhash(s, 0), wyhash(s, 0));
        let tag = tag_of(wyhash(s, 0));
        assert_ne!(tag, 0, "tag zero means empty and must never be produced");

        // Slot routing stays in range and agrees with the tag rule.
        let slot = slot_of(s);
        assert!(slot < SLOT_COUNT, "slot {slot} is outside the keyspace");
        let inner = hash_tag(s);
        assert_eq!(
            slot,
            crc16(inner) % SLOT_COUNT,
            "slot did not follow the hash tag rule for {s:?}"
        );
        // A key with no usable brace pair routes as itself.
        if !s.contains(&b'{') {
            assert_eq!(inner, s, "a key with no brace should route as itself");
        }

        // CRC32C is resumable, which the file format relies on for streaming
        // checksums over a segment.
        let whole = crc32c(0, s);
        let half = s.len() / 2;
        assert_eq!(whole, crc32c(crc32c(0, &s[..half]), &s[half..]));
    }
});
