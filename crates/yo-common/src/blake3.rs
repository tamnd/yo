//! BLAKE3, the hash the shape tag is made of (`15` section 3.1).
//!
//! Written here rather than depended on, for the same reason `crc`, `wyhash`
//! and `xxh3` are written here: the shape tag is a number six language bindings
//! have to agree on byte for byte forever, so the algorithm is part of the
//! format and belongs next to the other two hashes the format already pins.
//! The published crate also compiles assembly through a C toolchain by default,
//! which is a build dependency the whole workspace would inherit for a hash
//! that runs once when a collection is opened.
//!
//! This is the portable implementation, without the SIMD kernels. A shape
//! description is a few hundred bytes and it is hashed at open time, never on
//! a hot path, so a kernel that is four times faster would save nothing that
//! could be measured. If a caller ever needs BLAKE3 on a hot path, that is the
//! moment to add the kernels, and the test vectors here already cover them.
//!
//! Only the plain hash is implemented. Keyed hashing, key derivation and the
//! extended output are the parts of BLAKE3 nothing in `yo` uses.
//!
//! ```
//! use yo_common::blake3;
//!
//! let h = blake3::hash(b"");
//! assert_eq!(
//!     blake3::to_hex(&h),
//!     "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262",
//! );
//! ```

const OUT_LEN: usize = 32;
const BLOCK_LEN: usize = 64;
const CHUNK_LEN: usize = 1024;

const CHUNK_START: u32 = 1 << 0;
const CHUNK_END: u32 = 1 << 1;
const PARENT: u32 = 1 << 2;
const ROOT: u32 = 1 << 3;

const IV: [u32; 8] = [
    0x6A09_E667,
    0xBB67_AE85,
    0x3C6E_F372,
    0xA54F_F53A,
    0x510E_527F,
    0x9B05_688C,
    0x1F83_D9AB,
    0x5BE0_CD19,
];

const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

/// The 32 byte hash of a whole input.
///
/// The streaming form is [`Hasher`]; this is the one call every caller in the
/// tree actually wants.
#[must_use]
pub fn hash(input: &[u8]) -> [u8; OUT_LEN] {
    let mut h = Hasher::new();
    h.update(input);
    h.finalize()
}

/// Lower case hex, which is the spelling every binding prints and every test
/// vector is written in.
#[must_use]
pub fn to_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(DIGITS[usize::from(b >> 4)] as char);
        s.push(DIGITS[usize::from(b & 0x0f)] as char);
    }
    s
}

/// The BLAKE3 mixing function on one word quadruple.
#[inline]
fn g(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(mx);
    state[d] = (state[d] ^ state[a]).rotate_right(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(12);
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(my);
    state[d] = (state[d] ^ state[a]).rotate_right(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(7);
}

#[inline]
fn round(state: &mut [u32; 16], m: &[u32; 16]) {
    // Columns.
    g(state, 0, 4, 8, 12, m[0], m[1]);
    g(state, 1, 5, 9, 13, m[2], m[3]);
    g(state, 2, 6, 10, 14, m[4], m[5]);
    g(state, 3, 7, 11, 15, m[6], m[7]);
    // Diagonals.
    g(state, 0, 5, 10, 15, m[8], m[9]);
    g(state, 1, 6, 11, 12, m[10], m[11]);
    g(state, 2, 7, 8, 13, m[12], m[13]);
    g(state, 3, 4, 9, 14, m[14], m[15]);
}

#[inline]
fn permute(m: &mut [u32; 16]) {
    let old = *m;
    for (i, &p) in MSG_PERMUTATION.iter().enumerate() {
        m[i] = old[p];
    }
}

/// The compression function, returning the whole 16 word state because the
/// root node needs the second half and a chaining value needs only the first.
fn compress(
    chaining_value: &[u32; 8],
    block_words: &[u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
) -> [u32; 16] {
    let counter_low = counter as u32;
    let counter_high = (counter >> 32) as u32;
    let mut state = [
        chaining_value[0],
        chaining_value[1],
        chaining_value[2],
        chaining_value[3],
        chaining_value[4],
        chaining_value[5],
        chaining_value[6],
        chaining_value[7],
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        counter_low,
        counter_high,
        block_len,
        flags,
    ];
    let mut block = *block_words;

    for _ in 0..6 {
        round(&mut state, &block);
        permute(&mut block);
    }
    round(&mut state, &block);

    for i in 0..8 {
        state[i] ^= state[i + 8];
        state[i + 8] ^= chaining_value[i];
    }
    state
}

fn first_8(state: [u32; 16]) -> [u32; 8] {
    let mut cv = [0u32; 8];
    cv.copy_from_slice(&state[..8]);
    cv
}

fn words_from_le(block: &[u8; BLOCK_LEN]) -> [u32; 16] {
    let mut words = [0u32; 16];
    let (quads, _) = block.as_chunks::<4>();
    for (word, quad) in words.iter_mut().zip(quads) {
        *word = u32::from_le_bytes(*quad);
    }
    words
}

/// A node that is ready to be either chained into its parent or, if it turns
/// out to be the root, finalized.
struct Output {
    input_chaining_value: [u32; 8],
    block_words: [u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
}

impl Output {
    fn chaining_value(&self) -> [u32; 8] {
        first_8(compress(
            &self.input_chaining_value,
            &self.block_words,
            self.counter,
            self.block_len,
            self.flags,
        ))
    }

    fn root_bytes(&self) -> [u8; OUT_LEN] {
        let state = compress(
            &self.input_chaining_value,
            &self.block_words,
            0,
            self.block_len,
            self.flags | ROOT,
        );
        let mut out = [0u8; OUT_LEN];
        let (quads, _) = out.as_chunks_mut::<4>();
        for (word, quad) in state[..8].iter().zip(quads) {
            *quad = word.to_le_bytes();
        }
        out
    }
}

/// One 1 KiB chunk being filled a block at a time.
struct ChunkState {
    chaining_value: [u32; 8],
    counter: u64,
    block: [u8; BLOCK_LEN],
    block_len: u8,
    blocks_compressed: u8,
}

impl ChunkState {
    fn new(counter: u64) -> ChunkState {
        ChunkState {
            chaining_value: IV,
            counter,
            block: [0; BLOCK_LEN],
            block_len: 0,
            blocks_compressed: 0,
        }
    }

    fn len(&self) -> usize {
        BLOCK_LEN * usize::from(self.blocks_compressed) + usize::from(self.block_len)
    }

    fn start_flag(&self) -> u32 {
        if self.blocks_compressed == 0 {
            CHUNK_START
        } else {
            0
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        while !input.is_empty() {
            if usize::from(self.block_len) == BLOCK_LEN {
                let words = words_from_le(&self.block);
                self.chaining_value = first_8(compress(
                    &self.chaining_value,
                    &words,
                    self.counter,
                    BLOCK_LEN as u32,
                    self.start_flag(),
                ));
                self.blocks_compressed += 1;
                self.block = [0; BLOCK_LEN];
                self.block_len = 0;
            }

            let want = BLOCK_LEN - usize::from(self.block_len);
            let take = want.min(input.len());
            let at = usize::from(self.block_len);
            self.block[at..at + take].copy_from_slice(&input[..take]);
            self.block_len += take as u8;
            input = &input[take..];
        }
    }

    fn output(&self) -> Output {
        Output {
            input_chaining_value: self.chaining_value,
            block_words: words_from_le(&self.block),
            counter: self.counter,
            block_len: u32::from(self.block_len),
            flags: self.start_flag() | CHUNK_END,
        }
    }
}

fn parent_output(left: [u32; 8], right: [u32; 8]) -> Output {
    let mut block_words = [0u32; 16];
    block_words[..8].copy_from_slice(&left);
    block_words[8..].copy_from_slice(&right);
    Output {
        input_chaining_value: IV,
        block_words,
        counter: 0,
        block_len: BLOCK_LEN as u32,
        flags: PARENT,
    }
}

/// The streaming hasher.
///
/// Feed it with [`update`](Hasher::update) as many times as suits the caller
/// and the answer does not depend on where the boundaries fell.
///
/// The stack is 54 chaining values because that is the deepest a tree can get:
/// one entry per bit of the chunk counter, and a chunk is a kibibyte.
pub struct Hasher {
    chunk: ChunkState,
    stack: [[u32; 8]; 54],
    stack_len: u8,
}

impl Default for Hasher {
    fn default() -> Hasher {
        Hasher::new()
    }
}

impl Hasher {
    /// An empty hasher.
    #[must_use]
    pub fn new() -> Hasher {
        Hasher {
            chunk: ChunkState::new(0),
            stack: [[0; 8]; 54],
            stack_len: 0,
        }
    }

    fn push(&mut self, cv: [u32; 8]) {
        self.stack[usize::from(self.stack_len)] = cv;
        self.stack_len += 1;
    }

    fn pop(&mut self) -> [u32; 8] {
        self.stack_len -= 1;
        self.stack[usize::from(self.stack_len)]
    }

    /// Merge a finished chunk in, collapsing every subtree the new chunk
    /// completes. The trailing zeros of the chunk count say how many merges
    /// that is, which is the whole trick that makes the tree implicit.
    fn add_chunk(&mut self, mut cv: [u32; 8], total_chunks: u64) {
        let mut chunks = total_chunks;
        while chunks & 1 == 0 {
            let left = self.pop();
            cv = parent_output(left, cv).chaining_value();
            chunks >>= 1;
        }
        self.push(cv);
    }

    /// Add more input.
    pub fn update(&mut self, mut input: &[u8]) {
        while !input.is_empty() {
            if self.chunk.len() == CHUNK_LEN {
                let cv = self.chunk.output().chaining_value();
                let total = self.chunk.counter + 1;
                self.add_chunk(cv, total);
                self.chunk = ChunkState::new(total);
            }

            let want = CHUNK_LEN - self.chunk.len();
            let take = want.min(input.len());
            self.chunk.update(&input[..take]);
            input = &input[take..];
        }
    }

    /// The 32 byte hash of everything fed in so far.
    #[must_use]
    pub fn finalize(&self) -> [u8; OUT_LEN] {
        let mut output = self.chunk.output();
        let mut remaining = usize::from(self.stack_len);
        while remaining > 0 {
            remaining -= 1;
            output = parent_output(self.stack[remaining], output.chaining_value());
        }
        output.root_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The input the official test vectors use: bytes 0, 1, 2 ... 250, then
    /// back to 0. Written out here so the vectors below are checked against
    /// the same thing every other implementation checks them against.
    fn pattern(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 251) as u8).collect()
    }

    /// Every case from the BLAKE3 team's `test_vectors.json`, unkeyed. The
    /// lengths are chosen to land on every boundary that matters: empty, part
    /// of a block, a whole block, part of a chunk, a whole chunk, and the
    /// chunk counts that force one, two and many levels of parent nodes.
    const VECTORS: &[(usize, &str)] = &[
        (
            0,
            "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262",
        ),
        (
            1,
            "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213",
        ),
        (
            2,
            "7b7015bb92cf0b318037702a6cdd81dee41224f734684c2c122cd6359cb1ee63",
        ),
        (
            3,
            "e1be4d7a8ab5560aa4199eea339849ba8e293d55ca0a81006726d184519e647f",
        ),
        (
            4,
            "f30f5ab28fe047904037f77b6da4fea1e27241c5d132638d8bedce9d40494f32",
        ),
        (
            5,
            "b40b44dfd97e7a84a996a91af8b85188c66c126940ba7aad2e7ae6b385402aa2",
        ),
        (
            6,
            "06c4e8ffb6872fad96f9aaca5eee1553eb62aed0ad7198cef42e87f6a616c844",
        ),
        (
            7,
            "3f8770f387faad08faa9d8414e9f449ac68e6ff0417f673f602a646a891419fe",
        ),
        (
            8,
            "2351207d04fc16ade43ccab08600939c7c1fa70a5c0aaca76063d04c3228eaeb",
        ),
        (
            63,
            "e9bc37a594daad83be9470df7f7b3798297c3d834ce80ba85d6e207627b7db7b",
        ),
        (
            64,
            "4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98",
        ),
        (
            65,
            "de1e5fa0be70df6d2be8fffd0e99ceaa8eb6e8c93a63f2d8d1c30ecb6b263dee",
        ),
        (
            127,
            "d81293fda863f008c09e92fc382a81f5a0b4a1251cba1634016a0f86a6bd640d",
        ),
        (
            128,
            "f17e570564b26578c33bb7f44643f539624b05df1a76c81f30acd548c44b45ef",
        ),
        (
            129,
            "683aaae9f3c5ba37eaaf072aed0f9e30bac0865137bae68b1fde4ca2aebdcb12",
        ),
        (
            1023,
            "10108970eeda3eb932baac1428c7a2163b0e924c9a9e25b35bba72b28f70bd11",
        ),
        (
            1024,
            "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7",
        ),
        (
            1025,
            "d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444",
        ),
        (
            2048,
            "e776b6028c7cd22a4d0ba182a8bf62205d2ef576467e838ed6f2529b85fba24a",
        ),
        (
            2049,
            "5f4d72f40d7a5f82b15ca2b2e44b1de3c2ef86c426c95c1af0b6879522563030",
        ),
        (
            3072,
            "b98cb0ff3623be03326b373de6b9095218513e64f1ee2edd2525c7ad1e5cffd2",
        ),
        (
            3073,
            "7124b49501012f81cc7f11ca069ec9226cecb8a2c850cfe644e327d22d3e1cd3",
        ),
        (
            4096,
            "015094013f57a5277b59d8475c0501042c0b642e531b0a1c8f58d2163229e969",
        ),
        (
            4097,
            "9b4052b38f1c5fc8b1f9ff7ac7b27cd242487b3d890d15c96a1c25b8aa0fb995",
        ),
        (
            5120,
            "9cadc15fed8b5d854562b26a9536d9707cadeda9b143978f319ab34230535833",
        ),
        (
            5121,
            "628bd2cb2004694adaab7bbd778a25df25c47b9d4155a55f8fbd79f2fe154cff",
        ),
        (
            6144,
            "3e2e5b74e048f3add6d21faab3f83aa44d3b2278afb83b80b3c35164ebeca205",
        ),
        (
            6145,
            "f1323a8631446cc50536a9f705ee5cb619424d46887f3c376c695b70e0f0507f",
        ),
        (
            7168,
            "61da957ec2499a95d6b8023e2b0e604ec7f6b50e80a9678b89d2628e99ada77a",
        ),
        (
            7169,
            "a003fc7a51754a9b3c7fae0367ab3d782dccf28855a03d435f8cfe74605e7817",
        ),
        (
            8192,
            "aae792484c8efe4f19e2ca7d371d8c467ffb10748d8a5a1ae579948f718a2a63",
        ),
        (
            8193,
            "bab6c09cb8ce8cf459261398d2e7aef35700bf488116ceb94a36d0f5f1b7bc3b",
        ),
        (
            16_384,
            "f875d6646de28985646f34ee13be9a576fd515f76b5b0a26bb324735041ddde4",
        ),
        (
            31_744,
            "62b6960e1a44bcc1eb1a611a8d6235b6b4b78f32e7abc4fb4c6cdcce94895c47",
        ),
        (
            102_400,
            "bc3e3d41a1146b069abffad3c0d44860cf664390afce4d9661f7902e7943e085",
        ),
    ];

    #[test]
    fn the_official_vectors_pass() {
        for &(len, want) in VECTORS {
            let got = to_hex(&hash(&pattern(len)));
            assert_eq!(got, want, "length {len}");
        }
    }

    /// Where the caller cut its input cannot change the answer, which is the
    /// property the shape writer relies on when it emits a description field
    /// by field instead of in one buffer.
    #[test]
    fn the_split_does_not_matter() {
        let input = pattern(4097);
        let want = hash(&input);
        for step in [1usize, 7, 63, 64, 65, 1023, 1024, 1025, 1500] {
            let mut h = Hasher::new();
            for piece in input.chunks(step) {
                h.update(piece);
            }
            assert_eq!(h.finalize(), want, "step {step}");
        }
    }

    /// An empty update is not an event.
    #[test]
    fn empty_updates_are_free() {
        let mut h = Hasher::new();
        h.update(b"");
        h.update(b"yo");
        h.update(b"");
        assert_eq!(h.finalize(), hash(b"yo"));
    }

    #[test]
    fn hex_is_lower_case_and_padded() {
        assert_eq!(to_hex(&[0x00, 0x0f, 0xa0, 0xff]), "000fa0ff");
        assert_eq!(to_hex(&[]), "");
    }
}
