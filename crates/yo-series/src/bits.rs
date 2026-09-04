//! A bit at a time, into a byte vector and back out of it.
//!
//! Both compressed encodings in this crate write fields that are not a whole
//! number of bytes, a one bit tag in front of a nine bit number and so on, so
//! everything they write goes through here. The writer appends and the reader
//! walks, and neither of them can be asked to do anything else.
//!
//! Bits go in from the top of a byte down, which is the order the field tags
//! are read in and the order a hexadecimal dump of a chunk reads in.

/// Bits appended to a byte vector, most significant first.
#[derive(Debug, Default)]
pub struct Writer {
    /// What has been filled so far. The last byte is only partly written when
    /// [`Writer::used`] is not a multiple of eight.
    bytes: Vec<u8>,
    /// How many bits of `bytes` are real.
    used: usize,
}

impl Writer {
    /// A writer that carries on from a buffer this crate wrote earlier.
    ///
    /// The bit count has to be the one that came back from [`Writer::finish`],
    /// because the last byte of a stream is padded and there is nothing in the
    /// bytes themselves that says where the padding starts.
    #[must_use]
    pub fn resume(bytes: Vec<u8>, used: usize) -> Self {
        Self { bytes, used }
    }

    /// Appends the low `count` bits of `value`, most significant of those first.
    pub fn put(&mut self, value: u64, count: u32) {
        debug_assert!(count <= 64, "a u64 has sixty four bits to give");
        let mut left = count;
        while left > 0 {
            let free = 8 - (self.used % 8) as u32;
            if free == 8 {
                self.bytes.push(0);
            }
            let take = free.min(left);
            // The `take` bits of `value` that come next, slid down to the
            // bottom and then up to where the byte still has room.
            let chunk = (value >> (left - take)) & ((1u64 << take) - 1);
            let at = self.bytes.len() - 1;
            self.bytes[at] |= (chunk as u8) << (free - take);
            self.used += take as usize;
            left -= take;
        }
    }

    /// Appends one bit.
    pub fn put_bit(&mut self, bit: bool) {
        self.put(u64::from(bit), 1);
    }

    /// The bytes and the bit count, which is what [`Writer::resume`] wants back.
    #[must_use]
    pub fn finish(self) -> (Vec<u8>, usize) {
        (self.bytes, self.used)
    }
}

/// Bits read back out of what a [`Writer`] wrote.
#[derive(Debug)]
pub struct Reader<'a> {
    /// The stream, padding and all.
    bytes: &'a [u8],
    /// How far in the next bit is.
    at: usize,
    /// Where the real bits stop and the padding starts.
    end: usize,
}

impl<'a> Reader<'a> {
    /// A reader over `used` bits of `bytes`.
    #[must_use]
    pub fn new(bytes: &'a [u8], used: usize) -> Self {
        Self {
            bytes,
            at: 0,
            end: used.min(bytes.len() * 8),
        }
    }

    /// Whether every bit has been read.
    #[must_use]
    pub fn done(&self) -> bool {
        self.at >= self.end
    }

    /// The next `count` bits, or `None` if the stream ends first.
    ///
    /// A stream that ends early is a bug in this crate rather than something a
    /// client can cause, but a decoder that walks off the end of a chunk would
    /// be a far worse bug, so the check is here rather than in an assertion.
    pub fn take(&mut self, count: u32) -> Option<u64> {
        if self.at + count as usize > self.end {
            return None;
        }
        let mut left = count;
        let mut value = 0u64;
        while left > 0 {
            let free = 8 - (self.at % 8) as u32;
            let take = free.min(left);
            let byte = u64::from(self.bytes[self.at / 8]);
            let chunk = (byte >> (free - take)) & ((1u64 << take) - 1);
            value = (value << take) | chunk;
            self.at += take as usize;
            left -= take;
        }
        Some(value)
    }

    /// The next bit.
    pub fn take_bit(&mut self) -> Option<bool> {
        self.take(1).map(|bit| bit == 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn what_goes_in_comes_back_out() {
        let fields: &[(u64, u32)] = &[
            (1, 1),
            (0, 1),
            (0b101, 3),
            (0, 7),
            (0x1234_5678_9abc_def0, 64),
            (511, 9),
            (7, 12),
        ];
        let mut w = Writer::default();
        for &(value, count) in fields {
            w.put(value, count);
        }
        let (bytes, used) = w.finish();
        assert_eq!(used, fields.iter().map(|&(_, n)| n as usize).sum::<usize>());
        let mut r = Reader::new(&bytes, used);
        for &(value, count) in fields {
            assert_eq!(r.take(count), Some(value), "reading {count} bits");
        }
        assert!(r.done());
        assert_eq!(r.take(1), None);
    }

    #[test]
    fn a_writer_carries_on_where_it_left_off() {
        let mut w = Writer::default();
        w.put(0b110, 3);
        let (bytes, used) = w.finish();
        let mut w = Writer::resume(bytes, used);
        w.put(0b1010, 4);
        let (bytes, used) = w.finish();
        assert_eq!(used, 7);
        let mut r = Reader::new(&bytes, used);
        assert_eq!(r.take(3), Some(0b110));
        assert_eq!(r.take(4), Some(0b1010));
        assert!(r.done());
    }

    #[test]
    fn the_padding_is_never_read_as_a_field() {
        let mut w = Writer::default();
        w.put_bit(true);
        let (bytes, used) = w.finish();
        assert_eq!(bytes.len(), 1);
        let mut r = Reader::new(&bytes, used);
        assert_eq!(r.take_bit(), Some(true));
        assert_eq!(r.take_bit(), None);
    }
}
