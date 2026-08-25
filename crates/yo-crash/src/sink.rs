//! A sink that can be killed, and that knows what the device would have kept.
//!
//! [`MemorySink`](yo_record::sink::MemorySink) applies a write the moment it
//! arrives, which makes it a fine store and a useless crash victim: by the time
//! anything goes wrong there is no record of which bytes had reached the device
//! and which were still in flight, and that difference is the entire question.
//!
//! So this one keeps the two apart. Writes go into a pending list and into a
//! working image the writer can read back. A sync moves the pending list into
//! the durable image and empties it. A crash starts from the durable image and
//! puts back whichever pending writes the fault model says survived.
//!
//! The rule the whole harness rests on: **anything a sync covered is in the
//! durable image and a crash fault never touches it.** A device that loses
//! acknowledged data is a broken device, not a case the engine has to survive,
//! and mixing the two would turn every real bug into noise. Losing durable
//! bytes is modelled too, as [`Fault::RotBit`](crate::fault::Fault::RotBit),
//! but it is media rot and it is judged against a different rule.

use std::collections::HashMap;

use yo_common::Result;
use yo_record::sink::{PageSink, PageSource, PageWrite};

/// One byte range that was handed over and has not been synced.
#[derive(Debug, Clone)]
pub struct Pending {
    /// Which page it belongs to.
    pub page_addr: u64,
    /// Where in the physical page, header included.
    pub offset: usize,
    /// The bytes.
    pub bytes: Vec<u8>,
}

/// A set of pages, which is what a device holds and what a replay walks.
#[derive(Debug, Clone, Default)]
pub struct Image {
    pages: HashMap<u64, Vec<u8>>,
}

impl Image {
    /// An empty device.
    #[must_use]
    pub fn new() -> Image {
        Image::default()
    }

    /// How many pages have anything in them.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    /// Whether the device is blank.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Every page, lowest address first, which is the order a replay wants.
    #[must_use]
    pub fn pages(&self) -> Vec<(u64, &[u8])> {
        let mut v: Vec<(u64, &[u8])> = self.pages.iter().map(|(a, b)| (*a, b.as_slice())).collect();
        v.sort_unstable_by_key(|(a, _)| *a);
        v
    }

    /// One page, to be corrupted.
    pub fn page_mut(&mut self, page_addr: u64) -> Option<&mut Vec<u8>> {
        self.pages.get_mut(&page_addr)
    }

    /// Lands a write, growing the page and zero filling any gap in front of it.
    ///
    /// The zero fill matters and is not tidiness. A device that has never been
    /// written reads as zeroes, and a zero length is the end of log sentinel, so
    /// filling a gap with zeroes is what the real thing does and is what makes
    /// replay stop in the right place.
    pub fn apply(&mut self, page_addr: u64, offset: usize, bytes: &[u8]) {
        let end = offset + bytes.len();
        let page = self.pages.entry(page_addr).or_default();
        if page.len() < end {
            page.resize(end, 0);
        }
        page[offset..end].copy_from_slice(bytes);
    }
}

impl PageSource for Image {
    fn page_bytes(&self, page_addr: u64) -> Option<&[u8]> {
        self.pages.get(&page_addr).map(Vec::as_slice)
    }
}

/// A store that remembers what it would have kept.
#[derive(Debug, Default)]
pub struct CrashSink {
    /// Everything a sync has covered. A crash cannot take any of this.
    durable: Image,
    /// What the writer would read back, pending writes included.
    live: Image,
    /// Handed over since the last sync, in the order it was handed over.
    pending: Vec<Pending>,
    written_upto: u64,
    durable_upto: u64,
    writes: u64,
    syncs: u64,
}

impl CrashSink {
    /// An empty store.
    #[must_use]
    pub fn new() -> CrashSink {
        CrashSink::default()
    }

    /// How many byte ranges were handed over in total.
    #[must_use]
    pub const fn writes(&self) -> u64 {
        self.writes
    }

    /// How many syncs were asked for.
    #[must_use]
    pub const fn syncs(&self) -> u64 {
        self.syncs
    }

    /// Writes handed over that no sync has covered, oldest first.
    ///
    /// These are the ones a crash gets to play with, and the count is how many
    /// choices the fault model has.
    #[must_use]
    pub fn pending(&self) -> &[Pending] {
        &self.pending
    }

    /// The image as it stands on the device right now.
    #[must_use]
    pub const fn durable(&self) -> &Image {
        &self.durable
    }

    /// The image the pending writes land on when a crash lets them through.
    #[must_use]
    pub fn crash_base(&self) -> Image {
        self.durable.clone()
    }

    /// Everything handed over, synced or not.
    ///
    /// What the device would hold if the run had ended and every pending write
    /// had made it. This is the baseline media rot is judged against: rot can
    /// only eat bytes that reached the device, so records still sitting in the
    /// log's own pages were never its to lose.
    #[must_use]
    pub fn live_image(&self) -> Image {
        self.live.clone()
    }
}

impl PageSink for CrashSink {
    fn write(&mut self, w: PageWrite<'_>) -> Result<()> {
        self.writes += 1;
        self.written_upto = self.written_upto.max(w.covers_upto);
        self.live.apply(w.page_addr, w.offset, w.bytes);
        self.pending.push(Pending {
            page_addr: w.page_addr,
            offset: w.offset,
            bytes: w.bytes.to_vec(),
        });
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        self.syncs += 1;
        for p in self.pending.drain(..) {
            self.durable.apply(p.page_addr, p.offset, &p.bytes);
        }
        self.durable_upto = self.written_upto;
        Ok(())
    }

    fn durable_upto(&self) -> u64 {
        self.durable_upto
    }
}

impl PageSource for CrashSink {
    fn page_bytes(&self, page_addr: u64) -> Option<&[u8]> {
        // The live image, not the durable one. A read during the run has to see
        // what was written, or the log reads back its own tail page and finds a
        // hole where the records it just appended should be.
        self.live.page_bytes(page_addr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(page_addr: u64, offset: usize, bytes: &[u8], covers_upto: u64) -> PageWrite<'_> {
        PageWrite {
            page_addr,
            offset,
            bytes,
            covers_upto,
        }
    }

    #[test]
    fn a_write_is_readable_and_not_yet_durable() {
        let mut s = CrashSink::new();
        s.write(w(0, 0, b"hello", 5)).unwrap();
        assert_eq!(s.page_bytes(0).unwrap(), b"hello");
        assert!(s.durable().is_empty(), "nothing has reached the device");
        assert_eq!(s.durable_upto(), 0);
        assert_eq!(s.pending().len(), 1);
    }

    #[test]
    fn a_sync_moves_everything_pending_onto_the_device() {
        let mut s = CrashSink::new();
        s.write(w(0, 0, b"hello", 5)).unwrap();
        s.write(w(0, 5, b" there", 11)).unwrap();
        s.sync().unwrap();
        assert_eq!(s.durable().page_bytes(0).unwrap(), b"hello there");
        assert_eq!(s.durable_upto(), 11);
        assert!(s.pending().is_empty());
        assert_eq!(s.syncs(), 1);
    }

    #[test]
    fn writes_after_a_sync_are_pending_again_and_the_durable_image_does_not_move() {
        let mut s = CrashSink::new();
        s.write(w(0, 0, b"aaaa", 4)).unwrap();
        s.sync().unwrap();
        s.write(w(0, 0, b"bbbb", 8)).unwrap();
        assert_eq!(
            s.page_bytes(0).unwrap(),
            b"bbbb",
            "the writer sees its write"
        );
        assert_eq!(
            s.durable().page_bytes(0).unwrap(),
            b"aaaa",
            "the device does not"
        );
        assert_eq!(s.durable_upto(), 4);
    }

    #[test]
    fn a_gap_in_a_page_reads_as_zeroes() {
        let mut i = Image::new();
        i.apply(0, 8, b"xy");
        assert_eq!(i.page_bytes(0).unwrap(), b"\0\0\0\0\0\0\0\0xy");
    }

    #[test]
    fn pages_come_back_in_address_order() {
        let mut i = Image::new();
        i.apply(300, 0, b"c");
        i.apply(100, 0, b"a");
        i.apply(200, 0, b"b");
        let addrs: Vec<u64> = i.pages().iter().map(|(a, _)| *a).collect();
        assert_eq!(addrs, vec![100, 200, 300]);
    }

    #[test]
    fn the_crash_base_is_the_durable_image_and_nothing_else() {
        let mut s = CrashSink::new();
        s.write(w(0, 0, b"kept", 4)).unwrap();
        s.sync().unwrap();
        s.write(w(0, 4, b"lost", 8)).unwrap();
        let base = s.crash_base();
        assert_eq!(base.page_bytes(0).unwrap(), b"kept");
    }
}
