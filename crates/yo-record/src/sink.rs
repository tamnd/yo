//! Where a page goes when it leaves memory.
//!
//! The log does not open files. It fills pages and hands them to a sink, and
//! the sink is what knows about descriptors, io_uring and `fsync`. Two reasons
//! for the split, and the second is the one that pays.
//!
//! The obvious reason is testing: a log with a memory sink runs in a
//! microsecond and a log with a real file runs in milliseconds, and every
//! ordering rule in `06` is testable without a disk.
//!
//! The reason that pays is `07` section 7. In memory mode is the same format
//! over anonymous mappings, which means it is this crate with a different sink,
//! not a second implementation of everything with an `if in_memory` in the
//! middle of it.

use yo_common::Result;

/// One write, which is a byte range inside one page.
///
/// A struct rather than four arguments because the last field is the one that
/// gets forgotten, and forgetting it means a sink that reports commits durable
/// before they are.
#[derive(Debug, Clone, Copy)]
pub struct PageWrite<'a> {
    /// The log address of the first payload byte of this page, which is what
    /// the page header carries and how a sink that reorders writes puts them
    /// back in the right place.
    pub page_addr: u64,
    /// Where in the physical page these bytes go, header included, so an offset
    /// of 0 is the page header itself.
    pub offset: usize,
    /// The bytes.
    pub bytes: &'a [u8],
    /// The log address one past the last record byte this write covers.
    ///
    /// This is what a group commit waits on. It is separate from the byte range
    /// because a write is 4 KiB aligned and a commit is not, so the last aligned
    /// block usually contains bytes that no record has claimed yet.
    pub covers_upto: u64,
}

/// A page's worth of bytes, on its way out of memory.
///
/// Implementations may be asynchronous. [`Self::write`] hands bytes over and
/// returns; [`Self::durable_upto`] is how the log finds out what actually
/// landed, and the log never assumes anything the sink has not said out loud.
pub trait PageSink {
    /// Hands over a byte range.
    ///
    /// # Errors
    ///
    /// Whatever the underlying store returns. An error here does not lose the
    /// page: the log keeps it resident and the caller decides whether to retry
    /// or to stop accepting writes.
    fn write(&mut self, w: PageWrite<'_>) -> Result<()>;

    /// Asks for everything handed over so far to be made durable.
    ///
    /// This is the one `fsync` that a whole page of commits waits on (`06`
    /// section 3). It may return before the data is durable;
    /// [`Self::durable_upto`] is the answer to whether it is.
    ///
    /// # Errors
    ///
    /// Whatever the underlying store returns.
    fn sync(&mut self) -> Result<()>;

    /// The log address below which everything is durable.
    ///
    /// Monotonic. A sink that has synced nothing returns 0, and a sink with no
    /// durability at all returns `u64::MAX`, because for that sink every commit
    /// is as durable as it is ever going to be and parking a caller would be a
    /// wait that never ends.
    fn durable_upto(&self) -> u64;
}

/// Where a page comes back from, once it has left memory.
///
/// The read counterpart of [`PageSink`]. Two callers need it and they need it
/// for the same reason: the pages they want are in the stable region, which is
/// in the file and not in the log's resident window. Compaction reads the pages
/// it is rewriting, and recovery reads the page a partially filled tail is
/// sitting in so that it does not write zeroes over the records already there.
///
/// This is the seam. The file layer implements it against a mapping, and the
/// tests implement it against a `Vec`.
pub trait PageSource {
    /// The full physical bytes of the page whose payload starts at `page_addr`,
    /// header included, or `None` if this source does not have it.
    fn page_bytes(&self, page_addr: u64) -> Option<&[u8]>;
}

/// A sink that keeps pages in memory and calls that durable.
///
/// This is in memory mode (`07` section 7) and it is also what every test in
/// this crate writes to. Nothing here survives the process, which is exactly
/// what `:memory:` promises, so reporting a synced write as durable is honest
/// rather than a shortcut: there is no weaker guarantee available and no
/// stronger one to wait for.
#[derive(Debug, Default)]
pub struct MemorySink {
    pages: Vec<(u64, Vec<u8>)>,
    writes: u64,
    syncs: u64,
    written_upto: u64,
    durable_upto: u64,
}

impl MemorySink {
    /// An empty sink.
    #[must_use]
    pub fn new() -> MemorySink {
        MemorySink::default()
    }

    /// How many byte ranges were handed over.
    ///
    /// Tests assert on this, because "one flush per page" is a claim about a
    /// count and not a feeling.
    #[must_use]
    pub const fn writes(&self) -> u64 {
        self.writes
    }

    /// How many syncs were asked for. The group commit claim in `06` section 3
    /// is that this stays far below the commit count.
    #[must_use]
    pub const fn syncs(&self) -> u64 {
        self.syncs
    }

    /// The current image of the page whose payload starts at `page_addr`.
    #[must_use]
    pub fn page(&self, page_addr: u64) -> Option<&[u8]> {
        self.pages
            .iter()
            .find(|(a, _)| *a == page_addr)
            .map(|(_, b)| b.as_slice())
    }

    /// The page whose payload starts at `page_addr`, to be written on.
    ///
    /// This exists for fault injection. `06` section 6 asks for a hundred
    /// thousand injected faults with zero silent corruptions, and a harness
    /// that cannot reach into a page and flip a bit cannot inject anything. It
    /// is public rather than test only because the crash harness is its own
    /// binary and lives outside this crate.
    #[must_use]
    pub fn page_mut(&mut self, page_addr: u64) -> Option<&mut [u8]> {
        self.pages
            .iter_mut()
            .find(|(a, _)| *a == page_addr)
            .map(|(_, b)| b.as_mut_slice())
    }

    /// Every page held, ordered by address, which is what a replay walks.
    #[must_use]
    pub fn pages(&self) -> Vec<(u64, &[u8])> {
        let mut v: Vec<(u64, &[u8])> = self.pages.iter().map(|(a, b)| (*a, b.as_slice())).collect();
        v.sort_by_key(|(a, _)| *a);
        v
    }
}

impl PageSink for MemorySink {
    fn write(&mut self, w: PageWrite<'_>) -> Result<()> {
        self.writes += 1;
        self.written_upto = self.written_upto.max(w.covers_upto);
        let end = w.offset + w.bytes.len();
        let page = match self.pages.iter_mut().find(|(a, _)| *a == w.page_addr) {
            Some((_, b)) => b,
            None => {
                self.pages.push((w.page_addr, Vec::new()));
                &mut self.pages.last_mut().expect("just pushed").1
            }
        };
        if page.len() < end {
            page.resize(end, 0);
        }
        page[w.offset..end].copy_from_slice(w.bytes);
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        self.syncs += 1;
        // Everything handed over before this call is now durable, and nothing
        // handed over after it is. That is the whole contract, and it is the
        // same one a real `fsync` offers.
        self.durable_upto = self.written_upto;
        Ok(())
    }

    fn durable_upto(&self) -> u64 {
        self.durable_upto
    }
}

impl PageSource for MemorySink {
    fn page_bytes(&self, page_addr: u64) -> Option<&[u8]> {
        self.page(page_addr)
    }
}

/// A sink that drops everything and never claims durability.
///
/// Durability mode `none` with nowhere to write. Useful on its own for a cache
/// that is allowed to vanish, and useful in a benchmark that wants the append
/// path with no store underneath it, so that the number is the append and not
/// the disk.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullSink;

impl PageSink for NullSink {
    fn write(&mut self, _w: PageWrite<'_>) -> Result<()> {
        Ok(())
    }

    fn sync(&mut self) -> Result<()> {
        Ok(())
    }

    fn durable_upto(&self) -> u64 {
        // Everything, because nothing is ever going to become more durable than
        // it already is and a caller waiting on this address would wait forever.
        u64::MAX
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
    fn writes_land_at_their_offset_and_later_writes_overwrite() {
        let mut s = MemorySink::new();
        s.write(w(0, 0, b"aaaaaaaa", 8)).unwrap();
        s.write(w(0, 4, b"BBBB", 8)).unwrap();
        assert_eq!(s.page(0).unwrap(), b"aaaaBBBB");
        assert_eq!(s.writes(), 2);
    }

    #[test]
    fn a_write_past_the_end_grows_the_page_and_zero_fills_the_gap() {
        let mut s = MemorySink::new();
        s.write(w(0, 4, b"xy", 6)).unwrap();
        assert_eq!(s.page(0).unwrap(), b"\0\0\0\0xy");
    }

    #[test]
    fn pages_are_separate_and_come_back_in_address_order() {
        let mut s = MemorySink::new();
        s.write(w(4096, 0, b"second", 0)).unwrap();
        s.write(w(0, 0, b"first", 0)).unwrap();
        let pages = s.pages();
        assert_eq!(pages.len(), 2);
        assert_eq!(pages[0].0, 0);
        assert_eq!(pages[1].0, 4096);
    }

    #[test]
    fn a_sync_makes_durable_exactly_what_was_written_before_it() {
        let mut s = MemorySink::new();
        assert_eq!(s.durable_upto(), 0);
        s.write(w(0, 0, b"abc", 100)).unwrap();
        assert_eq!(s.durable_upto(), 0, "written is not durable");
        s.sync().unwrap();
        assert_eq!(s.durable_upto(), 100);
        s.write(w(0, 3, b"def", 200)).unwrap();
        assert_eq!(s.durable_upto(), 100, "the new bytes are not covered yet");
        s.sync().unwrap();
        assert_eq!(s.durable_upto(), 200);
        assert_eq!(s.syncs(), 2);
    }

    #[test]
    fn a_null_sink_says_everything_is_as_durable_as_it_gets() {
        let mut s = NullSink;
        s.write(w(0, 0, b"gone", 4)).unwrap();
        s.sync().unwrap();
        assert_eq!(s.durable_upto(), u64::MAX);
    }
}
