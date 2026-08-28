//! Replay. `06` section 4.
//!
//! Start at the checkpoint's tail, walk forward, stop at the first thing that
//! is not a record. That is the whole algorithm, and it is short because the
//! append path did the work: the length is written last, the four bytes past
//! the tail are always zero, and every record carries its own checksum. So the
//! three ways a log can end all look the same from here.
//!
//! - The writer stopped cleanly. The next four bytes are the sentinel zero.
//! - The machine lost power mid append. The length was never stored, so the
//!   next four bytes are still the sentinel zero.
//! - The device tore a write. The length is there but the bytes behind it are
//!   half of one record and half of another, so the checksum fails.
//!
//! The first two are indistinguishable and both mean stop. The third is
//! distinguishable and also means stop, but it is worth telling the caller
//! about, which is what [`ReplayReport::truncated_at`] is for: `yodb check`
//! prints it and recovery truncates to it.
//!
//! **There is no write ahead log.** Not as a simplification, as a consequence.
//! A write ahead log exists to make a second structure recoverable, and here
//! the log is the structure. Nothing is written twice, which is why the write
//! amplification of a durable commit is one.

use yo_common::{Code, Error, Result};
use yo_format::PAGE_HEADER_LEN;
use yo_format::page::PageHeader;
use yo_format::record::{RecordIter, RecordRef};

use crate::sink::PageSource;

/// What a replay found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReplayReport {
    /// Records handed to the callback.
    pub records: u64,
    /// Record bytes, trailers included. Not stride, so this is smaller than the
    /// address range covered.
    pub bytes: u64,
    /// Pages walked.
    pub pages: u64,
    /// Where the log ends, which is where the next append goes.
    pub tail: u64,
    /// The highest epoch seen in a page header, which is where the shard
    /// carries on from.
    pub epoch: u32,
    /// Set when the walk stopped because bytes were damaged rather than because
    /// it reached the end. The address is the first byte that could not be
    /// read, which is exactly what a truncation truncates to.
    pub truncated_at: Option<u64>,
    /// Why it stopped, in a form worth printing. `None` when it stopped at a
    /// clean end of log.
    pub reason: Option<&'static str>,
}

impl ReplayReport {
    /// Whether the log ended cleanly.
    #[must_use]
    pub const fn is_clean(&self) -> bool {
        self.truncated_at.is_none()
    }
}

/// Walks the log from `from` and hands every record to `f`.
///
/// `payload_len` has to match what the writer used, because it is what turns a
/// log address into a page and an offset. It is not stored per page: it comes
/// out of the superblock's page size, and a file whose page size does not match
/// its pages fails the header check on the first page rather than reading
/// garbage.
///
/// The callback gets the record's address and the record. Returning an error
/// from it stops the walk and the error comes back, which is how a caller that
/// finds a record it cannot apply refuses the file instead of half loading it.
///
/// Damage is not an error. A torn tail is the expected result of a crash, so it
/// comes back in [`ReplayReport::truncated_at`] and the caller decides. An
/// error is for a callback that said no, or a page that claims to be a page it
/// is not, which means the file layer handed over the wrong bytes.
///
/// # Errors
///
/// [`Code::Invalid`] if `payload_len` is zero. [`Code::Corrupt`] if a page
/// header decodes but names a different address than the one it was fetched
/// for. Whatever the callback returns.
pub fn replay<S, F>(source: &S, payload_len: usize, from: u64, mut f: F) -> Result<ReplayReport>
where
    S: PageSource + ?Sized,
    F: FnMut(u64, &RecordRef<'_>) -> Result<()>,
{
    if payload_len == 0 {
        return Err(Error::new(Code::Invalid, "a page with no payload"));
    }
    let plen = payload_len as u64;
    let mut rep = ReplayReport {
        tail: from,
        ..ReplayReport::default()
    };

    let mut page_addr = from - (from % plen);
    let mut start_off = (from % plen) as usize;

    loop {
        let Some(bytes) = source.page_bytes(page_addr) else {
            // No page here. The log ends at the end of the previous one, which
            // is where `tail` already is. This is the ordinary way a walk ends
            // when the last page was full.
            rep.reason = Some("no page at the next address");
            return Ok(rep);
        };
        // Where a truncation goes if this page turns out to be unreadable. The
        // start of the page, because everything before it is fine, except when
        // the walk started inside this page, in which case it is where the walk
        // started. A truncation point that moved backwards would throw away
        // records a checkpoint has already promised are there.
        let cut = page_addr.max(rep.tail);

        if bytes.len() < PAGE_HEADER_LEN {
            rep.tail = cut;
            rep.truncated_at = Some(cut);
            rep.reason = Some("a page shorter than its own header");
            return Ok(rep);
        }
        let header = match PageHeader::decode(bytes) {
            Ok(h) => h,
            Err(_) => {
                // A page header that will not decode is a page that was never
                // written or was written over. Either way the log stops here.
                rep.tail = cut;
                rep.truncated_at = Some(cut);
                rep.reason = Some("a page header that does not decode");
                return Ok(rep);
            }
        };
        if header.page_addr != page_addr {
            return Err(
                Error::new(Code::Corrupt, "a page that is not the page asked for")
                    .with_detail(format!("want={page_addr} have={}", header.page_addr)),
            );
        }
        rep.epoch = rep.epoch.max(header.epoch);
        rep.pages += 1;

        let payload = &bytes[PAGE_HEADER_LEN..];
        let avail = payload.len().min(payload_len);
        let begin = start_off.min(avail);
        let mut it = RecordIter::new(&payload[begin..avail]);
        loop {
            let off = begin + it.offset();
            let addr = page_addr + off as u64;
            let Some(next) = it.next() else {
                // No more records in this page. Which is not the same thing as
                // no more records, and the difference is the one mistake this
                // function must not make. A page is turned early whenever the
                // next record would have straddled its end, so the zero the
                // walk just found may be a gap of up to a record's width with
                // several more pages of log behind it. The end of the log is
                // where there is no next page, not where there is a zero.
                rep.tail = addr;

                // Except that a page header says how many payload bytes it
                // claims, and finding the zero short of that means records the
                // page promised did not land. That is damage, and it is the one
                // case where a zero really does end the walk.
                let claimed = page_addr + u64::from(header.used);
                if addr < claimed {
                    rep.truncated_at = Some(addr);
                    rep.reason = Some("fewer records than the page header claims");
                    return Ok(rep);
                }
                if source.page_bytes(page_addr + plen).is_none() {
                    rep.reason = Some("end of log");
                    return Ok(rep);
                }
                break;
            };
            match next {
                Ok(r) => {
                    f(addr, &r)?;
                    rep.records += 1;
                    rep.bytes += u64::from(r.len);
                    rep.tail = addr + r.stride() as u64;
                }
                Err(_) => {
                    // A length that is there but bytes that are not what was
                    // written. The log ends at the start of this record, not
                    // after it.
                    rep.tail = addr;
                    rep.truncated_at = Some(addr);
                    rep.reason = Some("a record that does not check out");
                    return Ok(rep);
                }
            }
        }

        page_addr += plen;
        start_off = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::log::{Durability, Log, LogConfig};
    use crate::sink::{MemorySink, PageSink, PageWrite};
    use yo_format::record::{RecordHeader, RecordKind};

    const PAGE: usize = 8192;
    const PAYLOAD: usize = PAGE - PAGE_HEADER_LEN;

    /// A prefix and a decimal number, the same bytes `format!` would give.
    ///
    /// Hand written for the same reason as the copy in `compact.rs`: Miri
    /// charges per operation, and the formatting machinery costs milliseconds
    /// per call where this costs microseconds.
    fn nkey(prefix: &str, n: u32) -> Vec<u8> {
        let mut out = prefix.as_bytes().to_vec();
        push_num(&mut out, n);
        out
    }

    fn push_num(out: &mut Vec<u8>, n: u32) {
        if n >= 10 {
            push_num(out, n / 10);
        }
        out.push(b'0' + (n % 10) as u8);
    }

    /// A log with `n` records in it, flushed, plus what went in.
    fn written(n: u32) -> (Log<MemorySink>, Vec<(u64, Vec<u8>)>) {
        let mut log = Log::new(
            LogConfig {
                shard: 2,
                page_len: PAGE,
                resident_pages: 3,
                mutable_fraction: 0.40,
                durability: Durability::Group,
            },
            MemorySink::new(),
        )
        .unwrap();
        let h = RecordHeader::new(RecordKind::String);
        let mut want = Vec::new();
        for i in 0..n {
            let value = nkey("value number ", i);
            let a = log.append(&h, &nkey("key", i), &value).unwrap();
            want.push((a.addr, value));
        }
        log.commit_pending().unwrap();
        (log, want)
    }

    /// Replays a sink and collects what came back.
    fn walk(sink: &MemorySink, from: u64) -> (ReplayReport, Vec<(u64, Vec<u8>)>) {
        let mut got = Vec::new();
        let rep = replay(sink, PAYLOAD, from, |addr, r| {
            got.push((addr, r.value.to_vec()));
            Ok(())
        })
        .unwrap();
        (rep, got)
    }

    #[test]
    fn a_clean_log_replays_every_record_in_order_and_lands_on_the_tail() {
        let (log, want) = written(600);
        let (rep, got) = walk(log.sink(), 0);
        assert!(rep.is_clean(), "{rep:?}");
        assert_eq!(rep.reason, Some("end of log"));
        assert_eq!(rep.records as usize, want.len());
        assert_eq!(got, want, "replay did not produce what was written");
        assert_eq!(rep.tail, log.tail(), "replay disagrees about the tail");
        assert!(rep.pages >= 3);
    }

    #[test]
    fn an_empty_log_replays_to_nothing() {
        let log = Log::new(
            LogConfig {
                page_len: PAGE,
                durability: Durability::Group,
                ..LogConfig::default()
            },
            MemorySink::new(),
        )
        .unwrap();
        let (rep, got) = walk(log.sink(), 0);
        assert_eq!(rep.records, 0);
        assert_eq!(rep.tail, 0);
        assert!(got.is_empty());
        assert!(rep.is_clean());
    }

    #[test]
    fn replay_from_a_checkpoint_skips_what_the_checkpoint_covered() {
        let (log, want) = written(600);
        let from = want[300].0;
        let (rep, got) = walk(log.sink(), from);
        assert_eq!(got.len(), 300, "should have replayed the last 300 only");
        assert_eq!(got[0], want[300]);
        assert_eq!(rep.tail, log.tail());
    }

    #[test]
    fn the_epoch_that_comes_back_is_the_highest_any_page_carried() {
        let mut log = Log::new(
            LogConfig {
                page_len: PAGE,
                durability: Durability::Group,
                ..LogConfig::default()
            },
            MemorySink::new(),
        )
        .unwrap();
        let h = RecordHeader::new(RecordKind::String);
        for i in 0..600u32 {
            log.append(&h, &nkey("key", i), &[0u8; 40]).unwrap();
            if i % 100 == 0 {
                log.advance_epoch();
            }
        }
        log.commit_pending().unwrap();
        let (rep, _) = walk(log.sink(), 0);
        assert_eq!(rep.epoch, log.epoch());
    }

    #[test]
    fn a_torn_tail_stops_the_walk_and_says_where() {
        let (log, want) = written(600);
        let mut sink = clone_sink(log.sink());

        // Damage the last record's payload, which is what a torn write does:
        // the length landed and the bytes behind it did not.
        let (addr, _) = *want.last().unwrap();
        flip(&mut sink, addr + 20);

        let (rep, got) = walk(&sink, 0);
        assert_eq!(rep.truncated_at, Some(addr), "{rep:?}");
        assert_eq!(rep.tail, addr, "the tail is the start of the bad record");
        assert_eq!(got.len(), want.len() - 1);
        assert_eq!(rep.reason, Some("a record that does not check out"));
        assert!(!rep.is_clean());
    }

    #[test]
    fn damage_in_the_middle_stops_there_rather_than_being_walked_past() {
        // Not a torn tail: a bad block in the middle of the log. Replay has no
        // way to resynchronise, because a record's length is the only thing
        // that says where the next one starts, so it stops. Everything before
        // it is still good and still gets applied, which is the difference
        // between losing a file and losing the writes after a bad block.
        let (log, want) = written(600);
        let mut sink = clone_sink(log.sink());
        let (addr, _) = want[100];
        flip(&mut sink, addr + 20);

        let (rep, got) = walk(&sink, 0);
        assert_eq!(rep.truncated_at, Some(addr));
        assert_eq!(got.len(), 100);
        assert_eq!(got, want[..100].to_vec());
    }

    #[test]
    fn a_flip_in_any_byte_of_a_record_is_caught() {
        // Every byte, not a sample. A checksum that covers most of a record is
        // not a checksum, and the cost of knowing rather than believing is a
        // few milliseconds.
        //
        // The flips are the point and there are only forty of them, one per
        // byte of the record. What costs is the walk after each one, which is
        // over the whole log, so Miri shortens the log rather than skipping
        // flips: every byte still gets its turn.
        let (log, want) = written(if cfg!(miri) { 60 } else { 200 });
        let (addr, _) = want[50];
        let len = 16 + 5 + "value number 50".len() + 4;

        for byte in 0..len {
            let mut sink = clone_sink(log.sink());
            flip(&mut sink, addr + byte as u64);
            let (rep, got) = walk(&sink, 0);
            assert!(
                !rep.is_clean() || got.len() < want.len(),
                "a flip at byte {byte} of the record at {addr} went unnoticed"
            );
        }
    }

    #[test]
    fn a_page_header_that_does_not_decode_ends_the_walk() {
        let (log, _) = written(600);
        let mut sink = clone_sink(log.sink());
        // Break the magic of the second page.
        let second = PAYLOAD as u64;
        let page = sink.page_mut(second).expect("there is a second page");
        page[0] ^= 0xff;

        let (rep, got) = walk(&sink, 0);
        assert_eq!(rep.reason, Some("a page header that does not decode"));
        assert_eq!(rep.truncated_at, Some(second));
        assert!(!got.is_empty(), "the first page should still have replayed");
    }

    #[test]
    fn a_callback_that_refuses_stops_the_whole_replay() {
        let (log, want) = written(600);
        let mut seen = 0;
        let err = replay(log.sink(), PAYLOAD, 0, |_, _| {
            seen += 1;
            if seen == 10 {
                Err(Error::new(
                    Code::Unsupported,
                    "this build cannot apply that",
                ))
            } else {
                Ok(())
            }
        })
        .unwrap_err();
        assert_eq!(err.code(), Code::Unsupported);
        assert_eq!(seen, 10);
        assert!(want.len() > 10);
    }

    #[test]
    fn a_log_replayed_then_reopened_appends_where_the_old_one_left_off() {
        // The round trip that recovery actually is: walk, take the tail, open
        // there, and carry on as if nothing happened.
        let (log, want) = written(600);
        let (rep, _) = walk(log.sink(), 0);
        let sink = clone_sink(log.sink());

        let mut second = Log::recover(
            LogConfig {
                shard: 2,
                page_len: PAGE,
                resident_pages: 3,
                mutable_fraction: 0.40,
                durability: Durability::Group,
            },
            sink,
            rep.tail,
        )
        .unwrap();
        let h = RecordHeader::new(RecordKind::String);
        let a = second.append(&h, b"after", b"recovery").unwrap();
        assert_eq!(a.addr, rep.tail, "the new record did not go at the tail");
        second.commit_pending().unwrap();

        let (rep2, got2) = walk(second.sink(), 0);
        assert!(rep2.is_clean(), "{rep2:?}");
        assert_eq!(got2.len(), want.len() + 1);
        assert_eq!(got2.last().unwrap().1, b"recovery");
        assert_eq!(rep2.tail, second.tail());
    }

    #[test]
    fn a_page_with_no_payload_is_refused_rather_than_looped_on() {
        let (log, _) = written(10);
        let err = replay(log.sink(), 0, 0, |_, _| Ok(())).unwrap_err();
        assert_eq!(err.code(), Code::Invalid);
    }

    // -- helpers ------------------------------------------------------------

    /// A copy of a sink's pages, so a test can damage one without damaging the
    /// log it came from.
    fn clone_sink(src: &MemorySink) -> MemorySink {
        let mut out = MemorySink::new();
        for (page_addr, bytes) in src.pages() {
            out.write(PageWrite {
                page_addr,
                offset: 0,
                bytes,
                covers_upto: 0,
            })
            .unwrap();
        }
        out
    }

    /// Flips a bit at a log address.
    fn flip(sink: &mut MemorySink, addr: u64) {
        let page_addr = addr - (addr % PAYLOAD as u64);
        let off = PAGE_HEADER_LEN + (addr - page_addr) as usize;
        let page = sink
            .page_mut(page_addr)
            .expect("flipping a byte in a page that is not there");
        page[off] ^= 0b0001_0000;
    }
}
