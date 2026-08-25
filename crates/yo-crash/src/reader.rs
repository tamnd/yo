//! The same wreckage, read by code that shares nothing with the engine.
//!
//! Every trial produces a damaged file. Replaying it with `yo-record` says what
//! the engine thinks is in there, and that answer is only worth something if
//! something else, written from the specification rather than from the writer,
//! reaches the same one.
//!
//! That is what this is. It walks an [`Image`] with `yo-reader` and nothing
//! else: its own record parser, its own page header decode, its own CRC. If the
//! two disagree about a record, or about where the log ends, then one of the two
//! transcriptions of the format is wrong and every other result in the harness
//! is a measurement of the same mistake made twice.
//!
//! The interesting disagreements are about stopping. A crash leaves a tail that
//! is half a record, and the two sides have to decide independently that it is
//! not a record. One deciding it is, on any of a hundred thousand damaged files,
//! is exactly the bug the gate is written to catch.

use yo_reader::format::{PAGE_HEADER_LEN, PageHeader, parse_record};

use crate::sink::Image;

/// One record, in the form both sides can be compared in.
pub type Seen = (u64, Vec<u8>, Vec<u8>);

/// What the independent walk found.
#[derive(Debug, Clone, Default)]
pub struct Walk {
    /// The records, in log order.
    pub records: Vec<Seen>,
    /// Where it stopped, and why, when it did not reach a clean end.
    pub stopped: Option<String>,
}

/// Walks an image with `yo-reader` alone.
///
/// Mirrors what replay does, deliberately, because the point is to arrive at
/// the same place by a different road. Start at log address 0, decode the page
/// header, walk records to the page's used mark, move to the next page. Stop at
/// a zero length, a page that is not there, a header that will not decode, or a
/// record that will not parse.
#[must_use]
pub fn walk(image: &Image, payload_len: usize) -> Walk {
    let mut out = Walk::default();
    let plen = payload_len as u64;
    let mut page_addr = 0u64;

    loop {
        let Some(page) = page_of(image, page_addr) else {
            // Not an error and not a stop worth naming. A page that is not in
            // the image is the end of what the device kept.
            return out;
        };

        let header = match PageHeader::decode(page) {
            Ok(h) => h,
            Err(e) => {
                out.stopped = Some(format!("page {page_addr}: {e}"));
                return out;
            }
        };
        if header.page_addr != page_addr {
            out.stopped = Some(format!(
                "page at {page_addr} says it belongs at {}",
                header.page_addr
            ));
            return out;
        }

        let limit = page.len().saturating_sub(PAGE_HEADER_LEN).min(payload_len);
        let mut at = 0usize;

        loop {
            if at >= limit {
                // Ran out of page with no zero to stop on, so the next page is
                // where the log carries on, if there is one.
                break;
            }
            let addr = page_addr + at as u64;
            let rest = &page[PAGE_HEADER_LEN + at..PAGE_HEADER_LEN + limit];
            match parse_record(rest) {
                Ok(Some(r)) => {
                    out.records.push((addr, r.key.clone(), r.value.clone()));
                    at += r.stride();
                }
                Err(e) => {
                    out.stopped = Some(format!("at {addr}: {e}"));
                    return out;
                }
                // A zero length. Which is not the end of the log, and working
                // out why not is the whole subtlety of this walk.
                //
                // A page is turned early whenever the next record would have
                // straddled its end, so the zeroes at the back of a page are a
                // gap of up to a record's width with more pages behind them. The
                // log ends where there is no next page, not where there is a
                // zero.
                //
                // Unless the zero turns up before the page header's used mark.
                // The header says how many payload bytes this page claims, so a
                // zero short of that is records the page promised and did not
                // deliver, and that is the one case where a zero really is the
                // end. Getting this wrong in either direction is a silent bug:
                // stop too early and acknowledged records are thrown away, carry
                // on too far and a gap becomes a hole in the middle of a log
                // nobody was told about.
                Ok(None) => {
                    if addr < page_addr + u64::from(header.used) {
                        out.stopped = Some(format!(
                            "at {addr}: fewer records than the page header claims"
                        ));
                    }
                    if addr < page_addr + u64::from(header.used)
                        || page_of(image, page_addr + plen).is_none()
                    {
                        return out;
                    }
                    break;
                }
            }
        }

        page_addr += plen;
    }
}

/// The bytes of the page whose payload starts at `page_addr`.
fn page_of(image: &Image, page_addr: u64) -> Option<&[u8]> {
    image
        .pages()
        .into_iter()
        .find(|(a, _)| *a == page_addr)
        .and_then(|(_, b)| {
            if b.len() >= PAGE_HEADER_LEN {
                Some(b)
            } else {
                None
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::CrashSink;
    use yo_format::{RecordHeader, RecordKind};
    use yo_record::sink::PageSink;
    use yo_record::{Durability, Log, LogConfig};

    fn a_log_of(n: u64) -> (Image, usize) {
        let page_len = 16 * 1024;
        let mut log = Log::new(
            LogConfig {
                page_len,
                durability: Durability::Group,
                ..LogConfig::default()
            },
            CrashSink::new(),
        )
        .unwrap();
        for i in 0..n {
            log.append(
                &RecordHeader::new(RecordKind::String),
                format!("k{i:07}").as_bytes(),
                &vec![i as u8; (i % 61) as usize],
            )
            .unwrap();
        }
        log.commit_pending().unwrap();
        let mut sink = log.into_sink();
        sink.sync().unwrap();
        (sink.durable().clone(), page_len - PAGE_HEADER_LEN)
    }

    #[test]
    fn a_whole_log_walks_end_to_end() {
        let (img, plen) = a_log_of(400);
        let w = walk(&img, plen);
        assert_eq!(w.records.len(), 400);
        assert_eq!(w.stopped, None, "a clean log has nothing to report");
        assert_eq!(w.records[0].1, b"k0000000");
        assert_eq!(w.records[399].1, b"k0000399");
    }

    #[test]
    fn the_walk_crosses_a_page_boundary() {
        // 400 records of up to 60 bytes will not fit in one 16 KiB page, so if
        // this comes back with everything then the page step is working. A test
        // that never crosses a boundary would pass with the step removed.
        let (img, plen) = a_log_of(400);
        assert!(
            img.len() > 1,
            "the log has to span pages for this to mean anything"
        );
        assert_eq!(walk(&img, plen).records.len(), 400);
    }

    #[test]
    fn a_flipped_bit_stops_the_walk_and_says_where() {
        let (mut img, plen) = a_log_of(50);
        // Into the first record's key.
        if let Some(p) = img.page_mut(0) {
            p[PAGE_HEADER_LEN + 18] ^= 0x40;
        }
        let w = walk(&img, plen);
        assert!(w.records.is_empty());
        assert!(w.stopped.unwrap().contains("checksum mismatch"));
    }

    #[test]
    fn a_torn_page_header_stops_the_walk() {
        let (mut img, plen) = a_log_of(50);
        if let Some(p) = img.page_mut(0) {
            p[6] ^= 0xff;
        }
        let w = walk(&img, plen);
        assert!(w.records.is_empty());
        assert!(w.stopped.is_some());
    }

    #[test]
    fn a_truncated_tail_stops_where_the_records_stop() {
        let (mut img, plen) = a_log_of(50);
        let before = walk(&img, plen).records.len();
        if let Some(p) = img.page_mut(0) {
            let half = p.len() / 2;
            p.truncate(half);
        }
        let w = walk(&img, plen);
        assert!(w.records.len() < before);
        assert!(!w.records.is_empty(), "the front of the page is still good");
    }

    #[test]
    fn an_empty_image_walks_to_nothing() {
        let w = walk(&Image::new(), 4096);
        assert!(w.records.is_empty());
        assert_eq!(w.stopped, None);
    }
}
