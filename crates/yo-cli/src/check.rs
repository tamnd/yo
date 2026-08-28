//! `yodb check`: everything that can be said about a file without changing it.
//!
//! The checker reads with [`yo_reader`], which shares no code with the engine.
//! That is the whole reason it is trustworthy. A checker built on the engine's
//! own parser agrees with the engine about everything, including the places the
//! engine is wrong, and those are exactly the places worth checking.
//!
//! It never writes and never repairs. A tool that offers to fix a file is a
//! tool that will eventually fix one that was fine, and the first thing anyone
//! should do with a damaged database is copy it.
//!
//! # What it can actually prove
//!
//! Checksums say a run of bytes is the run that was written. They say nothing
//! about whether it was the right run to write. So the checks worth having are
//! the ones that hold two independently stored facts up against each other:
//!
//! - the superblock says the file was this big, and the file is this big
//! - the checkpoint says shard 3's log ends at this address, and shard 3's last
//!   segment says it holds bytes up to that same address
//! - every segment carries the address it belongs at, and no two carry the same
//!   one
//!
//! Each of those can only pass by accident once. The record checksums on their
//! own cannot catch a segment handed to two shards, because both shards write
//! perfectly good records into it.

use std::collections::HashMap;
use std::path::Path;

use yo_reader::format::{DATA_START, LOG_PAGE_LEN, PAYLOAD_LEN};
use yo_reader::{Reader, SlotStatus};

/// How much a finding matters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Worth knowing, and the file is still sound.
    Note,
    /// The file works but has lost something it is meant to have, usually its
    /// redundancy. Fine today, one more fault from not fine.
    Warn,
    /// Something is wrong.
    Error,
}

impl Severity {
    const fn label(self) -> &'static str {
        match self {
            Severity::Note => "note ",
            Severity::Warn => "warn ",
            Severity::Error => "ERROR",
        }
    }
}

/// One thing the checker noticed.
#[derive(Debug, Clone)]
pub struct Finding {
    /// How much it matters.
    pub severity: Severity,
    /// What it is, in a sentence.
    pub what: String,
    /// Where in the file, when that is known.
    pub at: Option<u64>,
}

impl Finding {
    fn new(severity: Severity, what: impl Into<String>) -> Finding {
        Finding {
            severity,
            what: what.into(),
            at: None,
        }
    }

    fn at(mut self, off: u64) -> Finding {
        self.at = Some(off);
        self
    }
}

impl std::fmt::Display for Finding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.severity.label(), self.what)?;
        if let Some(at) = self.at {
            write!(f, " (at byte {at})")?;
        }
        Ok(())
    }
}

/// What the file turned out to hold.
#[derive(Debug, Default, Clone, Copy)]
pub struct Counts {
    /// Segments with a header in them.
    pub regions: u64,
    /// Records walked, when the walk happened.
    pub records: u64,
    /// Record bytes, trailers included.
    pub record_bytes: u64,
    /// Bytes the pages say belong to records nothing points at any more.
    pub dead_bytes: u64,
}

/// The whole verdict.
#[derive(Debug, Default)]
pub struct Report {
    /// Everything noticed, in the order it was noticed.
    pub findings: Vec<Finding>,
    /// What was in the file.
    pub counts: Counts,
}

impl Report {
    /// Whether anything is actually wrong.
    #[must_use]
    pub fn is_sound(&self) -> bool {
        !self.findings.iter().any(|f| f.severity == Severity::Error)
    }

    /// How many findings there are at this severity.
    #[must_use]
    pub fn count(&self, s: Severity) -> usize {
        self.findings.iter().filter(|f| f.severity == s).count()
    }

    fn push(&mut self, f: Finding) {
        self.findings.push(f);
    }
}

/// Reads a file and says what is wrong with it.
///
/// `walk_records` reads the used part of every segment and verifies every
/// record in it. Without it the check reads two superblock slots and one 32
/// byte header per segment, which is fast enough to run on a ten gigabyte file
/// without thinking about it. With it, the check reads as much of the file as
/// has records in it.
///
/// # Errors
///
/// Only when the file cannot be opened far enough to say anything at all, which
/// means it is not a `.yo` file or both superblocks are gone. Everything else
/// comes back as a finding, because a checker that stops at the first problem
/// makes you run it once per problem.
pub fn check(path: &Path, walk_records: bool) -> yo_reader::Result<Report> {
    let r = Reader::open(path)?;
    let sb = r.superblock();
    let mut rep = Report::default();

    // -- the two slots ------------------------------------------------------

    for (i, s) in r.slots().iter().enumerate() {
        let name = if i == 0 { "A" } else { "B" };
        if let SlotStatus::Bad(e) = s {
            let live = i == r.live_slot();
            rep.push(
                Finding::new(
                    if live {
                        Severity::Error
                    } else {
                        Severity::Warn
                    },
                    format!(
                        "superblock slot {name} does not decode: {e}. {}",
                        if live {
                            "This is the live slot."
                        } else {
                            "The other slot is carrying the file, so there is no spare left."
                        }
                    ),
                )
                .at(i as u64 * 16384),
            );
        }
    }

    if !sb.clean_shutdown() {
        rep.push(Finding::new(
            Severity::Note,
            "the file was not closed cleanly, so the log past the checkpoint has not been replayed",
        ));
    }

    // -- the size the superblock claims -------------------------------------

    let actual = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    if sb.file_size > actual {
        rep.push(Finding::new(
            Severity::Error,
            format!(
                "the superblock says the file is {} bytes and it is {actual}, so {} bytes are missing from the end",
                sb.file_size,
                sb.file_size - actual
            ),
        ));
    } else if actual > sb.file_size {
        // Not a problem. A segment gets allocated by growing the file, and the
        // superblock only learns about it at the next checkpoint, so a file
        // that is bigger than its last checkpoint is a file that was being
        // written to.
        rep.push(Finding::new(
            Severity::Note,
            format!(
                "the file has grown {} bytes past the last checkpoint, which is {} segments allocated since",
                actual - sb.file_size,
                (actual - sb.file_size).div_ceil(LOG_PAGE_LEN)
            ),
        ));
    }
    if actual > DATA_START && !(actual - DATA_START).is_multiple_of(LOG_PAGE_LEN) {
        rep.push(Finding::new(
            Severity::Warn,
            format!(
                "the data area is {} bytes, which is not a whole number of {LOG_PAGE_LEN} byte segments",
                actual - DATA_START
            ),
        ));
    }

    // -- the shard table ----------------------------------------------------

    match r.shard_table() {
        Ok(t) => {
            let mut seen = vec![false; sb.shard_count as usize];
            for &s in &t {
                if let Some(slot) = seen.get_mut(s as usize) {
                    *slot = true;
                }
            }
            let idle: Vec<String> = seen
                .iter()
                .enumerate()
                .filter(|(_, v)| !**v)
                .map(|(i, _)| i.to_string())
                .collect();
            if !idle.is_empty() {
                rep.push(Finding::new(
                    Severity::Note,
                    format!(
                        "nothing is routed to shard{} {}, so {} idle",
                        if idle.len() == 1 { "" } else { "s" },
                        idle.join(", "),
                        if idle.len() == 1 { "it is" } else { "they are" }
                    ),
                ));
            }
        }
        Err(e) => rep.push(Finding::new(
            Severity::Error,
            format!("the shard table does not decode: {e}"),
        )),
    }

    // -- the checkpoint entries ---------------------------------------------

    let checkpoints = match r.checkpoints() {
        Ok(c) => c,
        Err(e) => {
            rep.push(Finding::new(
                Severity::Error,
                format!("the checkpoint entries do not decode: {e}"),
            ));
            Vec::new()
        }
    };

    // -- the segments -------------------------------------------------------

    // Two segments claiming the same place in the same shard's log is the one
    // corruption no checksum can find, because both of them are internally
    // perfect. This map is the only thing that catches it.
    let mut claimed: HashMap<(u32, u64), u64> = HashMap::new();
    // Which page addresses each shard has, so the gaps can be found later.
    let mut by_shard: HashMap<u32, Vec<(u64, u32, u64)>> = HashMap::new();

    for region in r.regions() {
        rep.counts.regions += 1;
        if let Some(d) = &region.damage {
            rep.push(
                Finding::new(
                    Severity::Error,
                    format!("segment {} has an unreadable header: {d}", region.index),
                )
                .at(region.offset),
            );
            continue;
        }
        let h = &region.header;

        if h.shard >= sb.shard_count {
            rep.push(
                Finding::new(
                    Severity::Error,
                    format!(
                        "segment {} says it belongs to shard {} and the file has {}",
                        region.index, h.shard, sb.shard_count
                    ),
                )
                .at(region.offset),
            );
            continue;
        }
        if !h.page_addr.is_multiple_of(PAYLOAD_LEN) {
            rep.push(
                Finding::new(
                    Severity::Error,
                    format!(
                        "segment {} claims log address {}, which is not a multiple of the {PAYLOAD_LEN} byte payload",
                        region.index, h.page_addr
                    ),
                )
                .at(region.offset),
            );
            continue;
        }
        if u64::from(h.used) > PAYLOAD_LEN {
            rep.push(
                Finding::new(
                    Severity::Error,
                    format!(
                        "segment {} says {} of its bytes are used and it only holds {PAYLOAD_LEN}",
                        region.index, h.used
                    ),
                )
                .at(region.offset),
            );
            continue;
        }
        if h.dead_bytes > h.used {
            rep.push(
                Finding::new(
                    Severity::Error,
                    format!(
                        "segment {} says {} bytes are dead out of {} used",
                        region.index, h.dead_bytes, h.used
                    ),
                )
                .at(region.offset),
            );
        }
        if let Some(other) = claimed.insert((h.shard, h.page_addr), region.index) {
            rep.push(
                Finding::new(
                    Severity::Error,
                    format!(
                        "segments {other} and {} both claim log address {} of shard {}, so one of them is being written over",
                        region.index, h.page_addr, h.shard
                    ),
                )
                .at(region.offset),
            );
        }
        rep.counts.dead_bytes += u64::from(h.dead_bytes);
        by_shard
            .entry(h.shard)
            .or_default()
            .push((h.page_addr, h.used, region.offset));
    }

    // -- what the segments say against what the checkpoint says --------------

    for (shard, entry) in checkpoints.iter().enumerate() {
        let shard = shard as u32;
        let Some(pages) = by_shard.get_mut(&shard) else {
            if entry.log_tail > entry.log_begin {
                rep.push(Finding::new(
                    Severity::Error,
                    format!(
                        "the checkpoint says shard {shard} has a log from {} to {} and there is not one segment for it in the file",
                        entry.log_begin, entry.log_tail
                    ),
                ));
            }
            continue;
        };
        pages.sort_unstable();

        // Every page address between the oldest byte still kept and the tail
        // has to be present. A hole means a segment was lost, and replay would
        // walk straight over it into the next one and believe what it found.
        if entry.log_tail > 0 {
            let first = entry.log_begin / PAYLOAD_LEN;
            let last = (entry.log_tail - 1) / PAYLOAD_LEN;
            for k in first..=last {
                let want = k * PAYLOAD_LEN;
                if !pages.iter().any(|&(a, _, _)| a == want) {
                    rep.push(Finding::new(
                        Severity::Error,
                        format!(
                            "shard {shard} has no segment for log address {want}, and the checkpoint needs everything from {} to {}",
                            entry.log_begin, entry.log_tail
                        ),
                    ));
                }
            }

            // And the last one has to end exactly where the checkpoint says the
            // log ends. These two numbers are written at different times into
            // different parts of the file, so agreeing is meaningful.
            let want = last * PAYLOAD_LEN;
            if let Some(&(_, used, off)) = pages.iter().find(|&&(a, _, _)| a == want) {
                let ends_at = want + u64::from(used);
                if ends_at < entry.log_tail {
                    rep.push(
                        Finding::new(
                            Severity::Error,
                            format!(
                                "the checkpoint says shard {shard} ends at {} and its last segment only holds up to {ends_at}, so {} bytes of committed log are gone",
                                entry.log_tail,
                                entry.log_tail - ends_at
                            ),
                        )
                        .at(off),
                    );
                } else if ends_at > entry.log_tail {
                    // The other way round is normal: appends land in the page
                    // before the checkpoint that records them, so a page that
                    // runs past the tail is a page that was still being written.
                    rep.push(
                        Finding::new(
                            Severity::Note,
                            format!(
                                "shard {shard} has {} bytes written past the checkpoint tail, which replay will pick up",
                                ends_at - entry.log_tail
                            ),
                        )
                        .at(off),
                    );
                }
            }
        }

        for &(addr, _, off) in pages.iter() {
            if entry.log_tail > 0 && addr >= entry.log_tail.div_ceil(PAYLOAD_LEN) * PAYLOAD_LEN {
                rep.push(
                    Finding::new(
                        Severity::Warn,
                        format!(
                            "shard {shard} has a segment at log address {addr}, past everything the checkpoint knows about",
                            ),
                    )
                    .at(off),
                );
            }
        }
    }

    // -- the records --------------------------------------------------------

    if walk_records {
        for region in r.regions() {
            if region.damage.is_some() {
                continue;
            }
            match r.records(region) {
                Ok(rs) => {
                    rep.counts.records += rs.len() as u64;
                    rep.counts.record_bytes += rs.iter().map(|x| u64::from(x.len)).sum::<u64>();
                }
                Err(e) => rep.push(Finding::new(
                    Severity::Error,
                    format!("segment {} stops parsing: {e}", region.index),
                )),
            }
        }
    }

    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn severity_sorts_by_how_much_it_matters() {
        assert!(Severity::Error > Severity::Warn);
        assert!(Severity::Warn > Severity::Note);
    }

    #[test]
    fn a_report_with_only_notes_is_sound() {
        let mut r = Report::default();
        r.push(Finding::new(Severity::Note, "something"));
        r.push(Finding::new(Severity::Warn, "something else"));
        assert!(r.is_sound());
        r.push(Finding::new(Severity::Error, "the bad one"));
        assert!(!r.is_sound());
        assert_eq!(r.count(Severity::Note), 1);
        assert_eq!(r.count(Severity::Error), 1);
    }

    #[test]
    fn a_finding_says_where_when_it_knows() {
        let f = Finding::new(Severity::Error, "the tail is short").at(4096);
        assert_eq!(f.to_string(), "ERROR the tail is short (at byte 4096)");
        assert_eq!(
            Finding::new(Severity::Note, "fine").to_string(),
            "note  fine"
        );
    }
}
