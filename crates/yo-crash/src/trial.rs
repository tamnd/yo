//! One trial: write a log, kill it, read it back, and judge the result.
//!
//! # The two rules
//!
//! Everything here comes down to two sentences, and they are deliberately not
//! the same sentence.
//!
//! **A crash must not lose an acknowledged commit.** If the caller was told a
//! commit was durable, the record is in the file after the crash. This is the
//! promise `Durability::Group` makes and it is the one worth breaking a build
//! over. It applies only to faults that touch bytes no sync had covered, which
//! is every fault except rot, because a device that eats data it already
//! promised to keep is a broken device rather than a case to survive.
//!
//! **Nothing may come back wrong, ever.** Whatever replay hands over has to be
//! a record that was really appended, at the address it is claimed to be at,
//! with the bytes it was written with. This holds under every fault including
//! rot. Losing data loudly is recoverable, because there is a replica or a
//! backup and the operator finds out. Handing back a record that was never
//! written is not recoverable, because nothing downstream has any reason to
//! doubt it, and it is the failure this whole harness exists to find.
//!
//! # Why the oracle can be exact
//!
//! The workload knows every record it wrote and the address it went to, so the
//! answer is not a checksum or a count, it is the list. A trial does not ask
//! "does this look plausible", it asks "is this exactly the prefix of what I
//! wrote that should have survived". Everything else is a violation with a
//! name.

use yo_format::{RecordHeader, RecordKind};
use yo_record::{Durability, Log, LogConfig, replay};

use crate::fault::Fault;
use crate::rng::Rng;
use crate::sink::{CrashSink, Image};

/// How a trial is shaped.
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    /// The physical page size. Small on purpose: the behaviour worth testing
    /// happens at a page boundary, and a trial that cannot reach one in a few
    /// hundred records is a trial that only ever tests the middle of a page.
    pub page_len: usize,
    /// How many records to write.
    pub records: usize,
    /// The largest value. Values run from nothing up to this, so records that
    /// straddle a sector boundary and records that do not both turn up.
    pub max_value: usize,
    /// What a commit is worth.
    pub durability: Durability,
}

impl Default for Shape {
    fn default() -> Shape {
        Shape {
            page_len: 16 * 1024,
            records: 200,
            max_value: 300,
            durability: Durability::Group,
        }
    }
}

/// One record that really was appended.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Written {
    addr: u64,
    end: u64,
    key: Vec<u8>,
    value: Vec<u8>,
}

/// Something the engine did that it is not allowed to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Violation {
    /// Replay handed back something that was never written that way.
    ///
    /// The one that matters. Either the bytes are not the bytes that went in,
    /// or the address is not where they went, and in both cases a caller has no
    /// way to know.
    CameBackWrong {
        /// Where replay says it found it.
        addr: u64,
        /// What replay says is there.
        got: String,
        /// What is really there, or a note that nothing is.
        want: String,
    },
    /// Replay skipped a record and carried on past it.
    ///
    /// Distinct from stopping early, which is allowed. Stopping is a truncation
    /// and the caller is told. Skipping is a hole in the middle of data the
    /// caller believes is whole.
    SkippedOne {
        /// Which record, counting from the start of the log.
        index: usize,
        /// The address it should have been at.
        expected_at: u64,
    },
    /// A commit that was acknowledged durable is not in the file.
    LostAnAcknowledgedCommit {
        /// The address of the oldest one lost.
        addr: u64,
        /// What the sink said was durable when the crash happened.
        durable_upto: u64,
        /// How many were lost in total.
        count: usize,
    },
    /// Data went missing and replay said the log ended cleanly.
    ///
    /// The rot verdict. Losing bytes to a bad device is allowed; not mentioning
    /// it is not, because the caller carries on with a log it believes is whole.
    LostQuietly {
        /// How many records went missing.
        count: usize,
    },
    /// Replay returned an error where it should have truncated.
    ///
    /// A torn tail is the expected result of a crash and has a defined answer,
    /// which is `truncated_at`. An error means the walk hit something it had no
    /// answer for.
    Errored {
        /// What it said.
        message: String,
    },
    /// The engine and the independent reader read the same file differently.
    ///
    /// Neither one is presumed right. What this says is that the format has two
    /// transcriptions in this project and they do not agree, which makes every
    /// other verdict in the run a measurement of one mistake made twice.
    ReaderDisagrees {
        /// How many records the engine's replay found.
        engine: usize,
        /// How many the independent reader found.
        reader: usize,
        /// The first record they differ on, and how.
        first: String,
    },
}

/// What one trial did.
#[derive(Debug, Clone)]
pub struct Outcome {
    /// The seed that reproduces it.
    pub seed: u64,
    /// The fault that was injected.
    pub fault: Fault,
    /// Records appended.
    pub written: usize,
    /// Records whose commit was acknowledged durable before the crash.
    pub acknowledged: usize,
    /// Records replay handed back.
    pub recovered: usize,
    /// Where replay stopped, when it stopped early.
    pub truncated_at: Option<u64>,
    /// Everything the engine did that it may not do. Empty is a pass.
    pub violations: Vec<Violation>,
}

impl Outcome {
    /// Whether the engine behaved.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.violations.is_empty()
    }
}

/// Runs one trial and judges it.
///
/// # Errors
///
/// Only when the trial could not be set up, which means the shape is not a
/// shape a log can have. A failure of the engine under test comes back as a
/// violation in the outcome, not as an error, because the caller wants to keep
/// going and count them.
pub fn run(seed: u64, shape: Shape) -> yo_common::Result<Outcome> {
    let mut rng = Rng::new(seed);
    let cfg = LogConfig {
        shard: 0,
        page_len: shape.page_len,
        durability: shape.durability,
        ..LogConfig::default()
    };
    let payload_len = shape.page_len - yo_format::PAGE_HEADER_LEN;

    let mut log = Log::new(cfg, CrashSink::new())?;
    let mut ledger: Vec<Written> = Vec::with_capacity(shape.records);

    for i in 0..shape.records {
        let key = format!("k{i:07}").into_bytes();
        // Deterministic from the index, so the oracle can say what should be at
        // an address without carrying the whole workload around.
        let n = rng.below(shape.max_value + 1);
        let value: Vec<u8> = (0..n)
            .map(|b| (b as u8).wrapping_mul(31) ^ (i as u8))
            .collect();

        let h = RecordHeader::new(RecordKind::String);
        let a = match log.append(&h, &key, &value) {
            Ok(a) => a,
            // A record that does not fit is not a fault in anything under test.
            // Stop the workload and crash what there is.
            Err(_) => break,
        };
        ledger.push(Written {
            addr: a.addr,
            end: a.addr + u64::from(a.len),
            key,
            value,
        });

        // Commit at intervals rather than every record, so that a crash lands
        // both just after a sync and a long way after one. Rarely, because a
        // sync empties the in-flight list and a trial that crashes with nothing
        // in flight has no crash to inject.
        if rng.chance(1, 24) {
            log.commit_pending()?;
        } else if rng.chance(1, 3) {
            // Handed to the store and not synced, which is where group commit
            // spends most of its time: the maintenance slice flushes as pages
            // fill and syncs on a timer, so a crash arriving at random finds
            // bytes in this state far more often than any other.
            log.flush()?;
        }
    }

    // How the run ends decides what a crash has to work with, so it is drawn
    // rather than fixed. Ending every trial with a commit leaves nothing in
    // flight, and a harness with nothing in flight has no crash to inject: it
    // spends a hundred thousand trials proving that a clean file reads back.
    match rng.below(4) {
        0 => {}
        3 => log.commit_pending()?,
        _ => log.flush()?,
    }

    // Everything at or below this was promised to somebody.
    let durable_upto = log.durable_upto();
    let sink = log.into_sink();

    let fault = Fault::pick(&sink, &mut rng);
    let image = fault.apply(&sink, &mut rng);

    // Rot is judged against what the file held before the bit flipped, not
    // against everything the workload wrote. Records that were still in the
    // log's own pages when the run ended never reached the device, so they were
    // never the device's to lose, and counting them would report a violation on
    // every trial.
    let baseline = if fault.is_crash() {
        None
    } else {
        Some(count_records(&sink.live_image(), payload_len))
    };

    judge(
        seed,
        fault,
        &ledger,
        durable_upto,
        &image,
        payload_len,
        baseline,
    )
}

/// How many records an undamaged image holds.
fn count_records(image: &Image, payload_len: usize) -> usize {
    let mut n = 0usize;
    let _ = replay::replay(image, payload_len, 0, |_, _| {
        n += 1;
        Ok(())
    });
    n
}

/// Reads the wreckage and compares it against what really went in.
fn judge(
    seed: u64,
    fault: Fault,
    ledger: &[Written],
    durable_upto: u64,
    image: &Image,
    payload_len: usize,
    baseline: Option<usize>,
) -> yo_common::Result<Outcome> {
    let acknowledged = ledger.iter().filter(|w| w.end <= durable_upto).count();
    let mut got: Vec<(u64, Vec<u8>, Vec<u8>)> = Vec::new();
    let mut violations = Vec::new();

    let report = replay::replay(image, payload_len, 0, |addr, r| {
        got.push((addr, r.key.to_vec(), r.value.to_vec()));
        Ok(())
    });

    let report = match report {
        Ok(r) => r,
        Err(e) => {
            violations.push(Violation::Errored {
                message: e.to_string(),
            });
            return Ok(Outcome {
                seed,
                fault,
                written: ledger.len(),
                acknowledged,
                recovered: 0,
                truncated_at: None,
                violations,
            });
        }
    };

    // Rule two, checked first because it is the one that matters. Whatever came
    // back has to be exactly the front of what went in.
    for (i, (addr, key, value)) in got.iter().enumerate() {
        match ledger.get(i) {
            Some(w) if w.addr == *addr && w.key == *key && w.value == *value => {}
            Some(w) if w.addr != *addr => {
                violations.push(Violation::SkippedOne {
                    index: i,
                    expected_at: w.addr,
                });
                break;
            }
            Some(w) => {
                violations.push(Violation::CameBackWrong {
                    addr: *addr,
                    got: describe(key, value),
                    want: describe(&w.key, &w.value),
                });
                break;
            }
            None => {
                violations.push(Violation::CameBackWrong {
                    addr: *addr,
                    got: describe(key, value),
                    want: format!("nothing, the log only had {} records", ledger.len()),
                });
                break;
            }
        }
    }

    // Rule one. Only for faults that touched bytes nothing had promised.
    if fault.is_crash() && got.len() < acknowledged {
        violations.push(Violation::LostAnAcknowledgedCommit {
            addr: ledger[got.len()].addr,
            durable_upto,
            count: acknowledged - got.len(),
        });
    }

    // Rule three: the two transcriptions of the format have to agree about this
    // file, records and stopping point alike.
    let theirs = crate::reader::walk(image, payload_len);
    if let Some(v) = disagreement(&got, &theirs.records) {
        violations.push(v);
    }

    // And the rot rule: losing data is allowed, saying nothing about it is not.
    if let Some(baseline) = baseline
        && got.len() < baseline
        && report.is_clean()
    {
        violations.push(Violation::LostQuietly {
            count: baseline - got.len(),
        });
    }

    Ok(Outcome {
        seed,
        fault,
        written: ledger.len(),
        acknowledged,
        recovered: got.len(),
        truncated_at: report.truncated_at,
        violations,
    })
}

/// Compares the two walks and names the first place they part company.
fn disagreement(
    engine: &[(u64, Vec<u8>, Vec<u8>)],
    reader: &[(u64, Vec<u8>, Vec<u8>)],
) -> Option<Violation> {
    let mut first = None;
    for i in 0..engine.len().max(reader.len()) {
        match (engine.get(i), reader.get(i)) {
            (Some(a), Some(b)) if a == b => {}
            (Some(a), Some(b)) => {
                first = Some(format!(
                    "record {i}: the engine has {} at {}, the reader has {} at {}",
                    describe(&a.1, &a.2),
                    a.0,
                    describe(&b.1, &b.2),
                    b.0
                ));
                break;
            }
            (Some(a), None) => {
                first = Some(format!(
                    "record {i}: the engine has {} at {}, the reader stopped before it",
                    describe(&a.1, &a.2),
                    a.0
                ));
                break;
            }
            (None, Some(b)) => {
                first = Some(format!(
                    "record {i}: the reader has {} at {}, the engine stopped before it",
                    describe(&b.1, &b.2),
                    b.0
                ));
                break;
            }
            (None, None) => break,
        }
    }
    first.map(|first| Violation::ReaderDisagrees {
        engine: engine.len(),
        reader: reader.len(),
        first,
    })
}

/// A record in a form that fits on one line of a failure report.
fn describe(key: &[u8], value: &[u8]) -> String {
    format!(
        "key {}, {} value bytes starting {:02x?}",
        String::from_utf8_lossy(key),
        value.len(),
        &value[..value.len().min(4)]
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use yo_record::sink::PageSink;

    #[test]
    fn a_run_with_no_fault_at_all_recovers_everything() {
        // Not a real trial, a check on the oracle. If a whole image does not
        // come back whole then every other verdict in this file is noise.
        let mut log = Log::new(
            LogConfig {
                page_len: 16 * 1024,
                durability: Durability::Group,
                ..LogConfig::default()
            },
            CrashSink::new(),
        )
        .unwrap();
        let mut ledger = Vec::new();
        for i in 0..300u64 {
            let key = format!("k{i:07}").into_bytes();
            let value = vec![i as u8; (i % 97) as usize];
            let a = log
                .append(&RecordHeader::new(RecordKind::String), &key, &value)
                .unwrap();
            ledger.push(Written {
                addr: a.addr,
                end: a.addr + u64::from(a.len),
                key,
                value,
            });
        }
        log.commit_pending().unwrap();
        let durable = log.durable_upto();
        let mut sink = log.into_sink();
        sink.sync().unwrap();
        let image = sink.durable().clone();

        let out = judge(
            0,
            Fault::LoseAll,
            &ledger,
            durable,
            &image,
            16 * 1024 - yo_format::PAGE_HEADER_LEN,
            None,
        )
        .unwrap();
        assert!(out.passed(), "{:?}", out.violations);
        assert_eq!(out.recovered, 300);
        assert_eq!(out.truncated_at, None);
    }

    #[test]
    fn the_oracle_notices_a_record_coming_back_wrong() {
        // Feed it a ledger that disagrees with the image and check that it says
        // so. An oracle that cannot fail has never proved anything.
        let mut log = Log::new(
            LogConfig {
                page_len: 16 * 1024,
                durability: Durability::Group,
                ..LogConfig::default()
            },
            CrashSink::new(),
        )
        .unwrap();
        let mut ledger = Vec::new();
        for i in 0..20u64 {
            let key = format!("k{i:07}").into_bytes();
            let value = vec![7u8; 40];
            let a = log
                .append(&RecordHeader::new(RecordKind::String), &key, &value)
                .unwrap();
            ledger.push(Written {
                addr: a.addr,
                end: a.addr + u64::from(a.len),
                key,
                value,
            });
        }
        log.commit_pending().unwrap();
        let durable = log.durable_upto();
        let mut sink = log.into_sink();
        sink.sync().unwrap();
        let image = sink.durable().clone();

        ledger[5].value = vec![8u8; 40];

        let out = judge(
            0,
            Fault::LoseAll,
            &ledger,
            durable,
            &image,
            16 * 1024 - yo_format::PAGE_HEADER_LEN,
            None,
        )
        .unwrap();
        assert!(!out.passed(), "a wrong ledger should be caught");
        assert!(matches!(out.violations[0], Violation::CameBackWrong { .. }));
    }

    #[test]
    fn a_short_log_is_a_lost_commit_and_not_something_else() {
        let mut log = Log::new(
            LogConfig {
                page_len: 16 * 1024,
                durability: Durability::Group,
                ..LogConfig::default()
            },
            CrashSink::new(),
        )
        .unwrap();
        let mut ledger = Vec::new();
        for i in 0..20u64 {
            let key = format!("k{i:07}").into_bytes();
            let value = vec![7u8; 40];
            let a = log
                .append(&RecordHeader::new(RecordKind::String), &key, &value)
                .unwrap();
            ledger.push(Written {
                addr: a.addr,
                end: a.addr + u64::from(a.len),
                key,
                value,
            });
        }
        log.commit_pending().unwrap();
        let durable = log.durable_upto();
        let mut sink = log.into_sink();
        sink.sync().unwrap();
        // An image that stops after the first few records, with the ledger still
        // saying twenty were acknowledged.
        let mut image = sink.durable().clone();
        let cut = (ledger[5].addr as usize) + yo_format::PAGE_HEADER_LEN;
        if let Some(p) = image.page_mut(0) {
            p.truncate(cut);
        }

        let out = judge(
            0,
            Fault::LoseAll,
            &ledger,
            durable,
            &image,
            16 * 1024 - yo_format::PAGE_HEADER_LEN,
            None,
        )
        .unwrap();
        assert!(!out.passed());
        assert!(
            matches!(
                out.violations[0],
                Violation::LostAnAcknowledgedCommit { count: 15, .. }
            ),
            "{:?}",
            out.violations
        );
    }

    #[test]
    fn a_trial_is_the_same_trial_every_time_it_runs() {
        let a = run(4242, Shape::default()).unwrap();
        let b = run(4242, Shape::default()).unwrap();
        assert_eq!(a.fault, b.fault);
        assert_eq!(a.written, b.written);
        assert_eq!(a.acknowledged, b.acknowledged);
        assert_eq!(a.recovered, b.recovered);
        assert_eq!(a.truncated_at, b.truncated_at);
    }

    #[test]
    // Fifteen hundred trials of two hundred records each, which is minutes on a
    // real machine and hours under Miri. There is no unsafe in this crate for
    // Miri to check, so what it would be doing is interpreting a fuzzer, and the
    // cheap trials below already prove the harness still runs under it.
    #[cfg_attr(miri, ignore)]
    fn a_few_hundred_trials_pass() {
        // The real run is a hundred thousand and lives in the binary. This is
        // enough to fail the build if something obvious breaks.
        for seed in 0..500u64 {
            let out = run(seed, Shape::default()).unwrap();
            assert!(
                out.passed(),
                "seed {seed}, fault {:?}: {:?}",
                out.fault,
                out.violations
            );
        }
    }

    #[test]
    // Fifteen hundred trials of two hundred records each, which is minutes on a
    // real machine and hours under Miri. There is no unsafe in this crate for
    // Miri to check, so what it would be doing is interpreting a fuzzer, and the
    // cheap trials below already prove the harness still runs under it.
    #[cfg_attr(miri, ignore)]
    fn the_shape_that_caught_the_stale_page_tail() {
        // Small pages and enough records to turn the ring several times, which
        // is what it takes to get an old page's records sitting under a new
        // page's flush block. Both of these failed before `yo-record` stopped
        // sending the store the part of the block past the sentinel, and the
        // default shape did not: a hundred thousand trials at 16 KiB pages went
        // clean while this found it in the first thirty thousand.
        let shape = Shape {
            page_len: 8192,
            records: 400,
            ..Shape::default()
        };
        for seed in [26281u64, 31175] {
            let out = run(seed, shape).unwrap();
            assert!(
                out.passed(),
                "seed {seed}, fault {:?}: {:?}",
                out.fault,
                out.violations
            );
        }
    }

    #[test]
    // Fifteen hundred trials of two hundred records each, which is minutes on a
    // real machine and hours under Miri. There is no unsafe in this crate for
    // Miri to check, so what it would be doing is interpreting a fuzzer, and the
    // cheap trials below already prove the harness still runs under it.
    #[cfg_attr(miri, ignore)]
    fn the_trials_actually_reach_every_fault() {
        // A suite that passes because it never injected anything is the failure
        // mode worth guarding, so count what the seeds reached.
        let mut seen = std::collections::HashMap::new();
        for seed in 0..500u64 {
            let out = run(seed, Shape::default()).unwrap();
            *seen.entry(out.fault.kind()).or_insert(0usize) += 1;
        }
        for want in [
            "lose-all",
            "lose-prefix",
            "reorder",
            "tear",
            "scatter",
            "rot-bit",
        ] {
            assert!(seen.get(want).copied().unwrap_or(0) > 0, "never hit {want}");
        }
    }

    #[test]
    // Fifteen hundred trials of two hundred records each, which is minutes on a
    // real machine and hours under Miri. There is no unsafe in this crate for
    // Miri to check, so what it would be doing is interpreting a fuzzer, and the
    // cheap trials below already prove the harness still runs under it.
    #[cfg_attr(miri, ignore)]
    fn the_trials_actually_lose_things() {
        // And a suite where every trial recovers everything is a suite whose
        // faults are landing somewhere harmless.
        let mut truncated = 0;
        let mut lost = 0;
        for seed in 0..500u64 {
            let out = run(seed, Shape::default()).unwrap();
            if out.truncated_at.is_some() {
                truncated += 1;
            }
            if out.recovered < out.written {
                lost += 1;
            }
        }
        assert!(lost > 100, "only {lost} of 500 trials lost anything");
        assert!(truncated > 10, "only {truncated} of 500 tore a tail");
    }
}
