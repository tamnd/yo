//! The M1 exit gate, as a fuzzer: the independent reader agrees with the
//! engine about every file the engine can be made to write.
//!
//! `crates/yo-reader/tests/agrees_with_the_engine.rs` checks the same property
//! over a handful of shapes chosen by hand. Chosen by hand is the problem: the
//! shapes that break a format walk are the ones nobody thought to write down,
//! and they are all about lengths. A key of exactly the length that fills a
//! page, a value that leaves four bytes of a page free, a record whose total is
//! one short of a multiple of eight, a shard that got nothing while its
//! neighbour got everything. This drives all of that from the fuzzer.
//!
//! Three things have to agree, and they are three separate claims:
//!
//! - The reader sees the same records the test wrote, in order, byte for byte.
//! - The engine's own replay sees the same records too, so a disagreement can
//!   be pinned on one side rather than left as "the two differ".
//! - The reader's superblock and checkpoint entries match the engine's field
//!   for field, which is where a wrong offset in one of the two transcriptions
//!   of `06` and `07` shows up.
//!
//! The file is real and on disk, because the thing under test is the on disk
//! layout and a memory sink would let a wrong offset cancel itself out.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use yo_file::{Checkpoint, CreateOptions, Yo};
use yo_format::{PAGE_HEADER_LEN, RecordHeader, RecordKind};
use yo_reader::Reader;
use yo_record::{Durability, Log, LogConfig, replay};

/// One record to write.
#[derive(Arbitrary, Debug)]
struct Rec {
    key: Vec<u8>,
    value: Vec<u8>,
    ttl: Option<u64>,
    tombstone: bool,
}

#[derive(Arbitrary, Debug)]
struct Input {
    shards: u8,
    clean: bool,
    /// Which shard each record goes to, taken modulo the shard count, so a run
    /// where one shard gets everything and the rest get nothing is reachable.
    recs: Vec<(u8, Rec)>,
}

/// What went into the log, which is what both readers have to come back with.
struct Wrote {
    kind: u8,
    ttl: Option<u64>,
    key: Vec<u8>,
    value: Vec<u8>,
}

/// A distinct path per iteration, so a crash inside the engine cannot leave a
/// half written file that the next iteration then blames the reader for.
fn next_path() -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "yo-fuzz-file-{}-{}.yo",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_file(&p);
    p
}

/// Deletes the file when the iteration ends, however it ends.
struct Tmp(PathBuf);

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fuzz_target!(|input: Input| {
    // Four shards is enough for the interleaving to matter and small enough
    // that the per shard fixed cost does not dominate the iteration. A hundred
    // and twenty eight records at up to a few hundred bytes each crosses a
    // default page boundary without making every iteration a megabyte of IO.
    let shards = u32::from(input.shards % 4) + 1;
    let path = Tmp(next_path());

    let mut db = match Yo::create(
        &path.0,
        &CreateOptions {
            shard_count: shards,
            file_uuid: [0xa5; 16],
            created_unix_ms: 1_700_000_000_000,
            ..CreateOptions::default()
        },
    ) {
        Ok(db) => db,
        // Out of disk is not a finding.
        Err(_) => return,
    };

    // Bucket the records by shard first, because a log owns its shard's tail
    // and taking the same shard's log twice is documented to hand the second
    // caller an empty history.
    let mut per_shard: Vec<Vec<Wrote>> = (0..shards).map(|_| Vec::new()).collect();
    for (target, r) in input.recs.into_iter().take(128) {
        let s = usize::from(target) % shards as usize;
        // The key length is a u16 on the wire and the interesting sizes are all
        // small. An empty key is legal and worth having in the mix.
        let mut key = r.key;
        key.truncate(192);
        let kind = if r.tombstone {
            RecordKind::Tombstone
        } else {
            RecordKind::String
        };
        let mut value = r.value;
        value.truncate(320);
        // A tombstone carries no value. Writing one with a value would be a
        // file the engine never produces, and the gate is about files the
        // engine produces.
        if r.tombstone {
            value.clear();
        }
        per_shard[s].push(Wrote {
            kind: kind.as_u8(),
            ttl: r.ttl,
            key,
            value,
        });
    }

    let cfg = LogConfig::default();
    let payload_len = cfg.page_len - PAGE_HEADER_LEN;
    let mut entries = vec![yo_format::CheckpointEntry::default(); shards as usize];
    // Kept open past the write so that the engine side replay reads through the
    // same log the writer used, rather than through a fresh scan that could
    // agree with the reader by accident.
    let mut logs = Vec::new();

    for shard in 0..shards {
        let sink = match db.log(shard) {
            Ok(s) => s,
            Err(_) => return,
        };
        let mut log = match Log::new(
            LogConfig {
                shard,
                // The production default. Dropping to `None` was tried and
                // bought nothing: the iteration rate is the same either way,
                // because what costs the time here is creating and deleting a
                // file and the checkpoint's own two syncs, not the commits.
                durability: Durability::Group,
                ..LogConfig::default()
            },
            sink,
        ) {
            Ok(l) => l,
            Err(_) => return,
        };

        for w in &per_shard[shard as usize] {
            let mut h = RecordHeader::new(RecordKind::from_u8(w.kind).expect("a kind we just set"));
            if let Some(ttl) = w.ttl {
                h = h.with_ttl(ttl);
            }
            // A record too big for a page is refused, and refusing it is the
            // right answer rather than a finding. Everything here is well under
            // that, so this only ever fires if the sizes above change.
            if log.append(&h, &w.key, &w.value).is_err() {
                return;
            }
        }
        if log.commit_pending().is_err() {
            return;
        }
        entries[shard as usize] =
            log.checkpoint_entry(0, 0, per_shard[shard as usize].len() as u64);
        logs.push(log);
    }

    if db
        .checkpoint(&Checkpoint {
            clean_shutdown: input.clean,
            unix_ms: 1_700_000_100_000,
            ..Checkpoint::new(&entries)
        })
        .is_err()
    {
        return;
    }

    // The engine side walk, through the log the writer just used.
    let mut engine: Vec<Vec<Wrote>> = (0..shards).map(|_| Vec::new()).collect();
    for (shard, log) in logs.iter().enumerate() {
        let begin = entries[shard].log_begin;
        let rep = replay(log.sink(), payload_len, begin, |_, r| {
            engine[shard].push(Wrote {
                kind: r.kind,
                ttl: r.ttl_ms,
                key: r.key.to_vec(),
                value: r.value.to_vec(),
            });
            Ok(())
        })
        .expect("the engine cannot fail to replay a log it just wrote");
        assert!(
            rep.is_clean(),
            "shard {shard} replay stopped at {:?}: {:?}",
            rep.truncated_at,
            rep.reason
        );
        same(&engine[shard], &per_shard[shard], "engine", shard);
    }
    drop(logs);
    drop(db);

    // The independent side. Everything from here on shares no code with the
    // engine, which is the entire point of the gate.
    let r = Reader::open(&path.0).expect("the reader opens a file the engine just closed");

    let mut mine: Vec<Vec<Wrote>> = (0..shards).map(|_| Vec::new()).collect();
    let mut regions: Vec<_> = r.regions().iter().collect();
    // Regions come back in file order, which is allocation order and therefore
    // interleaved across shards. Each shard's own pages are in address order
    // within that shard, and that is the order the records were written in.
    regions.sort_by_key(|g| (g.header.shard, g.header.page_addr));
    for g in regions {
        assert!(g.is_good(), "region {}: {:?}", g.index, g.damage);
        let shard = g.header.shard as usize;
        assert!(
            shard < mine.len(),
            "region {} claims shard {shard}",
            g.index
        );
        for rec in r.records(g).expect("walk a region the engine wrote") {
            mine[shard].push(Wrote {
                kind: rec.kind,
                ttl: rec.ttl_ms,
                key: rec.key,
                value: rec.value,
            });
        }
    }
    for shard in 0..shards as usize {
        same(&mine[shard], &per_shard[shard], "reader", shard);
    }

    // And the two views of the metadata, which is the half of the format that
    // no record walk would ever notice being wrong.
    let engine_again = Yo::open(&path.0).expect("the engine reopens its own file");
    let sb = r.superblock();
    let esb = engine_again.superblock();
    assert_eq!(sb.seq, esb.seq, "checkpoint sequence");
    assert_eq!(sb.file_size, esb.file_size, "file size");
    assert_eq!(sb.shard_count, esb.shard_count, "shard count");
    assert_eq!(sb.page_size, esb.page_size, "page size");
    assert_eq!(sb.flags, esb.flags, "flags");
    assert_eq!(sb.catalog_addr, esb.catalog_addr, "catalogue address");
    assert_eq!(sb.free_list_addr, esb.free_list_addr, "free list address");
    assert_eq!(sb.clean_shutdown(), input.clean, "the clean shutdown bit");

    let cps = r.checkpoints().expect("checkpoint entries");
    assert_eq!(cps.len(), shards as usize, "one entry per shard");
    for (shard, m) in cps.iter().enumerate() {
        let e = entries[shard];
        assert_eq!(m.log_begin, e.log_begin, "shard {shard} begin");
        assert_eq!(m.log_head, e.log_head, "shard {shard} head");
        assert_eq!(m.log_read_only, e.log_read_only, "shard {shard} read only");
        assert_eq!(m.log_tail, e.log_tail, "shard {shard} tail");
        assert_eq!(m.key_count, e.key_count, "shard {shard} key count");
        assert_eq!(m.epoch, e.epoch, "shard {shard} epoch");
    }
});

/// Compares one side's walk against what was written, and says which side and
/// which record when it does not match.
fn same(got: &[Wrote], want: &[Wrote], side: &str, shard: usize) {
    assert_eq!(
        got.len(),
        want.len(),
        "{side} found {} records on shard {shard}, {} went in",
        got.len(),
        want.len()
    );
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert_eq!(a.key, b.key, "{side} key of record {i} on shard {shard}");
        assert_eq!(
            a.value, b.value,
            "{side} value of record {i} on shard {shard}"
        );
        assert_eq!(a.kind, b.kind, "{side} kind of record {i} on shard {shard}");
        assert_eq!(a.ttl, b.ttl, "{side} ttl of record {i} on shard {shard}");
    }
}
