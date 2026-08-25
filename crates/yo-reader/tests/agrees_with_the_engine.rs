//! The M1 exit gate, in test form: the independent reader agrees with the
//! engine about the files the engine writes.
//!
//! Every other test in this crate checks the reader against itself, which is
//! worth having and is not evidence of anything. These build real files with
//! `yo-file` and `yo-record` and then read them with code that shares nothing
//! with either. A disagreement here means one of the two transcriptions of the
//! format is wrong, and until this test exists there is no way for the project
//! to find that out.

use std::path::PathBuf;

use yo_file::{Checkpoint, CreateOptions, Yo};
use yo_format::{RecordHeader, RecordKind, record_flags};
use yo_reader::{Reader, SlotStatus};
use yo_record::{Durability, Log, LogConfig};

/// A file that deletes itself.
struct Tmp(PathBuf);

impl Tmp {
    fn new(name: &str) -> Tmp {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "yo-reader-agree-{name}-{}-{:?}.yo",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&p);
        Tmp(p)
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Writes `count` records into shard 0 and checkpoints, then returns what went
/// in so the reader's answer can be compared against it.
fn write_a_file(path: &std::path::Path, count: u64, shards: u32) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut db = Yo::create(
        path,
        &CreateOptions {
            shard_count: shards,
            file_uuid: [7; 16],
            created_unix_ms: 1_700_000_000_000,
            ..CreateOptions::default()
        },
    )
    .expect("create");

    let sink = db.log(0).expect("a log for shard 0");
    let mut log = Log::new(
        LogConfig {
            shard: 0,
            durability: Durability::Group,
            ..LogConfig::default()
        },
        sink,
    )
    .expect("a log");

    let mut wrote = Vec::new();
    for i in 0..count {
        let key = format!("key:{i:08}").into_bytes();
        // Values of every length, so a record whose total is not a multiple of
        // eight turns up. Walking by `len` instead of by `stride` survives a
        // suite where every value happens to be the same size, and this is
        // where that would show.
        let value: Vec<u8> = (0..(i % 37) as usize).map(|b| (b as u8) ^ 0x5a).collect();
        let mut h = RecordHeader::new(RecordKind::String);
        // Some with a TTL, so the eight extra header bytes get exercised on
        // both sides.
        if i % 5 == 0 {
            h = h.with_ttl(1_800_000_000_000 + i);
        }
        log.append(&h, &key, &value).expect("append");
        wrote.push((key, value));
    }
    log.commit_pending().expect("commit");

    let entry = log.checkpoint_entry(0, 0, count);
    let mut entries = vec![yo_format::CheckpointEntry::default(); shards as usize];
    entries[0] = entry;
    drop(log);

    db.checkpoint(&Checkpoint {
        clean_shutdown: true,
        unix_ms: 1_700_000_100_000,
        ..Checkpoint::new(&entries)
    })
    .expect("checkpoint");
    drop(db);

    wrote
}

#[test]
fn the_superblock_reads_the_same_from_both_sides() {
    let t = Tmp::new("sb");
    write_a_file(&t.0, 40, 4);

    let r = Reader::open(&t.0).expect("the independent reader opens it");
    let sb = r.superblock();

    assert_eq!(sb.format_version, yo_format::FORMAT_VERSION);
    assert_eq!(sb.page_size, yo_format::DEFAULT_PAGE_SIZE);
    assert_eq!(sb.shard_count, 4);
    assert_eq!(sb.db_count, 16);
    assert_eq!(sb.file_uuid, [7; 16]);
    assert_eq!(sb.created_unix_ms, 1_700_000_000_000);
    assert_eq!(sb.checkpoint_unix_ms, 1_700_000_100_000);
    assert!(sb.clean_shutdown(), "the file was closed on purpose");

    // Both slots are written at create time, so both should decode even though
    // only one is live.
    assert!(r.slots()[0].is_good(), "slot A: {:?}", r.slots()[0]);
    assert!(r.slots()[1].is_good(), "slot B: {:?}", r.slots()[1]);

    // And the engine's own view of the same file matches, field for field on
    // everything both types carry.
    let engine = Yo::open(&t.0).expect("the engine opens it");
    let e = engine.superblock();
    assert_eq!(sb.seq, e.seq);
    assert_eq!(sb.file_size, e.file_size);
    assert_eq!(sb.shard_count, e.shard_count);
    assert_eq!(sb.page_size, e.page_size);
    assert_eq!(sb.flags, e.flags);
    assert_eq!(sb.catalog_addr, e.catalog_addr);
    assert_eq!(sb.free_list_addr, e.free_list_addr);
    assert_eq!(sb.shard_table_off, e.shard_table_off);
    assert_eq!(sb.shard_table_len, e.shard_table_len);
}

#[test]
fn every_record_comes_back_byte_for_byte() {
    let t = Tmp::new("records");
    let wrote = write_a_file(&t.0, 500, 1);

    let r = Reader::open(&t.0).expect("open");
    let mut got = Vec::new();
    for region in r.regions() {
        assert!(
            region.is_good(),
            "region {}: {:?}",
            region.index,
            region.damage
        );
        got.extend(r.records(region).expect("walk the region"));
    }

    assert_eq!(got.len(), wrote.len(), "record count");
    for (i, (rec, (key, value))) in got.iter().zip(&wrote).enumerate() {
        assert_eq!(&rec.key, key, "key of record {i}");
        assert_eq!(&rec.value, value, "value of record {i}");
        assert_eq!(rec.kind, RecordKind::String.as_u8(), "kind of record {i}");
        assert!(
            rec.flags & record_flags::CHECKSUMMED != 0,
            "record {i} should be checksummed"
        );
        let wants_ttl = i % 5 == 0;
        assert_eq!(
            rec.ttl_ms.is_some(),
            wants_ttl,
            "ttl presence on record {i}"
        );
        if wants_ttl {
            assert_eq!(rec.ttl_ms, Some(1_800_000_000_000 + i as u64));
        }
    }
}

#[test]
fn the_checkpoint_entries_agree() {
    let t = Tmp::new("cp");
    write_a_file(&t.0, 120, 3);

    let r = Reader::open(&t.0).expect("open");
    let mine = r.checkpoints().expect("checkpoints");
    assert_eq!(mine.len(), 3);
    assert_eq!(mine[0].key_count, 120);
    assert!(mine[0].log_tail > 0, "shard 0 wrote something");
    // The shards that never ran are all zeroes, which is ordered and therefore
    // legal, and is what an untouched shard looks like.
    assert_eq!(mine[1], yo_reader::CheckpointEntry::default());
    assert_eq!(mine[2], yo_reader::CheckpointEntry::default());

    let engine = Yo::open(&t.0).expect("engine open");
    for (shard, m) in mine.iter().enumerate() {
        let e = engine
            .checkpoint_entry(shard as u32)
            .expect("the engine has an entry for every shard");
        assert_eq!(m.log_begin, e.log_begin, "shard {shard} begin");
        assert_eq!(m.log_head, e.log_head, "shard {shard} head");
        assert_eq!(m.log_read_only, e.log_read_only, "shard {shard} read_only");
        assert_eq!(m.log_tail, e.log_tail, "shard {shard} tail");
        assert_eq!(m.key_count, e.key_count, "shard {shard} keys");
        assert_eq!(m.epoch, e.epoch, "shard {shard} epoch");
    }
}

#[test]
fn the_two_checksums_are_the_same_function() {
    // The reader builds its table from the polynomial and the engine uses the
    // CPU instruction where it has one. If these ever disagree, every checksum
    // in the file is a coin flip, so it is worth checking over more than the
    // one published test vector.
    let mut data = Vec::new();
    for n in 0..600usize {
        data.push((n * 31 + 7) as u8);
        assert_eq!(
            yo_reader::crc::crc32c(0, &data),
            yo_common::crc32c(0, &data),
            "over {n} bytes"
        );
    }
    // And with a seed, which is the path the record trailer takes.
    assert_eq!(
        yo_reader::crc::crc32c(0xdead_beef, &data),
        yo_common::crc32c(0xdead_beef, &data)
    );
}

#[test]
fn an_empty_file_is_a_file_with_nothing_in_it() {
    let t = Tmp::new("empty");
    drop(Yo::create(&t.0, &CreateOptions::default()).expect("create"));

    let r = Reader::open(&t.0).expect("open");
    assert_eq!(r.regions().len(), 0);
    assert_eq!(r.checkpoints().expect("checkpoints").len(), 1);
    assert_eq!(r.shard_table().expect("shard table").len(), 16384);
    assert!(
        r.shard_table()
            .expect("shard table")
            .iter()
            .all(|&s| s == 0),
        "an empty table means shard 0 owns everything"
    );
}

#[test]
fn a_torn_slot_does_not_stop_the_reader() {
    let t = Tmp::new("torn");
    write_a_file(&t.0, 30, 1);

    // The live slot after a checkpoint is B, so damaging A leaves the file
    // readable and is what a crash halfway through a root flip looks like.
    {
        use std::io::{Seek, SeekFrom, Write};
        let mut f = std::fs::OpenOptions::new().write(true).open(&t.0).unwrap();
        f.seek(SeekFrom::Start(200)).unwrap();
        f.write_all(&[0xff; 64]).unwrap();
    }

    let r = Reader::open(&t.0).expect("the surviving slot carries it");
    assert!(matches!(r.slots()[0], SlotStatus::Bad(_)), "slot A is gone");
    assert!(r.slots()[1].is_good(), "slot B survived");
    assert_eq!(r.live_slot(), 1);

    let mut n = 0;
    for region in r.regions() {
        n += r.records(region).expect("walk").len();
    }
    assert_eq!(n, 30, "the records are still all there");
}

/// Rewrites one region's header with a different `used`, leaving the payload
/// alone. This is what a crash that loses the header write but keeps the record
/// writes leaves behind, and there is no way to produce it through the engine's
/// own API because the engine writes the two together.
fn stale_used(path: &std::path::Path, offset: u64, used: u32) {
    use std::io::{Read, Seek, SeekFrom, Write};
    let mut f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    let mut head = [0u8; 32];
    f.seek(SeekFrom::Start(offset)).unwrap();
    f.read_exact(&mut head).unwrap();
    let mut h = yo_format::PageHeader::decode(&head).expect("the header we are about to spoil");
    h.used = used;
    h.dead_bytes = 0;
    h.encode(&mut head);
    f.seek(SeekFrom::Start(offset)).unwrap();
    f.write_all(&head).unwrap();
}

#[test]
fn a_stale_used_mark_does_not_hide_records() {
    // Enough records to run past the first read, which is sized from `used` and
    // rounded up to 64 KiB. At roughly fifty bytes each this is a few hundred
    // KiB of payload, so the walk has to ask for more of the segment several
    // times over rather than once.
    let t = Tmp::new("stale-used");
    let wrote = write_a_file(&t.0, 4000, 1);

    let r = Reader::open(&t.0).expect("open");
    let region = r.regions()[0].clone();
    let real = region.header.used;
    let all = r.records(&region).expect("walk").len();
    assert_eq!(all, wrote.len(), "the file is intact to begin with");
    assert!(
        real > 64 * 1024,
        "under one read block of payload, so this would pass without the walk ever asking for more of the segment: used {real}"
    );
    drop(r);

    // Zero is the header a page starts life with, so a lost header write leaves
    // exactly this. Half is a header from an earlier flush of the same page.
    for used in [0, real / 2, real - 1] {
        stale_used(&t.0, region.offset, used);
        let r = Reader::open(&t.0).expect("open");
        let got = r.records(&r.regions()[0]).expect("walk");
        assert_eq!(
            got.len(),
            wrote.len(),
            "used {used} of {real} hid {} records",
            wrote.len() - got.len()
        );
        for (i, (rec, (key, value))) in got.iter().zip(&wrote).enumerate() {
            assert_eq!(&rec.key, key, "key of record {i} at used {used}");
            assert_eq!(&rec.value, value, "value of record {i} at used {used}");
        }
    }
}

#[test]
fn a_flipped_bit_in_a_value_is_caught() {
    let t = Tmp::new("flip");
    write_a_file(&t.0, 20, 1);

    let r = Reader::open(&t.0).expect("open");
    // Find a record with a value in it and corrupt a byte of it in the file.
    let region = r.regions()[0].clone();
    let before = r.records(&region).expect("walk");
    let target = before
        .iter()
        .position(|rec| !rec.value.is_empty())
        .expect("some record has a value");
    drop(r);

    // Walk forward by stride to find where that record's value sits.
    let mut at = region.offset + 32;
    for rec in &before[..target] {
        at += rec.stride() as u64;
    }
    let value_at = at + 16 + before[target].key.len() as u64;
    {
        use std::io::{Seek, SeekFrom, Write};
        let mut f = std::fs::OpenOptions::new().write(true).open(&t.0).unwrap();
        f.seek(SeekFrom::Start(value_at)).unwrap();
        let mut b = [0u8; 1];
        f.write_all({
            b[0] = before[target].value[0] ^ 0x01;
            &b
        })
        .unwrap();
    }

    let r = Reader::open(&t.0).expect("the file still opens");
    let e = r
        .records(&r.regions()[0])
        .expect_err("the walk should fail");
    assert!(e.to_string().contains("checksum mismatch"), "{e}");
    assert!(e.offset().is_some(), "the error says where: {e}");
}

#[test]
fn the_dump_binary_prints_what_is_in_the_file() {
    // A library that works and a binary that does not is a thing nobody notices
    // until they reach for the binary, which by definition is a bad moment.
    let t = Tmp::new("dump");
    write_a_file(&t.0, 12, 2);

    let out = std::process::Command::new(env!("CARGO_BIN_EXE_yo-dump"))
        .arg(&t.0)
        .arg("--records")
        .output()
        .expect("run yo-dump");
    assert!(
        out.status.success(),
        "yo-dump failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).expect("utf8");

    assert!(text.contains("2 shards"), "{text}");
    assert!(text.contains("clean_shutdown"), "{text}");
    assert!(text.contains("slot A      good"), "{text}");
    assert!(text.contains("slot B      good"), "{text}");
    assert!(text.contains("key:00000007"), "{text}");
    assert!(text.contains("12 records"), "{text}");
}

#[test]
fn the_dump_binary_says_no_to_a_file_that_is_not_ours() {
    let t = Tmp::new("notours");
    std::fs::write(&t.0, vec![0x41u8; 40000]).unwrap();

    let out = std::process::Command::new(env!("CARGO_BIN_EXE_yo-dump"))
        .arg(&t.0)
        .output()
        .expect("run yo-dump");
    assert!(!out.status.success(), "it should not have liked that");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("neither superblock decodes"), "{err}");
    assert!(err.contains("not the magic"), "{err}");
}
