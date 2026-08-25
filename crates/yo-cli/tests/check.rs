//! `yo check` against files that are fine and files that are not.
//!
//! The interesting tests here are the ones that break a file in a way no
//! checksum can notice. Anyone can catch a flipped bit. Catching a segment that
//! two shards both think is theirs, where both of them have written perfectly
//! valid records into it, is what a checker is for.

use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use yo_file::{Checkpoint, CreateOptions, Yo};
use yo_format::{CheckpointEntry, PAGE_HEADER_LEN, PageHeader, RecordHeader, RecordKind};
use yo_record::{Durability, Log, LogConfig};

struct Tmp(PathBuf);

impl Tmp {
    fn new(name: &str) -> Tmp {
        let mut p = std::env::temp_dir();
        p.push(format!("yo-check-{name}-{}.yo", std::process::id()));
        let _ = std::fs::remove_file(&p);
        Tmp(p)
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// A file with `count` records in shard 0, checkpointed and closed cleanly.
fn good_file(path: &Path, count: u64, shards: u32) {
    let mut db = Yo::create(
        path,
        &CreateOptions {
            shard_count: shards,
            ..CreateOptions::default()
        },
    )
    .expect("create");
    let sink = db.log(0).expect("log");
    let mut log = Log::new(
        LogConfig {
            shard: 0,
            durability: Durability::Group,
            ..LogConfig::default()
        },
        sink,
    )
    .expect("log");
    let h = RecordHeader::new(RecordKind::String);
    for i in 0..count {
        log.append(&h, format!("key:{i:06}").as_bytes(), b"a value")
            .expect("append");
    }
    log.commit_pending().expect("commit");
    let entry = log.checkpoint_entry(0, 0, count);
    drop(log);

    let mut entries = vec![CheckpointEntry::default(); shards as usize];
    entries[0] = entry;
    db.checkpoint(&Checkpoint {
        clean_shutdown: true,
        unix_ms: 1,
        ..Checkpoint::new(&entries)
    })
    .expect("checkpoint");
    drop(db);
}

/// Runs `yo check` and hands back the exit code and everything it printed.
fn run(path: &Path, extra: &[&str]) -> (i32, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_yo"))
        .arg("check")
        .arg(path)
        .args(extra)
        .output()
        .expect("run yo check");
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.code().unwrap_or(-1), text)
}

/// Overwrites bytes in place.
fn poke(path: &Path, off: u64, bytes: &[u8]) {
    let mut f = std::fs::OpenOptions::new().write(true).open(path).unwrap();
    f.seek(SeekFrom::Start(off)).unwrap();
    f.write_all(bytes).unwrap();
}

#[test]
fn a_good_file_passes() {
    let t = Tmp::new("good");
    good_file(&t.0, 200, 2);

    let (code, out) = run(&t.0, &[]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("OK"), "{out}");
    assert!(out.contains("200 records"), "{out}");
    assert!(!out.contains("ERROR"), "{out}");
}

#[test]
fn a_lost_header_write_does_not_hide_records() {
    // A crash can take the header write and leave the record writes, and this
    // is the file somebody runs the checker on. Counting only as far as `used`
    // would report fewer records than the engine's recovery is going to find,
    // which is a worse answer than saying nothing.
    let t = Tmp::new("staleused");
    good_file(&t.0, 200, 1);

    let off = yo_format::DATA_START;
    let mut head = [0u8; PAGE_HEADER_LEN];
    {
        use std::io::Read;
        let mut f = std::fs::File::open(&t.0).unwrap();
        f.seek(SeekFrom::Start(off)).unwrap();
        f.read_exact(&mut head).unwrap();
    }
    let mut h = PageHeader::decode(&head).expect("the header we are about to spoil");
    // Zero is what a page starts life with, so a header write that never landed
    // leaves exactly this behind.
    h.used = 0;
    h.dead_bytes = 0;
    h.encode(&mut head);
    poke(&t.0, off, &head);

    let (code, out) = run(&t.0, &[]);
    assert!(out.contains("200 records"), "code {code}: {out}");
}

#[test]
fn quick_skips_the_records_and_says_so() {
    let t = Tmp::new("quick");
    good_file(&t.0, 50, 1);

    let (code, out) = run(&t.0, &["--quick"]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("records not walked"), "{out}");
}

#[test]
fn an_empty_file_passes() {
    let t = Tmp::new("emptyok");
    drop(Yo::create(&t.0, &CreateOptions::default()).expect("create"));

    let (code, out) = run(&t.0, &[]);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("0 segments"), "{out}");
}

#[test]
fn a_file_that_is_not_ours_is_a_usage_failure_and_not_a_verdict() {
    let t = Tmp::new("alien");
    std::fs::write(&t.0, vec![0x5au8; 40000]).unwrap();

    // Exit 2, not 1. There is a difference between "this database has problems"
    // and "this is not a database", and a script that treats them the same will
    // eventually delete the wrong thing.
    let (code, out) = run(&t.0, &[]);
    assert_eq!(code, 2, "{out}");
    assert!(out.contains("neither superblock decodes"), "{out}");
}

#[test]
fn a_missing_file_is_a_usage_failure() {
    let (code, out) = run(Path::new("/nowhere/at/all.yo"), &[]);
    assert_eq!(code, 2, "{out}");
}

#[test]
fn a_damaged_spare_slot_is_a_warning_and_not_a_failure() {
    let t = Tmp::new("spare");
    good_file(&t.0, 20, 1);

    // Ask which slot is carrying the file rather than assuming. Which one it is
    // depends on how many checkpoints have run, and a test that hard codes it
    // starts failing the day somebody adds one.
    let live = yo_reader::Reader::open(&t.0).expect("open").live_slot();
    poke(&t.0, (1 - live) as u64 * 16384 + 300, &[0xff; 32]);

    let (code, out) = run(&t.0, &[]);
    assert_eq!(code, 0, "the file still works: {out}");
    assert!(out.contains("warn "), "{out}");
    assert!(out.contains("no spare left"), "{out}");
    assert!(out.contains("OK, with 1 warning"), "{out}");
}

#[test]
fn a_flipped_bit_in_a_record_is_found_and_located() {
    let t = Tmp::new("bitflip");
    good_file(&t.0, 40, 1);

    // Into the first record's value, which is past the page header and the
    // sixteen byte record header and the key.
    poke(&t.0, 32768 + PAGE_HEADER_LEN as u64 + 16 + 10 + 1, &[0x00]);

    let (code, out) = run(&t.0, &[]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("ERROR"), "{out}");
    assert!(out.contains("checksum mismatch"), "{out}");
    assert!(out.contains("at byte"), "{out}");
    assert!(out.contains("FAILED: 1 problem"), "{out}");
}

#[test]
fn two_segments_claiming_the_same_place_is_found() {
    // The corruption a checksum cannot see. Both segments are internally
    // perfect and every record in both of them verifies. The only thing wrong
    // is that they both say they are the same page of the same shard's log,
    // which means one is being written over the other.
    let t = Tmp::new("double");
    good_file(&t.0, 20, 1);

    // Grow the file by one segment and give it a header identical to the first.
    let region_len = 32 * 1024 * 1024u64;
    let second = 32768 + region_len;
    {
        let f = std::fs::OpenOptions::new().write(true).open(&t.0).unwrap();
        f.set_len(second + region_len).unwrap();
    }
    let mut head = [0u8; PAGE_HEADER_LEN];
    PageHeader {
        shard: 0,
        page_addr: 0,
        used: 64,
        dead_bytes: 0,
        epoch: 1,
    }
    .encode(&mut head);
    poke(&t.0, second, &head);

    let (code, out) = run(&t.0, &["--quick"]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("both claim log address 0"), "{out}");
    assert!(out.contains("written over"), "{out}");
}

#[test]
fn a_segment_that_belongs_to_no_shard_is_found() {
    let t = Tmp::new("noshard");
    good_file(&t.0, 20, 2);

    let mut head = [0u8; PAGE_HEADER_LEN];
    PageHeader {
        shard: 9,
        page_addr: 0,
        used: 64,
        dead_bytes: 0,
        epoch: 1,
    }
    .encode(&mut head);
    poke(&t.0, 32768, &head);

    let (code, out) = run(&t.0, &["--quick"]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("belongs to shard 9"), "{out}");
    assert!(out.contains("the file has 2"), "{out}");
}

#[test]
fn a_log_that_ends_before_the_checkpoint_says_it_does_is_found() {
    // The checkpoint promises committed data up to some address. The segment
    // says it holds less than that. One of the two is lying and either way
    // somebody has lost an acknowledged write, which is the worst thing this
    // tool can find.
    let t = Tmp::new("shorttail");
    good_file(&t.0, 100, 1);

    let mut head = [0u8; PAGE_HEADER_LEN];
    PageHeader {
        shard: 0,
        page_addr: 0,
        used: 64,
        dead_bytes: 0,
        epoch: 1,
    }
    .encode(&mut head);
    poke(&t.0, 32768, &head);

    let (code, out) = run(&t.0, &["--quick"]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("committed log are gone"), "{out}");
}

#[test]
fn a_truncated_file_is_found() {
    let t = Tmp::new("truncated");
    good_file(&t.0, 100, 1);

    let len = std::fs::metadata(&t.0).unwrap().len();
    let f = std::fs::OpenOptions::new().write(true).open(&t.0).unwrap();
    f.set_len(len - 4096).unwrap();
    drop(f);

    let (code, out) = run(&t.0, &["--quick"]);
    assert_eq!(code, 1, "{out}");
    assert!(out.contains("missing from the end"), "{out}");
}

#[test]
fn the_help_comes_out_without_a_file() {
    let out = Command::new(env!("CARGO_BIN_EXE_yo"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("yo check FILE"), "{text}");
    assert!(text.contains("Never writes"), "{text}");

    let out = Command::new(env!("CARGO_BIN_EXE_yo"))
        .arg("check")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));

    let out = Command::new(env!("CARGO_BIN_EXE_yo"))
        .arg("frobnicate")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("no such command"),
        "it should say which"
    );
}

#[test]
fn checking_never_changes_the_file() {
    // Stated in the documentation, so it should be true in a way that fails
    // loudly if somebody adds a repair flag and wires it up by accident.
    let t = Tmp::new("readonly");
    good_file(&t.0, 60, 1);
    let before = std::fs::read(&t.0).unwrap();

    let (code, _) = run(&t.0, &[]);
    assert_eq!(code, 0);

    let after = std::fs::read(&t.0).unwrap();
    assert_eq!(before.len(), after.len(), "the size changed");
    assert!(before == after, "the bytes changed");
}
