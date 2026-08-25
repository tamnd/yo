//! Prints what an independent reader makes of a `.yo` file.
//!
//! This is the thing you run when a file will not open and you want a second
//! opinion from something that does not share a line of code with whatever
//! refused it. It writes nothing and repairs nothing.
//!
//! ```text
//! yo-dump FILE [--records] [--limit N]
//! ```
//!
//! By default it prints the superblock, both slot verdicts, the checkpoint
//! entries and one line per region. With `--records` it walks the records in
//! every region too, which reads the used part of each one.

use std::path::PathBuf;
use std::process::ExitCode;

use yo_reader::format::superblock_flags;
use yo_reader::{Reader, SlotStatus};

fn main() -> ExitCode {
    let mut path: Option<PathBuf> = None;
    let mut records = false;
    let mut limit = usize::MAX;

    let mut args = std::env::args_os().skip(1);
    while let Some(a) = args.next() {
        match a.to_string_lossy().as_ref() {
            "--records" => records = true,
            "--limit" => {
                let Some(n) = args.next() else {
                    eprintln!("--limit wants a number");
                    return ExitCode::from(2);
                };
                match n.to_string_lossy().parse() {
                    Ok(n) => limit = n,
                    Err(e) => {
                        eprintln!("--limit wants a number: {e}");
                        return ExitCode::from(2);
                    }
                }
            }
            "-h" | "--help" => {
                println!("yo-dump FILE [--records] [--limit N]");
                return ExitCode::SUCCESS;
            }
            _ if path.is_none() => path = Some(PathBuf::from(a)),
            other => {
                eprintln!("unexpected argument: {other}");
                return ExitCode::from(2);
            }
        }
    }

    let Some(path) = path else {
        eprintln!("yo-dump FILE [--records] [--limit N]");
        return ExitCode::from(2);
    };

    match run(&path, records, limit) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{}: {e}", path.display());
            ExitCode::FAILURE
        }
    }
}

fn run(path: &std::path::Path, records: bool, limit: usize) -> yo_reader::Result<()> {
    let r = Reader::open(path)?;
    let sb = r.superblock();

    println!("file        {}", path.display());
    println!(
        "format      version {}, needs reader {}",
        sb.format_version, sb.min_reader_version
    );
    println!(
        "geometry    {} byte segments, {} shards, {} databases",
        sb.page_size, sb.shard_count, sb.db_count
    );
    println!("sequence    {} (slot {})", sb.seq, r.live_slot());
    println!("size        {} bytes at the checkpoint", sb.file_size);
    println!("uuid        {}", hex(&sb.file_uuid));
    println!("created     {} ms", sb.created_unix_ms);
    println!("checkpoint  {} ms", sb.checkpoint_unix_ms);
    println!("flags       {}", flags(sb.flags));
    println!("catalog     {}", addr(sb.catalog_addr));
    println!("free list   {}", addr(sb.free_list_addr));
    println!("archival    {}", addr(sb.archival_root));

    for (i, s) in r.slots().iter().enumerate() {
        let name = if i == 0 { "A" } else { "B" };
        match s {
            SlotStatus::Good { seq } => println!("slot {name}      good, sequence {seq}"),
            SlotStatus::Bad(e) => println!("slot {name}      BAD: {e}"),
        }
    }

    println!();
    println!("checkpoints");
    for (shard, e) in r.checkpoints()?.iter().enumerate() {
        println!(
            "  shard {shard:<4} begin {} head {} read_only {} tail {} keys {} epoch {}",
            e.log_begin, e.log_head, e.log_read_only, e.log_tail, e.key_count, e.epoch
        );
    }

    println!();
    println!("regions ({} written)", r.regions().len());
    let mut total = 0usize;
    for region in r.regions() {
        if let Some(d) = &region.damage {
            println!(
                "  region {:<5} offset {:<12} DAMAGED: {d}",
                region.index, region.offset
            );
            continue;
        }
        let h = &region.header;
        println!(
            "  region {:<5} offset {:<12} shard {:<4} page_addr {:<12} used {:<10} dead {:<10} epoch {}",
            region.index, region.offset, h.shard, h.page_addr, h.used, h.dead_bytes, h.epoch
        );
        if !records {
            continue;
        }
        match r.records(region) {
            Ok(rs) => {
                total += rs.len();
                for rec in rs.iter().take(limit) {
                    println!(
                        "      kind {:<3} flags {:#04x} len {:<7} key {:<28} value {} bytes{}",
                        rec.kind,
                        rec.flags,
                        rec.len,
                        printable(&rec.key),
                        rec.value.len(),
                        rec.ttl_ms.map_or(String::new(), |t| format!(" ttl {t}"))
                    );
                }
                if rs.len() > limit {
                    println!("      ... {} more", rs.len() - limit);
                }
            }
            // A region that will not walk is worth printing and carrying on
            // from. The next region is independent of this one, and the whole
            // reason to reach for this tool is to find out how much of the file
            // is still there.
            Err(e) => println!("      WALK FAILED: {e}"),
        }
    }
    if records {
        println!();
        println!("{total} records");
    }
    Ok(())
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn addr(a: u64) -> String {
    if a == 0 {
        "none".to_string()
    } else {
        a.to_string()
    }
}

fn flags(f: u32) -> String {
    let mut out = Vec::new();
    if f & superblock_flags::CLEAN_SHUTDOWN != 0 {
        out.push("clean_shutdown");
    }
    if f & superblock_flags::ENCRYPTED != 0 {
        out.push("encrypted");
    }
    if f & superblock_flags::HAS_ARCHIVAL != 0 {
        out.push("has_archival");
    }
    if f & superblock_flags::TIERING_ENGAGED != 0 {
        out.push("tiering_engaged");
    }
    if out.is_empty() {
        // Worth saying out loud rather than printing an empty line, because on
        // a file that was not closed cleanly this is the field somebody is
        // looking for.
        return format!("{f:#010x} (no clean shutdown)");
    }
    format!("{f:#010x} {}", out.join(" "))
}

fn printable(k: &[u8]) -> String {
    if k.is_empty() {
        return "(none)".to_string();
    }
    let s: String = k
        .iter()
        .take(24)
        .map(|&b| {
            if b.is_ascii_graphic() || b == b' ' {
                b as char
            } else {
                '.'
            }
        })
        .collect();
    if k.len() > 24 { format!("{s}...") } else { s }
}
