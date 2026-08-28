//! The runner. `yo-crash --trials 100000` is the M1 exit gate.
//!
//! Prints what it reached as well as what it found, because a run of a hundred
//! thousand trials that reports no violations proves nothing unless it also
//! shows that the faults landed somewhere. A harness whose injection is broken
//! passes every time, silently, and looks exactly like a correct engine.

use std::process::ExitCode;
use std::time::Instant;

use yo_crash::{Shape, trial};
use yo_record::Durability;

const USAGE: &str = "\
yo-crash, fault injection for the log

usage:
  yo-crash [--trials N] [--seed N] [--records N] [--page-len N] [--sync] [--quiet]

  --trials N     how many trials to run. Default 10000, the gate is 100000
  --seed N       where the seed sequence starts. Default 0, so a plain run is
                   the same run on every machine and reproduces on any of them
  --records N    records per trial. Default 200
  --page-len N   physical page size, a multiple of 4096 and larger than one
                   block, so 8192 at the smallest. Default 16384, small enough
                   that a trial reaches a page boundary
  --sync         commit every record instead of grouping. Far fewer bytes in
                   flight when the fault lands, so it reaches a different shape
                   of wreckage than the default
  --quiet        print the summary and nothing else

exit codes:
  0  every trial passed
  1  at least one violation, with the seed that reproduces it
  2  the arguments did not make sense
";

/// Running totals for one population of trials.
#[derive(Default)]
struct Tally {
    trials: u64,
    written: u64,
    acked: u64,
    recovered: u64,
}

impl Tally {
    fn print(&self) {
        println!(
            "  {} trials, {} records written, {} acknowledged durable, {} recovered",
            self.trials, self.written, self.acked, self.recovered
        );
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut trials = 10_000u64;
    let mut seed0 = 0u64;
    let mut shape = Shape::default();
    let mut quiet = false;

    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let mut value = |name: &str| -> Option<u64> {
            i += 1;
            match args.get(i).and_then(|v| v.parse::<u64>().ok()) {
                Some(v) => Some(v),
                None => {
                    eprintln!("yo-crash: {name} needs a number");
                    None
                }
            }
        };
        match a {
            "--trials" => match value("--trials") {
                Some(v) => trials = v,
                None => return ExitCode::from(2),
            },
            "--seed" => match value("--seed") {
                Some(v) => seed0 = v,
                None => return ExitCode::from(2),
            },
            "--records" => match value("--records") {
                Some(v) => shape.records = v as usize,
                None => return ExitCode::from(2),
            },
            "--page-len" => match value("--page-len") {
                Some(v) => shape.page_len = v as usize,
                None => return ExitCode::from(2),
            },
            "--sync" => shape.durability = Durability::Sync,
            "--quiet" => quiet = true,
            "-h" | "--help" => {
                print!("{USAGE}");
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("yo-crash: no such option: {other}\n");
                eprint!("{USAGE}");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }

    if !quiet {
        println!(
            "{trials} trials, {} records each, {} byte pages, seeds {seed0}..{}",
            shape.records,
            shape.page_len,
            seed0 + trials
        );
    }

    let started = Instant::now();
    let mut by_fault: Vec<(&'static str, u64)> = Vec::new();
    let mut failures = 0u64;
    let mut truncated = 0u64;
    let mut lost_something = 0u64;
    // Crash trials and rot trials are counted apart, and that is not a
    // presentation choice. They are judged by different rules: a crash may not
    // lose an acknowledged commit, and rot may. Adding the two together
    // produces a recovered count well below the acknowledged count on a run
    // where nothing went wrong, which reads exactly like the failure this
    // harness exists to detect.
    let mut crash = Tally::default();
    let mut rot = Tally::default();

    for n in 0..trials {
        let seed = seed0.wrapping_add(n);
        let out = match trial::run(seed, shape) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("yo-crash: trial {seed} could not be set up: {e}");
                return ExitCode::from(2);
            }
        };

        let kind = out.fault.kind();
        match by_fault.iter_mut().find(|(k, _)| *k == kind) {
            Some((_, c)) => *c += 1,
            None => by_fault.push((kind, 1)),
        }
        let t = if out.fault.is_crash() {
            &mut crash
        } else {
            &mut rot
        };
        t.trials += 1;
        t.written += out.written as u64;
        t.acked += out.acknowledged as u64;
        t.recovered += out.recovered as u64;
        if out.truncated_at.is_some() {
            truncated += 1;
        }
        if out.recovered < out.written {
            lost_something += 1;
        }

        if !out.passed() {
            failures += 1;
            println!("\nFAILED trial {seed}");
            println!(
                "  reproduce with: yo-crash --trials 1 --seed {seed} --records {} --page-len {}",
                shape.records, shape.page_len
            );
            println!("  fault: {:?}", out.fault);
            println!(
                "  wrote {} records, {} acknowledged durable, {} came back{}",
                out.written,
                out.acknowledged,
                out.recovered,
                match out.truncated_at {
                    Some(at) => format!(", truncated at {at}"),
                    None => ", clean end of log".to_string(),
                }
            );
            for v in &out.violations {
                println!("  {v:?}");
            }
            // Twenty is enough to see a pattern, and past that the output is
            // longer than anyone reads.
            if failures >= 20 {
                println!("\nstopping after 20 failures");
                break;
            }
        }
    }

    let took = started.elapsed();
    if !quiet {
        by_fault.sort_unstable_by_key(|a| std::cmp::Reverse(a.1));
        println!("\nfaults injected");
        for (kind, count) in &by_fault {
            println!("  {kind:<12} {count}");
        }
        println!("\ncrash trials, where an acknowledged commit may not be lost");
        crash.print();
        println!(
            "  recovered is at least acknowledged in every one of them, or this run would have \
             failed"
        );

        println!("\nrot trials, where the data may be gone and may not come back wrong");
        rot.print();
        println!("  recovered below acknowledged here is the bad device doing its job");

        println!(
            "\n{truncated} trials tore a tail, {lost_something} lost at least one record. A run \
             where neither number is large is a run whose faults are landing somewhere harmless."
        );
        println!(
            "{:.0} trials a second, {:.1}s total",
            trials as f64 / took.as_secs_f64().max(1e-9),
            took.as_secs_f64()
        );
    }

    if failures == 0 {
        println!("OK, {trials} trials, no silent corruptions");
        ExitCode::SUCCESS
    } else {
        println!("FAILED: {failures} trials found something");
        ExitCode::FAILURE
    }
}
