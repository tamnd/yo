//! What the cold form costs on a graph somebody else made.
//!
//! The bits an edge numbers in `csr.rs` are measured on graphs this repo
//! generates, which is the right way to hold the encoder to account but the
//! wrong way to answer whether the target in `11` is met. R-MAT is a synthetic
//! stand in and it is known to cluster less than the social graphs it stands in
//! for, so the honest number needs a real edge list.
//!
//! ```text
//! cargo run --release -p yo-graph --example compress -- soc-LiveJournal1.txt
//! ```
//!
//! The format is what SNAP publishes and what almost every public graph is
//! distributed as: two whitespace separated node ids a line, lines starting
//! with `#` ignored. Ids are taken as they come and the graph is sized to the
//! largest of them, so an id space with holes in it costs empty runs, which is
//! what it would cost in production too.
//!
//! It prints the hot form for comparison, the cold form as the ids arrive, and
//! the cold form after [`order_by_degree`](yo_graph::csr::order_by_degree),
//! because on a graph with hubs the ordering is worth more than anything in the
//! encoder.

use std::time::Instant;
use yo_graph::{Adjacency, Csr, csr};

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: compress <edge list> [--no-hot] [--codes]");
        std::process::exit(2);
    };
    let flags: Vec<String> = args.collect();
    let hot_too = !flags.iter().any(|a| a == "--no-hot");
    let codes = flags.iter().any(|a| a == "--codes");

    let t = Instant::now();
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        std::process::exit(1);
    });
    let mut edges: Vec<(u32, u32)> = Vec::new();
    let mut top = 0u32;
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let mut f = line.split_whitespace();
        let (Some(s), Some(d)) = (f.next(), f.next()) else {
            continue;
        };
        let (s, d): (u32, u32) = (s.parse().unwrap(), d.parse().unwrap());
        top = top.max(s).max(d);
        edges.push((s, d));
    }
    drop(text);
    let nodes = top + 1;
    let m = edges.len() as f64;
    println!(
        "{path}: {nodes} nodes, {} edges, read in {:?}",
        edges.len(),
        t.elapsed()
    );

    if hot_too {
        let t = Instant::now();
        let mut hot = Adjacency::out_only();
        for (s, d) in &edges {
            hot.link(u64::from(*s), u64::from(*d), 1, 0);
        }
        let built = t.elapsed();
        let raw = hot.bytes() as f64 / m;
        hot.compact();
        println!(
            "hot                {raw:6.2} bytes an edge, {:6.2} swept, built in {built:?}",
            hot.bytes() as f64 / m
        );
    }

    println!(
        "{:19}{:>7}{:>9}{:>9}{:>9}{:>8}{:>8}{:>8}{:>8}",
        "", "total", "offsets", "degrees", "firsts", "widths", "gaps", "groups", "slack"
    );

    let t = Instant::now();
    let cold = Csr::build(nodes, &mut edges.clone());
    report("cold", &cold, m, t.elapsed());

    let t = Instant::now();
    let to = csr::order_by_degree(nodes, &edges);
    csr::renumber(&mut edges, &to);
    let ordered = t.elapsed();
    let cold = Csr::build(nodes, &mut edges);
    report("cold, deg ordered", &cold, m, ordered);

    if codes {
        edges.sort_unstable();
        gap_shapes(&edges, m);
    }
}

/// What the gaps look like and what other codes would charge for them, none of
/// which changes the format. This is the diagnosis that says whether the
/// payload is close to what the graph is or whether the encoder is leaving
/// something behind, and on a real graph it is the only way to know which.
fn gap_shapes(edges: &[(u32, u32)], m: f64) {
    let mut lens = [0u64; 33];
    let (mut ones, mut gaps) = (0u64, 0u64);
    // What each alternative would charge, in bits.
    let (mut b32, mut b16, mut b8, mut pfor, mut delta) = (0u64, 0u64, 0u64, 0u64, 0u64);
    // Neighbours covered by a consecutive run of four or more, which is what
    // WebGraph's interval encoding takes out before it codes anything.
    let mut in_intervals = 0u64;

    let mut run: Vec<u32> = Vec::new();
    let mut i = 0usize;
    while i < edges.len() {
        let src = edges[i].0;
        run.clear();
        while i < edges.len() && edges[i].0 == src {
            run.push(edges[i].1);
            i += 1;
        }
        let mut streak = 1u64;
        for j in 1..run.len() {
            let g = run[j] - run[j - 1];
            gaps += 1;
            lens[bits(g) as usize] += 1;
            if g == 1 {
                ones += 1;
                streak += 1;
            } else {
                if streak >= 4 {
                    in_intervals += streak;
                }
                streak = 1;
            }
            delta += u64::from(elias_delta(g));
        }
        if streak >= 4 {
            in_intervals += streak;
        }
        let g: Vec<u32> = run.windows(2).map(|p| p[1] - p[0]).collect();
        b32 += blocked(&g, 32);
        b16 += blocked(&g, 16);
        b8 += blocked(&g, 8);
        pfor += patched(&g, 32);
    }

    // The entropy of the gaps under a coder that knows how long they are and
    // nothing else, which is the floor for anything that codes each gap on its
    // own. Everything below it has to come from the structure between runs.
    let n = gaps as f64;
    let mut h = 0.0f64;
    for (l, c) in lens.iter().enumerate() {
        if *c > 0 {
            let p = *c as f64 / n;
            h += p * (-p.log2() + (l.max(1) - 1) as f64);
        }
    }

    println!();
    println!("what the gaps look like, on the degree ordered graph");
    println!(
        "gaps                 {gaps}, {:.1}% of them 1",
        ones as f64 * 100.0 / n
    );
    println!(
        "in runs of 4 or more {:.1}% of all neighbours",
        in_intervals as f64 * 100.0 / m
    );
    println!("by length floor      {h:6.2} bits a gap");
    println!(
        "block of 32          {:6.2} bits a gap   (this is the format)",
        b32 as f64 / n
    );
    println!("block of 16          {:6.2}", b16 as f64 / n);
    println!("block of 8           {:6.2}", b8 as f64 / n);
    println!("block of 32, patched {:6.2}", pfor as f64 / n);
    println!("elias delta          {:6.2}", delta as f64 / n);
}

fn bits(v: u32) -> u32 {
    32 - v.leading_zeros()
}

/// A width per block, which is what the format does.
fn blocked(gaps: &[u32], block: usize) -> u64 {
    gaps.chunks(block)
        .map(|c| 6 + c.len() as u64 * u64::from(bits(c.iter().copied().max().unwrap_or(0))))
        .sum()
}

/// A width per block with the outliers left behind, at the width that comes out
/// cheapest.
///
/// Counted off a histogram of the widths in the block rather than by filtering
/// the block once per candidate width, because this runs over every gap in the
/// graph and the naive shape is quadratic in the block.
fn patched(gaps: &[u32], block: usize) -> u64 {
    let mut hist = [0u32; 33];
    gaps.chunks(block)
        .map(|c| {
            hist.fill(0);
            let mut top = 0u32;
            for g in c {
                let b = bits(*g);
                hist[b as usize] += 1;
                top = top.max(b);
            }
            // Walking the candidate width down from the top, `over` is how many
            // gaps do not fit and `wide` is the widest of them.
            let (mut over, mut wide) = (0u64, 0u32);
            let mut best = u64::MAX;
            for w in (0..=top).rev() {
                let cost = 6
                    + 5
                    + if over == 0 { 0 } else { 6 }
                    + c.len() as u64 * u64::from(w)
                    + over * (5 + u64::from(wide));
                best = best.min(cost);
                if w > 0 {
                    over += u64::from(hist[w as usize]);
                    if hist[w as usize] > 0 {
                        wide = wide.max(w);
                    }
                }
            }
            best
        })
        .sum()
}

/// Elias delta, which is the self delimiting code that does best on a gap
/// distribution with a long tail.
fn elias_delta(g: u32) -> u32 {
    let v = g + 1;
    let n = bits(v) - 1;
    let k = bits(n + 1) - 1;
    2 * k + 1 + n
}

fn report(name: &str, cold: &Csr, m: f64, took: std::time::Duration) {
    let c = cold.cost();
    let per = |bits: u64| bits as f64 / m;
    println!(
        "{name:19}{:>7.2}{:>9.2}{:>9.2}{:>9.2}{:>9.2}{:>8.2}{:>8.2}{:>8.2}   {:>6.1} MB in {took:?}",
        cold.bits_per_edge(),
        per(c.offsets),
        per(c.degrees),
        per(c.firsts),
        per(c.widths),
        per(c.gaps),
        per(c.groups),
        per(c.slack),
        cold.bytes() as f64 / (1 << 20) as f64,
    );
}
