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
//! It prints the hot form for comparison, the cold form as the ids arrive, the
//! cold form after [`order_by_degree`](yo_graph::csr::order_by_degree), and with
//! `--bisect` the cold form after [`bisect::order`](yo_graph::bisect::order),
//! which `--order <file>` will write out and read back so that a numbering that
//! took minutes can be asked more than one question,
//! because on a graph with hubs the ordering is worth more than anything in the
//! encoder. `--codes` says what the gaps under each of those look like and what
//! other codes would charge for them.

use std::time::Instant;
use yo_graph::{Adjacency, Csr, bisect, csr};

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: compress <edge list> [--no-hot] [--codes] [--bisect] [--order <file>]");
        std::process::exit(2);
    };
    let flags: Vec<String> = args.collect();
    let hot_too = !flags.iter().any(|a| a == "--no-hot");
    let codes = flags.iter().any(|a| a == "--codes");
    let bisected = flags.iter().any(|a| a == "--bisect");
    // A bisection of a real graph is minutes, and the encoder questions asked of
    // one are seconds, so the numbering can be written out once and read back.
    let keep = flags
        .iter()
        .position(|a| a == "--order")
        .and_then(|i| flags.get(i + 1))
        .cloned();

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
        "{:19}{:>7}{:>9}{:>9}{:>9}{:>8}{:>8}{:>9}{:>8}{:>8}",
        "", "total", "offsets", "degrees", "firsts", "widths", "gaps", "patches", "groups", "slack"
    );

    // The time reported against a row is what the ordering took, since that is
    // the part that differs; the build itself is the same work every time.
    let show = |name: &str, e: &[(u32, u32)], took: Option<std::time::Duration>| {
        let mut copy = e.to_vec();
        let t = Instant::now();
        let cold = Csr::build(nodes, &mut copy);
        report(name, &cold, m, took.unwrap_or_else(|| t.elapsed()));
        if codes {
            gap_shapes(name, &copy, m);
        }
    };

    show("cold", &edges, None);

    let t = Instant::now();
    let to = csr::order_by_degree(nodes, &edges);
    let ordered = t.elapsed();
    let mut deg = edges.clone();
    csr::renumber(&mut deg, &to);
    show("cold, deg ordered", &deg, Some(ordered));

    if bisected {
        let t = Instant::now();
        let to = match keep.as_deref().map(std::fs::read) {
            Some(Ok(raw)) => raw
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(u32::from_le_bytes)
                .collect(),
            _ => {
                let to = bisect::order(nodes, &edges);
                if let Some(path) = keep.as_deref() {
                    let mut raw = Vec::with_capacity(to.len() * 4);
                    for v in &to {
                        raw.extend_from_slice(&v.to_le_bytes());
                    }
                    let _ = std::fs::write(path, raw);
                }
                to
            }
        };
        let took = t.elapsed();
        let mut b = edges.clone();
        csr::renumber(&mut b, &to);
        show("cold, bisected", &b, Some(took));
    }
}

/// What the gaps look like and what other codes would charge for them, none of
/// which changes the format. This is the diagnosis that says whether the
/// payload is close to what the graph is or whether the encoder is leaving
/// something behind, and on a real graph it is the only way to know which.
fn gap_shapes(name: &str, edges: &[(u32, u32)], m: f64) {
    let mut lens = [0u64; 33];
    let (mut ones, mut gaps) = (0u64, 0u64);
    // What each alternative would charge, in bits.
    let (mut b32, mut b16, mut b8, mut pfor, mut delta) = (0u64, 0u64, 0u64, 0u64, 0u64);
    // What the same gaps cost if the consecutive stretches come out as intervals
    // first, which is what WebGraph does and what only pays once the numbering
    // puts neighbours next to each other.
    let (mut i32s, mut i8s) = (0u64, 0u64);
    // A width per block again, but with each run free to say how long its blocks
    // are, which costs two bits a run and lets a hub keep long blocks while a
    // clustered run cuts them short.
    let mut perrun = 0u64;
    let (mut pf16, mut pf8, mut ipf) = (0u64, 0u64, 0u64);
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
        pf16 += patched(&g, 16);
        pf8 += patched(&g, 8);
        ipf += intervals_patched(&run, 32, 4);
        i32s += intervals(&run, 32, 4);
        i8s += intervals(&run, 8, 4);
        perrun += 2 + [8usize, 16, 32, 64]
            .into_iter()
            .map(|b| blocked(&g, b))
            .min()
            .unwrap_or(0);
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
    println!("what the gaps look like, {name}");
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
    println!("block of 16, patched {:6.2}", pf16 as f64 / n);
    println!("block of 8, patched  {:6.2}", pf8 as f64 / n);
    println!("block per run        {:6.2}", perrun as f64 / n);
    println!("intervals, patched   {:6.2}", ipf as f64 / n);
    println!("intervals, block 32  {:6.2}", i32s as f64 / n);
    println!("intervals, block 8   {:6.2}", i8s as f64 / n);
}

/// What it costs if the consecutive stretches come out as intervals and only
/// what is left is gap coded.
///
/// An interval is `min` or more ids in a row. It goes out as the gap to where it
/// starts and eight bits of length, and what is left is blocked at `block` the
/// way the format blocks it. Cheap to model and honest about the header: a run
/// with no interval in it pays six bits for saying so.
fn intervals_patched(run: &[u32], block: usize, min: usize) -> u64 {
    let (kept, count) = split_off_intervals(run, min);
    let gaps: Vec<u32> = kept.windows(2).map(|p| p[1] - p[0]).collect();
    6 + count * 8 + patched(&gaps, block)
}

/// The consecutive stretches of `min` or more, and what is left over.
fn split_off_intervals(run: &[u32], min: usize) -> (Vec<u32>, u64) {
    let (mut kept, mut count) = (Vec::new(), 0u64);
    let mut i = 0;
    while i < run.len() {
        let mut j = i + 1;
        while j < run.len() && run[j] == run[j - 1] + 1 {
            j += 1;
        }
        if j - i >= min {
            count += 1;
            kept.push(run[i]);
        } else {
            kept.extend_from_slice(&run[i..j]);
        }
        i = j;
    }
    (kept, count)
}

fn intervals(run: &[u32], block: usize, min: usize) -> u64 {
    let (kept, count) = split_off_intervals(run, min);
    let gaps: Vec<u32> = kept.windows(2).map(|p| p[1] - p[0]).collect();
    6 + count * 8 + blocked(&gaps, block)
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
        "{name:19}{:>7.2}{:>9.2}{:>9.2}{:>9.2}{:>8.2}{:>8.2}{:>9.2}{:>8.2}{:>8.2}   {:>6.1} MB in {took:?}",
        cold.bits_per_edge(),
        per(c.offsets),
        per(c.degrees),
        per(c.firsts),
        per(c.widths),
        per(c.gaps),
        per(c.patches),
        per(c.groups),
        per(c.slack),
        cold.bytes() as f64 / (1 << 20) as f64,
    );
}
