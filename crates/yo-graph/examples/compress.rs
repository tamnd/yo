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
        eprintln!("usage: compress <edge list> [--no-hot]");
        std::process::exit(2);
    };
    let hot_too = !args.any(|a| a == "--no-hot");

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

    let t = Instant::now();
    let cold = Csr::build(nodes, &mut edges.clone());
    println!(
        "cold               {:6.2} bits an edge, {:>6.1} MB, built in {:?}",
        cold.bits_per_edge(),
        cold.bytes() as f64 / (1 << 20) as f64,
        t.elapsed()
    );

    let t = Instant::now();
    let to = csr::order_by_degree(nodes, &edges);
    csr::renumber(&mut edges, &to);
    let ordered = t.elapsed();
    let cold = Csr::build(nodes, &mut edges);
    println!(
        "cold, deg ordered  {:6.2} bits an edge, {:>6.1} MB, ordered in {ordered:?}",
        cold.bits_per_edge(),
        cold.bytes() as f64 / (1 << 20) as f64,
    );
}
