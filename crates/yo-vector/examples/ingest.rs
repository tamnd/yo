//! How fast the vector index takes vectors, and where that time goes.
//!
//! The M6 gate is fifty thousand vectors a second on one core with recall stable
//! over a continuous write stream and no rebuild. `examples/recall.rs` measures
//! the recall half on SIFT1M. This measures the rate half, and it measures it at
//! several collection sizes rather than one, because the failure mode worth
//! catching is not a slow constant, it is a rate that falls as the collection
//! grows. A single number at a hundred thousand vectors cannot tell those apart
//! and would have passed the gate while the index was quadratic.
//!
//! ```text
//! cargo run --release -p yo-vector --example ingest
//! cargo run --release -p yo-vector --example ingest -- 400000
//! ```
//!
//! The vectors are synthetic on purpose. Ingest rate is a function of the
//! partition count and the dimension, and clustered synthetic data at a hundred
//! and twenty eight dimensions gives the same partition count per vector that
//! SIFT does, so this measures the same thing without a hundred and forty
//! megabytes of download. Recall is the thing that needs a real dataset, and
//! recall is not what this looks at.
//!
//! The breakdown is the point. `insert` is finding the nearest centroid and
//! writing one code. `maintain` is splitting, merging and LIRE's sweep. They
//! grow differently and they have different fixes, so a total that is falling
//! tells you nothing about which one to go and look at.

use std::time::{Duration, Instant};
use yo_common::Rng;
use yo_vector::{Bits, Partitions, Tuning, Vectors};

/// The vectors, in full precision, which is what a rerank and a sweep read.
struct Store {
    dim: usize,
    data: Vec<f32>,
}

impl Vectors for Store {
    fn get(&self, id: u64, into: &mut [f32]) -> bool {
        let at = id as usize * self.dim;
        let Some(v) = self.data.get(at..at + self.dim) else {
            return false;
        };
        into.copy_from_slice(v);
        true
    }
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(200_000);
    let dim = 128;
    // Enough clusters that the index has real structure to find and few enough
    // that each one is far bigger than a partition, which is the shape of every
    // real embedding collection.
    let store = corpus(dim, n, 200, 0x9e37);

    println!("{n} vectors, {dim} dimensions, one core");
    println!(
        "{:>10}{:>12}{:>12}{:>12}{:>12}{:>10}",
        "at", "partitions", "a second", "insert", "maintain", "touched"
    );

    let mut ix = Partitions::new(dim, Bits::One, 0x51f7, Tuning::default());
    let mut buf = vec![0f32; dim];
    // Report at every doubling, because a rate that halves each time the
    // collection doubles is exactly what a quadratic looks like and a table on a
    // log scale is where that is visible by eye.
    let mut mark = 12_500usize;
    let (mut inserting, mut maintaining) = (Duration::ZERO, Duration::ZERO);
    let (mut was_i, mut was_m, mut was_at) = (Duration::ZERO, Duration::ZERO, 0usize);
    // How many vectors maintenance has moved or looked at. This is the number
    // that says whether a slowdown is work getting more expensive or there being
    // more of it, and those have nothing to do with each other.
    let (mut touched, mut was_t) = (0usize, 0usize);

    for id in 0..n as u64 {
        store.get(id, &mut buf);
        let t = Instant::now();
        ix.insert(id, &buf);
        inserting += t.elapsed();
        let t = Instant::now();
        if ix.needs_maintenance() {
            touched += ix.maintain(&store, 4);
        }
        maintaining += t.elapsed();

        let at = id as usize + 1;
        if at == mark || at == n {
            // Per interval rather than cumulative. A cumulative rate at a
            // million vectors is mostly made of what happened at ten thousand,
            // which is the number that is not in question.
            let took = (inserting - was_i) + (maintaining - was_m);
            println!(
                "{at:>10}{:>12}{:>12.0}{:>11.1}%{:>11.1}%{:>10.1}",
                ix.partitions(),
                (at - was_at) as f64 / took.as_secs_f64(),
                (inserting - was_i).as_secs_f64() * 100.0 / took.as_secs_f64(),
                (maintaining - was_m).as_secs_f64() * 100.0 / took.as_secs_f64(),
                (touched - was_t) as f64 / (at - was_at) as f64,
            );
            (was_i, was_m, was_at, was_t) = (inserting, maintaining, at, touched);
            mark *= 2;
        }
    }

    let total = inserting + maintaining;
    println!();
    println!(
        "{n} vectors in {total:?}, {:.0} a second overall, {:?} inserting and {:?} maintaining",
        n as f64 / total.as_secs_f64(),
        inserting,
        maintaining,
    );
}

/// `n` vectors on the unit sphere, gathered around `clusters` centres.
fn corpus(dim: usize, n: usize, clusters: usize, seed: u64) -> Store {
    let mut rng = Rng::new(seed);
    let centres: Vec<Vec<f32>> = (0..clusters).map(|_| draw(dim, &mut rng)).collect();
    let mut data = Vec::with_capacity(n * dim);
    for i in 0..n {
        let centre = &centres[i % clusters];
        let mut v = draw(dim, &mut rng);
        for (x, c) in v.iter_mut().zip(centre) {
            *x = *x * 0.35 + c;
        }
        unit(&mut v);
        data.extend_from_slice(&v);
    }
    Store { dim, data }
}

/// One gaussian vector, by Box-Muller off the uniform generator.
fn draw(dim: usize, rng: &mut Rng) -> Vec<f32> {
    let mut v = Vec::with_capacity(dim);
    while v.len() < dim {
        let u1 = ((rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64).max(1e-12);
        let u2 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let r = (-2.0 * u1.ln()).sqrt();
        v.push((r * (std::f64::consts::TAU * u2).cos()) as f32);
        if v.len() < dim {
            v.push((r * (std::f64::consts::TAU * u2).sin()) as f32);
        }
    }
    v
}

fn unit(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in v {
            *x /= n;
        }
    }
}
