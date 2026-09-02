//! Temporary: does raising the reach of the centroid ranking cost anything on
//! an unfiltered search? Same index, same query, only Tuning::widen differs.
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use yo_common::Rng;
use yo_vector::{Bits, Partitions, Tuning, Vectors};

struct Store(Vec<f32>, usize);
impl Vectors for Store {
    fn get(&self, id: u64, into: &mut [f32]) -> bool {
        let at = id as usize * self.1;
        match self.0.get(at..at + self.1) {
            Some(v) => { into.copy_from_slice(v); true }
            None => false,
        }
    }
}
impl Store {
    fn at(&self, i: usize) -> &[f32] { &self.0[i * self.1..(i + 1) * self.1] }
    fn len(&self) -> usize { self.0.len() / self.1 }
}

fn corpus(dim: usize, n: usize, clusters: usize, seed: u64) -> Store {
    let mut rng = Rng::new(seed);
    let centres: Vec<Vec<f32>> = (0..clusters).map(|_| draw(dim, &mut rng)).collect();
    let mut all = Vec::with_capacity(n * dim);
    for i in 0..n {
        let off = draw(dim, &mut rng);
        let mut v: Vec<f32> = centres[i % clusters].iter().zip(&off).map(|(c, o)| c + o * 0.7).collect();
        unit(&mut v);
        all.extend_from_slice(&v);
    }
    Store(all, dim)
}
fn draw(dim: usize, rng: &mut Rng) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim).map(|i| {
        let u = (rng.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
        let heavy = if i < dim / 16 { 6.0 } else { 1.0 };
        (u * 2.0 - 1.0) * heavy
    }).collect();
    unit(&mut v);
    v
}
fn unit(v: &mut [f32]) {
    let len = v.iter().map(|c| c * c).sum::<f32>().sqrt();
    for c in v { *c /= len; }
}
fn build(store: &Store, tuning: Tuning) -> Partitions {
    let mut ix = Partitions::new(store.1, Bits::One, 7, tuning);
    for id in 0..store.len() as u64 {
        ix.insert(id, store.at(id as usize));
        if id % 256 == 0 { ix.maintain(store, 4096); }
    }
    ix.maintain(store, 1 << 24);
    ix
}

fn bench(c: &mut Criterion) {
    let dim = 128;
    let store = corpus(dim, 100_000, 64, 0x5eed);
    let queries = corpus(dim, 64, 64, 0xc0ffee);
    let wide = build(&store, Tuning::default());
    let narrow = build(&store, Tuning { widen: 1, ..Tuning::default() });
    let mut g = c.benchmark_group("widen");
    g.sample_size(40);
    g.bench_function("reach 64", |b| {
        let mut i = 0usize;
        b.iter(|| { i = (i + 1) % queries.len(); black_box(wide.candidates(black_box(queries.at(i)), 40)) });
    });
    g.bench_function("reach 8", |b| {
        let mut i = 0usize;
        b.iter(|| { i = (i + 1) % queries.len(); black_box(narrow.candidates(black_box(queries.at(i)), 40)) });
    });
    g.finish();
}
criterion_group!(benches, bench);
criterion_main!(benches);
