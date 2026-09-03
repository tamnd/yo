//! MS-MARCO-v2 embeddings, turned into a dataset the other examples can read.
//!
//! M6's recall gate names SIFT1M and MS-MARCO-v2, and the second one is the
//! one that matters, because SIFT descriptors are byte valued gradient
//! histograms from 2009 and nobody stores those any more. What people actually
//! put in a vector index is the output of an embedding model, which is a
//! thousand odd dimensions of float, and a quantiser that holds on 128
//! dimensions of SIFT is not thereby known to hold on that.
//!
//! The corpus is the TREC-RAG 2024 one, which is MS-MARCO v2.1 passages
//! embedded with Cohere Embed English v3, published as
//! `CohereLabs/msmarco-v2.1-embed-english-v3`. It is 113,520,750 passages in
//! sixty shards, so a shard is about 1.76 million vectors at 1024 dimensions,
//! which is already more than the million the gate asks for.
//!
//! ```text
//! U=https://huggingface.co/datasets/CohereLabs/msmarco-v2.1-embed-english-v3/resolve/main
//! curl -L -O $U/passages_npy/msmarco_v2.1_doc_segmented_00.npy
//! curl -L -O $U/queries_jsonl/queries.jsonl.gz
//! gunzip queries.jsonl.gz
//! mkdir msmarco
//! cargo run --release -p yo-vector --example msmarco -- \
//!     msmarco 1000000 queries.jsonl msmarco_v2.1_doc_segmented_00.npy
//! ```
//!
//! That writes `msmarco/msmarco_base.fvecs` and `msmarco/msmarco_query.fvecs`,
//! after which `truth.rs` makes the ground truth and `recall.rs` and `drift.rs`
//! run on it exactly as they run on SIFT. The last argument is repeatable, so
//! six shards and a larger count is the ten million scale row.
//!
//! # Why this needs a ground truth computed
//!
//! The queries file already carries a `top1k_offsets` per query, worked out on
//! a flat index, and it cannot be used here. Those offsets are positions in all
//! sixty shards stacked end to end, so they are the true neighbours in a corpus
//! of 113 million and not in the one to ten million this builds. A true
//! neighbour of a query is very unlikely to be in the first shard. So the
//! ground truth for a subset has to be computed for that subset, which is what
//! `truth.rs` is for.
//!
//! # The two formats
//!
//! The passages are a numpy `.npy` file, which is a short ASCII header holding
//! a Python dict and then the raw array. The dict says the element type and the
//! shape and there is nothing else in the file, so reading one is parsing three
//! fields and then a `read_exact` per row. The elements here are `<f2`, little
//! endian half precision, which is half the download and is why nothing widens
//! them until they are written out.
//!
//! The queries are one JSON object per line with an `emb` array on each. Only
//! that array is wanted, and pulling one array of numbers out of a line does
//! not need a JSON library, so it is a scan for the key and then a run of
//! floats. Every other field on the line is text, ids, and the top 1000 hits
//! that this cannot use.
//!
//! # Whether L2 is the right metric here
//!
//! The corpus is published with cosine similarities and this writes a file that
//! everything downstream measures in L2. Those two rank the same way when the
//! vectors are unit length and do not otherwise, so the norms are measured on
//! the way past and printed. Cohere v3 embeddings come out normalised and the
//! printed range says so rather than assuming it, because a silently
//! unnormalised corpus would give a recall number that is wrong in a way
//! nothing downstream could notice.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [dir, want, queries, shards @ ..] = &args[..] else {
        eprintln!("usage: msmarco <out directory> <vectors> <queries.jsonl> <passages.npy>...");
        std::process::exit(2);
    };
    if shards.is_empty() {
        eprintln!("usage: msmarco <out directory> <vectors> <queries.jsonl> <passages.npy>...");
        std::process::exit(2);
    }
    let want: usize = want
        .parse()
        .expect("the second argument is how many vectors to take");
    let set = prefix(dir);
    std::fs::create_dir_all(dir).expect("could not make the output directory");

    let dim = passages(dir, set, want, shards);
    let n = query(dir, set, dim, queries);
    println!("{n} queries at {dim} dimensions");
    println!("now: cargo run --release -p yo-vector --example truth -- {dir}");
}

/// Every shard in turn, up to `want` vectors, as one `fvecs` file.
///
/// Nothing is held in memory beyond a row, because a shard is 3.6 GB on disk
/// and 7.2 GB once widened, and there is no reason for a format conversion to
/// need either. It reads a row, widens it, writes it, and forgets it.
fn passages(dir: &str, set: &str, want: usize, shards: &[String]) -> usize {
    let out = format!("{dir}/{set}_base.fvecs");
    let mut file = BufWriter::with_capacity(1 << 20, File::create(&out).expect("could not write"));
    let mut dim = 0usize;
    let mut wrote = 0usize;
    let mut norms = Norms::new();

    for path in shards {
        if wrote == want {
            break;
        }
        let f = File::open(path).unwrap_or_else(|e| {
            eprintln!("{path}: {e}");
            std::process::exit(1);
        });
        let mut r = BufReader::with_capacity(1 << 22, f);
        let (half, rows, cols) = npy(&mut r, path);
        if dim == 0 {
            dim = cols;
        }
        assert_eq!(cols, dim, "{path} is {cols} dimensional, not {dim}");
        println!(
            "{path}: {rows} by {cols}, {}",
            if half { "f16" } else { "f32" }
        );

        let take = rows.min(want - wrote);
        let mut raw = vec![0u8; cols * if half { 2 } else { 4 }];
        let mut v = vec![0f32; cols];
        for _ in 0..take {
            r.read_exact(&mut raw).expect("short shard");
            if half {
                for (x, b) in v.iter_mut().zip(raw.as_chunks::<2>().0) {
                    *x = f16(u16::from_le_bytes(*b));
                }
            } else {
                for (x, b) in v.iter_mut().zip(raw.as_chunks::<4>().0) {
                    *x = f32::from_le_bytes(*b);
                }
            }
            norms.add(&v);
            write_vec(&mut file, &v);
            wrote += 1;
        }
    }

    file.flush().expect("flush");
    assert!(wrote > 0, "no vectors were written");
    println!("wrote {out}, {wrote} vectors at {dim} dimensions");
    norms.report();
    dim
}

/// The `emb` array off every line of the queries file, as one `fvecs` file.
fn query(dir: &str, set: &str, dim: usize, path: &str) -> usize {
    let out = format!("{dir}/{set}_query.fvecs");
    let f = File::open(path).unwrap_or_else(|e| {
        eprintln!("{path}: {e}");
        eprintln!("this wants the plain jsonl, so gunzip queries.jsonl.gz first");
        std::process::exit(1);
    });
    let mut r = BufReader::with_capacity(1 << 20, f);
    let mut file = BufWriter::with_capacity(1 << 20, File::create(&out).expect("could not write"));

    let mut line = String::new();
    let mut n = 0usize;
    let mut norms = Norms::new();
    loop {
        line.clear();
        if r.read_line(&mut line).expect("could not read") == 0 {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }
        let v = embedding(&line).unwrap_or_else(|| {
            eprintln!("{path} line {} has no emb array", n + 1);
            std::process::exit(1);
        });
        assert_eq!(v.len(), dim, "query {n} is {} dimensional", v.len());
        norms.add(&v);
        write_vec(&mut file, &v);
        n += 1;
    }
    file.flush().expect("flush");
    println!("wrote {out}, {n} queries");
    norms.report();
    n
}

/// The numbers in the `emb` array on one line of JSON.
///
/// This is not a JSON parser and does not want to be. The array holds numbers
/// and nothing else, so the end of it is the next `]` and the numbers in
/// between are what `f32::from_str` already knows how to read. Anything that
/// does not parse is a line this has misread, and stopping there is better than
/// quietly writing a short vector.
fn embedding(line: &str) -> Option<Vec<f32>> {
    let at = line.find("\"emb\"")?;
    let open = line[at..].find('[')? + at + 1;
    let close = line[open..].find(']')? + open;
    let mut v = Vec::with_capacity(1024);
    for part in line[open..close].split(',') {
        v.push(part.trim().parse().ok()?);
    }
    Some(v)
}

/// One vector, in the record format `fvecs` uses.
fn write_vec(file: &mut BufWriter<File>, v: &[f32]) {
    file.write_all(&(v.len() as i32).to_le_bytes())
        .expect("write");
    for x in v {
        file.write_all(&x.to_le_bytes()).expect("write");
    }
}

/// The header of a numpy array file, as whether it is half precision and the
/// two dimensions of the array.
///
/// The reader is left sitting on the first element, which is the whole point:
/// there is no index and no per row framing in this format, so once the header
/// is past, the rest of the file is rows end to end.
fn npy<R: Read>(r: &mut R, path: &str) -> (bool, usize, usize) {
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic).expect("short file");
    assert_eq!(&magic[..6], b"\x93NUMPY", "{path} is not a npy file");
    // Version 1 writes the header length as two bytes and every later version
    // writes four, which is the only difference that matters here.
    let len = if magic[6] == 1 {
        let mut n = [0u8; 2];
        r.read_exact(&mut n).expect("short file");
        u16::from_le_bytes(n) as usize
    } else {
        let mut n = [0u8; 4];
        r.read_exact(&mut n).expect("short file");
        u32::from_le_bytes(n) as usize
    };
    let mut head = vec![0u8; len];
    r.read_exact(&mut head).expect("short file");
    let head = String::from_utf8(head).expect("the npy header is not text");

    let half = match field(&head, "'descr'") {
        Some(d) if d.contains("<f2") => true,
        Some(d) if d.contains("<f4") => false,
        d => panic!("{path} holds {d:?}, and this reads <f2 and <f4"),
    };
    assert!(
        field(&head, "'fortran_order'").is_some_and(|f| f.contains("False")),
        "{path} is in column order, and this reads row order"
    );
    let shape = field(&head, "'shape'").expect("the npy header has no shape");
    let mut dims = shape
        .trim_matches(|c| c == '(' || c == ')')
        .split(',')
        .filter_map(|d| d.trim().parse::<usize>().ok());
    let rows = dims.next().expect("the npy shape has no rows");
    let cols = dims.next().expect("the npy shape is not two dimensional");
    (half, rows, cols)
}

/// One value out of the Python dict a npy header is.
///
/// The keys are known and the values are short, so this finds the key and takes
/// everything up to the next comma that is not inside the shape tuple. Writing a
/// Python literal parser for three fields would be the wrong shape of effort.
fn field<'a>(head: &'a str, key: &str) -> Option<&'a str> {
    let at = head.find(key)? + key.len();
    let rest = head[at..].trim_start().strip_prefix(':')?.trim_start();
    let end = if rest.starts_with('(') {
        rest.find(')')? + 1
    } else {
        rest.find(',').unwrap_or(rest.len())
    };
    Some(rest[..end].trim())
}

/// A half precision float, widened.
///
/// Every case is here rather than only the one embeddings use, because the
/// cases that never happen are the ones that would be wrong for years. The
/// awkward one is a subnormal half, which is an ordinary float once the leading
/// bit is found, so the exponent has to be worked out from where that bit is
/// rather than copied across.
fn f16(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from(bits >> 10) & 0x1f;
    let mant = u32::from(bits & 0x3ff);
    let rest = match exp {
        0 if mant == 0 => 0,
        0 => {
            let lz = mant.leading_zeros();
            ((134 - lz) << 23) | (((mant << (lz - 21)) & 0x3ff) << 13)
        }
        // An infinity or a NaN, which keep the payload they came with.
        0x1f => 0x7f80_0000 | (mant << 13),
        // The exponent bias goes from 15 to 127 and the mantissa from 10 bits
        // to 23, and neither can lose anything on the way.
        _ => ((exp + 112) << 23) | (mant << 13),
    };
    f32::from_bits(sign | rest)
}

/// How long the vectors are, which is the whole question of whether measuring
/// them in L2 ranks them the way the corpus was published to be ranked.
struct Norms {
    n: usize,
    sum: f64,
    low: f64,
    high: f64,
}

impl Norms {
    fn new() -> Norms {
        Norms {
            n: 0,
            sum: 0.0,
            low: f64::INFINITY,
            high: 0.0,
        }
    }

    fn add(&mut self, v: &[f32]) {
        let len = v
            .iter()
            .map(|&x| f64::from(x) * f64::from(x))
            .sum::<f64>()
            .sqrt();
        self.n += 1;
        self.sum += len;
        self.low = self.low.min(len);
        self.high = self.high.max(len);
    }

    fn report(&self) {
        if self.n == 0 {
            return;
        }
        let mean = self.sum / self.n as f64;
        println!(
            "  norms: mean {mean:.6}, from {:.6} to {:.6}",
            self.low, self.high
        );
        if (self.high - self.low).abs() > 1e-3 {
            println!("  these are not all the same length, so L2 and cosine will not agree");
        }
    }
}

/// What the files in a directory are called, which every dataset published in
/// this format names after itself.
fn prefix(dir: &str) -> &str {
    dir.trim_end_matches('/')
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(dir)
}
