//! Sixty two queries put to an 8.10.1 and put again to the walk here.
//!
//! The corpus is twelve documents over two text fields, a tag and a number,
//! written into a real server and into an index built the same way, and every
//! query in `reference.txt` is the answer that server gave: which keys came
//! back and what each one scored, sorted by key so the file does not depend on
//! the order the two of them answer in.
//!
//! Regenerating it is a `FT.CREATE` with this schema, twelve `HSET`s with these
//! values and an `FT.SEARCH ... WITHSCORES NOCONTENT LIMIT 0 100` per line, so
//! nothing in the file is written by hand and nothing in it is a guess about
//! what the other side would do.

use yo_search::query::{Ask, parse};
use yo_search::score::Scorer;
use yo_search::{Definition, English, Field, Index, Kind, Tag, Text, walk};

/// The documents, as key, the two text fields, the tag and the number.
const DOCS: [(&str, &str, &str, &str, &str); 12] = [
    ("d:1", "the dog runs fast", "red hat", "aa,bb", "1"),
    ("d:2", "dogs run in the park", "blue hat", "aa bb", "2"),
    ("d:3", "a cat naps quietly", "green shoes", "cc", "3"),
    ("d:4", "cats and dogs together", "red shoes", "aa,cc", "4"),
    ("d:5", "running water", "blue hat", "dd", "5"),
    ("d:6", "the quick brown fox", "red hat", "ee,ff", "10"),
    ("d:7", "quicker brown foxes", "green hat", "ee", "20"),
    ("d:8", "jumping over lazy dogs", "blue shoes", "ff", "-3"),
    ("d:9", "abacus abandon ability", "red hat", "gg", "0"),
    ("d:10", "abacus only", "blue hat", "gg,hh", "2.5"),
    ("d:11", "hello world", "hello there", "ii", "100"),
    ("d:12", "world of dogs and cats", "green hat", "aa,ii", "7"),
];

/// The index the corpus was written into, with the schema the file was made
/// under.
fn corpus() -> Index {
    let mut index = Index::new(
        b"ix",
        Definition::default(),
        vec![
            Field::new(b"t", Kind::Text(Text::default())),
            Field::new(b"b", Kind::Text(Text::default())),
            Field::new(b"g", Kind::Tag(Tag::default())),
            Field::new(b"n", Kind::Numeric),
        ],
    );
    let mut english = English::new();
    for (key, t, b, g, n) in DOCS {
        index
            .write(
                &mut english,
                key.as_bytes(),
                &[
                    (b"t", t.as_bytes()),
                    (b"b", b.as_bytes()),
                    (b"g", g.as_bytes()),
                    (b"n", n.as_bytes()),
                ],
            )
            .expect("a document a real server indexed too");
    }
    index.held.settle();
    index
}

/// What this build answers one query with, as the file writes it.
fn answer(index: &Index, query: &str) -> String {
    let node = parse(query.as_bytes(), index, &Ask::default()).expect("a query that parses");
    let facts = index.held.facts();
    let mut pairs: Vec<String> = walk::run(&index.held, &node)
        .into_iter()
        .map(|hit| {
            let doc = index.held.docs.get(hit.id).expect("a live document");
            let score = Scorer::default_scorer().of(&facts, doc, &hit.found, None);
            format!("{}={score}", String::from_utf8_lossy(&doc.key))
        })
        .collect();
    pairs.sort();
    pairs.join(" ")
}

/// Whether two of these lines say the same thing, which means the same keys in
/// the same order with scores that agree to the last few bits of a double.
fn same(got: &str, want: &str) -> bool {
    let (got, want): (Vec<&str>, Vec<&str>) = (
        got.split_whitespace().collect(),
        want.split_whitespace().collect(),
    );
    if got.len() != want.len() {
        return false;
    }
    got.iter().zip(&want).all(|(got, want)| {
        let (Some((got_key, got_score)), Some((want_key, want_score))) =
            (got.rsplit_once('='), want.rsplit_once('='))
        else {
            return got == want;
        };
        let (Ok(got_score), Ok(want_score)) = (got_score.parse::<f64>(), want_score.parse::<f64>())
        else {
            return false;
        };
        got_key == want_key
            && (got_score - want_score).abs() <= f64::EPSILON * want_score.abs().max(1.0) * 8.0
    })
}

#[test]
fn every_query_answers_what_a_real_server_answered() {
    let index = corpus();
    let mut wrong = Vec::new();
    let mut checked = 0;
    for line in include_str!("reference.txt").lines() {
        let Some((query, want)) = line.split_once('\t') else {
            continue;
        };
        checked += 1;
        let got = answer(&index, query);
        if !same(&got, want) {
            wrong.push(format!("{query}\n  want {want}\n  got  {got}"));
        }
    }
    assert_eq!(checked, 62, "the file lost a query");
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
