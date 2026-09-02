//! The query a property graph exists for, over a graph small enough to check by
//! hand.
//!
//! `adjacency` has its own tests for the structure and `graph` has its own for
//! the bookkeeping. Neither of them asks the question a caller asks, which is
//! two hops with a filter on the node properties at the far end and a filter on
//! the edge properties along the way. That question is where the two halves have
//! to line up: the slot beside a neighbour has to be the slot the edge's
//! document is under, in the same order, after the graph has been edited.
//!
//! The graph here is a tiny film ratings set. People rate films, people follow
//! people, and the query is what the people I follow rated highly, which is a
//! recommendation and the thing every property graph benchmark is a version of.

use yo_common::Result;
use yo_doc::{Builder, IndexKind, Key};
use yo_graph::{Dir, Graph};

const FOLLOWS: u32 = 1;
const RATED: u32 = 2;

/// The bytes of a value built by `f`.
fn built(f: impl FnOnce(&mut Builder) -> Result<()>) -> Vec<u8> {
    let mut b = Builder::new();
    f(&mut b).expect("built");
    b.finish().expect("finished").to_vec()
}

fn person(name: &str) -> Vec<u8> {
    built(|b| {
        b.begin_object()?;
        b.key(b"kind")?;
        b.text("person")?;
        b.key(b"name")?;
        b.text(name)?;
        b.end_object()
    })
}

fn film(title: &str, year: i64) -> Vec<u8> {
    built(|b| {
        b.begin_object()?;
        b.key(b"kind")?;
        b.text("film")?;
        b.key(b"title")?;
        b.text(title)?;
        b.key(b"year")?;
        b.int(year)?;
        b.end_object()
    })
}

fn rating(score: i64) -> Vec<u8> {
    built(|b| {
        b.begin_object()?;
        b.key(b"score")?;
        b.int(score)?;
        b.end_object()
    })
}

/// The whole set. People are 1 to 4 and films are 101 to 104.
fn films_and_friends() -> Graph {
    let mut g = Graph::new();
    for (id, name) in [(1u64, "ada"), (2, "grace"), (3, "edsger"), (4, "barbara")] {
        g.put_node(id, &person(name)).expect("a person");
    }
    for (id, title, year) in [
        (101u64, "stalker", 1979),
        (102, "solaris", 1972),
        (103, "arrival", 2016),
        (104, "primer", 2004),
    ] {
        g.put_node(id, &film(title, year)).expect("a film");
    }

    g.link(1, 2, FOLLOWS, &built(|b| b.null())).expect("linked");
    g.link(1, 3, FOLLOWS, &built(|b| b.null())).expect("linked");
    g.link(2, 4, FOLLOWS, &built(|b| b.null())).expect("linked");

    for (who, what, score) in [
        (2u64, 101u64, 5i64),
        (2, 102, 3),
        (2, 103, 5),
        (3, 101, 2),
        (3, 104, 5),
        (4, 103, 4),
    ] {
        g.link(who, what, RATED, &rating(score)).expect("rated");
    }
    g
}

/// What the people `me` follows rated at `at_least` or better, as titles.
fn recommended(g: &Graph, me: u64, at_least: i64) -> Vec<String> {
    let mut out = Vec::new();
    for friend in g.neighbours(me, FOLLOWS, Dir::Out) {
        for (film, edge) in g.hop(*friend, RATED, Dir::Out) {
            let score = g
                .edge(edge)
                .and_then(|d| d.get(b"score"))
                .and_then(|v| v.as_int())
                .expect("a rating has a score");
            if score < at_least {
                continue;
            }
            let title = g
                .node(film)
                .and_then(|d| d.get(b"title"))
                .and_then(|v| v.as_text())
                .expect("a film has a title")
                .to_string();
            out.push(title);
        }
    }
    out.sort();
    out.dedup();
    out
}

#[test]
fn two_hops_with_a_filter_on_the_edge() {
    let g = films_and_friends();
    assert_eq!(g.nodes(), 8);
    assert_eq!(g.edges(), 9);
    assert_eq!(g.labels(), [FOLLOWS, RATED]);

    // Ada follows Grace and Edsger. Grace gave stalker and arrival a 5 and
    // solaris a 3, Edsger gave primer a 5 and stalker a 2. Barbara's 4 for
    // arrival is two hops away through Grace and so is not in this answer.
    assert_eq!(
        recommended(&g, 1, 5),
        vec!["arrival".to_string(), "primer".into(), "stalker".into()]
    );
    // Dropping the bar to a 3 lets solaris in and nothing else, since the only
    // rating between is Edsger's 2 for stalker, which is already in.
    assert_eq!(
        recommended(&g, 1, 3),
        vec![
            "arrival".to_string(),
            "primer".into(),
            "solaris".into(),
            "stalker".into()
        ]
    );
    // Grace follows only Barbara.
    assert_eq!(recommended(&g, 2, 4), vec!["arrival".to_string()]);
    // Barbara follows nobody, so there is nothing to recommend her.
    assert!(recommended(&g, 4, 1).is_empty());
}

#[test]
fn the_slots_still_line_up_after_the_graph_is_edited() {
    // The failure this file is really about. Unlinking moves the last entry of
    // a run into the hole it made, so if the neighbour array and the slot array
    // were ever moved apart, an edit would leave a film paired with somebody
    // else's rating and every read after it would be quietly wrong.
    let mut g = films_and_friends();

    // Grace changes her mind about solaris and stops following Barbara.
    assert!(g.unlink(2, 102, RATED).is_some());
    assert!(g.unlink(2, 4, FOLLOWS).is_some());
    assert_eq!(g.edges(), 7);
    assert_eq!(
        recommended(&g, 1, 3),
        vec!["arrival".to_string(), "primer".into(), "stalker".into()]
    );
    assert!(recommended(&g, 2, 1).is_empty());

    // Every rating Grace has left is still paired with the film it was for.
    let mut hers: Vec<(String, i64)> = g
        .hop(2, RATED, Dir::Out)
        .map(|(film, edge)| {
            let title = g
                .node(film)
                .and_then(|d| d.get(b"title"))
                .and_then(|v| v.as_text())
                .expect("a title")
                .to_string();
            let score = g
                .edge(edge)
                .and_then(|d| d.get(b"score"))
                .and_then(|v| v.as_int())
                .expect("a score");
            (title, score)
        })
        .collect();
    hers.sort();
    assert_eq!(
        hers,
        vec![("arrival".to_string(), 5), ("stalker".into(), 5)]
    );

    // And a new rating reuses one of the two freed slots without inheriting
    // what was under it.
    let e = g.link(2, 104, RATED, &rating(1)).expect("rated");
    assert_eq!(
        g.edge(e)
            .and_then(|d| d.get(b"score"))
            .and_then(|v| v.as_int()),
        Some(1)
    );
    assert_eq!(g.edge_props().len(), 8);
}

#[test]
fn an_index_starts_the_walk() {
    // The other half of a property graph: a query does not begin at a node id,
    // it begins at a property. Finding the films of a year and then walking
    // back along the ratings is the shape, and it needs the incoming direction
    // that `Adjacency::new` indexes.
    let mut g = films_and_friends();
    g.index_nodes("$.kind", IndexKind::Equality)
        .expect("indexed");
    g.index_edges("$.score", IndexKind::Equality)
        .expect("indexed");

    let mut films = Vec::new();
    let n = g
        .find_nodes("$.kind", &Key::text("film"), |id, _| films.push(id))
        .expect("found");
    assert_eq!(n, 4);
    films.sort_unstable();
    assert_eq!(films, vec![101, 102, 103, 104]);

    // Who rated stalker, and what did they give it.
    let mut raters: Vec<(String, i64)> = g
        .hop(101, RATED, Dir::In)
        .map(|(who, edge)| {
            let name = g
                .node(who)
                .and_then(|d| d.get(b"name"))
                .and_then(|v| v.as_text())
                .expect("a name")
                .to_string();
            let score = g
                .edge(edge)
                .and_then(|d| d.get(b"score"))
                .and_then(|v| v.as_int())
                .expect("a score");
            (name, score)
        })
        .collect();
    raters.sort();
    assert_eq!(raters, vec![("edsger".to_string(), 2), ("grace".into(), 5)]);

    // Every five star rating, straight out of the edge index.
    let mut fives = Vec::new();
    let n = g
        .find_edges("$.score", &Key::int(5), |slot, _| fives.push(slot))
        .expect("found");
    assert_eq!(n, 3);
    assert_eq!(fives.len(), 3);

    // The index follows the graph. Removing Edsger takes his two ratings with
    // him, and the index says so without being rebuilt.
    assert!(g.remove_node(3).expect("removed"));
    assert_eq!(g.find_edges("$.score", &Key::int(5), |_, _| {}).unwrap(), 2);
    assert_eq!(
        g.find_nodes("$.kind", &Key::text("person"), |_, _| {})
            .unwrap(),
        3
    );
}

#[test]
fn the_field_names_are_paid_for_once_across_the_whole_graph() {
    // A thousand people each following ten others, which is ten thousand edges
    // carrying a score. The names of the fields are what an edge property store
    // is mostly made of, and interning is why they are not.
    let mut g = Graph::new();
    for id in 0..1000u64 {
        g.put_node(id, &person("someone")).expect("a person");
    }
    for id in 0..1000u64 {
        for n in 1..11u64 {
            g.link(id, (id + n) % 1000, FOLLOWS, &rating((id % 5) as i64 + 1))
                .expect("linked");
        }
    }
    assert_eq!(g.edges(), 10_000);
    assert_eq!(g.node_props().keys().len(), 2, "kind and name, once");
    assert_eq!(g.edge_props().keys().len(), 1, "score, once");
}
