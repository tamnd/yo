//! The typed graph, over a set small enough to check by hand.
//!
//! `yo-graph` has its own tests for the plane and for the documents behind it.
//! What is not tested down there is the layer that makes a wrong traversal a
//! compile error rather than an empty answer, and the bookkeeping that layer
//! needs: the label that keeps two node types apart in one store, the lookup
//! from the id in a struct to the dense id the plane is keyed by, and what
//! happens to both when a node is removed.
//!
//! The graph is people, companies and films. People follow people, people work
//! at companies and people rate films, which is three edge types with three
//! different pairs of ends and so the shape that would catch a layer that had
//! quietly collapsed them into one.

use yo::{Edge, Node, Yo};

#[derive(Yo, Debug, PartialEq, Clone)]
struct Person {
    #[yo(id)]
    id: u64,
    #[yo(index)]
    city: String,
    #[yo(ordered)]
    age: i64,
}

#[derive(Yo, Debug, PartialEq)]
struct Company {
    #[yo(id)]
    id: u64,
    #[yo(index)]
    city: String,
}

#[derive(Yo, Debug, PartialEq)]
struct Film {
    #[yo(id)]
    id: String,
    #[yo(ordered)]
    year: i64,
}

#[derive(Yo, Debug, PartialEq)]
struct Follows {
    since: i64,
}

#[derive(Yo, Debug, PartialEq)]
struct WorksAt {
    title: String,
}

#[derive(Yo, Debug, PartialEq)]
struct Rated {
    #[yo(index)]
    score: i64,
}

impl Node for Person {
    const LABEL: &'static str = "Person";
}
impl Node for Company {
    const LABEL: &'static str = "Company";
}
impl Node for Film {
    const LABEL: &'static str = "Film";
}

impl Edge for Follows {
    type From = Person;
    type To = Person;
    const LABEL: &'static str = "FOLLOWS";
}
impl Edge for WorksAt {
    type From = Person;
    type To = Company;
    const LABEL: &'static str = "WORKS_AT";
}
impl Edge for Rated {
    type From = Person;
    type To = Film;
    const LABEL: &'static str = "RATED";
}

fn person(id: u64, city: &str, age: i64) -> Person {
    Person {
        id,
        city: city.to_owned(),
        age,
    }
}

#[test]
fn a_walk_goes_where_the_edges_go() -> yo::Result<()> {
    let db = yo::open(yo::MEMORY)?;
    let g = db.graph("social")?;

    let ada = g.add(&person(1, "london", 36))?;
    let grace = g.add(&person(2, "london", 45))?;
    let edsger = g.add(&person(3, "austin", 51))?;
    let barbara = g.add(&person(4, "austin", 62))?;

    g.link(ada, grace, &Follows { since: 2024 })?;
    g.link(ada, edsger, &Follows { since: 2025 })?;
    g.link(grace, barbara, &Follows { since: 2026 })?;
    g.link(edsger, barbara, &Follows { since: 2023 })?;

    assert_eq!(g.nodes()?, 4);
    assert_eq!(g.edges()?, 4);
    assert_eq!(g.count::<Person>()?, 4);
    assert_eq!(g.degree::<Follows>(ada)?, 2);

    let mut one = g.out::<Follows>(ada)?;
    one.sort_by_key(|id| format!("{id:?}"));
    assert_eq!(one.len(), 2);
    assert!(one.contains(&grace) && one.contains(&edsger));

    // Both of Ada's friends follow Barbara, and a walk asks which nodes it can
    // reach rather than by how many routes, so the frontier is a set.
    let two = g.walk(ada).out::<Follows>()?.out::<Follows>()?;
    assert_eq!(two.len(), 1);
    assert_eq!(two.nodes()?, vec![person(4, "austin", 62)]);

    // And backwards from the far end gets to both of them.
    assert_eq!(g.walk(barbara).incoming::<Follows>()?.len(), 2);
    Ok(())
}

#[test]
fn a_walk_can_change_type_along_the_way() -> yo::Result<()> {
    let db = yo::open(yo::MEMORY)?;
    let g = db.graph("social")?;

    let ada = g.add(&person(1, "london", 36))?;
    let grace = g.add(&person(2, "london", 45))?;
    let acme = g.add(&Company {
        id: 100,
        city: "london".to_owned(),
    })?;
    let globex = g.add(&Company {
        id: 200,
        city: "austin".to_owned(),
    })?;

    g.link(ada, grace, &Follows { since: 2024 })?;
    g.link(
        grace,
        acme,
        &WorksAt {
            title: "engineer".to_owned(),
        },
    )?;
    g.link(
        ada,
        globex,
        &WorksAt {
            title: "director".to_owned(),
        },
    )?;

    // Where the people I follow work. The walk starts on a Person and ends on
    // a Company, and the type it hands back changes with it.
    let where_they_work: Vec<Company> = g.walk(ada).out::<Follows>()?.out::<WorksAt>()?.nodes()?;
    assert_eq!(
        where_they_work,
        vec![Company {
            id: 100,
            city: "london".to_owned()
        }]
    );

    // Backwards from a company gets the people, not the companies.
    let staff: Vec<Person> = g.walk(acme).incoming::<WorksAt>()?.nodes()?;
    assert_eq!(staff, vec![person(2, "london", 45)]);
    Ok(())
}

#[test]
fn two_node_types_can_hold_the_same_id() -> yo::Result<()> {
    // The failure a label is for. Person 1 and Company 1 are two nodes, and a
    // lookup table keyed by the id alone would make them one.
    let db = yo::open(yo::MEMORY)?;
    let g = db.graph("social")?;

    let ada = g.add(&person(1, "london", 36))?;
    let acme = g.add(&Company {
        id: 1,
        city: "austin".to_owned(),
    })?;

    assert_eq!(g.nodes()?, 2);
    assert_eq!(g.count::<Person>()?, 1);
    assert_eq!(g.count::<Company>()?, 1);
    assert_eq!(g.get(ada)?, Some(person(1, "london", 36)));
    assert_eq!(
        g.get(acme)?,
        Some(Company {
            id: 1,
            city: "austin".to_owned()
        })
    );

    // And the index they both declare on `$.city` is one index, so a find has
    // to keep the two apart by label rather than by what it read.
    g.add(&person(2, "austin", 45))?;
    assert_eq!(g.find(Person::CITY, "austin")?.len(), 1);
    assert_eq!(g.find(Company::CITY, "austin")?.len(), 1);
    assert_eq!(g.find(Person::CITY, "london")?.len(), 1);
    assert_eq!(g.find(Company::CITY, "london")?.len(), 0);
    Ok(())
}

#[test]
fn an_id_is_a_handle_and_not_the_id_you_wrote() -> yo::Result<()> {
    let db = yo::open(yo::MEMORY)?;
    let g = db.graph("social")?;

    let ada = g.add(&person(41_920, "london", 36))?;
    assert_eq!(g.id_of::<Person>(&41_920)?, Some(ada));
    assert_eq!(g.id_of::<Person>(&7)?, None);
    // A film's id is a string, and the same lookup works, which is the point of
    // the id in the struct not being what the plane is keyed by.
    let arrival = g.add(&Film {
        id: "arrival".to_owned(),
        year: 2016,
    })?;
    assert_eq!(g.id_of::<Film>("arrival")?, Some(arrival));
    assert_eq!(g.id_of::<Film>("primer")?, None);

    // Adding the same struct id again is the same node, updated.
    let again = g.add(&person(41_920, "turin", 37))?;
    assert_eq!(again, ada);
    assert_eq!(g.nodes()?, 2);
    assert_eq!(g.get(ada)?.map(|p| p.city), Some("turin".to_owned()));
    Ok(())
}

#[test]
fn an_edge_carries_its_own_fields() -> yo::Result<()> {
    let db = yo::open(yo::MEMORY)?;
    let g = db.graph("social")?;

    let ada = g.add(&person(1, "london", 36))?;
    let arrival = g.add(&Film {
        id: "arrival".to_owned(),
        year: 2016,
    })?;
    let primer = g.add(&Film {
        id: "primer".to_owned(),
        year: 2004,
    })?;

    let a = g.link(ada, arrival, &Rated { score: 5 })?;
    let b = g.link(ada, primer, &Rated { score: 3 })?;
    assert_eq!(g.edge(a)?, Some(Rated { score: 5 }));
    assert_eq!(g.edge(b)?, Some(Rated { score: 3 }));

    // What Ada rated, with the score she gave it, which is the query an edge's
    // fields exist for.
    let mut rated: Vec<(String, i64)> = Vec::new();
    for hop in g.out_edges::<Rated>(ada)? {
        let film = g.get(hop.to)?.expect("a film");
        let score = g.edge(hop.edge)?.expect("a rating").score;
        rated.push((film.id, score));
    }
    rated.sort();
    assert_eq!(
        rated,
        vec![("arrival".to_owned(), 5), ("primer".to_owned(), 3)]
    );
    Ok(())
}

#[test]
fn two_edges_between_the_same_pair_are_two_edges() -> yo::Result<()> {
    let db = yo::open(yo::MEMORY)?;
    let g = db.graph("social")?;
    let ada = g.add(&person(1, "london", 36))?;
    let arrival = g.add(&Film {
        id: "arrival".to_owned(),
        year: 2016,
    })?;

    let first = g.link(ada, arrival, &Rated { score: 3 })?;
    let again = g.link(ada, arrival, &Rated { score: 5 })?;
    assert_ne!(first, again);
    assert_eq!(g.edges()?, 2);
    assert_eq!(g.edge(first)?, Some(Rated { score: 3 }));
    assert_eq!(g.edge(again)?, Some(Rated { score: 5 }));

    // Unlinking takes one of them.
    assert!(g.unlink::<Rated>(ada, arrival)?);
    assert_eq!(g.edges()?, 1);
    assert!(g.unlink::<Rated>(ada, arrival)?);
    assert!(!g.unlink::<Rated>(ada, arrival)?);
    Ok(())
}

#[test]
fn removing_a_node_takes_its_edges_and_frees_its_id() -> yo::Result<()> {
    let db = yo::open(yo::MEMORY)?;
    let g = db.graph("social")?;

    let ada = g.add(&person(1, "london", 36))?;
    let grace = g.add(&person(2, "london", 45))?;
    let edsger = g.add(&person(3, "austin", 51))?;
    g.link(ada, grace, &Follows { since: 2024 })?;
    g.link(grace, edsger, &Follows { since: 2025 })?;

    assert!(g.remove(grace)?);
    assert!(!g.has(grace)?);
    assert!(!g.remove(grace)?, "removing twice is not two removals");
    assert_eq!(g.nodes()?, 2);
    assert_eq!(g.count::<Person>()?, 2);
    // Both edges touched her, so both are gone.
    assert_eq!(g.edges()?, 0);
    assert!(g.out::<Follows>(ada)?.is_empty());

    // The lookup no longer answers for her, and the index does not either.
    assert_eq!(g.id_of::<Person>(&2)?, None);
    assert_eq!(
        g.find(Person::CITY, "london")?,
        vec![person(1, "london", 36)]
    );

    // Putting her back is a new handle, and the old one is not it.
    let back = g.add(&person(2, "turin", 46))?;
    assert_ne!(back, grace);
    assert_eq!(g.count::<Person>()?, 3);
    assert!(
        g.get(grace)?.is_none(),
        "the old handle is not the new node"
    );
    Ok(())
}

#[test]
fn an_edge_needs_both_of_its_ends() -> yo::Result<()> {
    let db = yo::open(yo::MEMORY)?;
    let g = db.graph("social")?;
    let ada = g.add(&person(1, "london", 36))?;
    let grace = g.add(&person(2, "london", 45))?;
    assert!(g.remove(grace)?);

    // An edge onto a node that is gone would be a dangling reference that every
    // later read had to guard against, so it is refused here instead.
    let e = g.link(ada, grace, &Follows { since: 2026 });
    assert!(e.is_err());
    assert_eq!(e.unwrap_err().code(), yo::Code::NotFound);
    assert_eq!(g.edges()?, 0);
    Ok(())
}

#[test]
fn a_walk_can_start_at_an_index_and_be_filtered() -> yo::Result<()> {
    let db = yo::open(yo::MEMORY)?;
    let g = db.graph("social")?;

    let ada = g.add(&person(1, "london", 36))?;
    let grace = g.add(&person(2, "london", 45))?;
    let edsger = g.add(&person(3, "austin", 51))?;
    let barbara = g.add(&person(4, "london", 62))?;
    g.link(ada, edsger, &Follows { since: 2024 })?;
    g.link(grace, barbara, &Follows { since: 2025 })?;

    // Everyone in london, and then who they follow.
    let reached: Vec<Person> = g
        .walk_from(Person::CITY, "london")?
        .out::<Follows>()?
        .nodes()?;
    let mut ages: Vec<i64> = reached.iter().map(|p| p.age).collect();
    ages.sort_unstable();
    assert_eq!(ages, vec![51, 62]);

    // The same walk, keeping only the ones over sixty.
    let older: Vec<Person> = g
        .walk_from(Person::CITY, "london")?
        .out::<Follows>()?
        .filter(|p| p.age > 60)?
        .nodes()?;
    assert_eq!(older, vec![person(4, "london", 62)]);

    assert_eq!(g.count_at(Person::CITY, "london")?, 3);
    assert_eq!(g.count_at(Person::CITY, "austin")?, 1);
    Ok(())
}

#[test]
fn a_graph_and_a_collection_cannot_share_a_name() -> yo::Result<()> {
    let db = yo::open(yo::MEMORY)?;
    let _ = db.graph("social")?;
    let clash = db.docs::<Person>("social");
    assert!(clash.is_err());
    assert_eq!(clash.unwrap_err().code(), yo::Code::ShapeMismatch);

    let _ = db.docs::<Person>("people")?;
    let other = db.graph("people");
    assert!(other.is_err());
    Ok(())
}

#[test]
fn a_second_handle_is_the_same_graph() -> yo::Result<()> {
    let db = yo::open(yo::MEMORY)?;
    let one = db.graph("social")?;
    let ada = one.add(&person(1, "london", 36))?;

    let two = db.graph("social")?;
    assert_eq!(two.nodes()?, 1);
    assert_eq!(two.get(ada)?, Some(person(1, "london", 36)));
    assert_eq!(two.name()?, "social");
    assert_eq!(one.clone().nodes()?, 1);
    Ok(())
}

#[test]
fn a_hop_that_no_edge_of_that_kind_has_taken_is_empty_and_not_an_error() -> yo::Result<()> {
    // A label registers itself the first time it is used, so a read of a label
    // that has never been written has nothing to look in. It should answer
    // nothing rather than create the label or complain.
    let db = yo::open(yo::MEMORY)?;
    let g = db.graph("social")?;
    let ada = g.add(&person(1, "london", 36))?;
    assert!(g.out::<Follows>(ada)?.is_empty());
    assert_eq!(g.degree::<Follows>(ada)?, 0);
    assert!(!g.unlink::<Follows>(ada, ada)?);
    assert!(g.walk(ada).out::<Follows>()?.is_empty());
    Ok(())
}

#[test]
fn a_big_enough_walk_still_lands_where_it_should() -> yo::Result<()> {
    // A thousand people in a ring of ten, so a walk of ten hops comes back to
    // where it started. Enough nodes that the dense ids are not all in one run
    // and enough hops that a frontier that had lost a node would show up.
    let db = yo::open(yo::MEMORY)?;
    let g = db.graph("social")?;

    let mut ids = Vec::new();
    for id in 0..1000u64 {
        ids.push(g.add(&person(id, "london", id as i64 % 90))?);
    }
    for id in 0..1000u64 {
        let next = (id + 1) % 1000;
        g.link(
            ids[id as usize],
            ids[next as usize],
            &Follows { since: 2026 },
        )?;
    }
    assert_eq!(g.edges()?, 1000);

    let mut w = g.walk(ids[0]);
    for _ in 0..1000 {
        w = w.out::<Follows>()?;
        assert_eq!(w.len(), 1, "a ring has one node at every distance");
    }
    assert_eq!(w.ids(), vec![ids[0]]);
    Ok(())
}
