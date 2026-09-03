//! HNSW as a compatibility view, not as an index (`10` section 7).
//!
//! Clients pass `M`, `EF_CONSTRUCTION` and `EF_RUNTIME` to `VADD` and
//! `FT.CREATE` and expect them to do something, because against Redis and
//! valkey they do. There is no graph here to point them at.
//!
//! There are three things you can do about that and two of them are bad. You
//! can reject the parameters, which breaks every client that has ever created a
//! vector index. You can accept them and do nothing, which is worse, because
//! someone raises `EF_RUNTIME` to fix their recall and nothing happens and they
//! have no way to find out why. Or you can work out what each one was for and
//! do that thing, which is what this is.
//!
//! # The mapping
//!
//! `EF_CONSTRUCTION` is how hard the index works while building. Here that is
//! the size a posting is split and merged around, which is what decides how
//! many vectors a probe reads and how finely the space is cut.
//!
//! `EF_RUNTIME` is the search beam, the pool of candidates a query keeps. Here
//! that is [`Tuning::probe`] and [`Tuning::rerank`], which are the two things
//! that decide how much a query looks at.
//!
//! `INITIAL_CAP` is how many vectors are coming, so the postings for them can
//! be allocated up front instead of grown into.
//!
//! `M` is the out degree of a graph. There is no graph, so there is nothing for
//! it to mean, and it is recorded and echoed back by `FT.INFO` and `VINFO` and
//! changes nothing. That is not a fudge. Quietly mapping it onto some unrelated
//! knob so the number looks used would be worse than admitting it does not
//! apply.
//!
//! # What the client is actually promised
//!
//! Not that a given `EF_RUNTIME` reads the same number of vectors it would
//! under HNSW. It does not, and it could not: a graph walk at `EF_RUNTIME` 10
//! touches tens of vectors and a probe of eight postings here touches two
//! thousand, because measuring one against a code is a popcount and following a
//! graph edge is a cache miss. The absolute numbers are not comparable and
//! pretending they are would be the lie.
//!
//! What is promised is the thing a client relies on, which is that the knob
//! responds: raise `EF_RUNTIME` and the search does proportionally more work
//! and finds more, lower it and it does less and finds less. That is what
//! somebody turning it up at three in the morning needs to be true, and
//! `an_ef_runtime_client_gets_what_it_turned_the_knob_for` measures it rather
//! than asserting it.
//!
//! # When somebody really does want a graph
//!
//! [`Compat::Strict`] exists for that, and it does not build one. It reports
//! [`Plan::Graph`], and the layer above turns that into a refusal, so a client
//! that asked for HNSW and meant HNSW is told it is not here rather than being
//! served something else under the name. The default is [`Compat::Permissive`],
//! where the parameters are honoured as above and the partition index serves.
//!
//! ```
//! use yo_vector::hnsw::{Compat, Plan, Requested};
//!
//! // What FT.CREATE sends when nobody set anything.
//! let asked = Requested::default();
//! let Plan::Partitions { tuning, .. } = asked.plan(Compat::Permissive) else {
//!     panic!("permissive serves it")
//! };
//! // Redis's defaults come out as ours, so a client that set nothing gets the
//! // index tuned the way it would have been anyway.
//! assert_eq!(tuning.posting, yo_vector::Tuning::default().posting);
//!
//! // And a client that meant it is not quietly served something else.
//! assert!(matches!(asked.plan(Compat::Strict), Plan::Graph));
//! ```

use crate::Tuning;

/// Redis's default `M`, which is echoed and otherwise ignored.
pub const M: usize = 16;
/// Redis's default `EF_CONSTRUCTION`, which maps to our default posting size.
pub const EF_CONSTRUCTION: usize = 200;
/// Redis's default `EF_RUNTIME`, which maps to our default probe and rerank.
pub const EF_RUNTIME: usize = 10;

/// What a client asked for when it said HNSW.
///
/// Every field is optional because every one of them is optional on the wire,
/// and the whole thing is `Copy` and public because most of what happens to it
/// is being handed straight back out by `FT.INFO`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Requested {
    /// The graph out degree, which is recorded and echoed and does nothing.
    pub m: usize,
    /// How hard to work while building, which sets the posting size.
    pub ef_construction: usize,
    /// The search beam, which sets the probe and the rerank width.
    pub ef_runtime: usize,
    /// How many vectors are coming, so the postings can be there already.
    pub initial_cap: Option<usize>,
}

impl Default for Requested {
    /// What Redis uses when the client set nothing.
    fn default() -> Requested {
        Requested {
            m: M,
            ef_construction: EF_CONSTRUCTION,
            ef_runtime: EF_RUNTIME,
            initial_cap: None,
        }
    }
}

/// Whether an HNSW request is served by the partition index or refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Compat {
    /// Honour the parameters, serve with the partition index, and say in
    /// `FT.INFO` what is really running. This is the default and it is what
    /// almost everybody wants, because almost nobody asked for a graph, they
    /// asked for vector search and HNSW is what the last engine called it.
    #[default]
    Permissive,
    /// A request for HNSW means HNSW. Nothing here builds one, so this refuses
    /// rather than substituting.
    Strict,
}

/// What to do with an HNSW request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Plan {
    /// Serve it here, tuned like this, with this many postings ready.
    Partitions {
        /// The partition index tuning the parameters map to.
        tuning: Tuning,
        /// How many postings to allocate up front, from `INITIAL_CAP`.
        capacity: usize,
    },
    /// The client asked for a real graph under [`Compat::Strict`]. There is not
    /// one, so the layer above refuses.
    Graph,
}

impl Requested {
    /// What to do about it.
    #[must_use]
    pub fn plan(&self, compat: Compat) -> Plan {
        match compat {
            Compat::Strict => Plan::Graph,
            Compat::Permissive => Plan::Partitions {
                tuning: self.tuning(),
                capacity: self.capacity(),
            },
        }
    }

    /// The partition index tuning these parameters ask for.
    ///
    /// Everything is scaled against the defaults, so a client that set nothing
    /// gets exactly [`Tuning::default`] and a client that doubled a parameter
    /// gets roughly double whatever it controls. The clamps at either end are
    /// there because these numbers arrive from the network and nothing stops
    /// somebody sending `EF_RUNTIME 4000000000`.
    #[must_use]
    pub fn tuning(&self) -> Tuning {
        let base = Tuning::default();
        Tuning {
            posting: scale(base.posting, self.ef_construction, EF_CONSTRUCTION).clamp(32, 4096),
            probe: scale(base.probe, self.ef_runtime, EF_RUNTIME).clamp(1, 256),
            rerank: scale(base.rerank, self.ef_runtime, EF_RUNTIME).clamp(1, 64),
            // Not derived from anything HNSW has, because HNSW has no filters
            // to widen for.
            ..base
        }
    }

    /// How many postings to have ready, from `INITIAL_CAP`.
    ///
    /// Zero when the client did not say, which means grow into it as usual.
    #[must_use]
    pub fn capacity(&self) -> usize {
        match self.initial_cap {
            Some(n) if n > 0 => n.div_ceil(self.tuning().posting),
            _ => 0,
        }
    }
}

/// `base * asked / default`, rounded rather than truncated so that halving a
/// small number does not land on zero by accident.
fn scale(base: usize, asked: usize, default: usize) -> usize {
    let asked = asked.min(1 << 24);
    (base * asked + default / 2) / default
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bits, Partitions, Vectors};
    use yo_common::Rng;

    #[test]
    fn the_defaults_a_client_did_not_set_are_the_defaults_it_would_have_got() {
        let tuning = Requested::default().tuning();
        let base = Tuning::default();
        assert_eq!(tuning.posting, base.posting);
        assert_eq!(tuning.probe, base.probe);
        assert_eq!(tuning.rerank, base.rerank);
        assert_eq!(tuning.sweep, base.sweep);
        assert_eq!(tuning.widen, base.widen);
        assert_eq!(Requested::default().capacity(), 0);
    }

    #[test]
    fn m_is_echoed_and_changes_nothing() {
        let base = Requested::default();
        let plenty = Requested { m: 512, ..base };
        assert_eq!(plenty.tuning().posting, base.tuning().posting);
        assert_eq!(plenty.tuning().probe, base.tuning().probe);
        // And it comes back out for FT.INFO, which is all it was ever for.
        assert_eq!(plenty.m, 512);
    }

    #[test]
    fn ef_construction_moves_the_posting_size_and_nothing_else() {
        let base = Requested::default();
        let harder = Requested {
            ef_construction: EF_CONSTRUCTION * 2,
            ..base
        };
        assert_eq!(harder.tuning().posting, base.tuning().posting * 2);
        assert_eq!(harder.tuning().probe, base.tuning().probe);
    }

    #[test]
    fn ef_runtime_moves_the_probe_and_the_rerank_and_nothing_else() {
        let base = Requested::default();
        let wider = Requested {
            ef_runtime: EF_RUNTIME * 4,
            ..base
        };
        assert_eq!(wider.tuning().probe, base.tuning().probe * 4);
        assert_eq!(wider.tuning().rerank, base.tuning().rerank * 4);
        assert_eq!(wider.tuning().posting, base.tuning().posting);
    }

    #[test]
    fn a_number_off_the_network_cannot_ask_for_something_absurd() {
        let silly = Requested {
            m: usize::MAX,
            ef_construction: usize::MAX,
            ef_runtime: usize::MAX,
            initial_cap: Some(usize::MAX),
        };
        let tuning = silly.tuning();
        assert_eq!(tuning.posting, 4096);
        assert_eq!(tuning.probe, 256);
        assert_eq!(tuning.rerank, 64);

        // And nothing goes to zero at the other end, because a probe of zero is
        // an index that answers nothing.
        let none = Requested {
            ef_construction: 0,
            ef_runtime: 0,
            ..Requested::default()
        };
        assert_eq!(none.tuning().posting, 32);
        assert_eq!(none.tuning().probe, 1);
        assert_eq!(none.tuning().rerank, 1);
    }

    #[test]
    fn initial_cap_asks_for_the_postings_the_vectors_will_need() {
        let asked = Requested {
            initial_cap: Some(100_000),
            ..Requested::default()
        };
        // A hundred thousand vectors at 256 to a posting.
        assert_eq!(asked.capacity(), 391);

        // And it follows EF_CONSTRUCTION, because that is what sets the size.
        let bigger = Requested {
            ef_construction: EF_CONSTRUCTION * 2,
            ..asked
        };
        assert_eq!(bigger.capacity(), 196);
    }

    #[test]
    fn strict_refuses_rather_than_serving_something_else() {
        let asked = Requested::default();
        assert_eq!(asked.plan(Compat::Strict), Plan::Graph);
        assert_eq!(
            asked.plan(Compat::Permissive),
            Plan::Partitions {
                tuning: asked.tuning(),
                capacity: 0,
            }
        );
        // Permissive is the default, because almost nobody asked for a graph.
        assert_eq!(Compat::default(), Compat::Permissive);
    }

    struct Store(Vec<Vec<f32>>);

    impl Vectors for Store {
        fn get(&self, id: u64, into: &mut [f32]) -> bool {
            match self.0.get(id as usize) {
                Some(v) => {
                    into.copy_from_slice(v);
                    true
                }
                None => false,
            }
        }
    }

    fn corpus(dim: usize, n: usize, clusters: usize, seed: u64) -> Store {
        let mut rng = Rng::new(seed);
        let centres: Vec<Vec<f32>> = (0..clusters).map(|_| draw(dim, &mut rng)).collect();
        Store(
            (0..n)
                .map(|i| {
                    let off = draw(dim, &mut rng);
                    let mut v: Vec<f32> = centres[i % clusters]
                        .iter()
                        .zip(&off)
                        .map(|(c, o)| c + o * 0.7)
                        .collect();
                    unit(&mut v);
                    v
                })
                .collect(),
        )
    }

    fn draw(dim: usize, rng: &mut Rng) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim)
            .map(|i| {
                let u = (rng.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
                let heavy = if i < dim / 16 { 6.0 } else { 1.0 };
                (u * 2.0 - 1.0) * heavy
            })
            .collect();
        unit(&mut v);
        v
    }

    fn unit(v: &mut [f32]) {
        let len = v.iter().map(|c| c * c).sum::<f32>().sqrt();
        for c in v {
            *c /= len;
        }
    }

    fn truth(q: &[f32], store: &Store, k: usize) -> Vec<u64> {
        let mut all: Vec<(u64, f32)> = store
            .0
            .iter()
            .enumerate()
            .map(|(i, v)| {
                (
                    i as u64,
                    q.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum::<f32>(),
                )
            })
            .collect();
        all.sort_by(|a, b| a.1.total_cmp(&b.1));
        all[..k].iter().map(|(i, _)| *i).collect()
    }

    /// Build one index at the posting size the parameters ask for, then search
    /// it at the probe and rerank they ask for, and say what fraction of the
    /// true nearest ten came back.
    fn recall(dim: usize, asked: Requested, seed: u64) -> f32 {
        let store = corpus(dim, 4000, 24, seed);
        let mut ix = Partitions::new(dim, Bits::One, 7, asked.tuning());
        for (id, v) in store.0.iter().enumerate() {
            ix.insert(id as u64, v);
            if id % 128 == 0 {
                ix.maintain(&store, 4096);
            }
        }
        ix.maintain(&store, 1 << 20);

        let queries = corpus(dim, 30, 24, seed ^ 0x5eed);
        let mut hits = 0usize;
        for q in &queries.0 {
            let want = truth(q, &store, 10);
            let got: Vec<u64> = ix.search(q, 10, &store).into_iter().map(|h| h.id).collect();
            hits += want.iter().filter(|id| got.contains(id)).count();
        }
        hits as f32 / (queries.0.len() * 10) as f32
    }

    /// The promise the mapping actually makes, measured rather than asserted.
    ///
    /// Not that a given `EF_RUNTIME` reads the same vectors it would under a
    /// graph, which it does not and could not. That turning it up finds more,
    /// which is the thing somebody turning it up is relying on.
    #[test]
    fn an_ef_runtime_client_gets_what_it_turned_the_knob_for() {
        let dim = 64;
        let low = recall(
            dim,
            Requested {
                ef_runtime: 1,
                ..Requested::default()
            },
            0x1379,
        );
        let high = recall(
            dim,
            Requested {
                ef_runtime: EF_RUNTIME * 8,
                ..Requested::default()
            },
            0x1379,
        );
        assert!(
            high > low + 0.05,
            "raising EF_RUNTIME went from {low} to {high}, which is not a knob doing anything"
        );
        assert!(high >= 0.95, "the top of the range only reached {high}");
    }

    /// The same for `EF_CONSTRUCTION`, which is a build time knob, so what it
    /// buys is a finer cut of the space for the same probe.
    #[test]
    fn an_ef_construction_client_gets_what_it_turned_the_knob_for() {
        let dim = 64;
        let coarse = recall(
            dim,
            Requested {
                ef_construction: EF_CONSTRUCTION * 8,
                ef_runtime: 2,
                ..Requested::default()
            },
            0x2468,
        );
        let fine = recall(
            dim,
            Requested {
                ef_construction: EF_CONSTRUCTION / 4,
                ef_runtime: 2,
                ..Requested::default()
            },
            0x2468,
        );
        assert!(
            fine > coarse + 0.05,
            "smaller postings for the same probe went from {coarse} to {fine}"
        );
    }
}
