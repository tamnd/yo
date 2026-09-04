//! One column of a schema: where it reads from, what it holds and what may be
//! asked of it.

use yo_shape::Metric;

/// The default weight a `TEXT` field carries when nobody names one.
pub const WEIGHT: f64 = 1.0;
/// The character a `TAG` field splits on when nobody names one.
pub const SEPARATOR: u8 = b',';
/// The out degree a vector field's graph is built with when nobody names one.
pub const M: u64 = 16;
/// The build beam width a vector field uses when nobody names one.
pub const EF_CONSTRUCTION: u64 = 200;
/// The search beam width a vector field uses when nobody names one.
pub const EF_RUNTIME: u64 = 10;
/// How far past the requested count a range search reaches by default.
pub const EPSILON: f64 = 0.01;
/// The out degree the Vamana form builds with when nobody names one, which is
/// not the same number the graph form uses.
pub const GRAPH_MAX_DEGREE: u64 = 32;
/// The build window the Vamana form uses when nobody names one.
pub const CONSTRUCTION_WINDOW: u64 = 200;
/// What the Vamana form reports when the client asked for no compression,
/// which is what it does when the client did not mention compression at all.
pub const NO_COMPRESSION: &str = "NO_COMPRESSION";
/// How many vectors the Vamana form gathers before it works a compression out,
/// when the client asked for compression and did not say how many.
pub const TRAINING_THRESHOLD: u64 = 10240;
/// The smallest training threshold that means anything, which is the block the
/// compression is worked out over.
pub const MIN_TRAINING: u64 = 1024;

/// What a field holds, and the options that only make sense for that.
#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    /// Free text, tokenised into words and scored.
    Text(Text),
    /// A short string split on one character, matched whole and never scored.
    Tag(Tag),
    /// A number, matched by value or by range.
    Numeric,
    /// A longitude and latitude pair, matched by distance from a point.
    Geo,
    /// A well known text shape, matched by containment or intersection.
    GeoShape(Coords),
    /// An embedding, matched by nearest neighbour.
    Vector(Vector),
}

impl Kind {
    /// The word `FT.INFO` reports this kind under, which is the same word
    /// `FT.CREATE` took.
    #[must_use]
    pub const fn token(&self) -> &'static str {
        match self {
            Kind::Text(_) => "TEXT",
            Kind::Tag(_) => "TAG",
            Kind::Numeric => "NUMERIC",
            Kind::Geo => "GEO",
            Kind::GeoShape(_) => "GEOSHAPE",
            Kind::Vector(_) => "VECTOR",
        }
    }

    /// Whether a field of this kind may be asked to index the empty string.
    ///
    /// Text and tag only. A number has no empty form and a coordinate pair has
    /// none either, so `INDEXEMPTY` on one of those is not a flag that does
    /// nothing, it is an argument the parser has never heard of and says so.
    #[must_use]
    pub const fn takes_empty(&self) -> bool {
        matches!(self, Kind::Text(_) | Kind::Tag(_))
    }
}

/// A `TEXT` field's own options.
#[derive(Debug, Clone, PartialEq)]
pub struct Text {
    /// How much a match in this field counts for against a match in another
    /// one. Any float, negative included, because a real server takes any float
    /// here and reports it back.
    pub weight: f64,
    /// Whether words in this field keep the ending they arrived with.
    pub nostem: bool,
    /// The phonetic matcher, as the client spelled it.
    ///
    /// Kept because the parser validates it and refuses the ones it does not
    /// know, and dropped from every reply, which is what a real server does
    /// with it: `FT.INFO` never mentions phonetic at all.
    pub phonetic: Option<Box<[u8]>>,
}

impl Default for Text {
    fn default() -> Text {
        Text {
            weight: WEIGHT,
            nostem: false,
            phonetic: None,
        }
    }
}

/// A `TAG` field's own options.
#[derive(Debug, Clone, PartialEq)]
pub struct Tag {
    /// The one character the value is split on.
    pub separator: u8,
    /// Whether two tags that differ only in case are two tags.
    pub casesensitive: bool,
}

impl Default for Tag {
    fn default() -> Tag {
        Tag {
            separator: SEPARATOR,
            casesensitive: false,
        }
    }
}

/// Which plane a `GEOSHAPE` field's shapes live on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Coords {
    /// Cartesian, where the coordinates are just numbers.
    Flat,
    /// Longitude and latitude on a sphere.
    Spherical,
}

impl Coords {
    /// The word `FT.INFO` reports the coordinate system under.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Coords::Flat => "FLAT",
            Coords::Spherical => "SPHERICAL",
        }
    }
}

/// Which index a vector field asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algo {
    /// Every vector measured, which is exact and costs the collection.
    Flat,
    /// A navigable small world graph, which is approximate.
    Hnsw,
    /// The tiered form, which writes into a flat buffer and moves into the
    /// graph behind the client.
    Svs,
}

impl Algo {
    /// The word `FT.CREATE` takes and `FT.INFO` gives back.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Algo::Flat => "FLAT",
            Algo::Hnsw => "HNSW",
            Algo::Svs => "SVS-VAMANA",
        }
    }
}

/// How wide one coordinate of a vector is on the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Width {
    /// A signed byte, which is what a quantised model emits.
    Int8,
    /// An unsigned byte.
    Uint8,
    /// Half precision.
    Float16,
    /// Brain float, which is half precision with the exponent of a float.
    BFloat16,
    /// Single precision, which is what nearly every model emits.
    Float32,
    /// Double precision.
    Float64,
}

impl Width {
    /// The word `FT.CREATE` takes and `FT.INFO` gives back.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Width::Int8 => "INT8",
            Width::Uint8 => "UINT8",
            Width::Float16 => "FLOAT16",
            Width::BFloat16 => "BFLOAT16",
            Width::Float32 => "FLOAT32",
            Width::Float64 => "FLOAT64",
        }
    }

    /// How many bytes one coordinate takes.
    #[must_use]
    pub const fn bytes(self) -> usize {
        match self {
            Width::Int8 | Width::Uint8 => 1,
            Width::Float16 | Width::BFloat16 => 2,
            Width::Float32 => 4,
            Width::Float64 => 8,
        }
    }
}

/// A `VECTOR` field's own options.
///
/// Three of them are required and the rest have defaults, which is why the
/// count in front of them on the wire is a count of words rather than a count
/// of options: a client sends `6 TYPE FLOAT32 DIM 4 DISTANCE_METRIC COSINE` and
/// the six is the six words.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    /// Which index was asked for.
    pub algo: Algo,
    /// How wide one coordinate is.
    pub width: Width,
    /// How many coordinates a vector has.
    pub dim: u64,
    /// What the index measures.
    pub metric: Metric,
    /// How many vectors to make room for up front, if the client said.
    pub initial_cap: Option<u64>,
    /// How many vectors a flat index puts in one block, if the client said.
    pub block_size: Option<u64>,
    /// The out degree of the graph, for the graph forms.
    pub m: u64,
    /// The beam width a build uses.
    pub ef_construction: u64,
    /// The beam width a search uses when the query does not name one.
    pub ef_runtime: u64,
    /// How far past the requested range a range query reaches.
    pub epsilon: f64,
    /// The out degree the Vamana form builds with.
    pub graph_max_degree: u64,
    /// The build window the Vamana form uses.
    pub construction_window: u64,
    /// What the Vamana form compresses its vectors to, in the spelling the
    /// module knows the scheme by, or `None` for no compression at all.
    pub compression: Option<Box<[u8]>>,
    /// How many vectors are gathered before the compression is worked out,
    /// which only means anything next to a compression and is reported only
    /// when there is one.
    pub training_threshold: Option<u64>,
}

impl Vector {
    /// A vector field with the three things that have no default filled in and
    /// everything else where a real server leaves it.
    #[must_use]
    pub fn new(algo: Algo, width: Width, dim: u64, metric: Metric) -> Vector {
        Vector {
            algo,
            width,
            dim,
            metric,
            initial_cap: None,
            block_size: None,
            m: M,
            ef_construction: EF_CONSTRUCTION,
            ef_runtime: EF_RUNTIME,
            epsilon: EPSILON,
            graph_max_degree: GRAPH_MAX_DEGREE,
            construction_window: CONSTRUCTION_WINDOW,
            compression: None,
            training_threshold: None,
        }
    }

    /// The word `FT.INFO` reports the metric under, which is upper case where
    /// the rest of this build writes it lower case.
    #[must_use]
    pub const fn metric_token(&self) -> &'static str {
        match self.metric {
            Metric::L2 => "L2",
            Metric::Cosine => "COSINE",
            Metric::Ip => "IP",
            Metric::Hamming => "HAMMING",
        }
    }
}

/// One field of a schema.
///
/// The identifier is where the value is read from, which is a hash field name
/// or a JSON path, and the attribute is what a query calls it. They are the
/// same bytes unless the client said `AS`.
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    /// Where the value comes from.
    pub identifier: Box<[u8]>,
    /// What a query names it.
    pub attribute: Box<[u8]>,
    /// What it holds.
    pub kind: Kind,
    /// Whether the value is kept beside the document so a sort does not have to
    /// read the document back.
    pub sortable: bool,
    /// Whether a sortable copy is kept exactly as it arrived rather than folded
    /// and trimmed the way the index folds it.
    pub unf: bool,
    /// Whether the field is left out of the index entirely, which leaves it
    /// usable for sorting and for returning and useless for matching.
    pub noindex: bool,
    /// Whether a suffix trie is built, which is what makes a query ending in a
    /// wildcard cheap.
    pub suffix_trie: bool,
    /// Whether a document with an empty value at this field is indexed under
    /// the empty value rather than left out.
    pub index_empty: bool,
    /// Whether a document with no value at this field at all is recorded, so a
    /// query can ask for the ones that are missing it.
    pub index_missing: bool,
}

impl Field {
    /// A field with the identifier standing in for the attribute and every
    /// option off, which is what `SCHEMA name TYPE` means.
    #[must_use]
    pub fn new(identifier: &[u8], kind: Kind) -> Field {
        Field {
            identifier: identifier.into(),
            attribute: identifier.into(),
            kind,
            sortable: false,
            unf: false,
            noindex: false,
            suffix_trie: false,
            index_empty: false,
            index_missing: false,
        }
    }

    /// The same with a different name to query it by, which is `AS`.
    #[must_use]
    pub fn named(mut self, attribute: &[u8]) -> Field {
        self.attribute = attribute.into();
        self
    }

    /// Whether a sortable copy of this field is normalised.
    ///
    /// A number that is sortable is always unnormalised, because there is
    /// nothing to normalise about it, and a real server reports `UNF` on every
    /// sortable numeric field whether or not the client asked. Everything else
    /// reports it only when asked.
    #[must_use]
    pub fn is_unf(&self) -> bool {
        self.unf || (self.sortable && self.kind == Kind::Numeric)
    }
}
