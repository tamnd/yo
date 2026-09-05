//! What an index is: which keys it follows and what it reads out of them.

use crate::field::Field;
use crate::held::Held;

/// The score every document in an index gets when nothing says otherwise.
pub const SCORE: f64 = 1.0;

/// Which kind of value an index follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// Hashes, where a field identifier is a field name.
    Hash,
    /// Documents, where a field identifier is a path into one.
    Json,
}

impl Source {
    /// The word `FT.CREATE` takes after `ON` and `FT.INFO` gives back.
    #[must_use]
    pub const fn token(self) -> &'static str {
        match self {
            Source::Hash => "HASH",
            Source::Json => "JSON",
        }
    }
}

/// The flags that go in `FT.INFO`'s `index_options` array.
///
/// Five bits and not five booleans on the definition, because the reply writes
/// them in a fixed order and reading them back in that order is what keeps the
/// reply and the parse from drifting apart.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Options {
    /// No term offsets, which costs highlighting and exact phrase matching.
    pub nooffsets: bool,
    /// No highlighting, which `NOOFFSETS` implies because there is nothing to
    /// highlight from.
    pub nohl: bool,
    /// No record of which field a term came from, so a query cannot ask.
    pub nofields: bool,
    /// No term frequencies, which costs scoring.
    pub nofreqs: bool,
    /// Room for the full number of text fields rather than the eight a
    /// compact index has.
    pub maxtextfields: bool,
}

impl Options {
    /// The words this set is reported as, in the order a real server writes
    /// them.
    #[must_use]
    pub fn tokens(self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.nooffsets {
            out.push("NOOFFSETS");
        }
        if self.nohl || self.nooffsets {
            out.push("NOHL");
        }
        if self.nofields {
            out.push("NOFIELDS");
        }
        if self.nofreqs {
            out.push("NOFREQS");
        }
        if self.maxtextfields {
            out.push("MAXTEXTFIELDS");
        }
        out
    }
}

/// Which keys an index follows and how it reads them.
#[derive(Debug, Clone)]
pub struct Definition {
    /// Hashes or documents.
    pub on: Source,
    /// The key prefixes this index follows. One empty prefix, which follows
    /// every key, when the client named none.
    pub prefixes: Vec<Box<[u8]>>,
    /// An expression a key has to satisfy on top of its prefix.
    pub filter: Option<Box<[u8]>>,
    /// The language documents are stemmed in when nothing else says.
    pub language: Option<Box<[u8]>>,
    /// A field naming the language per document.
    pub language_field: Option<Box<[u8]>>,
    /// The score every document gets when nothing else says.
    pub score: f64,
    /// A field naming the score per document.
    pub score_field: Option<Box<[u8]>>,
    /// A field whose value is carried through a search untouched.
    pub payload_field: Option<Box<[u8]>>,
    /// The five reported flags.
    pub options: Options,
    /// Whether the keys that were already there were left alone.
    pub skip_initial_scan: bool,
    /// How many seconds of idleness this index survives, if it is temporary.
    pub temporary: Option<u64>,
    /// The stop words this index uses, or `None` for the built in list.
    pub stopwords: Option<Vec<Box<[u8]>>>,
}

impl Default for Definition {
    fn default() -> Definition {
        Definition {
            on: Source::Hash,
            prefixes: vec![Box::default()],
            filter: None,
            language: None,
            language_field: None,
            score: SCORE,
            score_field: None,
            payload_field: None,
            options: Options::default(),
            skip_initial_scan: false,
            temporary: None,
            stopwords: None,
        }
    }
}

impl Definition {
    /// Whether this index follows every key of its kind.
    ///
    /// Which is what one empty prefix and no filter means, and is the field
    /// `FT.INFO` reports as `indexes_all`.
    #[must_use]
    pub fn indexes_all(&self) -> bool {
        self.filter.is_none() && self.prefixes.iter().all(|p| p.is_empty())
    }

    /// Whether a key is one this index follows, on its name alone.
    ///
    /// The filter is not applied here, because it is an expression over the
    /// value and this is the half that can be decided from the key.
    #[must_use]
    pub fn covers(&self, key: &[u8]) -> bool {
        self.prefixes.iter().any(|p| key.starts_with(p))
    }
}

/// One index: what it is called, which keys it follows and what it reads.
#[derive(Debug, Clone)]
pub struct Index {
    /// The name `FT.SEARCH` is given.
    pub name: Box<[u8]>,
    /// Which keys it follows.
    pub definition: Definition,
    /// What it reads out of them, in the order the client declared it.
    pub schema: Vec<Field>,
    /// How many times this index has been opened.
    ///
    /// A real server counts opens rather than commands and reports the count
    /// from `FT.INFO`, which is itself an open, so the first `FT.INFO` after a
    /// create answers one. Kept here because it belongs to the index and
    /// survives everything except dropping it.
    pub uses: u64,
    /// What it has read: the documents, the terms, the numbers and the values.
    pub held: Held,
}

impl Index {
    /// An index with a definition and a schema and nothing in it yet.
    #[must_use]
    pub fn new(name: &[u8], definition: Definition, schema: Vec<Field>) -> Index {
        Index {
            name: name.into(),
            definition,
            schema,
            uses: 0,
            held: Held::new(),
        }
    }

    /// The field a query means by a name, which is the attribute and not the
    /// identifier.
    #[must_use]
    pub fn field(&self, attribute: &[u8]) -> Option<&Field> {
        self.schema.iter().find(|f| &*f.attribute == attribute)
    }

    /// Whether the schema already has a field a query would call `attribute`.
    #[must_use]
    pub fn has(&self, attribute: &[u8]) -> bool {
        self.field(attribute).is_some()
    }
}
