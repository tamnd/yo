//! The knobs a client turns with `FT.CONFIG`, and what each of them will take.
//!
//! Sixty nine of them, in the order a real server dumps them, because the order
//! is the answer to `FT.CONFIG GET *` and a set would answer in whatever order
//! it happened to be in. The names, the help text and the defaults are all
//! copied from a running 8.10.1 rather than invented, so a client that reads
//! the dump and writes it back gets the same server it started with.
//!
//! # A name is matched whole, and case does not count
//!
//! `FT.CONFIG GET timeout` and `FT.CONFIG GET TIMEOUT` are the same question
//! and both answer under the spelling the table holds. Nothing globs. `TIME*`
//! and `*TIMEOUT*` both answer an empty list, and only the single character `*`
//! means everything, which is measured and is worth knowing because the sister
//! command `CONFIG GET` does glob. A name nobody knows answers an empty list
//! rather than an error, on `GET` and on `HELP` alike.
//!
//! # Reading is always allowed and writing often is not
//!
//! Twenty six of the settings are read only once the server is up, and asking
//! to change one is told `Not modifiable at runtime` whatever value came with
//! it. That check runs before the value is looked at, so `FT.CONFIG SET NOGC x`
//! complains about `NOGC` and never about `x`. An unknown name is refused
//! before either, and too many words are refused after both: the order is name,
//! then whether it may move, then the value, then the count.
//!
//! # Every number has a floor and a ceiling
//!
//! The floor is always the hard one and always complains the same way, with
//! `Value is outside acceptable bounds`. Some ceilings have a complaint of
//! their own, and those are the interesting ones, since a message like
//! `Number of worker threads cannot exceed 16` tells a client what to do next
//! where the general one does not. A handful of settings carry a second floor
//! with its own message too, and two of them are checked against each other
//! rather than against a constant.
//!
//! # A number is read twice before it is given up on
//!
//! The strict reader goes first: a plain decimal whole number, an optional
//! minus, and no leading zero unless the whole of it is a zero. When that says
//! no the loose reader has a go, which is C's `strtod` and takes a great deal
//! more: `+5` is five, `0x10` is sixteen, `010` is ten rather than eight, `1e3`
//! is a thousand, and `1.5` is refused for having a fraction rather than for
//! being a bad number. A space on either end is refused by both.
//!
//! Which reader answered is visible, because a number that is out of range
//! after the strict reader is told it is out of bounds and one that is out of
//! range after the loose reader is told it is the wrong shape. That is how
//! `TIMEOUT -16` and `TIMEOUT -0x10` end up with two different complaints for
//! the same number, and both of them are measured.
//!
//! # Two settings are two names for one number
//!
//! `MAXEXPANSIONS` and `MAXPREFIXEXPANSIONS` are the same number under two
//! spellings, and writing either moves both. So are
//! `FORK_GC_CLEAN_NUMERIC_EMPTY_NODES` and the one with the leading underscore,
//! except that the first of the pair takes no value at all: naming it turns it
//! on, and handing it a value is answered `EXCESSARGS`.
//!
//! # What is stored and what is honoured
//!
//! Everything here is remembered and read back exactly, and only some of it
//! changes what a search does. That gap is D-89 and it closes setting by
//! setting rather than all at once, because each one needs measuring against a
//! real server before it can be trusted to mean anything.

/// Every setting, in the order `FT.CONFIG GET *` answers with.
pub const TABLE: &[Setting] = &[
    fixed("EXTLOAD", "Load extension scoring/expansion module", None),
    fixed(
        "NOGC",
        "Disable garbage collection (for this process)",
        Some("false"),
    ),
    whole(
        "MINPREFIX",
        "Set the minimum prefix for expansions (`*`)",
        2,
        1,
        i64::MAX,
    ),
    Setting {
        name: "MINSTEMLEN",
        help: "Set the minimum word length to stem (default 4)",
        kind: Kind::Whole(Whole {
            start: 4,
            least: 0,
            most: U32,
            floor: Some((
                2,
                "SEARCH_SYNTAX Minimum stem length cannot be lower than 2",
            )),
            roof: None,
            beside: None,
        }),
        pair: 0,
    },
    whole(
        "FORKGC_SLEEP_BEFORE_EXIT",
        "set the amount of seconds for the fork GC to sleep before exists, should always be set to 0 (other then on tests).",
        0,
        0,
        i64::MAX,
    ),
    fixed(
        "MAXDOCTABLESIZE",
        "Maximum runtime document table size (for this process)",
        Some("1000000"),
    ),
    Setting {
        name: "MAXSEARCHRESULTS",
        help: "Maximum number of results from ft.search command",
        kind: Kind::Wide(1_000_000),
        pair: 0,
    },
    Setting {
        name: "MAXAGGREGATERESULTS",
        help: "Maximum number of results from ft.aggregate command",
        kind: Kind::Wide(-1),
        pair: 0,
    },
    Setting {
        name: "MAX_AGGREGATE_GROUPS",
        help: "Maximum number of GROUPBY groups materialized by ft.aggregate command",
        kind: Kind::Whole(Whole {
            start: 1_000_000,
            least: 1,
            most: i64::MAX,
            floor: None,
            roof: Some((
                67_108_864,
                "SEARCH_LIMIT_OVER Value exceeds maximum possible aggregate groups",
            )),
            beside: None,
        }),
        pair: 0,
    },
    Setting {
        name: "MAXEXPANSIONS",
        help: "Maximum prefix expansions to be used in a query",
        kind: Kind::Whole(Whole {
            start: 200,
            least: 1,
            most: i64::MAX,
            floor: None,
            roof: None,
            beside: None,
        }),
        pair: EXPANSIONS,
    },
    Setting {
        name: "MAXPREFIXEXPANSIONS",
        help: "Maximum prefix expansions to be used in a query",
        kind: Kind::Whole(Whole {
            start: 200,
            least: 1,
            most: i64::MAX,
            floor: None,
            roof: None,
            beside: None,
        }),
        pair: EXPANSIONS,
    },
    whole("TIMEOUT", "Query (search) timeout", 500, 0, i64::MAX),
    whole(
        "_MAX_FOREGROUND_TIMEOUT_LIMIT",
        "Maximum allowed value (ms) for search-timeout and per-query TIMEOUT when workers are disabled (0 = unlimited)",
        60_000,
        0,
        i64::MAX,
    ),
    Setting {
        name: "WORKERS",
        help: "Number of worker threads to use for query processing and background tasks. Default is 0. This configuration also affects the number of connections per shard. See CONN_PER_SHARD.",
        kind: Kind::Whole(Whole {
            start: 4,
            least: 0,
            most: i64::MAX,
            floor: None,
            roof: Some((16, THREADS)),
            beside: None,
        }),
        pair: 0,
    },
    Setting {
        name: "MIN_OPERATION_WORKERS",
        help: "Number of worker threads to use for background tasks when the server is in an operation event. Default is 4",
        kind: Kind::Whole(Whole {
            start: 4,
            least: 0,
            most: i64::MAX,
            floor: None,
            roof: Some((16, THREADS)),
            beside: None,
        }),
        pair: 0,
    },
    fixed(
        "WORKER_THREADS",
        "Deprecated, see WORKERS and MIN_OPERATION_WORKERS",
        Some("0"),
    ),
    fixed(
        "MT_MODE",
        "Deprecated, see WORKERS and MIN_OPERATION_WORKERS",
        Some("MT_MODE_OFF"),
    ),
    fixed(
        "TIERED_HNSW_BUFFER_LIMIT",
        "Use for setting the buffer limit threshold for vector similarity tiered HNSW index, so that if we are using WORKERS for indexing, and the number of vectors waiting in the buffer to be indexed exceeds this limit, we insert new vectors directly into HNSW",
        Some("1024"),
    ),
    fixed(
        "PRIVILEGED_THREADS_NUM",
        "Deprecated. See `WORKERS_PRIORITY_BIAS_THRESHOLD`",
        Some("1"),
    ),
    fixed(
        "WORKERS_PRIORITY_BIAS_THRESHOLD",
        "The number of high priority tasks to be executed at any given time by the worker thread pool, before executing low priority tasks. After this number of high priority tasks are being executed, the worker thread pool will execute high and low priority tasks alternately.",
        Some("1"),
    ),
    fixed(
        "FRISOINI",
        "Path to Chinese dictionary configuration file (for Chinese tokenization)",
        None,
    ),
    Setting {
        name: "DEFAULT_SCORER",
        help: "Default scorer to use when no scorer is specified in queries",
        kind: Kind::Word(Word {
            of: &[
                "TFIDF",
                "TFIDF.DOCNORM",
                "BM25",
                "DISMAX",
                "DOCSCORE",
                "HAMMING",
                "BM25STD",
                "BM25STD.TANH",
            ],
            start: 6,
            folded: false,
            bad: "SEARCH_VALUE_BAD Invalid default scorer value",
        }),
        pair: 0,
    },
    Setting {
        name: "ON_TIMEOUT",
        help: "Action to perform when search timeout is exceeded (choose RETURN or FAIL)",
        kind: Kind::Word(Word {
            of: &["return", "fail"],
            start: 0,
            folded: true,
            bad: "SEARCH_VALUE_BAD Invalid ON_TIMEOUT value",
        }),
        pair: 0,
    },
    whole(
        "GCSCANSIZE",
        "Scan this many documents at a time during every GC iteration",
        100,
        1,
        i64::MAX,
    ),
    whole(
        "MIN_PHONETIC_TERM_LEN",
        "Minimum length of term to be considered for phonetic matching",
        3,
        1,
        i64::MAX,
    ),
    fixed(
        "GC_POLICY",
        "gc policy to use (DEFAULT/LEGACY)",
        Some("fork"),
    ),
    whole(
        "FORK_GC_RUN_INTERVAL",
        "interval (in seconds) in which to run the fork gc (relevant only when fork gc is used)",
        30,
        1,
        i64::MAX,
    ),
    whole(
        "FORK_GC_CLEAN_THRESHOLD",
        "the fork gc will only start to clean when the number of not cleaned document will exceed this threshold",
        100,
        0,
        i64::MAX,
    ),
    whole(
        "FORK_GC_RETRY_INTERVAL",
        "interval (in seconds) in which to retry running the forkgc after failure.",
        5,
        1,
        i64::MAX,
    ),
    Setting {
        name: "FORK_GC_CLEAN_NUMERIC_EMPTY_NODES",
        help: "clean empty nodes from numeric tree",
        kind: Kind::Bare,
        pair: EMPTY_NODES,
    },
    Setting {
        name: "_FORK_GC_CLEAN_NUMERIC_EMPTY_NODES",
        help: "clean empty nodes from numeric tree",
        kind: Kind::Yes(true),
        pair: EMPTY_NODES,
    },
    whole(
        "UNION_ITERATOR_HEAP",
        "minimum number of iterators in a union from which the iterator willswitch to heap based implementation.",
        20,
        1,
        i64::MAX,
    ),
    whole(
        "CURSOR_MAX_IDLE",
        "max idle time allowed to be set for cursor, setting it height might cause high memory consumption.",
        300_000,
        1,
        i64::MAX,
    ),
    whole(
        "INDEX_CURSOR_LIMIT",
        "Max number of cursors for a given index that can be opened inside of a shard. Default is 128",
        128,
        0,
        i64::MAX,
    ),
    fixed(
        "NO_MEM_POOLS",
        "Set RediSearch to run without memory pools",
        Some("false"),
    ),
    fixed(
        "PARTIAL_INDEXED_DOCS",
        "Enable commands filter which optimize indexing on partial hash updates",
        Some("false"),
    ),
    fixed(
        "UPGRADE_INDEX",
        "Relevant only when loading an v1.x rdb, specify argument for upgrading the index.",
        Some("Upgrade config for upgrading"),
    ),
    yes(
        "_NUMERIC_COMPRESS",
        "Enable legacy compression of double to float.",
        false,
    ),
    yes(
        "_FREE_RESOURCE_ON_THREAD",
        "Determine whether some index resources are free on a second thread.",
        true,
    ),
    yes(
        "_PRINT_PROFILE_CLOCK",
        "Disable print of time for ft.profile. For testing only.",
        true,
    ),
    fixed(
        "RAW_DOCID_ENCODING",
        "Disable compression for DocID inverted index. Boost CPU performance.",
        Some("false"),
    ),
    Setting {
        name: "_NUMERIC_RANGES_PARENTS",
        help: "Keep numeric ranges in numeric tree parent nodes of leafs for `x` generations.",
        kind: Kind::Whole(Whole {
            start: 0,
            least: 0,
            most: i64::MAX,
            floor: None,
            roof: Some((
                2,
                "SEARCH_PARSE_ARGS Max depth for range cannot be higher than max depth for balance",
            )),
            beside: None,
        }),
        pair: 0,
    },
    Setting {
        name: "DEFAULT_DIALECT",
        help: "Set RediSearch default dialect version through the lifetime of the server.",
        kind: Kind::Whole(Whole {
            start: 1,
            least: 1,
            most: i64::MAX,
            floor: None,
            roof: Some((
                4,
                "SEARCH_VALUE_BAD Default dialect version cannot be higher than 4",
            )),
            beside: None,
        }),
        pair: 0,
    },
    whole(
        "VSS_MAX_RESIZE",
        "Set RediSearch vector indexes max resize (in bytes).",
        0,
        0,
        i64::MAX,
    ),
    fixed(
        "MULTI_TEXT_SLOP",
        "Set RediSearch delta used to increase positional offsets between array slots for multi text values.Can control the level of separation between phrases in different array slots (related to the SLOP parameter of ft.search command)",
        Some("100"),
    ),
    fixed(
        "BG_INDEX_SLEEP_GAP",
        "The number of iterations to run while performing background indexing before we call usleep(1) (sleep for 1 micro-second) and make sure that we allow redis process other commands.",
        Some("100"),
    ),
    yes(
        "_PRIORITIZE_INTERSECT_UNION_CHILDREN",
        "Intersection iterator orders the children iterators by their relative estimated number of results in ascending order, so that if we see first iterators with a lower count of results we will skip a larger number of results, which translates into faster iteration. If this flag is set, we use this optimization in a way where union iterators are being factorize by the number of their own children, so that we sort by the number of children times the overall estimated number of results instead.",
        false,
    ),
    yes(
        "ENABLE_UNSTABLE_FEATURES",
        "Enable unstable features.",
        false,
    ),
    Setting {
        name: "_BG_INDEX_MEM_PCT_THR",
        help: "Set the percentage of memory usage threshold (out of maxmemory) at which background indexing will stop. The default is 100 percent.",
        kind: Kind::Whole(Whole {
            start: 100,
            least: 0,
            most: i64::MAX,
            floor: None,
            roof: Some((
                100,
                "SEARCH_LIMIT_OVER Memory limit for indexing cannot be greater then 100%",
            )),
            beside: None,
        }),
        pair: 0,
    },
    Setting {
        name: "BM25STD_TANH_FACTOR",
        help: "Set the BM25STD.TANH stretch factor. This is an integer value that divides the argument of the tanh function that is used to normalize the score computed by the BM25STD scorer.The default value is 4.",
        kind: Kind::Whole(Whole {
            start: 4,
            least: 1,
            most: i64::MAX,
            floor: None,
            roof: Some((
                10_000,
                "SEARCH_LIMIT_OVER BM25STD_TANH_FACTOR must be between 1 and 10000 inclusive",
            )),
            beside: None,
        }),
        pair: 0,
    },
    whole(
        "_BG_INDEX_OOM_PAUSE_TIME",
        "Set the time (in seconds) given to the background indexing thread to sleep when it reaches the memory limit, giving time to reallocate memory.The default value is 5 seconds in Redis Enterprise, 0 in Redis OS.",
        0,
        0,
        U32,
    ),
    whole(
        "INDEXER_YIELD_EVERY_OPS",
        "The number of operations to perform before yielding to Redis during indexing while loading",
        1000,
        1,
        U32,
    ),
    Setting {
        name: "BG_INDEX_SLEEP_DURATION_US",
        help: "Sleep duration in microseconds during background indexing periodic sleep (max 999999, usleep POSIX limit)",
        kind: Kind::Whole(Whole {
            start: 1,
            least: 1,
            most: U32,
            floor: None,
            roof: Some((
                999_999,
                "SEARCH_LIMIT_OVER BG_INDEX_SLEEP_DURATION_US must be between 1 and 999999 (usleep POSIX limit)",
            )),
            beside: None,
        }),
        pair: 0,
    },
    Setting {
        name: "ON_OOM",
        help: "Action to perform when search OOM is exceeded (choose RETURN, FAIL or IGNORE)",
        kind: Kind::Word(Word {
            of: &["return", "fail", "ignore"],
            start: 0,
            folded: true,
            bad: "SEARCH_VALUE_BAD Invalid ON_OOM value",
        }),
        pair: 0,
    },
    Setting {
        name: "_MIN_TRIM_DELAY_MS",
        help: "Minimum delay before checking trimming state after slot migration (in milliseconds)",
        kind: Kind::Whole(Whole {
            start: 2000,
            least: 1,
            most: U32,
            floor: None,
            roof: None,
            beside: Some(("_MAX_TRIM_DELAY_MS", true)),
        }),
        pair: 0,
    },
    Setting {
        name: "_MAX_TRIM_DELAY_MS",
        help: "Maximum delay before enabling trimming after slot migration (in milliseconds)",
        kind: Kind::Whole(Whole {
            start: 5000,
            least: 1,
            most: U32,
            floor: None,
            roof: None,
            beside: Some(("_MIN_TRIM_DELAY_MS", false)),
        }),
        pair: 0,
    },
    whole(
        "_TRIMMING_STATE_CHECK_DELAY_MS",
        "Delay between trimming state checks (in milliseconds)",
        100,
        1,
        U32,
    ),
    fixed(
        "_SIMULATE_IN_FLEX",
        "Simulate working under Flex conditions. This is used for testing only.",
        Some("false"),
    ),
    fixed(
        "search-disk-drop-read-cache",
        "Drop OS read cache after each SpeedB read (yes/no, default no)",
        Some("false"),
    ),
    fixed(
        "search-disk-use-direct-reads",
        "Use O_DIRECT for SpeedB reads (yes/no, default no)",
        Some("false"),
    ),
    fixed(
        "PARTITIONS",
        "Number of RediSearch partitions to use",
        Some("AUTO"),
    ),
    fixed(
        "CLUSTER_TIMEOUT",
        "Cluster synchronization timeout",
        Some("0"),
    ),
    Setting {
        name: "OSS_GLOBAL_PASSWORD",
        help: "Deprecated, Global oss cluster password that will be used to connect to other shards",
        kind: Kind::Hidden("Password: *******"),
        pair: 0,
    },
    whole(
        "CONN_PER_SHARD",
        "Number of connections to each shard in the cluster. Default to 0. If 0, the number of connections is set to `WORKERS` + 1.",
        0,
        0,
        i64::MAX,
    ),
    whole(
        "CURSOR_REPLY_THRESHOLD",
        "Maximum number of replies to accumulate before triggering `_FT.CURSOR READ` on the shards",
        1,
        1,
        i64::MAX,
    ),
    fixed(
        "SEARCH_THREADS",
        "Sets the number of search threads in the coordinator thread pool",
        Some("20"),
    ),
    fixed(
        "SEARCH_IO_THREADS",
        "Sets the number of I/O threads in the coordinator",
        Some("1"),
    ),
    whole(
        "TOPOLOGY_VALIDATION_TIMEOUT",
        "Sets the timeout for topology validation (in milliseconds). After this timeout, any pending requests will be processed, even if the topology is not fully connected. Default is 30000 (30 seconds). 0 means no timeout.",
        30_000,
        0,
        i64::MAX,
    ),
    whole(
        "CONNECT_TIMEOUT",
        "Sets the per-attempt timeout for inter-shard connection setup (in milliseconds). Bounds the TCP+TLS handshake so a blackholed SYN does not stall a connection indefinitely. Default is 10000 (10 seconds). 0 disables the timeout.",
        10_000,
        0,
        I32,
    ),
];

/// A setting nobody can move, which is most of them.
const fn fixed(name: &'static str, help: &'static str, value: Option<&'static str>) -> Setting {
    Setting {
        name,
        help,
        kind: Kind::Fixed(value),
        pair: 0,
    }
}

/// A number with nothing but a hard floor and a hard ceiling on it.
const fn whole(
    name: &'static str,
    help: &'static str,
    start: i64,
    least: i64,
    most: i64,
) -> Setting {
    Setting {
        name,
        help,
        kind: Kind::Whole(Whole {
            start,
            least,
            most,
            floor: None,
            roof: None,
            beside: None,
        }),
        pair: 0,
    }
}

/// A `true` or a `false`.
const fn yes(name: &'static str, help: &'static str, start: bool) -> Setting {
    Setting {
        name,
        help,
        kind: Kind::Yes(start),
        pair: 0,
    }
}

/// As far as a thirty two bit unsigned setting reaches.
const U32: i64 = 0xFFFF_FFFF;
/// As far as a thirty two bit signed setting reaches.
const I32: i64 = 0x7FFF_FFFF;

/// The group the two expansion names share.
const EXPANSIONS: u8 = 1;
/// The group the two empty node names share.
const EMPTY_NODES: u8 = 2;

/// The one complaint two settings both make, so it is written once.
const THREADS: &str = "SEARCH_LIMIT_OVER Number of worker threads cannot exceed 16";

/// What a number that is out of its hard range is told.
const BOUNDS: &str = "SEARCH_PARSE_ARGS Value is outside acceptable bounds";
/// What a value that is not the shape the setting wanted is told.
const SHAPE: &str = "SEARCH_PARSE_ARGS Could not convert argument to expected type";
/// What a `SET` with nothing after the name is told.
const NOTHING: &str = "SEARCH_PARSE_ARGS Expected an argument, but none provided";
/// What a name nobody knows is told.
pub const UNKNOWN: &str = "SEARCH_OPTION_INVALID Invalid option";
/// What a name that cannot move is told.
pub const STUCK: &str = "SEARCH_OPTION_BAD Not modifiable at runtime";
/// What too many words after a good `SET` are told, which is not an error.
pub const EXCESS: &[u8] = b"EXCESSARGS";

/// One setting: how it is spelled, what it is for, and what it will take.
pub struct Setting {
    /// The spelling the server answers under, whatever the client typed.
    pub name: &'static str,
    /// The sentence `FT.CONFIG HELP` answers with.
    pub help: &'static str,
    /// The shape of the value and where it starts.
    pub kind: Kind,
    /// Which pair of names share one number, or zero for a setting on its own.
    pub pair: u8,
}

/// The shape of a setting's value.
pub enum Kind {
    /// A whole number, and the range it has to land in.
    Whole(Whole),
    /// A whole number kept in a thirty two bit signed slot, so a wider one
    /// wraps into it, and answered as `unlimited` once it has gone negative.
    /// There is no range on one of these at all: every number parses.
    Wide(i32),
    /// A `true` or a `false`, spelled out in full either way.
    Yes(bool),
    /// A `true` that is turned on by naming it and takes no value at all. It
    /// shares its number with the `Yes` it is paired with, which is where the
    /// `false` it can still show comes from.
    Bare,
    /// One of a handful of words.
    Word(Word),
    /// A value the client may write and will never read back.
    Hidden(&'static str),
    /// A value the client may read and will never write.
    Fixed(Option<&'static str>),
}

/// A number setting, floors and ceilings and all.
pub struct Whole {
    /// What it says before anybody moves it.
    pub start: i64,
    /// The lowest it will take, below which it is out of bounds.
    pub least: i64,
    /// The highest it will take, above which it is out of bounds.
    pub most: i64,
    /// A second floor with a complaint of its own, checked after the hard one.
    pub floor: Option<(i64, &'static str)>,
    /// A second ceiling with a complaint of its own, checked after the hard one.
    pub roof: Option<(i64, &'static str)>,
    /// Another setting this one is measured against, and whether it has to come
    /// out lower than that one rather than higher.
    pub beside: Option<(&'static str, bool)>,
}

/// A setting that takes one of a list of words.
pub struct Word {
    /// The words it will take, in the spelling it answers under.
    pub of: &'static [&'static str],
    /// Which of them it says before anybody moves it.
    pub start: usize,
    /// Whether the client's spelling is folded before it is looked for, in
    /// which case the list's spelling is what comes back. `ON_TIMEOUT` folds
    /// and `DEFAULT_SCORER` does not, which is measured: `RETURN` is taken and
    /// answered as `return`, while `bm25std` is refused outright.
    pub folded: bool,
    /// What a word that is not on the list is told.
    pub bad: &'static str,
}

/// A number a setting is holding right now.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Held {
    /// A [`Kind::Whole`].
    Whole(i64),
    /// A [`Kind::Wide`], already narrowed to the slot it lives in.
    Wide(i32),
    /// A [`Kind::Yes`] or the [`Kind::Bare`] that shares its number.
    Yes(bool),
    /// Which word of a [`Kind::Word`] is in force.
    Word(usize),
    /// A setting nobody can move, which holds nothing of its own.
    Still,
}

/// Every setting on the server, and what each is set to.
///
/// One of these per server rather than per index or per database, and it
/// outlives the keyspace: a `TIMEOUT` written before `FLUSHALL` is still there
/// after it, which is measured and is the one way this differs from the
/// dictionaries it sits beside.
#[derive(Debug)]
pub struct Config {
    /// One entry per row of [`TABLE`], in the same order.
    held: Vec<Held>,
}

impl Default for Config {
    fn default() -> Config {
        Config::new()
    }
}

impl Config {
    /// A server nobody has turned a knob on yet.
    #[must_use]
    pub fn new() -> Config {
        let held = TABLE
            .iter()
            .map(|s| match &s.kind {
                Kind::Whole(w) => Held::Whole(w.start),
                Kind::Wide(start) => Held::Wide(*start),
                Kind::Yes(start) => Held::Yes(*start),
                Kind::Bare => Held::Yes(true),
                Kind::Word(w) => Held::Word(w.start),
                Kind::Hidden(_) | Kind::Fixed(_) => Held::Still,
            })
            .collect();
        Config { held }
    }

    /// Which row a client's spelling names, if any.
    ///
    /// Case does not count and nothing globs, so this is a walk over sixty nine
    /// short names rather than a match. That is fast enough for a command a
    /// client sends once at startup, and it keeps the order of the table the
    /// one thing that decides the order of the dump.
    #[must_use]
    pub fn find(name: &[u8]) -> Option<usize> {
        TABLE.iter().position(|s| same(s.name.as_bytes(), name))
    }

    /// What a row says right now, or nothing where the value is a nil.
    #[must_use]
    pub fn value(&self, at: usize) -> Option<Vec<u8>> {
        match (&TABLE[at].kind, self.held[at]) {
            (Kind::Fixed(text), _) => text.map(|t| t.as_bytes().to_vec()),
            (Kind::Hidden(text), _) => Some(text.as_bytes().to_vec()),
            (_, Held::Whole(n)) => Some(n.to_string().into_bytes()),
            // A negative one is the way a real server says there is no limit,
            // and since the slot is thirty two bits wide anything past two
            // billion has wrapped into being negative and says the same.
            (_, Held::Wide(n)) if n < 0 => Some(b"unlimited".to_vec()),
            (_, Held::Wide(n)) => Some(n.to_string().into_bytes()),
            (_, Held::Yes(on)) => Some(if on {
                b"true".to_vec()
            } else {
                b"false".to_vec()
            }),
            (Kind::Word(w), Held::Word(which)) => Some(w.of[which].as_bytes().to_vec()),
            (_, Held::Word(_) | Held::Still) => None,
        }
    }

    /// Writes a row, and answers what the client is told when it will not take.
    ///
    /// The value is missing rather than empty when the client left it off
    /// altogether, which is not the same thing: `FT.CONFIG SET TIMEOUT` is told
    /// an argument was expected and `FT.CONFIG SET TIMEOUT ""` is told the
    /// value is the wrong shape.
    ///
    /// # Errors
    ///
    /// The message to answer with, already carrying its own code word.
    pub fn set(&mut self, at: usize, value: Option<&[u8]>) -> Result<(), Vec<u8>> {
        let setting = &TABLE[at];
        if matches!(setting.kind, Kind::Fixed(_)) {
            return Err(STUCK.as_bytes().to_vec());
        }
        // The one setting that is turned on by being named. It takes no value,
        // so there is nothing to parse and nothing that can go wrong.
        if matches!(setting.kind, Kind::Bare) {
            self.write(at, Held::Yes(true));
            return Ok(());
        }
        let Some(value) = value else {
            return Err(NOTHING.as_bytes().to_vec());
        };
        let held = match &setting.kind {
            Kind::Whole(w) => Held::Whole(self.bounded(setting.name, w, value)?),
            // No range at all on one of these, so which reader found the
            // number does not matter and neither does how wide it came out.
            Kind::Wide(_) => {
                let (read, _) = number(value).ok_or_else(|| SHAPE.as_bytes().to_vec())?;
                Held::Wide(read as i32)
            }
            Kind::Yes(_) => match value {
                v if same(b"true", v) => Held::Yes(true),
                v if same(b"false", v) => Held::Yes(false),
                _ => return Err(SHAPE.as_bytes().to_vec()),
            },
            Kind::Word(w) => {
                let found = w.of.iter().position(|word| {
                    if w.folded {
                        same(word.as_bytes(), value)
                    } else {
                        word.as_bytes() == value
                    }
                });
                Held::Word(found.ok_or_else(|| w.bad.as_bytes().to_vec())?)
            }
            // Written and never read, so there is nothing to keep.
            Kind::Hidden(_) => Held::Still,
            Kind::Fixed(_) | Kind::Bare => unreachable!("both are answered above"),
        };
        self.write(at, held);
        Ok(())
    }

    /// Reads a number and checks it against every bound the setting carries.
    fn bounded(&self, name: &str, w: &Whole, value: &[u8]) -> Result<i64, Vec<u8>> {
        let (read, strictly) = number(value).ok_or_else(|| SHAPE.as_bytes().to_vec())?;
        // Every number here is held in something unsigned, so a negative one
        // the loose reader found never reaches the range check at all and is
        // told it is the wrong shape instead.
        if !strictly && read < 0 {
            return Err(SHAPE.as_bytes().to_vec());
        }
        if read < w.least || read > w.most {
            return Err(BOUNDS.as_bytes().to_vec());
        }
        if let Some((floor, why)) = w.floor
            && read < floor
        {
            return Err(why.as_bytes().to_vec());
        }
        if let Some((roof, why)) = w.roof
            && read > roof
        {
            return Err(why.as_bytes().to_vec());
        }
        // The two trimming delays are measured against each other rather than
        // against a constant, and the complaint names both of them and both
        // their numbers, so it is built here rather than kept in the table.
        if let Some((other, lower)) = w.beside {
            let at = Config::find(other.as_bytes()).expect("a setting names a setting");
            let Held::Whole(theirs) = self.held[at] else {
                unreachable!("a setting beside another is a number")
            };
            if (lower && read >= theirs) || (!lower && read <= theirs) {
                let word = if lower { "less" } else { "greater" };
                return Err(format!(
                    "SEARCH_PARSE_ARGS {name} ({read}) must be {word} than {other} ({theirs})"
                )
                .into_bytes());
            }
        }
        Ok(read)
    }

    /// Puts a number in, and in the twin of the row when it has one.
    fn write(&mut self, at: usize, held: Held) {
        self.held[at] = held;
        let pair = TABLE[at].pair;
        if pair == 0 {
            return;
        }
        for (other, row) in TABLE.iter().enumerate() {
            if row.pair == pair {
                self.held[other] = held;
            }
        }
    }
}

/// Whether two names are the same word, ignoring the case of either.
fn same(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.eq_ignore_ascii_case(y))
}

/// A number, and whether the strict reader is the one that found it.
type Read = (i64, bool);

/// Reads a number the way a real server does, strictly first and loosely after.
fn number(value: &[u8]) -> Option<Read> {
    if let Some(n) = strict(value) {
        return Some((n, true));
    }
    loose(value).map(|n| (n, false))
}

/// A plain decimal whole number and nothing else.
///
/// This is Redis's own reader and it is stricter than it looks. No sign but a
/// minus, no leading zero unless the number is exactly `0`, no space, no
/// fraction, and anything too wide for sixty four bits is a refusal rather than
/// a clamp. `010` is not a number to it, which is the whole reason the loose
/// reader below gets a turn and reads it as ten.
fn strict(value: &[u8]) -> Option<i64> {
    let (sign, digits) = match value.split_first() {
        Some((b'-', rest)) => (-1i128, rest),
        _ => (1i128, value),
    };
    if digits.is_empty() || (digits[0] == b'0' && (digits.len() > 1 || sign < 0)) {
        return if digits == b"0" { Some(0) } else { None };
    }
    if !digits.iter().all(u8::is_ascii_digit) {
        return None;
    }
    let mut read: i128 = 0;
    for byte in digits {
        read = read * 10 + i128::from(byte - b'0');
        if read > 1 << 63 {
            return None;
        }
    }
    i64::try_from(read * sign).ok()
}

/// Anything C's `strtod` would take, so long as it comes out whole.
///
/// Redis refuses a leading space and a trailing anything, and Rust's own reader
/// refuses both already. What it does not do is hexadecimal, which C has taken
/// since C99 and which a client really does send, so `0x` is picked off here
/// before the rest is handed over.
fn loose(value: &[u8]) -> Option<i64> {
    let text = core::str::from_utf8(value).ok()?;
    let (sign, rest) = match text.strip_prefix('-') {
        Some(rest) => (-1i64, rest),
        None => (1i64, text.strip_prefix('+').unwrap_or(text)),
    };
    if let Some(digits) = rest.strip_prefix("0x").or_else(|| rest.strip_prefix("0X")) {
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        let read = i64::try_from(u64::from_str_radix(digits, 16).ok()?).ok()?;
        return Some(read * sign);
    }
    let read: f64 = text.parse().ok()?;
    // A fraction is a number that will not fit rather than a bad number, and so
    // is anything past the end of sixty four bits, and both are told the same
    // thing. The comparisons are written against the powers of two rather than
    // against `i64::MAX`, because that number does not survive the trip through
    // a double and the power of two does.
    if read.fract() != 0.0 || !(-(2f64.powi(63))..2f64.powi(63)).contains(&read) {
        return None;
    }
    Some(read as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(config: &Config, name: &str) -> String {
        let at = Config::find(name.as_bytes()).expect("a setting");
        String::from_utf8(config.value(at).unwrap_or_default()).expect("text")
    }

    fn set(config: &mut Config, name: &str, to: &str) -> Result<(), String> {
        let at = Config::find(name.as_bytes()).expect("a setting");
        config
            .set(at, Some(to.as_bytes()))
            .map_err(|e| String::from_utf8(e).expect("text"))
    }

    #[test]
    fn a_name_is_matched_whole_and_the_case_of_it_does_not_count() {
        assert_eq!(Config::find(b"TIMEOUT"), Config::find(b"timeout"));
        assert_eq!(Config::find(b"TiMeOuT"), Config::find(b"timeout"));
        assert_eq!(Config::find(b"TIME"), None);
        assert_eq!(Config::find(b"TIMEOUT*"), None);
        assert_eq!(Config::find(b"*"), None);
    }

    #[test]
    fn the_table_holds_every_setting_a_real_server_dumps() {
        assert_eq!(TABLE.len(), 69);
        let config = Config::new();
        assert_eq!(value(&config, "TIMEOUT"), "500");
        assert_eq!(value(&config, "MAXAGGREGATERESULTS"), "unlimited");
        assert_eq!(value(&config, "MAXSEARCHRESULTS"), "1000000");
        assert_eq!(value(&config, "DEFAULT_SCORER"), "BM25STD");
        assert_eq!(value(&config, "ON_TIMEOUT"), "return");
        assert_eq!(value(&config, "MT_MODE"), "MT_MODE_OFF");
        assert_eq!(value(&config, "OSS_GLOBAL_PASSWORD"), "Password: *******");
        let at = Config::find(b"EXTLOAD").expect("a setting");
        assert_eq!(config.value(at), None);
    }

    #[test]
    fn a_setting_that_cannot_move_says_so_before_it_reads_the_value() {
        let mut config = Config::new();
        assert_eq!(set(&mut config, "NOGC", "true"), Err(STUCK.to_owned()));
        assert_eq!(set(&mut config, "NOGC", "rubbish"), Err(STUCK.to_owned()));
        assert_eq!(set(&mut config, "GC_POLICY", "fork"), Err(STUCK.to_owned()));
    }

    #[test]
    fn a_number_is_read_strictly_and_then_loosely() {
        let mut config = Config::new();
        for (wrote, reads) in [
            ("0x10", "16"),
            ("0X1f", "31"),
            ("+0x10", "16"),
            ("+5", "5"),
            ("010", "10"),
            ("08", "8"),
            ("00", "0"),
            ("0777", "777"),
            ("1e3", "1000"),
            ("1E3", "1000"),
            ("0.0", "0"),
            ("-0.0", "0"),
            ("9223372036854775807", "9223372036854775807"),
        ] {
            assert_eq!(set(&mut config, "TIMEOUT", wrote), Ok(()), "{wrote}");
            assert_eq!(value(&config, "TIMEOUT"), reads, "{wrote}");
        }
        for bad in [
            " 5",
            "5 ",
            "1.5",
            "1e-3",
            "x",
            "",
            "0b11",
            "0xg",
            "nan",
            "inf",
            "1e100",
            "99999999999999999999",
        ] {
            assert_eq!(
                set(&mut config, "TIMEOUT", bad),
                Err(SHAPE.to_owned()),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn which_reader_found_a_negative_decides_what_it_is_told() {
        let mut config = Config::new();
        // The strict reader took these, so they reach the range check.
        assert_eq!(set(&mut config, "TIMEOUT", "-1"), Err(BOUNDS.to_owned()));
        assert_eq!(set(&mut config, "TIMEOUT", "-16"), Err(BOUNDS.to_owned()));
        // And the loose reader took these, so they never get that far.
        for bad in ["-0x10", "-1e3", "-010", "-2.0"] {
            assert_eq!(
                set(&mut config, "TIMEOUT", bad),
                Err(SHAPE.to_owned()),
                "{bad}"
            );
        }
        // A setting with no range at all takes them all and wraps them.
        for wrote in ["-1", "-0x10", "-1e3", "-010"] {
            assert_eq!(
                set(&mut config, "MAXSEARCHRESULTS", wrote),
                Ok(()),
                "{wrote}"
            );
            assert_eq!(value(&config, "MAXSEARCHRESULTS"), "unlimited", "{wrote}");
        }
    }

    #[test]
    fn a_number_out_of_its_range_is_told_which_way_it_went() {
        let mut config = Config::new();
        assert_eq!(set(&mut config, "TIMEOUT", "-1"), Err(BOUNDS.to_owned()));
        assert_eq!(set(&mut config, "MINPREFIX", "0"), Err(BOUNDS.to_owned()));
        assert_eq!(
            set(&mut config, "WORKERS", "17"),
            Err(THREADS.to_owned()),
            "a ceiling with a complaint of its own"
        );
        assert_eq!(set(&mut config, "WORKERS", "-1"), Err(BOUNDS.to_owned()));
        assert_eq!(
            set(&mut config, "MINSTEMLEN", "1"),
            Err("SEARCH_SYNTAX Minimum stem length cannot be lower than 2".to_owned()),
            "a floor with a complaint of its own"
        );
        assert_eq!(set(&mut config, "MINSTEMLEN", "-1"), Err(BOUNDS.to_owned()));
        assert_eq!(
            set(&mut config, "DEFAULT_DIALECT", "9"),
            Err("SEARCH_VALUE_BAD Default dialect version cannot be higher than 4".to_owned())
        );
        assert_eq!(
            set(&mut config, "DEFAULT_DIALECT", "0"),
            Err(BOUNDS.to_owned())
        );
    }

    #[test]
    fn the_two_trimming_delays_are_measured_against_each_other() {
        let mut config = Config::new();
        assert_eq!(
            set(&mut config, "_MIN_TRIM_DELAY_MS", "5000"),
            Err("SEARCH_PARSE_ARGS _MIN_TRIM_DELAY_MS (5000) must be less than _MAX_TRIM_DELAY_MS (5000)".to_owned())
        );
        assert_eq!(
            set(&mut config, "_MAX_TRIM_DELAY_MS", "1999"),
            Err("SEARCH_PARSE_ARGS _MAX_TRIM_DELAY_MS (1999) must be greater than _MIN_TRIM_DELAY_MS (2000)".to_owned())
        );
        assert_eq!(set(&mut config, "_MIN_TRIM_DELAY_MS", "4999"), Ok(()));
        assert_eq!(set(&mut config, "_MAX_TRIM_DELAY_MS", "5000"), Ok(()));
    }

    #[test]
    fn a_wide_number_wraps_into_its_slot_and_answers_unlimited_when_it_goes_under() {
        let mut config = Config::new();
        for (wrote, reads) in [
            ("-1", "unlimited"),
            ("0", "0"),
            ("2147483647", "2147483647"),
            ("2147483648", "unlimited"),
            ("4294967295", "unlimited"),
            ("9223372036854775807", "unlimited"),
        ] {
            assert_eq!(set(&mut config, "MAXSEARCHRESULTS", wrote), Ok(()));
            assert_eq!(value(&config, "MAXSEARCHRESULTS"), reads, "{wrote}");
        }
    }

    #[test]
    fn a_word_setting_folds_the_spelling_only_where_a_real_server_does() {
        let mut config = Config::new();
        assert_eq!(set(&mut config, "ON_TIMEOUT", "RETURN"), Ok(()));
        assert_eq!(value(&config, "ON_TIMEOUT"), "return");
        assert_eq!(set(&mut config, "ON_TIMEOUT", "fail"), Ok(()));
        assert_eq!(value(&config, "ON_TIMEOUT"), "fail");
        assert_eq!(
            set(&mut config, "ON_TIMEOUT", "ignore"),
            Err("SEARCH_VALUE_BAD Invalid ON_TIMEOUT value".to_owned())
        );
        assert_eq!(set(&mut config, "ON_OOM", "ignore"), Ok(()));
        assert_eq!(set(&mut config, "DEFAULT_SCORER", "TFIDF"), Ok(()));
        assert_eq!(
            set(&mut config, "DEFAULT_SCORER", "tfidf"),
            Err("SEARCH_VALUE_BAD Invalid default scorer value".to_owned())
        );
    }

    #[test]
    fn a_yes_or_no_takes_those_two_words_and_nothing_else() {
        let mut config = Config::new();
        assert_eq!(set(&mut config, "_NUMERIC_COMPRESS", "TRUE"), Ok(()));
        assert_eq!(value(&config, "_NUMERIC_COMPRESS"), "true");
        for bad in ["yes", "no", "1", "0", "enabled"] {
            assert_eq!(
                set(&mut config, "_NUMERIC_COMPRESS", bad),
                Err(SHAPE.to_owned()),
                "{bad}"
            );
        }
    }

    #[test]
    fn two_names_for_one_number_move_together() {
        let mut config = Config::new();
        assert_eq!(set(&mut config, "MAXEXPANSIONS", "321"), Ok(()));
        assert_eq!(value(&config, "MAXPREFIXEXPANSIONS"), "321");
        assert_eq!(set(&mut config, "MAXPREFIXEXPANSIONS", "7"), Ok(()));
        assert_eq!(value(&config, "MAXEXPANSIONS"), "7");
    }

    #[test]
    fn the_bare_setting_takes_no_value_and_turns_its_twin_on() {
        let mut config = Config::new();
        let bare = Config::find(b"FORK_GC_CLEAN_NUMERIC_EMPTY_NODES").expect("a setting");
        assert_eq!(
            set(&mut config, "_FORK_GC_CLEAN_NUMERIC_EMPTY_NODES", "false"),
            Ok(())
        );
        assert_eq!(value(&config, "FORK_GC_CLEAN_NUMERIC_EMPTY_NODES"), "false");
        assert_eq!(config.set(bare, None), Ok(()));
        assert_eq!(value(&config, "FORK_GC_CLEAN_NUMERIC_EMPTY_NODES"), "true");
        assert_eq!(value(&config, "_FORK_GC_CLEAN_NUMERIC_EMPTY_NODES"), "true");
    }

    #[test]
    fn a_written_password_is_never_read_back() {
        let mut config = Config::new();
        assert_eq!(set(&mut config, "OSS_GLOBAL_PASSWORD", "hunter2"), Ok(()));
        assert_eq!(value(&config, "OSS_GLOBAL_PASSWORD"), "Password: *******");
    }

    #[test]
    fn a_set_with_nothing_after_the_name_says_so() {
        let mut config = Config::new();
        let at = Config::find(b"TIMEOUT").expect("a setting");
        assert_eq!(config.set(at, None), Err(NOTHING.as_bytes().to_vec()));
    }
}
