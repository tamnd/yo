//! The API model.
//!
//! `dx/03` section 2 says the model is generated from the Rust source of truth
//! and then checked in. The proc macro that reads `#[yo_api]` attributes off the
//! engine is not written yet, so for now the source of truth is this table. The
//! important half is already true: there is exactly one place a signature, a
//! cost class or a doc line is written down, and everything downstream is
//! emitted from it.
//!
//! When the macro lands it replaces this table and nothing else changes.

/// What the entry does, which decides what the generator demands of it.
///
/// The full vocabulary is here even though the ABI does not use all of it yet.
/// Trimming it to what today's entries happen to need would mean the model's
/// meaning changes every time an entry is added, and the model is the thing six
/// binding generators read.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Produces a handle. Has exactly one matching destructor.
    Constructor,
    /// Releases a handle. Idempotent and null safe.
    Destructor,
    /// Reads. A single key read must have a `_many` form.
    Read,
    /// Writes.
    Write,
    /// Opens or advances a cursor.
    Iterator,
    /// Everything else: version, statistics, configuration.
    Admin,
}

impl Kind {
    /// The spelling used in the model file and in every generated binding.
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Constructor => "constructor",
            Kind::Destructor => "destructor",
            Kind::Read => "read",
            Kind::Write => "write",
            Kind::Iterator => "iterator",
            Kind::Admin => "admin",
        }
    }
}

/// One of the seven classes in `15` section 6.
///
/// Every entry carries one, and a binding that does not print it in its doc
/// comment fails CI. A user should never have to measure to learn that a call
/// is a scan.
///
/// All eight are listed for the same reason `Kind` is: the vocabulary is fixed
/// by `15` section 6 and not by which ones the current entries happen to use.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cost {
    /// One index probe and at most one dereference.
    Probe,
    /// A probe then a sequential run.
    ProbeRun,
    /// A tree descent.
    LogN,
    /// Proportional to the collection.
    Scan,
    /// Proportional to the inputs.
    Merge,
    /// Approximate and bounded by `nprobe`.
    Search,
    /// May touch the file.
    Fault,
    /// Constant and trivial. Version calls, handle release, arena reset.
    Free,
}

impl Cost {
    /// The spelling used in the model file and in every generated binding.
    pub fn as_str(self) -> &'static str {
        match self {
            Cost::Probe => "probe",
            Cost::ProbeRun => "probe+run",
            Cost::LogN => "log n",
            Cost::Scan => "scan",
            Cost::Merge => "merge",
            Cost::Search => "search",
            Cost::Fault => "fault",
            Cost::Free => "free",
        }
    }
}

/// How a parameter or a result relates to memory the caller can see.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Own {
    /// The engine reads it during the call and never keeps it.
    Borrowed,
    /// The engine writes it into the caller's arena.
    ArenaOwned,
    /// The engine owns it and it stays valid until the handle is closed.
    EngineOwned,
    /// A plain value, no memory involved.
    Value,
}

impl Own {
    /// The spelling used in the model file.
    pub fn as_str(self) -> &'static str {
        match self {
            Own::Borrowed => "borrowed",
            Own::ArenaOwned => "arena",
            Own::EngineOwned => "engine",
            Own::Value => "value",
        }
    }
}

/// One parameter.
#[derive(Debug, Clone, Copy)]
pub struct Param {
    /// The name, used verbatim in the header and in every binding.
    pub name: &'static str,
    /// The C type, spelled exactly as it appears in `yo.h`.
    pub ty: &'static str,
    /// Who owns the memory behind it.
    pub own: Own,
    /// One line, rendered into each language's comment syntax.
    pub doc: &'static str,
}

/// One ABI entry point.
#[derive(Debug, Clone, Copy)]
pub struct Func {
    /// The C symbol. Bindings never invent their own.
    pub symbol: &'static str,
    /// What it does.
    pub kind: Kind,
    /// What it costs.
    pub cost: Cost,
    /// The C return type.
    pub returns: &'static str,
    /// Who owns the returned memory.
    pub returns_own: Own,
    /// The parameters in order, not counting the trailing error out param.
    pub params: &'static [Param],
    /// Whether the entry takes a trailing `yo_error *err`.
    pub errors: bool,
    /// The `_many` form, if this is a single key read that has one.
    pub many: Option<&'static str>,
    /// The ABI version this entry appeared in, as major times 1000 plus minor.
    pub since: u32,
    /// The prose. One source, rendered into every language.
    pub doc: &'static [&'static str],
    /// The snippet in `yo-snippets` that must compile in every tier 1 language.
    pub example: Option<&'static str>,
}

/// An opaque handle type.
#[derive(Debug, Clone, Copy)]
pub struct Handle {
    /// The C type name.
    pub name: &'static str,
    /// Whether the handle may be used from more than one thread at a time.
    pub shareable: bool,
    /// The entry point that destroys it.
    pub closer: &'static str,
    /// One line of prose.
    pub doc: &'static str,
}

/// The ABI major version. Frozen at 1.0; see `dx/02` section 2.
pub const ABI_MAJOR: u32 = 0;
/// The ABI minor version. A bump adds symbols and never changes one.
pub const ABI_MINOR: u32 = 1;

/// Every handle in the ABI, with its lifetime class.
pub const HANDLES: &[Handle] = &[
    Handle {
        name: "yo_db",
        shareable: false,
        closer: "yo_close",
        doc: "The database. One per process per path. It belongs to the thread that opened it for as long as inline mode is the only mode, and it becomes shareable when owned mode lands; widening that guarantee later is compatible and claiming it now would not be true.",
    },
    Handle {
        name: "yo_arena",
        shareable: false,
        closer: "yo_arena_free",
        doc: "Caller scoped result storage. Everything the engine writes for you lands here.",
    },
];

const P_DB: Param = Param {
    name: "db",
    ty: "yo_db *",
    own: Own::EngineOwned,
    doc: "The database.",
};
const P_ARENA: Param = Param {
    name: "arena",
    ty: "yo_arena *",
    own: Own::EngineOwned,
    doc: "The arena the result is written into.",
};
const P_KEY: Param = Param {
    name: "key",
    ty: "yo_slice",
    own: Own::Borrowed,
    doc: "The key. Read during the call and not retained.",
};

/// Every entry point, in header order.
pub const FUNCS: &[Func] = &[
    Func {
        symbol: "yo_abi_version",
        kind: Kind::Admin,
        cost: Cost::Free,
        returns: "uint32_t",
        returns_own: Own::Value,
        params: &[],
        errors: false,
        many: None,
        since: 1,
        doc: &[
            "The ABI version as (major << 16) | minor.",
            "",
            "Call this at load time. A binding that sees a different major refuses to",
            "proceed and reports YO_ERR_ABI_MISMATCH naming both versions, because a",
            "major mismatch means the struct layouts below have moved.",
        ],
        example: None,
    },
    Func {
        symbol: "yo_version_string",
        kind: Kind::Admin,
        cost: Cost::Free,
        returns: "const char *",
        returns_own: Own::EngineOwned,
        params: &[],
        errors: false,
        many: None,
        since: 1,
        doc: &[
            "The engine version, as a NUL terminated string that lives forever.",
            "",
            "This is the crate version and it moves independently of the ABI version.",
        ],
        example: None,
    },
    Func {
        symbol: "yo_noop",
        kind: Kind::Admin,
        cost: Cost::Free,
        returns: "void",
        returns_own: Own::Value,
        params: &[],
        errors: false,
        many: None,
        since: 1,
        doc: &[
            "Does nothing, on purpose.",
            "",
            "Every binding publishes the round trip cost of this call next to its own",
            "overhead number, so that the split between what the FFI costs and what the",
            "binding costs is visible rather than argued about (dx/03 section 6).",
        ],
        example: None,
    },
    Func {
        symbol: "yo_code_name",
        kind: Kind::Admin,
        cost: Cost::Free,
        returns: "const char *",
        returns_own: Own::EngineOwned,
        params: &[Param {
            name: "code",
            ty: "uint32_t",
            own: Own::Value,
            doc: "The code to name. An integer rather than yo_code, so that a binding can pass one it has never heard of without a cast.",
        }],
        errors: false,
        many: None,
        since: 1,
        doc: &[
            "The spelling of a code, for a build that does not know it.",
            "",
            "A binding compiled against an older header will meet codes it has no name",
            "for. That is a value to report, not a crash, so this never returns NULL.",
        ],
        example: None,
    },
    Func {
        symbol: "yo_code_retryable",
        kind: Kind::Admin,
        cost: Cost::Free,
        returns: "uint8_t",
        returns_own: Own::Value,
        params: &[Param {
            name: "code",
            ty: "uint32_t",
            own: Own::Value,
            doc: "The code to classify. An integer, for the same reason as yo_code_name.",
        }],
        errors: false,
        many: None,
        since: 1,
        doc: &[
            "Whether the identical call could succeed later.",
            "",
            "Generated from errors.toml, so retryability is a property of the condition",
            "and never a judgement made at a call site.",
        ],
        example: None,
    },
    Func {
        symbol: "yo_arena_new",
        kind: Kind::Constructor,
        cost: Cost::Free,
        returns: "yo_arena *",
        returns_own: Own::EngineOwned,
        params: &[P_DB],
        errors: true,
        many: None,
        since: 1,
        doc: &[
            "Opens an arena for results.",
            "",
            "Make one per call site, or per request, or per iteration batch. Reset it",
            "between uses and free it once. This is what turns a loop over ten thousand",
            "results into one allocation rather than ten thousand, and it is the whole",
            "reason a binding in a managed language can stay inside the 3x budget.",
        ],
        example: Some("arena-basics"),
    },
    Func {
        symbol: "yo_arena_reset",
        kind: Kind::Admin,
        cost: Cost::Free,
        returns: "void",
        returns_own: Own::Value,
        params: &[Param {
            name: "arena",
            ty: "yo_arena *",
            own: Own::EngineOwned,
            doc: "The arena to rewind.",
        }],
        errors: false,
        many: None,
        since: 1,
        doc: &[
            "Rewinds the arena and keeps its capacity.",
            "",
            "Every borrowed view into this arena is invalid the instant this returns,",
            "and not one instruction longer. The debug build poisons the memory so that",
            "a binding which gets this wrong fails in its own test suite.",
        ],
        example: Some("arena-basics"),
    },
    Func {
        symbol: "yo_arena_free",
        kind: Kind::Destructor,
        cost: Cost::Free,
        returns: "void",
        returns_own: Own::Value,
        params: &[Param {
            name: "arena",
            ty: "yo_arena *",
            own: Own::EngineOwned,
            doc: "The arena, or NULL.",
        }],
        errors: false,
        many: None,
        since: 1,
        doc: &["Frees the arena. Idempotent in the sense that NULL is fine."],
        example: Some("arena-basics"),
    },
    Func {
        symbol: "yo_arena_used",
        kind: Kind::Admin,
        cost: Cost::Free,
        returns: "uint64_t",
        returns_own: Own::Value,
        params: &[Param {
            name: "arena",
            ty: "const yo_arena *",
            own: Own::EngineOwned,
            doc: "The arena to measure.",
        }],
        errors: false,
        many: None,
        since: 1,
        doc: &[
            "Bytes handed out since the last reset.",
            "",
            "Bindings use this in their own tests to prove that a batched read really",
            "did cost one arena bump per row and not one malloc per row.",
        ],
        example: None,
    },
    Func {
        symbol: "yo_open",
        kind: Kind::Constructor,
        cost: Cost::Fault,
        returns: "yo_db *",
        returns_own: Own::EngineOwned,
        params: &[
            Param {
                name: "path",
                ty: "const char *",
                own: Own::Borrowed,
                doc: "The path to the .yo file, or NULL for memory only.",
            },
            Param {
                name: "opts",
                ty: "const yo_open_options *",
                own: Own::Borrowed,
                doc: "Options, or NULL for the defaults.",
            },
        ],
        errors: true,
        many: None,
        since: 1,
        doc: &[
            "Opens a database.",
            "",
            "At this milestone the path is accepted and ignored: nothing is written to",
            "disk until the record plane lands. Passing NULL is the honest way to ask",
            "for what you actually get today.",
        ],
        example: Some("open-and-get"),
    },
    Func {
        symbol: "yo_close",
        kind: Kind::Destructor,
        cost: Cost::Free,
        returns: "void",
        returns_own: Own::Value,
        params: &[Param {
            name: "db",
            ty: "yo_db *",
            own: Own::EngineOwned,
            doc: "The database, or NULL.",
        }],
        errors: false,
        many: None,
        since: 1,
        doc: &[
            "Closes the database. NULL is fine.",
            "",
            "Closing with live children frees nothing and is a bug in the binding, not",
            "in the caller. The debug build asserts on it.",
        ],
        example: Some("open-and-get"),
    },
    Func {
        symbol: "yo_set",
        kind: Kind::Write,
        cost: Cost::Probe,
        returns: "int32_t",
        returns_own: Own::Value,
        params: &[
            P_DB,
            P_KEY,
            Param {
                name: "value",
                ty: "yo_slice",
                own: Own::Borrowed,
                doc: "The value. Copied into the engine during the call.",
            },
        ],
        errors: true,
        many: None,
        since: 1,
        doc: &["Stores a value under a key. Returns 0, or -1 with err populated."],
        example: Some("open-and-get"),
    },
    Func {
        symbol: "yo_get",
        kind: Kind::Read,
        cost: Cost::Probe,
        returns: "int32_t",
        returns_own: Own::Value,
        params: &[
            P_DB,
            P_KEY,
            Param {
                name: "out",
                ty: "yo_slice *",
                own: Own::EngineOwned,
                doc: "Receives a view into engine memory, valid until the next write.",
            },
        ],
        errors: true,
        many: Some("yo_get_many"),
        since: 1,
        doc: &[
            "Reads a value without copying it. Returns 1 found, 0 missing, -1 error.",
            "",
            "The view points into the engine and stays valid until the next write to",
            "this database. If you cannot promise that, use yo_get_copy.",
        ],
        example: Some("open-and-get"),
    },
    Func {
        symbol: "yo_get_copy",
        kind: Kind::Read,
        cost: Cost::Probe,
        returns: "int32_t",
        returns_own: Own::ArenaOwned,
        params: &[
            P_DB,
            P_KEY,
            P_ARENA,
            Param {
                name: "out",
                ty: "yo_slice *",
                own: Own::ArenaOwned,
                doc: "Receives a view into the arena, valid until it is reset or freed.",
            },
        ],
        errors: true,
        many: Some("yo_get_many"),
        since: 1,
        doc: &[
            "Reads a value into the arena. Returns 1 found, 0 missing, -1 error.",
            "",
            "This is the default in every managed language, and it costs one arena bump",
            "rather than one allocation.",
        ],
        example: None,
    },
    Func {
        symbol: "yo_get_many",
        kind: Kind::Read,
        cost: Cost::Probe,
        returns: "int32_t",
        returns_own: Own::ArenaOwned,
        params: &[
            P_DB,
            Param {
                name: "keys",
                ty: "const yo_slice *",
                own: Own::Borrowed,
                doc: "The keys, n of them.",
            },
            Param {
                name: "n",
                ty: "uint32_t",
                own: Own::Value,
                doc: "How many keys, at most YO_BATCH_MAX per call.",
            },
            P_ARENA,
            Param {
                name: "out",
                ty: "yo_slice *",
                own: Own::ArenaOwned,
                doc: "n results in key order. A missing key gets a NULL pointer and zero length.",
            },
        ],
        errors: true,
        many: None,
        since: 1,
        doc: &[
            "Reads many keys in one crossing. Returns how many were found, or -1.",
            "",
            "This exists because the per call overhead in Python and Node is the binding",
            "constraint, and amortising one crossing over 64 keys turns a 450 ns problem",
            "into a 40 ns one. Every point read in this ABI has a _many form, by rule of",
            "the generator rather than by anyone's judgement.",
        ],
        example: Some("get-many"),
    },
    Func {
        symbol: "yo_del",
        kind: Kind::Write,
        cost: Cost::Probe,
        returns: "int32_t",
        returns_own: Own::Value,
        params: &[P_DB, P_KEY],
        errors: true,
        many: None,
        since: 1,
        doc: &["Removes a key. Returns 1 removed, 0 absent, -1 error."],
        example: None,
    },
    Func {
        symbol: "yo_len",
        kind: Kind::Read,
        cost: Cost::Free,
        returns: "uint64_t",
        returns_own: Own::Value,
        params: &[Param {
            name: "db",
            ty: "const yo_db *",
            own: Own::EngineOwned,
            doc: "The database.",
        }],
        errors: false,
        many: None,
        since: 1,
        doc: &["How many keys are live."],
        example: None,
    },
];
