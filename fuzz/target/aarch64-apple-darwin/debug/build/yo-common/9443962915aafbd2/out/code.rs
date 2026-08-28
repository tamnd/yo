// Generated from errors.toml by build.rs. Do not edit.

/// A stable, wire visible condition code.
///
/// These numbers are frozen once released. They are the same integers the C
/// ABI exposes as `yo_code` and the same ones every binding reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u32)]
#[non_exhaustive]
pub enum Code {
    /// Not an error. Present so that the C enum has a zero value that means success.
    ///
    /// C name: `YO_OK`.
    Ok = 0,
    /// The type used to open a collection does not match the type the collection was created with. The message carries both shape descriptions in full, not just the tags.
    ///
    /// C name: `YO_ERR_SHAPE_MISMATCH`.
    ShapeMismatch = 1,
    /// Another process holds the writer lock on this file. Exactly one writer per file.
    ///
    /// C name: `YO_ERR_LOCKED`.
    Locked = 2,
    /// The handle has live children. Closing a parent with live children closes nothing.
    ///
    /// C name: `YO_ERR_BUSY`.
    Busy = 3,
    /// The key, field, member or node does not exist. This is a value and not a failure, and the typed API returns an Option rather than this error wherever it can.
    ///
    /// C name: `YO_ERR_NOT_FOUND`.
    NotFound = 4,
    /// The key holds a different data structure than the operation expects. This is Redis WRONGTYPE.
    ///
    /// C name: `YO_ERR_WRONG_TYPE`.
    WrongType = 5,
    /// The header the caller compiled against and the library it loaded disagree. The message names both versions.
    ///
    /// C name: `YO_ERR_ABI_MISMATCH`.
    AbiMismatch = 6,
    /// A checksum failed or a structure is self-inconsistent. The message names the offset and what was expected there.
    ///
    /// C name: `YO_ERR_CORRUPT`.
    Corrupt = 7,
    /// A bounded resource is exhausted: the address space, a fixed table, or a configured memory limit.
    ///
    /// C name: `YO_ERR_FULL`.
    Full = 8,
    /// The operating system refused an operation. The detail field carries errno and the path.
    ///
    /// C name: `YO_ERR_IO`.
    Io = 9,
    /// The operation is valid but this build or this platform does not implement it. The message says which build would.
    ///
    /// C name: `YO_ERR_UNSUPPORTED`.
    Unsupported = 10,
    /// The arguments are wrong. The position field points at which one.
    ///
    /// C name: `YO_ERR_INVALID`.
    Invalid = 11,
    /// An iterator has been open long enough to hold back reclamation. Finish it or drop it. The message carries the iterator age in microseconds.
    ///
    /// C name: `YO_ERR_EPOCH_STALLED`.
    EpochStalled = 12,
    /// The file's min_reader_version is higher than this build supports. The message names both numbers and the release that would read it.
    ///
    /// C name: `YO_ERR_VERSION_TOO_NEW`.
    VersionTooNew = 13,
}

impl Code {
    /// Every code, in wire order. Index equals the numeric value.
    pub const ALL: &'static [Code] = &[
        Code::Ok,
        Code::ShapeMismatch,
        Code::Locked,
        Code::Busy,
        Code::NotFound,
        Code::WrongType,
        Code::AbiMismatch,
        Code::Corrupt,
        Code::Full,
        Code::Io,
        Code::Unsupported,
        Code::Invalid,
        Code::EpochStalled,
        Code::VersionTooNew,
    ];

    /// The wire value.
    #[inline]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    /// The code for a wire value, or `None` if this build does not know it.
    ///
    /// An unknown code from a newer peer is a value to report, not a panic.
    #[inline]
    pub const fn from_u32(v: u32) -> Option<Code> {
        match v {
            0 => Some(Code::Ok),
            1 => Some(Code::ShapeMismatch),
            2 => Some(Code::Locked),
            3 => Some(Code::Busy),
            4 => Some(Code::NotFound),
            5 => Some(Code::WrongType),
            6 => Some(Code::AbiMismatch),
            7 => Some(Code::Corrupt),
            8 => Some(Code::Full),
            9 => Some(Code::Io),
            10 => Some(Code::Unsupported),
            11 => Some(Code::Invalid),
            12 => Some(Code::EpochStalled),
            13 => Some(Code::VersionTooNew),
            _ => None,
        }
    }

    /// The C ABI spelling, which is also what appears in log lines.
    #[inline]
    pub const fn c_name(self) -> &'static str {
        match self {
            Code::Ok => "YO_OK",
            Code::ShapeMismatch => "YO_ERR_SHAPE_MISMATCH",
            Code::Locked => "YO_ERR_LOCKED",
            Code::Busy => "YO_ERR_BUSY",
            Code::NotFound => "YO_ERR_NOT_FOUND",
            Code::WrongType => "YO_ERR_WRONG_TYPE",
            Code::AbiMismatch => "YO_ERR_ABI_MISMATCH",
            Code::Corrupt => "YO_ERR_CORRUPT",
            Code::Full => "YO_ERR_FULL",
            Code::Io => "YO_ERR_IO",
            Code::Unsupported => "YO_ERR_UNSUPPORTED",
            Code::Invalid => "YO_ERR_INVALID",
            Code::EpochStalled => "YO_ERR_EPOCH_STALLED",
            Code::VersionTooNew => "YO_ERR_VERSION_TOO_NEW",
        }
    }

    /// Whether the identical call could succeed later.
    ///
    /// This is a property of the condition and not of the caller, which is why
    /// it is generated rather than decided at each call site.
    #[inline]
    pub const fn is_retryable(self) -> bool {
        match self {
            Code::Ok => false,
            Code::ShapeMismatch => false,
            Code::Locked => true,
            Code::Busy => true,
            Code::NotFound => false,
            Code::WrongType => false,
            Code::AbiMismatch => false,
            Code::Corrupt => false,
            Code::Full => false,
            Code::Io => true,
            Code::Unsupported => false,
            Code::Invalid => false,
            Code::EpochStalled => true,
            Code::VersionTooNew => false,
        }
    }

    /// The C ABI spelling with a trailing NUL, ready to cross the boundary.
    ///
    /// The C ABI needs NUL terminated strings and Rust literals are not,
    /// so the terminator is put here where it is generated rather than in a
    /// second table somebody has to keep in step by hand.
    #[inline]
    pub const fn c_name_z(self) -> &'static str {
        match self {
            Code::Ok => "YO_OK\0",
            Code::ShapeMismatch => "YO_ERR_SHAPE_MISMATCH\0",
            Code::Locked => "YO_ERR_LOCKED\0",
            Code::Busy => "YO_ERR_BUSY\0",
            Code::NotFound => "YO_ERR_NOT_FOUND\0",
            Code::WrongType => "YO_ERR_WRONG_TYPE\0",
            Code::AbiMismatch => "YO_ERR_ABI_MISMATCH\0",
            Code::Corrupt => "YO_ERR_CORRUPT\0",
            Code::Full => "YO_ERR_FULL\0",
            Code::Io => "YO_ERR_IO\0",
            Code::Unsupported => "YO_ERR_UNSUPPORTED\0",
            Code::Invalid => "YO_ERR_INVALID\0",
            Code::EpochStalled => "YO_ERR_EPOCH_STALLED\0",
            Code::VersionTooNew => "YO_ERR_VERSION_TOO_NEW\0",
        }
    }

    /// The documentation page with a trailing NUL, if there is one.
    #[inline]
    pub const fn url_z(self) -> Option<&'static str> {
        match self {
            Code::Ok => None,
            Code::ShapeMismatch => Some("https://yo.tamnd.dev/errors/shape-mismatch\0"),
            Code::Locked => Some("https://yo.tamnd.dev/errors/locked\0"),
            Code::Busy => Some("https://yo.tamnd.dev/errors/busy\0"),
            Code::NotFound => None,
            Code::WrongType => Some("https://yo.tamnd.dev/errors/wrong-type\0"),
            Code::AbiMismatch => Some("https://yo.tamnd.dev/errors/abi-mismatch\0"),
            Code::Corrupt => Some("https://yo.tamnd.dev/errors/corrupt\0"),
            Code::Full => None,
            Code::Io => None,
            Code::Unsupported => None,
            Code::Invalid => None,
            Code::EpochStalled => Some("https://yo.tamnd.dev/errors/epoch-stalled\0"),
            Code::VersionTooNew => Some("https://yo.tamnd.dev/errors/version-too-new\0"),
        }
    }

    /// The documentation page for this condition, if it has one.
    #[inline]
    pub const fn url(self) -> Option<&'static str> {
        match self {
            Code::Ok => None,
            Code::ShapeMismatch => Some("https://yo.tamnd.dev/errors/shape-mismatch"),
            Code::Locked => Some("https://yo.tamnd.dev/errors/locked"),
            Code::Busy => Some("https://yo.tamnd.dev/errors/busy"),
            Code::NotFound => None,
            Code::WrongType => Some("https://yo.tamnd.dev/errors/wrong-type"),
            Code::AbiMismatch => Some("https://yo.tamnd.dev/errors/abi-mismatch"),
            Code::Corrupt => Some("https://yo.tamnd.dev/errors/corrupt"),
            Code::Full => None,
            Code::Io => None,
            Code::Unsupported => None,
            Code::Invalid => None,
            Code::EpochStalled => Some("https://yo.tamnd.dev/errors/epoch-stalled"),
            Code::VersionTooNew => Some("https://yo.tamnd.dev/errors/version-too-new"),
        }
    }
}
