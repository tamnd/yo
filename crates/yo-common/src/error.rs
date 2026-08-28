//! The error model. P5: errors are values with structure, never a string a
//! caller has to parse.
//!
//! The [`Code`] enum is generated from `errors.toml`. This module is the Rust
//! shaped wrapper around it, and it carries the four extra fields the C ABI
//! also carries so that nothing is lost crossing the boundary: a message, a
//! position, a documentation URL, and a free form detail.

use std::fmt;

include!(concat!(env!("OUT_DIR"), "/code.rs"));

/// An error, with everything a caller or an agent needs to act on it.
///
/// Cheap to construct on the failure path and never constructed on the success
/// path, so the size of this type does not touch the hot path. It is returned
/// by value rather than boxed because a boxed error means an allocation, and an
/// allocation on a shard thread aborts (`yo-alloc`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    code: Code,
    message: String,
    position: Option<u32>,
    detail: Option<String>,
}

impl Error {
    /// A new error with just a message.
    ///
    /// The message is copied into a `String` here, and that allocation is
    /// wrapped in [`yo_alloc::allow`] because a shard thread that allocates
    /// aborts and an error is by definition off the path the budget is for. A
    /// client that sends `INCR` at a key holding a word should get an error
    /// back, not a server that stops answering everybody else.
    ///
    /// The wrap only covers what happens inside this call, so a caller that
    /// builds its message with `format!` first has already allocated by the
    /// time it gets here. Use [`Error::fmt`] for those.
    pub fn new(code: Code, message: impl Into<String>) -> Error {
        Error {
            code,
            message: yo_alloc::allow(|| message.into()),
            position: None,
            detail: None,
        }
    }

    /// A new error whose message needs formatting, built without allocating
    /// outside the wrap.
    ///
    /// `Error::fmt(code, format_args!("no such thing: {name}"))` is the shape.
    /// `format_args!` builds nothing, so the only allocation is the one this
    /// does, and it happens where it is allowed to.
    pub fn fmt(code: Code, args: fmt::Arguments<'_>) -> Error {
        Error {
            code,
            message: yo_alloc::allow(|| fmt::format(args)),
            position: None,
            detail: None,
        }
    }

    /// Attach the argument index or byte offset the error is about.
    ///
    /// P10: the first error should teach. A position turns "invalid arguments"
    /// into "argument 3 is invalid", which is the difference between a user
    /// reading the docs and a user guessing.
    #[must_use]
    pub fn at(mut self, position: u32) -> Error {
        self.position = Some(position);
        self
    }

    /// Attach machine readable detail, such as `errno=13 path=/var/lib/app.yo`.
    #[must_use]
    pub fn with_detail(mut self, detail: impl Into<String>) -> Error {
        self.detail = Some(yo_alloc::allow(|| detail.into()));
        self
    }

    /// The stable condition code.
    #[inline]
    pub const fn code(&self) -> Code {
        self.code
    }

    /// Whether the identical call could succeed later.
    #[inline]
    pub const fn is_retryable(&self) -> bool {
        self.code.is_retryable()
    }

    /// The human readable message, without the code or the URL.
    #[inline]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The argument index or byte offset, if the error is about one.
    #[inline]
    pub const fn position(&self) -> Option<u32> {
        self.position
    }

    /// Machine readable detail, if any.
    #[inline]
    pub fn detail(&self) -> Option<&str> {
        self.detail.as_deref()
    }

    /// The documentation page for this condition, if it has one.
    #[inline]
    pub fn url(&self) -> Option<&'static str> {
        self.code.url()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code.c_name(), self.message)?;
        if let Some(p) = self.position {
            write!(f, " (at {p})")?;
        }
        if let Some(d) = &self.detail {
            write!(f, " [{d}]")?;
        }
        if let Some(u) = self.code.url() {
            write!(f, " see {u}")?;
        }
        Ok(())
    }
}

impl core::error::Error for Error {}

/// The crate wide result type.
pub type Result<T> = core::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_dense_and_stable() {
        for (i, &c) in Code::ALL.iter().enumerate() {
            assert_eq!(c.as_u32() as usize, i);
            assert_eq!(Code::from_u32(c.as_u32()), Some(c));
        }
    }

    /// These numbers are on the wire and in every binding. Changing one is a
    /// breaking change for every language at once, so it fails here first.
    #[test]
    fn wire_values_are_frozen() {
        assert_eq!(Code::Ok.as_u32(), 0);
        assert_eq!(Code::ShapeMismatch.as_u32(), 1);
        assert_eq!(Code::Locked.as_u32(), 2);
        assert_eq!(Code::Busy.as_u32(), 3);
        assert_eq!(Code::NotFound.as_u32(), 4);
        assert_eq!(Code::WrongType.as_u32(), 5);
        assert_eq!(Code::AbiMismatch.as_u32(), 6);
        assert_eq!(Code::Corrupt.as_u32(), 7);
        assert_eq!(Code::Full.as_u32(), 8);
        assert_eq!(Code::Io.as_u32(), 9);
        assert_eq!(Code::Unsupported.as_u32(), 10);
        assert_eq!(Code::Invalid.as_u32(), 11);
        assert_eq!(Code::EpochStalled.as_u32(), 12);
        assert_eq!(Code::VersionTooNew.as_u32(), 13);
    }

    #[test]
    fn an_unknown_code_is_a_value_not_a_panic() {
        assert_eq!(Code::from_u32(9999), None);
    }

    #[test]
    fn retryability_matches_the_model() {
        assert!(Code::Locked.is_retryable());
        assert!(Code::Busy.is_retryable());
        assert!(Code::Io.is_retryable());
        assert!(Code::EpochStalled.is_retryable());
        assert!(!Code::ShapeMismatch.is_retryable());
        assert!(!Code::Corrupt.is_retryable());
        assert!(!Code::WrongType.is_retryable());
    }

    #[test]
    fn display_carries_everything() {
        let e = Error::new(Code::Invalid, "expected an integer")
            .at(3)
            .with_detail("got=abc");
        let s = e.to_string();
        assert!(s.contains("YO_ERR_INVALID"), "{s}");
        assert!(s.contains("expected an integer"), "{s}");
        assert!(s.contains("at 3"), "{s}");
        assert!(s.contains("got=abc"), "{s}");
    }

    /// The rule this is protecting is that a shard thread aborts when it
    /// allocates, and an error message is a `String`. If building one were not
    /// allowed, the first client to send `INCR` at a key holding a word would
    /// take the server down with it, which is a worse failure than the one the
    /// rule exists to prevent.
    #[test]
    fn building_an_error_is_allowed_where_allocating_is_not() {
        yo_alloc::enter_no_alloc();
        let e = Error::fmt(Code::Invalid, format_args!("no such thing: {}", "x"))
            .with_detail("got=abc");
        assert_eq!(e.message(), "no such thing: x");
        // And the thread is still forbidden afterwards, because the wrap is
        // around the allocation and not around the caller.
        assert!(yo_alloc::is_forbidden());
        yo_alloc::exit_no_alloc();
        assert!(!yo_alloc::is_forbidden());
    }

    #[test]
    fn errors_that_need_a_page_have_one() {
        // Anything a user is likely to hit and unlikely to understand needs a
        // URL. NotFound and Full do not, because they explain themselves.
        for c in [
            Code::ShapeMismatch,
            Code::Locked,
            Code::Busy,
            Code::WrongType,
            Code::AbiMismatch,
            Code::Corrupt,
            Code::EpochStalled,
            Code::VersionTooNew,
        ] {
            assert!(c.url().is_some(), "{} has no documentation URL", c.c_name());
        }
    }
}
