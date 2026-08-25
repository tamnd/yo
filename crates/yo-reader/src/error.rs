//! One error type, with a message and sometimes a place.
//!
//! Not `yo_common::Error`. That type has a code enum with a frozen wire value
//! per variant, a retryability model and a documentation page per code, all of
//! which the engine needs and none of which a reader does. Sharing it would
//! also mean sharing a dependency with the thing this crate is meant to check.
//!
//! What a reader owes whoever is holding a file that will not open is a
//! sentence saying what is wrong and a byte offset saying where. That is the
//! whole type.

use std::fmt;

/// What went wrong, and where in the file if that is known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    what: String,
    at: Option<u64>,
}

impl Error {
    /// An error with no particular place.
    #[must_use]
    pub fn new(what: impl Into<String>) -> Error {
        Error {
            what: what.into(),
            at: None,
        }
    }

    /// The same error, pinned to a byte offset in the file.
    #[must_use]
    pub fn at(mut self, off: u64) -> Error {
        self.at = Some(off);
        self
    }

    /// The message on its own.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.what
    }

    /// The byte offset, if there is one.
    #[must_use]
    pub const fn offset(&self) -> Option<u64> {
        self.at
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.at {
            Some(off) => write!(f, "{} (at byte {off})", self.what),
            None => f.write_str(&self.what),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Error {
        Error::new(e.to_string())
    }
}

/// What every fallible thing in this crate returns.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_place_shows_up_in_the_message() {
        let e = Error::new("the tail is torn").at(32768);
        assert_eq!(e.to_string(), "the tail is torn (at byte 32768)");
        assert_eq!(e.offset(), Some(32768));
        assert_eq!(e.message(), "the tail is torn");
    }

    #[test]
    fn without_a_place_it_is_just_the_sentence() {
        let e = Error::new("not a .yo file");
        assert_eq!(e.to_string(), "not a .yo file");
        assert_eq!(e.offset(), None);
    }
}
