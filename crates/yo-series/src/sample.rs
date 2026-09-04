//! One reading.

/// A value and the moment it was read at.
///
/// Timestamps are milliseconds since the epoch as far as the commands are
/// concerned, but nothing in this crate cares what the unit is, so a series of
/// anything counted upwards works just as well.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    /// When it was read.
    pub at: i64,
    /// What was read.
    pub value: f64,
}

impl Sample {
    /// A sample at `at` holding `value`.
    #[must_use]
    pub fn new(at: i64, value: f64) -> Self {
        Self { at, value }
    }
}
