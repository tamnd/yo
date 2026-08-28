//! `INCREX`, which is a counter with a policy attached.
//!
//! Redis 8.8 put four separate ideas into one command: increment by an integer
//! or a float, refuse or clamp a result that leaves a range, and set, keep,
//! clear or conditionally set the key's deadline, all in the round trip that
//! used to be `INCR` followed by `EXPIRE`. It is the first Redis primitive that
//! implements a workload rather than a data structure, and it replaces a Lua
//! script for rate limiting, quota counting and stock levels.
//!
//! The arithmetic is here rather than in `strings.rs` because it is the part
//! with the edges. A rejected increment must not create the key and must not
//! touch the deadline, a saturated one must create it, and the amount actually
//! applied has to come back so the caller can tell the two apart without
//! comparing against a value it did not have.

use yo_common::{Code, Error, Result};

/// An integer or a float, which is what `INCREX` counts in.
///
/// The two never mix inside one call. `BYINT` with a `UBOUND` that is not an
/// integer is an error on a real server, and it is an error here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Num {
    /// `BYINT`, and the default when neither is given.
    Int(i64),
    /// `BYFLOAT`.
    Float(f64),
}

impl Num {
    /// Whether this is the integer kind.
    #[must_use]
    pub const fn is_int(self) -> bool {
        matches!(self, Num::Int(_))
    }

    /// Zero of the same kind, which is what a rejected increment applied.
    #[must_use]
    const fn zero_like(self) -> Num {
        match self {
            Num::Int(_) => Num::Int(0),
            Num::Float(_) => Num::Float(0.0),
        }
    }

    fn as_int(self, what: &str) -> Result<i64> {
        match self {
            Num::Int(n) => Ok(n),
            Num::Float(_) => Err(Error::fmt(
                Code::Invalid,
                format_args!("{what} is not an integer or out of range"),
            )),
        }
    }

    fn as_float(self, what: &str) -> Result<f64> {
        match self {
            Num::Float(f) => Ok(f),
            Num::Int(n) => {
                // An integer bound on a float increment is not the error a
                // float bound on an integer increment is, because every i64 is
                // a sensible float bound. It is accepted and widened.
                let _ = what;
                Ok(n as f64)
            }
        }
    }
}

/// What `INCREX` should do with the key's deadline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IncrExpire {
    /// No expiration option, which leaves whatever deadline the key had.
    #[default]
    Keep,
    /// `PERSIST`: drop the deadline.
    Persist,
    /// `EX`, `PX`, `EXAT` or `PXAT`, as an absolute unix millisecond.
    At(u64),
    /// The same, with `ENX`: set it only if the key has no deadline already.
    AtIfNone(u64),
}

/// Everything `INCREX` can be asked to do beyond adding one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IncrEx {
    /// `BYINT n` or `BYFLOAT f`. One by default.
    pub by: Num,
    /// `SATURATE`: clamp to the bound instead of refusing.
    pub saturate: bool,
    /// `LBOUND`. The type's own minimum when absent.
    pub lower: Option<Num>,
    /// `UBOUND`. The type's own maximum when absent.
    pub upper: Option<Num>,
    /// The expiration options.
    pub expire: IncrExpire,
}

impl Default for IncrEx {
    fn default() -> IncrEx {
        IncrEx {
            by: Num::Int(1),
            saturate: false,
            lower: None,
            upper: None,
            expire: IncrExpire::Keep,
        }
    }
}

impl IncrEx {
    /// Plain `INCREX key`, which adds one and leaves the deadline alone.
    pub const PLAIN: IncrEx = IncrEx {
        by: Num::Int(1),
        saturate: false,
        lower: None,
        upper: None,
        expire: IncrExpire::Keep,
    };

    /// This, by a different amount.
    #[must_use]
    pub const fn by(mut self, by: Num) -> IncrEx {
        self.by = by;
        self
    }

    /// This, clamping instead of refusing.
    #[must_use]
    pub const fn saturating(mut self) -> IncrEx {
        self.saturate = true;
        self
    }

    /// This, held between two bounds.
    #[must_use]
    pub const fn between(mut self, lower: Option<Num>, upper: Option<Num>) -> IncrEx {
        self.lower = lower;
        self.upper = upper;
        self
    }

    /// This, with something to say about the deadline.
    #[must_use]
    pub const fn expiring(mut self, expire: IncrExpire) -> IncrEx {
        self.expire = expire;
        self
    }
}

/// What `INCREX` did.
///
/// Both halves reach the client: the reply is the value and then the amount
/// applied, and an amount of zero is how a client tells a refused increment
/// from one that happened to add nothing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Counted {
    /// The value now, which is the value before when the increment was refused.
    pub value: Num,
    /// How much was actually added, which is zero when nothing was.
    pub applied: Num,
    /// Whether anything was written. A refused increment does not create the
    /// key and does not touch its deadline.
    pub stored: bool,
}

/// `current + by`, held inside the bounds, in the kind `by` is.
///
/// An out of range result is refused unless `saturate`, in which case it lands
/// on the bound it went past. Overflow counts as going past the bound in the
/// direction of the increment, so `INCREX` on `i64::MAX` refuses rather than
/// wrapping, and with `SATURATE` it stays where it is.
pub fn apply(current: Num, opts: &IncrEx) -> Result<Counted> {
    match opts.by {
        Num::Int(by) => {
            let now = current.as_int("value")?;
            let lo = opts.lower.map_or(Ok(i64::MIN), |b| b.as_int("LBOUND"))?;
            let hi = opts.upper.map_or(Ok(i64::MAX), |b| b.as_int("UBOUND"))?;
            if lo > hi {
                return Err(bounds_crossed());
            }
            let want = now.checked_add(by);
            let out = match want {
                Some(v) if v >= lo && v <= hi => Some(v),
                _ if !opts.saturate => None,
                // Which bound it landed on is decided by the direction of the
                // increment and not by the arithmetic, because the arithmetic
                // may have overflowed on the way there.
                _ if by >= 0 => Some(hi),
                _ => Some(lo),
            };
            Ok(match out {
                Some(v) => Counted {
                    value: Num::Int(v),
                    applied: Num::Int(v.saturating_sub(now)),
                    stored: true,
                },
                None => Counted {
                    value: Num::Int(now),
                    applied: Num::Int(0),
                    stored: false,
                },
            })
        }
        Num::Float(by) => {
            if by.is_nan() {
                return Err(Error::new(Code::Invalid, "value is not a valid float"));
            }
            let now = match current {
                Num::Float(f) => f,
                Num::Int(n) => n as f64,
            };
            let lo = opts.lower.map_or(Ok(f64::MIN), |b| b.as_float("LBOUND"))?;
            let hi = opts.upper.map_or(Ok(f64::MAX), |b| b.as_float("UBOUND"))?;
            if lo > hi {
                return Err(bounds_crossed());
            }
            let want = now + by;
            let out = if want.is_finite() && want >= lo && want <= hi {
                Some(want)
            } else if !opts.saturate {
                None
            } else if by >= 0.0 {
                Some(hi)
            } else {
                Some(lo)
            };
            Ok(match out {
                Some(v) => Counted {
                    value: Num::Float(v),
                    applied: Num::Float(v - now),
                    stored: true,
                },
                None => Counted {
                    value: Num::Float(now),
                    applied: opts.by.zero_like(),
                    stored: false,
                },
            })
        }
    }
}

/// What a real 8.8 says when the range is empty.
///
/// It refuses rather than treating it as a range nothing fits in, which is the
/// right call: a caller that has its bounds the wrong way round has a bug, and
/// silently refusing every increment forever is a hard bug to find.
fn bounds_crossed() -> Error {
    Error::new(Code::Invalid, "LBOUND can't be greater than UBOUND")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int(n: i64) -> Num {
        Num::Int(n)
    }

    #[test]
    fn the_plain_form_adds_one() {
        let c = apply(int(5), &IncrEx::PLAIN).unwrap();
        assert_eq!(c.value, int(6));
        assert_eq!(c.applied, int(1));
        assert!(c.stored);
    }

    #[test]
    fn a_result_past_a_bound_is_refused_and_nothing_is_written() {
        // What a real 8.8 replies to `INCREX b BYINT 10 UBOUND 5` on a key that
        // is not there: the value it would have had, zero applied, and the key
        // still not there afterwards.
        let opts = IncrEx::PLAIN.by(int(10)).between(None, Some(int(5)));
        let c = apply(int(0), &opts).unwrap();
        assert_eq!(c.value, int(0));
        assert_eq!(c.applied, int(0));
        assert!(!c.stored);
    }

    #[test]
    fn saturate_lands_on_the_bound_and_reports_what_it_managed() {
        let opts = IncrEx::PLAIN
            .by(int(10))
            .between(None, Some(int(5)))
            .saturating();
        let c = apply(int(0), &opts).unwrap();
        assert_eq!(c.value, int(5));
        assert_eq!(c.applied, int(5));
        assert!(c.stored);

        // Downwards, from 5 to a floor of 0, which is minus five and not minus
        // ten.
        let down = IncrEx::PLAIN
            .by(int(-10))
            .between(Some(int(0)), None)
            .saturating();
        let c = apply(int(5), &down).unwrap();
        assert_eq!(c.value, int(0));
        assert_eq!(c.applied, int(-5));
    }

    #[test]
    fn overflow_is_a_bound_and_not_a_wrap() {
        let c = apply(int(i64::MAX), &IncrEx::PLAIN).unwrap();
        assert_eq!(c.value, int(i64::MAX));
        assert_eq!(c.applied, int(0));
        assert!(!c.stored);

        let sat = apply(int(i64::MAX), &IncrEx::PLAIN.saturating()).unwrap();
        assert_eq!(sat.value, int(i64::MAX));
        assert_eq!(sat.applied, int(0));

        let down = apply(int(i64::MIN), &IncrEx::PLAIN.by(int(-1)).saturating()).unwrap();
        assert_eq!(down.value, int(i64::MIN));
        assert_eq!(down.applied, int(0));
    }

    #[test]
    fn bounds_the_wrong_way_round_are_refused_rather_than_obeyed() {
        // `INCREX c UBOUND 5 LBOUND 10` on a real 8.8.
        let opts = IncrEx::PLAIN.between(Some(int(10)), Some(int(5)));
        let e = apply(int(0), &opts).unwrap_err();
        assert_eq!(e.message(), "LBOUND can't be greater than UBOUND");

        let f = IncrEx::PLAIN
            .by(Num::Float(1.0))
            .between(Some(Num::Float(10.0)), Some(Num::Float(5.0)));
        assert!(apply(Num::Float(0.0), &f).is_err());
    }

    #[test]
    fn a_float_bound_on_an_integer_increment_is_an_error() {
        // `INCREX q BYINT 1 UBOUND 5.5` on a real 8.8 is
        // `ERR UBOUND is not an integer or out of range`.
        let opts = IncrEx::PLAIN.between(None, Some(Num::Float(5.5)));
        let e = apply(int(1), &opts).unwrap_err();
        assert!(e.message().contains("UBOUND"), "{e}");
    }

    #[test]
    fn a_float_increment_counts_in_floats() {
        let c = apply(Num::Float(1.0), &IncrEx::PLAIN.by(Num::Float(0.5))).unwrap();
        assert_eq!(c.value, Num::Float(1.5));
        assert_eq!(c.applied, Num::Float(0.5));

        // An integer bound is fine on a float increment, since every i64 is a
        // sensible float bound.
        let bounded = IncrEx::PLAIN
            .by(Num::Float(10.0))
            .between(None, Some(int(5)))
            .saturating();
        let c = apply(Num::Float(0.0), &bounded).unwrap();
        assert_eq!(c.value, Num::Float(5.0));
    }

    #[test]
    fn a_float_that_overflows_to_infinity_is_out_of_range() {
        let opts = IncrEx::PLAIN.by(Num::Float(f64::MAX));
        let c = apply(Num::Float(f64::MAX), &opts).unwrap();
        assert!(!c.stored);
        assert_eq!(c.value, Num::Float(f64::MAX));

        let sat = apply(Num::Float(f64::MAX), &opts.saturating()).unwrap();
        assert_eq!(sat.value, Num::Float(f64::MAX));
        assert_eq!(sat.applied, Num::Float(0.0));
    }
}
