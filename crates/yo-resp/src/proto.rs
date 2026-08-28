//! The protocol version a connection is speaking, and the limits it is held to.

/// RESP2 or RESP3, per connection.
///
/// A connection starts at RESP2 and moves to RESP3 when the client sends
/// `HELLO 3`. It can move back. The version is a property of the connection and
/// never of the command, which is why every reply writer takes it once at
/// construction rather than at each call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Proto {
    /// The protocol every client understands. Five types and nothing else.
    #[default]
    Resp2,
    /// Redis 6 and later. Maps, sets, doubles, booleans, push messages, and a
    /// null that is not a length of minus one.
    Resp3,
}

impl Proto {
    /// The number a `HELLO` reply reports, and the number `HELLO` takes.
    #[inline]
    pub const fn version(self) -> i64 {
        match self {
            Proto::Resp2 => 2,
            Proto::Resp3 => 3,
        }
    }

    /// The protocol for a `HELLO` argument, or `None` if there is no such
    /// version. `HELLO 4` is an error and this is where that is decided.
    #[inline]
    pub const fn from_version(v: i64) -> Option<Proto> {
        match v {
            2 => Some(Proto::Resp2),
            3 => Some(Proto::Resp3),
            _ => None,
        }
    }

    /// Whether the richer type set is available.
    #[inline]
    pub const fn is_resp3(self) -> bool {
        matches!(self, Proto::Resp3)
    }
}

/// The bounds the codec enforces, which are Redis's bounds.
///
/// These are not tuning knobs, they are the difference between a protocol error
/// and an allocation the size of whatever a stranger asked for. A count line
/// saying two billion arguments has to be refused before anything is reserved
/// for it, which is why the multibulk limit is checked against the parsed
/// number and not against what arrives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// The largest `*` count accepted. Redis's `PROTO_MAX_MULTIBULK`.
    pub max_multibulk: usize,
    /// The largest `$` length accepted. Redis's `proto-max-bulk-len`, which is
    /// configurable there and so is configurable here.
    pub max_bulk: usize,
    /// The longest an inline request or an unterminated count line may get
    /// before it is refused. Redis's `PROTO_INLINE_MAX_SIZE`.
    pub max_inline: usize,
    /// How deeply a reply may nest before the decoder gives up.
    ///
    /// Redis has no equivalent because Redis never parses a reply. This exists
    /// because the reply decoder recurses, and `*1\r\n` repeated a million
    /// times would otherwise be a stack overflow rather than an error. There is
    /// no legitimate reply anywhere near this deep.
    pub max_depth: usize,
}

impl Limits {
    /// Redis's defaults: a million arguments, a 512 MiB bulk, a 64 KiB inline
    /// request.
    pub const DEFAULT: Limits = Limits {
        max_multibulk: 1024 * 1024,
        max_bulk: 512 * 1024 * 1024,
        max_inline: 64 * 1024,
        max_depth: 128,
    };
}

impl Default for Limits {
    fn default() -> Limits {
        Limits::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_connection_starts_at_resp2() {
        assert_eq!(Proto::default(), Proto::Resp2);
        assert!(!Proto::default().is_resp3());
    }

    #[test]
    fn hello_takes_two_and_three_and_nothing_else() {
        assert_eq!(Proto::from_version(2), Some(Proto::Resp2));
        assert_eq!(Proto::from_version(3), Some(Proto::Resp3));
        for v in [-1, 0, 1, 4, 300] {
            assert_eq!(Proto::from_version(v), None, "HELLO {v}");
        }
        for p in [Proto::Resp2, Proto::Resp3] {
            assert_eq!(Proto::from_version(p.version()), Some(p));
        }
    }

    #[test]
    fn the_limits_are_the_redis_numbers() {
        let l = Limits::default();
        assert_eq!(l.max_multibulk, 1024 * 1024);
        assert_eq!(l.max_bulk, 512 * 1024 * 1024);
        assert_eq!(l.max_inline, 64 * 1024);
    }
}
