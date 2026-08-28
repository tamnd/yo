//! The protocol fuzzer `12` section 11 point 3 asks for.
//!
//! Everything here is bytes off a socket, which means every byte of it came
//! from somebody who is not us. The three properties being checked are the ones
//! that matter for a server that has to stay up:
//!
//! 1. No input panics, overflows or runs out of stack. A count line claiming
//!    four billion arguments must be an error and not an allocation.
//! 2. A decode that succeeds consumes exactly what it says it consumed, so a
//!    connection that trusts `consumed` stays in step with the client.
//! 3. Feeding the same bytes one at a time gives the same answer as feeding
//!    them all at once. This is the resume state's real test, because the whole
//!    point of it is that arrival order does not change the result.

#![no_main]

use libfuzzer_sys::fuzz_target;
use yo_resp::{Argv, Limits, Step, frame};

fuzz_target!(|data: &[u8]| {
    // Smaller than the defaults so that the interesting refusals are reachable
    // from inputs a fuzzer can actually produce in a few hundred bytes.
    let limits = Limits {
        max_multibulk: 1024,
        max_bulk: 4096,
        max_inline: 256,
        max_depth: 32,
    };

    // Requests, all at once.
    let mut argv = Argv::new();
    let whole = argv.decode(data, &limits);
    if let Ok(Step::Command { consumed }) = whole {
        assert!(consumed <= data.len(), "consumed past the end of the buffer");
        // Every argument it reported must be readable. A span pointing outside
        // the buffer would come back as `None` here rather than as a slice of
        // somebody else's memory, and either way it is a bug.
        for i in 0..argv.len() {
            assert!(argv.arg(data, i).is_some(), "argument {i} does not resolve");
        }
    }

    // Requests, one byte at a time. The same bytes in a different rhythm must
    // reach the same verdict.
    let mut piecemeal = Argv::new();
    let mut answer = None;
    for n in 0..=data.len() {
        match piecemeal.decode(&data[..n], &limits) {
            Ok(Step::Incomplete) => {}
            other => {
                answer = Some(other);
                break;
            }
        }
    }
    if let (Some(Ok(Step::Command { consumed })), Ok(Step::Command { consumed: all })) =
        (answer, whole)
    {
        assert_eq!(consumed, all, "the same command measured two ways");
    }
    // The two are only compared when both found a command. They are allowed to
    // disagree on an error, and the disagreement is not a bug: a size limit is
    // a limit on what is being held, so an inline request that passes the limit
    // before its newline arrives is refused, while the same bytes handed over
    // all at once contain the newline and are a command. Redis, which is also
    // streaming, refuses it the same way.

    // Replies. The server never parses one, but the replication client and the
    // differential harness do, and this is the side that recurses.
    if let Ok(Some((_, used))) = frame::decode(data, &limits) {
        assert!(used <= data.len(), "a frame used more than it was given");
        assert!(used > 0, "a frame used nothing and would loop forever");
    }
});
