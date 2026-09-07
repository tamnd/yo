//! Telling somebody what happened to a key when no command reported it.
//!
//! Almost everything a database does happens because a client asked, and the
//! client's own reply says it happened. A few things do not. A key whose
//! deadline passed goes on the way past, either the next time anything looks at
//! it or when the cycle in [`super::expiry`] gets to it, and a key a write
//! needed the memory of goes because the policy picked it. A key coming into
//! being is the other way round: a command did ask, but what it asked for was
//! `SET` or `RPUSH` or `XADD`, and whether the name was already taken is not
//! part of the answer it gets back. Neither is what was under the name before,
//! which a write that replaces the whole value throws away, nor whether it was
//! even the same kind of thing.
//!
//! All of them are things a client watching a key wants to hear about and
//! nothing on the way out would otherwise mention.
//!
//! # Why a hook and not a return value
//!
//! Because these happen a long way below the caller that would care. A `SADD`
//! that finds a dead key under the name it wants reaps it inside the lookup, and
//! the lookup answers "not there", which is all `SADD` needs and is exactly what
//! it would have been told about a key that never existed. Threading "and by the
//! way one went" back up through every lookup would put a return value nobody
//! reads on the funnel every command comes through, and it would still be wrong
//! for the expiry cycle, which has no caller in the command path at all.
//!
//! A key arriving is the same shape of problem from the other end. Every group
//! writes records and every group can create one, so a return value would have
//! to be added to a few dozen entry points and then carried back through all of
//! them, when what actually knows is the one function underneath that puts a
//! record in the map.
//!
//! # Why a thread local
//!
//! This crate does not know what a subscriber is and should not learn. What it
//! knows is which key and what happened to it. Somewhere above there is a layer
//! that knows who is listening, and the same reason
//! [`super::keyspace::Keyspace`] does not carry a pub/sub registry is the reason
//! it does not carry a notifier either.
//!
//! So the layer above installs a plain function pointer on the thread it is
//! about to run work on, and this crate calls it if it is there. On a thread
//! with nobody installed each of these costs a thread local read and a test
//! against null, on paths that are already doing a good deal more than that.
//!
//! Ordering falls out of this for free. The hook fires at the moment the thing
//! happens, which for a key that was just created is before the command that
//! created it has said what it did, so a listener that queues what it hears
//! keeps the order a real server publishes in without having to be told what
//! that order is.

use core::cell::Cell;

/// What happened to a key that the command's own reply does not cover.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum What {
    /// It was not there a moment ago and now it is.
    Born,
    /// The whole of what was under it has just been thrown away for something
    /// else, which is not the same as a write that changed part of it.
    Overwritten,
    /// And what replaced it is a different kind of thing. Always said straight
    /// after an [`What::Overwritten`] and never on its own.
    TypeChanged,
    /// Its deadline had passed, found on the way past or by the cycle.
    Expired,
    /// A write needed the memory and the policy chose this one.
    Evicted,
}

/// What gets called when one of them happens.
///
/// A bare function pointer and not a boxed closure, so that installing one is a
/// word written to a thread local and calling one is an indirect call, with
/// nothing allocated and nothing dropped. What a listener needs to know beyond
/// the key and what happened, the database number in practice, it keeps on the
/// side, since it is the one arranging for this to be installed at all.
pub type Told = fn(&[u8], What);

thread_local! {
    /// Whoever wants to hear about it on this thread, and usually nobody.
    static TELL: Cell<Option<Told>> = const { Cell::new(None) };
}

/// Ask to be told what happens to the keys on this thread, answering who was
/// being told before.
///
/// The answer goes back so that a caller which installs one around a piece of
/// work can put back whatever was there, which is what makes it safe to do this
/// inside something that has already done it.
pub fn tell(who: Option<Told>) -> Option<Told> {
    TELL.replace(who)
}

/// Whether anybody is listening at all.
///
/// For a caller that has to look something up before it can say anything. A
/// write that replaces the whole of what was under a key has to know what kind
/// of thing was there to say whether the kind changed, and that is a probe of
/// the map on the way into `SET`, which is not a thing to pay for on a server
/// where the answer would go nowhere.
pub(crate) fn listening() -> bool {
    TELL.get().is_some()
}

/// Say what happened to a key, if anybody asked to hear about it.
pub(crate) fn say(key: &[u8], what: What) {
    if let Some(told) = TELL.get() {
        told(key, what);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    thread_local! {
        static HEARD: RefCell<Vec<(Vec<u8>, What)>> = const { RefCell::new(Vec::new()) };
    }

    fn note(key: &[u8], what: What) {
        HEARD.with_borrow_mut(|heard| heard.push((key.to_vec(), what)));
    }

    /// Nobody listening is the usual case and it has to be the cheap one, which
    /// here means it has to be a case at all rather than a panic.
    #[test]
    fn news_nobody_asked_for_goes_nowhere() {
        assert!(tell(None).is_none());
        say(b"k", What::Expired);
    }

    /// And news somebody did ask for arrives with what happened on it, since the
    /// three are different events to a listener and only this knows which it was.
    #[test]
    fn a_listener_hears_the_key_and_what_happened_to_it() {
        let was = tell(Some(note));
        say(b"a", What::Born);
        say(b"b", What::Expired);
        say(b"c", What::Evicted);
        tell(was);
        HEARD.with_borrow(|heard| {
            assert_eq!(
                heard.as_slice(),
                [
                    (b"a".to_vec(), What::Born),
                    (b"b".to_vec(), What::Expired),
                    (b"c".to_vec(), What::Evicted)
                ]
            );
        });
        // And the thread is back to how it was found, which is what lets a
        // caller install one around a piece of work inside another.
        assert!(tell(None).is_none());
    }
}
