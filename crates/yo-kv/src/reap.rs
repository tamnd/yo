//! Telling somebody that a key went when no command asked for it.
//!
//! Almost everything a database does happens because a client asked, and the
//! client's own reply says it happened. Two things do not. A key whose deadline
//! passed goes on the way past, either the next time anything looks at it or
//! when the cycle in [`super::expiry`] gets to it, and a key a write needed the
//! memory of goes because the policy picked it. Nobody is waiting on either, so
//! nothing would say so, and a client that is watching a key wants to hear about
//! both.
//!
//! # Why a hook and not a return value
//!
//! Because a reap happens ten frames below the caller that would care. A `SADD`
//! that finds a dead key under the name it wants reaps it inside the lookup, and
//! the lookup answers "not there", which is all `SADD` needs and is exactly what
//! it would have been told about a key that never existed. Threading "and by the
//! way one went" back up through every lookup would put a return value nobody
//! reads on the funnel every command comes through, and it would still be wrong
//! for the cycle, which has no caller in the command path at all.
//!
//! # Why a thread local
//!
//! This crate does not know what a subscriber is and should not learn. What it
//! knows is that a key went and why. Somewhere above it there is a layer that
//! knows who is listening, and the same reason [`super::keyspace::Keyspace`]
//! does not carry a pub/sub registry is the reason it does not carry a
//! notifier either.
//!
//! So the layer above installs a plain function pointer on the thread it is
//! about to run work on, and this crate calls it if it is there. On a thread
//! with nobody installed a reap costs a thread local read and a test against
//! null, on a path that has just deleted a key and is therefore already paying
//! for a good deal more than that.
//!
//! Ordering falls out of this for free. The hook fires at the moment the key
//! goes, which is before the command that provoked it has done its own work, so
//! a listener that queues what it hears keeps the order a real server publishes
//! in without having to be told what that order is.

use core::cell::Cell;

/// Why a key went when nobody asked for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Why {
    /// Its deadline had passed, found on the way past or by the cycle.
    Expired,
    /// A write needed the memory and the policy chose this one.
    Evicted,
}

/// What gets called when one goes.
///
/// A bare function pointer and not a boxed closure, so that installing one is a
/// word written to a thread local and calling one is an indirect call, with
/// nothing allocated and nothing dropped. What a listener needs to know beyond
/// the key and the reason, the database number in practice, it keeps on the
/// side, since it is the one arranging for this to be installed at all.
pub type Told = fn(&[u8], Why);

thread_local! {
    /// Whoever wants to hear about it on this thread, and usually nobody.
    static TELL: Cell<Option<Told>> = const { Cell::new(None) };
}

/// Ask to be told about the keys this thread reaps, answering who was being
/// told before.
///
/// The answer goes back so that a caller which installs one around a piece of
/// work can put back whatever was there, which is what makes it safe to do this
/// inside something that has already done it.
pub fn tell(who: Option<Told>) -> Option<Told> {
    TELL.replace(who)
}

/// Say that a key went, if anybody asked to hear about it.
pub(crate) fn went(key: &[u8], why: Why) {
    if let Some(told) = TELL.get() {
        told(key, why);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    thread_local! {
        static HEARD: RefCell<Vec<(Vec<u8>, Why)>> = const { RefCell::new(Vec::new()) };
    }

    fn note(key: &[u8], why: Why) {
        HEARD.with_borrow_mut(|heard| heard.push((key.to_vec(), why)));
    }

    /// Nobody listening is the usual case and it has to be the cheap one, which
    /// here means it has to be a case at all rather than a panic.
    #[test]
    fn a_reap_nobody_asked_about_goes_nowhere() {
        assert!(tell(None).is_none());
        went(b"k", Why::Expired);
    }

    /// And one somebody did ask about arrives with the reason on it, since the
    /// two are different events to a listener and only this knows which it was.
    #[test]
    fn a_listener_hears_the_key_and_why_it_went() {
        let was = tell(Some(note));
        went(b"a", Why::Expired);
        went(b"b", Why::Evicted);
        tell(was);
        HEARD.with_borrow(|heard| {
            assert_eq!(
                heard.as_slice(),
                [(b"a".to_vec(), Why::Expired), (b"b".to_vec(), Why::Evicted)]
            );
        });
        // And the thread is back to how it was found, which is what lets a
        // caller install one around a piece of work inside another.
        assert!(tell(None).is_none());
    }
}
