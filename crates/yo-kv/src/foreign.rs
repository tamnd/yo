//! A body this crate holds without knowing what it is.
//!
//! Everything in `yo-kv` is a primitive that something else is built out of.
//! `yo-doc` uses this crate's `Set`, `Slab` and rank tree; `yo-graph` uses
//! `yo-doc`; a vector index will use both. So the engines that a key could hold
//! all sit above this crate and none of them can be named from inside it.
//!
//! That leaves two ways to put a graph under a key. Either the keyspace stops
//! being the only thing that knows which keys exist, and `DEL`, `TYPE`,
//! `EXISTS`, `KEYS`, `SCAN`, `RANDOMKEY`, `EXPIRE`, `DBSIZE` and `FLUSHDB` each
//! learn to ask a second table, or the keyspace holds the body as something it
//! cannot look inside. The first way is nine places that have to agree and will
//! not, and the drift shows up as a key that `EXISTS` can see and `DEL` cannot
//! remove. This is the second way.
//!
//! # What the keyspace needs to know
//!
//! Four things, and no more: what to call it, what `OBJECT ENCODING` says, what
//! it costs, and whether it is empty. Everything else about a graph is asked
//! through a downcast by the layer that put it there and knows what it is.
//!
//! # Downcasting
//!
//! [`Foreign`] requires [`Any`], so [`Keyspace::foreign`] and
//! [`Keyspace::foreign_mut`] hand back a `&dyn Foreign` that the caller turns
//! back into its own type with `downcast_ref`. That is a vtable dispatch and a
//! type id comparison, which is a few nanoseconds on a command that is about to
//! do a hash lookup and a traversal, so it is not on any path where it matters.
//!
//! [`Keyspace::foreign`]: crate::Keyspace::foreign
//! [`Keyspace::foreign_mut`]: crate::Keyspace::foreign_mut

use std::any::Any;
use std::fmt::Debug;

/// A body the keyspace holds and does not understand.
///
/// Implemented above this crate, on the engines the keyspace cannot name. The
/// keyspace owns the box and frees it when the key goes, so a foreign body
/// cannot outlive its key and cannot be left behind by a `DEL` that forgot
/// about it.
///
/// `Send` for the same reason a value of any other type is: the key it hangs
/// off lives in a stripe, and a stripe is worked on by whichever thread holds
/// its lock. Every body in this repo is plain owned data and meets the bound
/// already.
pub trait Foreign: Any + Debug + Send {
    /// The word `TYPE` replies with.
    ///
    /// Not `"foreign"`. A client asking `TYPE` about a graph is told `graph`,
    /// because the escape is this crate's problem and not the client's.
    fn type_name(&self) -> &'static str;

    /// The word `OBJECT ENCODING` replies with.
    fn encoding(&self) -> &'static str;

    /// What this body has allocated, for `MEMORY USAGE` and for `maxmemory`.
    ///
    /// The same contract every other body in here has: what it holds, not what
    /// it would hold if it were packed, and it is walked rather than tracked
    /// because a running total is a field that gets out of step.
    fn memory_bytes(&self) -> usize;

    /// Whether the body is empty, so the key can go when its last member does.
    ///
    /// Redis deletes a key when its collection empties and a client can see the
    /// difference, so this is what the commands above check after a removal
    /// rather than each of them deciding what empty means.
    fn is_empty(&self) -> bool;
}

impl dyn Foreign {
    /// This body as the type that put it here, or `None` if it is another one.
    ///
    /// Two graphs and a vector index share one tag, so a command has to ask
    /// rather than assume. `None` is not an error on its own: it is what
    /// `G.NGET` on a key holding a vector index sees, and the caller turns it
    /// into WRONGTYPE.
    #[must_use]
    pub fn downcast_ref<T: Foreign>(&self) -> Option<&T> {
        (self as &dyn Any).downcast_ref::<T>()
    }

    /// The same, with a mutable borrow.
    #[must_use]
    pub fn downcast_mut<T: Foreign>(&mut self) -> Option<&mut T> {
        (self as &mut dyn Any).downcast_mut::<T>()
    }
}
