//! The probabilistic sketches: answers that trade exactness for space.
//!
//! Every structure here answers a question about a set that would otherwise
//! need the set. A Bloom filter answers "have I seen this" in about ten bits an
//! item instead of the item, and it is wrong in one direction only, which is
//! what makes it useful in front of something expensive: a miss is certain, so
//! the expensive thing is never asked, and a hit is only probable, so it is.
//!
//! These are Redis's structures and not new ones. RedisBloom shipped the
//! command surface a decade ago and clients were written against it, and more
//! to the point `BF.SCANDUMP` hands a client the filter's actual bytes. A
//! sketch that answered the same commands over a different layout would be a
//! sketch nobody could move off this server, so the layouts here are copied
//! deliberately, down to the hash and the rounding. `11` section 8 has the
//! argument in full.
//!
//! ```
//! use yo_sketch::bloom::{Added, Bloom};
//!
//! let mut b = Bloom::new(1000, 0.01, 2, false);
//! assert_eq!(b.add(b"seen"), Added::Yes);
//! assert_eq!(b.add(b"seen"), Added::Already);
//! assert!(b.contains(b"seen"));
//! assert!(!b.contains(b"never"));
//! ```
//!
//! # What is here so far
//!
//! The scaling Bloom filter, which is [`bloom::Bloom`], and the cuckoo filter,
//! which is [`cuckoo::Cuckoo`]. The count min sketch, the top k and the t digest
//! go beside them and none of those is written yet.

pub mod bloom;
pub mod cuckoo;
mod hash;
