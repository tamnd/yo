//! The database handle.
//!
//! This is a stub and it says so out loud. There is no file, no record plane, no
//! expiry and no type beyond bytes, because none of those exist yet. What it does
//! have is the real index and the real arena underneath, which is enough for the
//! ABI, the header, the generator and the first C program to be exercised against
//! something that actually stores and returns data rather than a mock.
//!
//! # Execution mode
//!
//! Inline only, which is `15` section 7's first row: the calling thread is the
//! shard. That is the mode that makes a 150 ns point read possible at all, it is
//! the default whenever `shards == 1`, and it is what the sixty second snippet
//! uses. Owned mode arrives with the record plane, and asking for it here returns
//! `YO_ERR_UNSUPPORTED` rather than pretending.
//!
//! Because the mode is inline, the handle belongs to the thread that opened it.
//! The ABI documents `yo_db` as shareable from the version where owned mode
//! lands; widening a guarantee later is compatible, and claiming one now that the
//! implementation does not keep is not.

use yo_index::RawMap;

/// What is behind a `yo_db *`.
pub struct Db {
    /// The data. One map, no shards, no file.
    pub map: RawMap,
    /// Live arenas made from this database. Closing with children outstanding
    /// returns `YO_ERR_BUSY` and frees nothing, which is the ABI's uniform rule
    /// for every parent handle.
    pub children: usize,
    /// The thread that opened it, checked in debug builds.
    #[cfg(debug_assertions)]
    owner: std::thread::ThreadId,
}

impl Db {
    /// Opens an empty database.
    pub fn new() -> Db {
        Db {
            map: RawMap::new(),
            children: 0,
            #[cfg(debug_assertions)]
            owner: std::thread::current().id(),
        }
    }

    /// Fails loudly in debug if the handle has wandered onto another thread.
    #[inline]
    pub fn check_thread(&self) {
        #[cfg(debug_assertions)]
        assert_eq!(
            self.owner,
            std::thread::current().id(),
            "an inline mode database was used from a thread other than the one that opened it"
        );
    }
}

impl Default for Db {
    fn default() -> Db {
        Db::new()
    }
}

/// The C `yo_open_options`.
///
/// Caller laid out, so it carries the leading `size` and grows by appending. A
/// caller passing a struct from an older header gets defaults for everything
/// their build did not know about.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
pub struct yo_open_options {
    /// Set by the caller to `sizeof(yo_open_options)`.
    pub size: u32,
    /// Shard threads. Zero means one per core, one means inline.
    pub shards: u32,
    /// Non zero opens without taking the writer lock.
    pub read_only: u8,
}

impl yo_open_options {
    /// Reads an options struct the caller may have built from an older header.
    ///
    /// Fields past the caller's declared size are never read, which is the other
    /// half of the `size` discipline: the engine promises not to look at memory
    /// the caller did not allocate.
    ///
    /// # Safety
    ///
    /// `p` is null or points to a `yo_open_options` with `size` set.
    pub unsafe fn read(p: *const yo_open_options) -> yo_open_options {
        let mut out = yo_open_options {
            size: size_of::<yo_open_options>() as u32,
            shards: 1,
            read_only: 0,
        };
        if p.is_null() {
            return out;
        }
        // SAFETY: the caller promises the struct is initialised, and `size` is
        // the one field every version of it has.
        let size = unsafe { (*p).size };
        // SAFETY: guarded by the caller's own declared size, field by field.
        unsafe {
            if size >= (core::mem::offset_of!(yo_open_options, shards) + 4) as u32 {
                out.shards = (*p).shards;
            }
            if size >= (core::mem::offset_of!(yo_open_options, read_only) + 1) as u32 {
                out.read_only = (*p).read_only;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_options_mean_inline_and_writable() {
        // SAFETY: null is explicitly allowed and means the defaults.
        let o = unsafe { yo_open_options::read(core::ptr::null()) };
        assert_eq!(o.shards, 1);
        assert_eq!(o.read_only, 0);
    }

    #[test]
    fn a_short_options_struct_is_not_read_past() {
        let full = yo_open_options {
            size: 8,
            shards: 4,
            read_only: 1,
        };
        // SAFETY: `full` is a real struct; `size` of 8 says the caller's build
        // ends after `shards`.
        let o = unsafe { yo_open_options::read(&full) };
        assert_eq!(o.shards, 4, "shards is inside the declared size");
        assert_eq!(
            o.read_only, 0,
            "read_only is past it and must keep its default"
        );
    }

    #[test]
    fn the_stub_stores_and_returns_bytes() {
        let mut db = Db::new();
        assert_eq!(db.map.set(b"user:42", b"tam"), None);
        assert_eq!(db.map.get(b"user:42"), Some(&b"tam"[..]));
        assert_eq!(db.map.get(b"nobody"), None);
    }
}
