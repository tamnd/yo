//! The C ABI. Every binding except Rust reaches the engine through this and
//! nothing else.
//!
//! Three jobs, in priority order, from `dx/02` section 1:
//!
//! 1. Be stable. Once the major version reaches 1, no symbol here changes
//!    signature or meaning, ever. It is 0 today, and that is the only window in
//!    which any of this can still move.
//! 2. Be fast enough to disappear. A binding gets a 3x budget against in process
//!    Rust and spends most of it in the host language, so this layer's own share
//!    has to be a call and a dereference. No allocation, no locking, no `errno`,
//!    no thread local lookup on a read path.
//! 3. Be mechanically bindable. The header is generated from the same model the
//!    binding generators read, so nobody has to parse C.
//!
//! Not a job: being pleasant to write C against. `yo-kit` is the pleasant layer.
//! This one is allowed to be verbose.
//!
//! # What is real here and what is not
//!
//! The index and the arena underneath are the real ones. The database is a stub
//! with no file, no record plane and no types beyond bytes, because those land
//! with the record plane. The shape of the boundary is what this milestone is
//! for, and the shape is real.
//!
//! # Reading this file
//!
//! Every entry point follows the same three steps: check the arguments and
//! report `YO_ERR_INVALID` if they are wrong, clear the caller's error struct,
//! then do the work. A returned handle means success. An `int32_t` return means
//! 1 or a count for found, 0 for not found, and -1 with `err` populated.

#![deny(missing_docs)]
#![allow(non_camel_case_types)]

mod arena;
mod db;
mod error;

use core::ffi::{CStr, c_char};

pub use crate::arena::ResultArena;
pub use crate::db::{Db, yo_open_options};
pub use crate::error::yo_error;

use crate::error::{Report, clear};
use yo_common::{BATCH_MAX, Code};

/// The database handle, as C sees it.
pub type yo_db = Db;
/// The arena handle, as C sees it.
pub type yo_arena = ArenaHandle;

/// An arena plus the database it was made from.
///
/// The ABI's arena calls take only the arena, so the arena is what has to know
/// its parent. That is the price of the uniform rule that closing a parent with
/// live children frees nothing: somebody has to keep the count, and making the
/// caller pass the database back to free an arena would be the other way to do
/// it, at the cost of an argument on a call that has no use for one.
pub struct ArenaHandle {
    arena: ResultArena,
    db: *mut Db,
}

/// The ABI major version. See `yo_abi_version`.
pub const ABI_MAJOR: u32 = 0;
/// The ABI minor version. A bump adds symbols and never changes one.
pub const ABI_MINOR: u32 = 1;

/// A borrowed run of bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct yo_slice {
    /// The first byte, or null for an absent value.
    pub ptr: *const u8,
    /// How many bytes.
    pub len: u64,
}

impl yo_slice {
    /// The slice that means "there is nothing here".
    pub const NONE: yo_slice = yo_slice {
        ptr: core::ptr::null(),
        len: 0,
    };
}

/// Turns a caller's slice into bytes, or `None` if it is malformed.
///
/// A null pointer with a non zero length is the one shape that is not a value,
/// and it is worth catching here rather than letting it become a segfault three
/// frames down.
///
/// # Safety
///
/// If `s.len` is non zero then `s.ptr` points to that many readable bytes for
/// the duration of the call.
unsafe fn bytes<'a>(s: yo_slice) -> Option<&'a [u8]> {
    if s.len == 0 {
        return Some(&[]);
    }
    if s.ptr.is_null() {
        return None;
    }
    // SAFETY: the caller promises `len` readable bytes at `ptr`.
    Some(unsafe { core::slice::from_raw_parts(s.ptr, s.len as usize) })
}

/// Reports a bad argument and returns the failure value.
///
/// # Safety
///
/// `err` is null or a valid `yo_error` with `size` set.
unsafe fn invalid(err: *mut yo_error, message: &str) -> i32 {
    // SAFETY: the caller's promise about `err` is passed straight through.
    unsafe { Report::new(Code::Invalid, message).emit(err, None) };
    -1
}

/// The ABI version as `(major << 16) | minor`.
///
/// Call it at load time. A different major means the struct layouts have moved
/// and the only correct response is to refuse to proceed.
#[unsafe(no_mangle)]
pub extern "C" fn yo_abi_version() -> u32 {
    (ABI_MAJOR << 16) | ABI_MINOR
}

/// The engine version as a NUL terminated string that lives forever.
#[unsafe(no_mangle)]
pub extern "C" fn yo_version_string() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// Does nothing, on purpose.
///
/// Every binding publishes the round trip cost of this next to its own overhead
/// number, so the split between what the FFI costs and what the binding costs is
/// visible rather than argued about.
#[unsafe(no_mangle)]
pub extern "C" fn yo_noop() {}

/// The spelling of a code, for a build that does not know it.
///
/// Never returns null. A binding compiled against an older header will meet
/// codes it has no name for, and that is a value to report rather than a crash.
#[unsafe(no_mangle)]
pub extern "C" fn yo_code_name(code: u32) -> *const c_char {
    match Code::from_u32(code) {
        Some(c) => c.c_name_z().as_ptr().cast(),
        None => c"YO_ERR_UNKNOWN".as_ptr(),
    }
}

/// Whether the identical call could succeed later.
#[unsafe(no_mangle)]
pub extern "C" fn yo_code_retryable(code: u32) -> u8 {
    Code::from_u32(code).is_some_and(Code::is_retryable) as u8
}

/// Opens a database.
///
/// `path` is accepted and ignored at this version: nothing is written to disk
/// until the record plane lands, and passing null is the honest way to ask for
/// what you actually get today.
///
/// # Safety
///
/// `path` is null or a NUL terminated string, `opts` is null or a valid
/// `yo_open_options` with `size` set, and `err` is null or a valid `yo_error`
/// with `size` set.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn yo_open(
    path: *const c_char,
    opts: *const yo_open_options,
    err: *mut yo_error,
) -> *mut yo_db {
    // SAFETY: the caller's promise about `err`.
    unsafe { clear(err) };

    if !path.is_null() {
        // SAFETY: the caller promises a NUL terminated string.
        if unsafe { CStr::from_ptr(path) }.to_str().is_err() {
            // SAFETY: as above for `err`.
            unsafe {
                Report::new(Code::Invalid, "the path is not valid UTF-8\0").emit(err, None);
            }
            return core::ptr::null_mut();
        }
    }

    // SAFETY: the caller's promise about `opts`.
    let opts = unsafe { yo_open_options::read(opts) };
    if opts.shards != 1 {
        // SAFETY: as above for `err`.
        unsafe {
            Report::new(
                Code::Unsupported,
                "only inline mode is implemented at this version, pass shards = 1\0",
            )
            .emit(err, None);
        }
        return core::ptr::null_mut();
    }

    Box::into_raw(Box::new(Db::new()))
}

/// Closes the database. Null is fine.
///
/// Closing with live arenas frees nothing, because a freed database with a live
/// arena pointing into it is a use after free waiting to be blamed on the
/// caller. The debug build asserts so that the binding bug surfaces in the
/// binding's own tests.
///
/// # Safety
///
/// `db` is null or a handle from `yo_open` that has not been closed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn yo_close(db: *mut yo_db) {
    if db.is_null() {
        return;
    }
    // SAFETY: the caller promises a live handle.
    let live = unsafe { (*db).children };
    debug_assert_eq!(live, 0, "yo_close called with {live} live arena(s)");
    if live != 0 {
        return;
    }
    // SAFETY: the handle came from `Box::into_raw` and is being freed once.
    drop(unsafe { Box::from_raw(db) });
}

/// How many keys are live.
///
/// # Safety
///
/// `db` is a live handle from `yo_open`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn yo_len(db: *const yo_db) -> u64 {
    if db.is_null() {
        return 0;
    }
    // SAFETY: the caller promises a live handle.
    unsafe {
        (*db).check_thread();
        (*db).map.len() as u64
    }
}

/// Opens an arena for results.
///
/// # Safety
///
/// `db` is a live handle and `err` is null or a valid `yo_error` with `size` set.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn yo_arena_new(db: *mut yo_db, err: *mut yo_error) -> *mut yo_arena {
    // SAFETY: the caller's promise about `err`.
    unsafe { clear(err) };
    if db.is_null() {
        // SAFETY: as above.
        unsafe {
            Report::new(Code::Invalid, "yo_arena_new was given a null database\0").emit(err, None)
        };
        return core::ptr::null_mut();
    }
    // SAFETY: the caller promises a live handle.
    unsafe {
        (*db).check_thread();
        (*db).children += 1;
    }
    Box::into_raw(Box::new(ArenaHandle {
        arena: ResultArena::new(),
        db,
    }))
}

/// Rewinds the arena and keeps its capacity.
///
/// Every borrowed view into it is invalid the instant this returns.
///
/// # Safety
///
/// `arena` is null or a live handle from `yo_arena_new`, and nothing holds a
/// view into it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn yo_arena_reset(arena: *mut yo_arena) {
    if arena.is_null() {
        return;
    }
    // SAFETY: the caller promises a live handle.
    unsafe {
        (*arena).arena.check_thread();
        (*arena).arena.reset();
    }
}

/// Bytes handed out since the last reset.
///
/// # Safety
///
/// `arena` is null or a live handle from `yo_arena_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn yo_arena_used(arena: *const yo_arena) -> u64 {
    if arena.is_null() {
        return 0;
    }
    // SAFETY: the caller promises a live handle.
    unsafe { (*arena).arena.used() }
}

/// Frees the arena. Null is fine.
///
/// Every view into it dies here. The parent database learns it has one fewer
/// child, which is what makes `yo_close` able to refuse.
///
/// # Safety
///
/// `arena` is null or a live handle from `yo_arena_new` that has not been freed,
/// and the database it came from is still open.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn yo_arena_free(arena: *mut yo_arena) {
    if arena.is_null() {
        return;
    }
    // SAFETY: the handle came from `Box::into_raw` and is freed once.
    let handle = unsafe { Box::from_raw(arena) };
    if !handle.db.is_null() {
        // SAFETY: the caller promises the parent is still open, which is the
        // only order in which the ABI's ownership rules allow this to happen.
        unsafe { (*handle.db).children -= 1 };
    }
}

/// Stores a value under a key. Returns 0, or -1 with `err` populated.
///
/// # Safety
///
/// `db` is a live handle, the two slices describe readable memory for the
/// duration of the call, and `err` is null or a valid `yo_error`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn yo_set(
    db: *mut yo_db,
    key: yo_slice,
    value: yo_slice,
    err: *mut yo_error,
) -> i32 {
    // SAFETY: the caller's promise about `err`.
    unsafe { clear(err) };
    if db.is_null() {
        // SAFETY: as above.
        return unsafe { invalid(err, "yo_set was given a null database\0") };
    }
    // SAFETY: the caller's promise about the slices.
    let (Some(k), Some(v)) = (unsafe { bytes(key) }, unsafe { bytes(value) }) else {
        // SAFETY: as above for `err`.
        return unsafe { invalid(err, "a slice has a null pointer and a non zero length\0") };
    };
    if k.is_empty() {
        // SAFETY: as above.
        return unsafe { invalid(err, "the key is empty\0") };
    }
    // SAFETY: the caller promises a live handle.
    unsafe {
        (*db).check_thread();
        (*db).map.set(k, v);
    }
    0
}

/// Reads a value without copying it. Returns 1 found, 0 missing, -1 error.
///
/// The view points into the engine and stays valid until the next write to this
/// database. If you cannot promise that, use `yo_get_copy`.
///
/// # Safety
///
/// `db` is a live handle, `key` describes readable memory, `out` is a writable
/// `yo_slice`, and `err` is null or a valid `yo_error`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn yo_get(
    db: *mut yo_db,
    key: yo_slice,
    out: *mut yo_slice,
    err: *mut yo_error,
) -> i32 {
    // SAFETY: the caller's promise about `err`.
    unsafe { clear(err) };
    if db.is_null() || out.is_null() {
        // SAFETY: as above.
        return unsafe { invalid(err, "yo_get was given a null database or output\0") };
    }
    // SAFETY: the caller's promise about `key`.
    let Some(k) = (unsafe { bytes(key) }) else {
        // SAFETY: as above for `err`.
        return unsafe { invalid(err, "the key has a null pointer and a non zero length\0") };
    };
    // SAFETY: the caller promises a live handle and a writable output.
    unsafe {
        (*db).check_thread();
        match (*db).map.get(k) {
            Some(v) => {
                *out = yo_slice {
                    ptr: v.as_ptr(),
                    len: v.len() as u64,
                };
                1
            }
            None => {
                *out = yo_slice::NONE;
                0
            }
        }
    }
}

/// Reads a value into the arena. Returns 1 found, 0 missing, -1 error.
///
/// This is the default in every managed language, and it costs one arena bump
/// rather than one allocation.
///
/// # Safety
///
/// As `yo_get`, plus `arena` is a live handle from `yo_arena_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn yo_get_copy(
    db: *mut yo_db,
    key: yo_slice,
    arena: *mut yo_arena,
    out: *mut yo_slice,
    err: *mut yo_error,
) -> i32 {
    // SAFETY: the caller's promise about `err`.
    unsafe { clear(err) };
    if db.is_null() || arena.is_null() || out.is_null() {
        // SAFETY: as above.
        return unsafe {
            invalid(
                err,
                "yo_get_copy was given a null database, arena or output\0",
            )
        };
    }
    // SAFETY: the caller's promise about `key`.
    let Some(k) = (unsafe { bytes(key) }) else {
        // SAFETY: as above for `err`.
        return unsafe { invalid(err, "the key has a null pointer and a non zero length\0") };
    };
    // SAFETY: the caller promises live handles and a writable output.
    unsafe {
        (*db).check_thread();
        (*arena).arena.check_thread();
        match (*db).map.get(k) {
            Some(v) => {
                let p = (*arena).arena.put(v);
                *out = yo_slice {
                    ptr: p.as_ptr(),
                    len: v.len() as u64,
                };
                1
            }
            None => {
                *out = yo_slice::NONE;
                0
            }
        }
    }
}

/// Reads many keys in one crossing. Returns how many were found, or -1.
///
/// This exists because the per call overhead in Python and Node is the binding
/// constraint, and amortising one crossing over 64 keys turns a 450 ns problem
/// into a 40 ns one. A missing key gets a null pointer and a zero length in its
/// own output slot, so the caller can tell missing from empty without a second
/// array.
///
/// # Safety
///
/// `db` and `arena` are live handles, `keys` points to `n` readable slices,
/// `out` points to `n` writable slices, and `err` is null or a valid `yo_error`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn yo_get_many(
    db: *mut yo_db,
    keys: *const yo_slice,
    n: u32,
    arena: *mut yo_arena,
    out: *mut yo_slice,
    err: *mut yo_error,
) -> i32 {
    // SAFETY: the caller's promise about `err`.
    unsafe { clear(err) };
    if db.is_null() || arena.is_null() {
        // SAFETY: as above.
        return unsafe { invalid(err, "yo_get_many was given a null database or arena\0") };
    }
    if n == 0 {
        return 0;
    }
    if keys.is_null() || out.is_null() {
        // SAFETY: as above.
        return unsafe { invalid(err, "yo_get_many was given a null key or output array\0") };
    }
    if n as usize > BATCH_MAX {
        // SAFETY: as above.
        return unsafe { invalid(err, "a batch may carry at most YO_BATCH_MAX keys\0") };
    }

    // SAFETY: the caller promises `n` readable slices and `n` writable ones.
    let (ks, os) = unsafe {
        (
            core::slice::from_raw_parts(keys, n as usize),
            core::slice::from_raw_parts_mut(out, n as usize),
        )
    };

    // SAFETY: the caller promises live handles.
    unsafe {
        (*db).check_thread();
        (*arena).arena.check_thread();
    }

    let mut found = 0i32;
    for (i, key) in ks.iter().enumerate() {
        // SAFETY: the caller's promise about each key slice.
        let Some(k) = (unsafe { bytes(*key) }) else {
            // SAFETY: as above for `err`.
            return unsafe { invalid(err, "a key has a null pointer and a non zero length\0") };
        };
        // SAFETY: both handles are live and were checked above.
        os[i] = unsafe {
            match (*db).map.get(k) {
                Some(v) => {
                    let p = (*arena).arena.put(v);
                    found += 1;
                    yo_slice {
                        ptr: p.as_ptr(),
                        len: v.len() as u64,
                    }
                }
                None => yo_slice::NONE,
            }
        };
    }
    found
}

/// Removes a key. Returns 1 removed, 0 absent, -1 error.
///
/// # Safety
///
/// `db` is a live handle, `key` describes readable memory, and `err` is null or
/// a valid `yo_error`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn yo_del(db: *mut yo_db, key: yo_slice, err: *mut yo_error) -> i32 {
    // SAFETY: the caller's promise about `err`.
    unsafe { clear(err) };
    if db.is_null() {
        // SAFETY: as above.
        return unsafe { invalid(err, "yo_del was given a null database\0") };
    }
    // SAFETY: the caller's promise about `key`.
    let Some(k) = (unsafe { bytes(key) }) else {
        // SAFETY: as above for `err`.
        return unsafe { invalid(err, "the key has a null pointer and a non zero length\0") };
    };
    // SAFETY: the caller promises a live handle.
    unsafe {
        (*db).check_thread();
        i32::from((*db).map.del(k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err() -> yo_error {
        yo_error {
            size: size_of::<yo_error>() as u32,
            code: 0,
            retryable: 0,
            message: core::ptr::null(),
            position: core::ptr::null(),
            url: core::ptr::null(),
            detail: core::ptr::null(),
        }
    }

    fn slice(b: &[u8]) -> yo_slice {
        yo_slice {
            ptr: b.as_ptr(),
            len: b.len() as u64,
        }
    }

    #[test]
    fn every_code_crosses_the_boundary_terminated() {
        // A string that reaches C without a NUL is a read past the end of a
        // Rust literal, which is the kind of bug that shows up as somebody
        // else's log line three services away.
        for c in Code::ALL {
            let name = c.c_name_z();
            assert!(name.ends_with('\0'), "{} is not NUL terminated", c.c_name());
            assert_eq!(&name[..name.len() - 1], c.c_name());
            let p = yo_code_name(c.as_u32());
            // SAFETY: the pointer is the generated literal, which is terminated
            // by the assertion just above.
            assert_eq!(unsafe { CStr::from_ptr(p) }.to_str().unwrap(), c.c_name());
            if let Some(u) = c.url_z() {
                assert!(
                    u.ends_with('\0'),
                    "the url for {} is not NUL terminated",
                    c.c_name()
                );
                assert_eq!(&u[..u.len() - 1], c.url().unwrap());
            } else {
                assert_eq!(c.url(), None);
            }
        }
    }

    #[test]
    fn an_unknown_code_still_gets_a_name() {
        let p = yo_code_name(9999);
        assert!(!p.is_null());
        // SAFETY: the returned pointer is a NUL terminated literal.
        let name = unsafe { CStr::from_ptr(p) };
        assert_eq!(name.to_str().unwrap(), "YO_ERR_UNKNOWN");
        assert_eq!(yo_code_retryable(9999), 0);
    }

    #[test]
    fn the_abi_version_packs_the_way_the_header_says() {
        assert_eq!(yo_abi_version(), (ABI_MAJOR << 16) | ABI_MINOR);
        assert_eq!(yo_abi_version() >> 16, ABI_MAJOR);
        assert_eq!(yo_abi_version() & 0xffff, ABI_MINOR);
    }

    #[test]
    fn open_set_get_close() {
        let mut e = err();
        // SAFETY: every argument below is a real object owned by this test.
        unsafe {
            let db = yo_open(core::ptr::null(), core::ptr::null(), &mut e);
            assert!(!db.is_null(), "open failed with code {}", e.code);

            assert_eq!(yo_set(db, slice(b"user:42"), slice(b"tam"), &mut e), 0);
            assert_eq!(yo_len(db), 1);

            let mut out = yo_slice::NONE;
            assert_eq!(yo_get(db, slice(b"user:42"), &mut out, &mut e), 1);
            assert_eq!(
                core::slice::from_raw_parts(out.ptr, out.len as usize),
                b"tam"
            );

            assert_eq!(yo_get(db, slice(b"nobody"), &mut out, &mut e), 0);
            assert!(
                out.ptr.is_null(),
                "a miss must not leave a stale pointer behind"
            );

            assert_eq!(yo_del(db, slice(b"user:42"), &mut e), 1);
            assert_eq!(yo_del(db, slice(b"user:42"), &mut e), 0);
            assert_eq!(yo_len(db), 0);
            yo_close(db);
        }
    }

    #[test]
    fn owned_mode_is_refused_rather_than_faked() {
        let mut e = err();
        let opts = yo_open_options {
            size: size_of::<yo_open_options>() as u32,
            shards: 8,
            read_only: 0,
        };
        // SAFETY: both structs are real and owned by this test.
        let db = unsafe { yo_open(core::ptr::null(), &opts, &mut e) };
        assert!(db.is_null());
        assert_eq!(e.code, Code::Unsupported.as_u32());
        assert!(!e.message.is_null(), "a refusal with no message is useless");
    }

    #[test]
    fn a_batch_costs_one_arena_and_reports_misses_in_place() {
        let mut e = err();
        // SAFETY: every argument below is a real object owned by this test.
        unsafe {
            let db = yo_open(core::ptr::null(), core::ptr::null(), &mut e);
            let arena = yo_arena_new(db, &mut e);
            assert!(!arena.is_null());

            for i in 0..40u32 {
                let k = format!("k{i}");
                let v = format!("v{i}");
                assert_eq!(
                    yo_set(db, slice(k.as_bytes()), slice(v.as_bytes()), &mut e),
                    0
                );
            }

            let names: Vec<String> = (0..64u32).map(|i| format!("k{i}")).collect();
            let keys: Vec<yo_slice> = names.iter().map(|k| slice(k.as_bytes())).collect();
            let mut out = vec![yo_slice::NONE; 64];

            let found = yo_get_many(db, keys.as_ptr(), 64, arena, out.as_mut_ptr(), &mut e);
            assert_eq!(found, 40, "the batch lost or invented rows");
            for (i, o) in out.iter().enumerate() {
                if i < 40 {
                    let got = core::slice::from_raw_parts(o.ptr, o.len as usize);
                    assert_eq!(got, format!("v{i}").as_bytes());
                } else {
                    assert!(o.ptr.is_null(), "slot {i} should be a miss");
                }
            }

            let used = yo_arena_used(arena);
            assert!(
                used > 0 && used < 4096,
                "40 short values should be a few hundred bytes, got {used}"
            );
            yo_arena_reset(arena);
            assert_eq!(yo_arena_used(arena), 0);

            yo_arena_free(arena);
            yo_close(db);
        }
    }

    #[test]
    fn a_batch_larger_than_the_limit_is_refused() {
        let mut e = err();
        // SAFETY: real objects, and the call is expected to reject before it
        // reads either array.
        unsafe {
            let db = yo_open(core::ptr::null(), core::ptr::null(), &mut e);
            let arena = yo_arena_new(db, &mut e);
            let keys = [slice(b"a")];
            let mut out = [yo_slice::NONE];
            let n = (BATCH_MAX + 1) as u32;
            assert_eq!(
                yo_get_many(db, keys.as_ptr(), n, arena, out.as_mut_ptr(), &mut e),
                -1
            );
            assert_eq!(e.code, Code::Invalid.as_u32());
            yo_arena_free(arena);
            yo_close(db);
        }
    }

    #[test]
    fn null_handles_are_reported_and_never_crash() {
        let mut e = err();
        // SAFETY: null is the whole point of this test and every entry point
        // documents it as reportable rather than undefined.
        unsafe {
            assert_eq!(
                yo_set(core::ptr::null_mut(), slice(b"k"), slice(b"v"), &mut e),
                -1
            );
            assert_eq!(e.code, Code::Invalid.as_u32());
            assert_eq!(
                yo_get(
                    core::ptr::null_mut(),
                    slice(b"k"),
                    core::ptr::null_mut(),
                    &mut e
                ),
                -1
            );
            assert_eq!(yo_del(core::ptr::null_mut(), slice(b"k"), &mut e), -1);
            assert_eq!(yo_len(core::ptr::null()), 0);
            yo_close(core::ptr::null_mut());
            yo_arena_free(core::ptr::null_mut());
            yo_arena_reset(core::ptr::null_mut());
            assert_eq!(yo_arena_used(core::ptr::null()), 0);
        }
    }

    #[test]
    fn a_malformed_slice_is_reported_rather_than_dereferenced() {
        let mut e = err();
        let bad = yo_slice {
            ptr: core::ptr::null(),
            len: 8,
        };
        // SAFETY: the malformed slice is never dereferenced, which is what this
        // test exists to prove.
        unsafe {
            let db = yo_open(core::ptr::null(), core::ptr::null(), &mut e);
            assert_eq!(yo_set(db, bad, slice(b"v"), &mut e), -1);
            assert_eq!(e.code, Code::Invalid.as_u32());
            let mut out = yo_slice::NONE;
            assert_eq!(yo_get(db, bad, &mut out, &mut e), -1);
            yo_close(db);
        }
    }

    #[test]
    fn an_empty_value_is_a_value_and_not_a_miss() {
        let mut e = err();
        // SAFETY: real objects owned by this test.
        unsafe {
            let db = yo_open(core::ptr::null(), core::ptr::null(), &mut e);
            assert_eq!(yo_set(db, slice(b"k"), yo_slice::NONE, &mut e), 0);
            let mut out = yo_slice {
                ptr: core::ptr::dangling(),
                len: 99,
            };
            assert_eq!(
                yo_get(db, slice(b"k"), &mut out, &mut e),
                1,
                "an empty value must still be found"
            );
            assert_eq!(out.len, 0);
            yo_close(db);
        }
    }

    #[test]
    fn a_copy_survives_a_write_and_a_borrow_is_not_promised_to() {
        let mut e = err();
        // SAFETY: real objects owned by this test.
        unsafe {
            let db = yo_open(core::ptr::null(), core::ptr::null(), &mut e);
            let arena = yo_arena_new(db, &mut e);
            assert_eq!(yo_set(db, slice(b"k"), slice(b"first"), &mut e), 0);

            let mut copied = yo_slice::NONE;
            assert_eq!(yo_get_copy(db, slice(b"k"), arena, &mut copied, &mut e), 1);

            for i in 0..10_000u32 {
                let k = format!("filler{i}");
                yo_set(db, slice(k.as_bytes()), slice(b"x"), &mut e);
            }
            yo_set(db, slice(b"k"), slice(b"second"), &mut e);

            let got = core::slice::from_raw_parts(copied.ptr, copied.len as usize);
            assert_eq!(
                got, b"first",
                "the arena copy should be untouched by later writes"
            );

            yo_arena_free(arena);
            yo_close(db);
        }
    }
}
