//! The error struct and the rules for filling it in.
//!
//! Never `errno`, never a thread local last error, never a negative integer that
//! means seven things. The call says whether it worked and the out parameter
//! says what happened, with all five reportable fields intact.
//!
//! The leading `size` is the Win32 and Vulkan discipline. The caller sets it to
//! `sizeof(yo_error)` for the header it compiled against, and the engine writes
//! only the fields that fit. That is what lets this struct gain a field later
//! without breaking a binary built today.

use crate::arena::ResultArena;
use core::ffi::c_char;
use core::mem::{offset_of, size_of};
use yo_common::Code;

/// The C `yo_error`.
///
/// `code` is `u32` rather than a Rust enum on purpose. A C enum with no negative
/// enumerators is four bytes on every compiler that matters, and taking the
/// integer means a value from a newer engine arrives as a number to report
/// rather than as undefined behaviour in a match.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[allow(non_camel_case_types)]
pub struct yo_error {
    /// Set by the caller to `sizeof(yo_error)`.
    pub size: u32,
    /// The condition, from `errors.toml`.
    pub code: u32,
    /// Whether the identical call could succeed later.
    pub retryable: u8,
    /// What happened, in a sentence.
    pub message: *const c_char,
    /// The key, field, path or offset it happened at, or null.
    pub position: *const c_char,
    /// The documentation page. Never null for a real condition.
    pub url: *const c_char,
    /// The long form, such as a shape diff. Multi line, or null.
    pub detail: *const c_char,
}

/// The last byte of a field, for deciding whether the caller's struct has it.
const fn end_of<T>(offset: usize) -> u32 {
    (offset + size_of::<T>()) as u32
}

/// A condition on its way out to C.
///
/// Built by value so that the fill happens in one place and every field goes
/// through the same size check.
pub struct Report<'a> {
    code: Code,
    message: &'a str,
    position: Option<&'a str>,
    detail: Option<&'a str>,
}

impl<'a> Report<'a> {
    /// A condition with a message.
    pub fn new(code: Code, message: &'a str) -> Report<'a> {
        Report {
            code,
            message,
            position: None,
            detail: None,
        }
    }

    /// Attaches the key, field or offset it happened at.
    ///
    /// Nothing at this milestone has a position to report, because the only
    /// conditions it can raise are about the arguments rather than about the
    /// data. The field is still filled and tested, because the day a shape
    /// mismatch needs it is not the day to find out the plumbing is missing.
    #[allow(dead_code)]
    pub fn at(mut self, position: &'a str) -> Report<'a> {
        self.position = Some(position);
        self
    }

    /// Attaches the long form, such as a shape diff.
    #[allow(dead_code)]
    pub fn with_detail(mut self, detail: &'a str) -> Report<'a> {
        self.detail = Some(detail);
        self
    }

    /// Writes the condition into the caller's struct.
    ///
    /// A null `err` is legal and means the caller does not want the detail, so
    /// this does nothing. That is a real choice a C caller makes and refusing it
    /// would just push a dummy struct into every call site.
    ///
    /// Strings go in the arena when the message has to be built. Every message
    /// at this milestone is a literal instead, so `arena` goes unused, and the
    /// parameter is here rather than added later because adding it later means
    /// touching every call site at the moment the first shape diff needs it.
    ///
    /// # Safety
    ///
    /// `err` is null or points to a `yo_error` whose `size` field the caller has
    /// set to the size of the struct their build knows about.
    pub unsafe fn emit(self, err: *mut yo_error, arena: Option<&mut ResultArena>) {
        let _ = arena;
        if err.is_null() {
            return;
        }
        // SAFETY: the caller promises `err` points to an initialised struct, and
        // `size` is the one field every version has.
        let size = unsafe { (*err).size };
        if size < end_of::<u32>(offset_of!(yo_error, code)) {
            // A struct too small to hold a code is a caller that has not
            // initialised it. Writing anything would be a guess.
            return;
        }

        // SAFETY: `err` is a valid, caller owned struct and each write below is
        // guarded by the caller's own declared size.
        unsafe {
            (*err).code = self.code.as_u32();
            if size >= end_of::<u8>(offset_of!(yo_error, retryable)) {
                (*err).retryable = u8::from(self.code.is_retryable());
            }
            if size >= end_of::<*const c_char>(offset_of!(yo_error, message)) {
                (*err).message = cstr(self.message);
            }
            if size >= end_of::<*const c_char>(offset_of!(yo_error, position)) {
                (*err).position = self.position.map_or(core::ptr::null(), cstr);
            }
            if size >= end_of::<*const c_char>(offset_of!(yo_error, url)) {
                (*err).url = self.code.url_z().map_or(core::ptr::null(), cstr);
            }
            if size >= end_of::<*const c_char>(offset_of!(yo_error, detail)) {
                (*err).detail = self.detail.map_or(core::ptr::null(), cstr);
            }
        }
    }
}

/// Turns a `&'static str` that is already NUL terminated into a C string.
///
/// Every string that crosses at this milestone is a literal in this crate with
/// an explicit trailing NUL, checked by the assertion below rather than trusted.
fn cstr(s: &str) -> *const c_char {
    debug_assert!(
        s.as_bytes().last() == Some(&0),
        "a string crossing the ABI is not NUL terminated: {s:?}"
    );
    s.as_ptr().cast()
}

/// Marks a successful call, so a caller that reuses one struct across calls does
/// not see a stale message from three calls ago.
///
/// # Safety
///
/// `err` is null or a valid `yo_error` with `size` set.
pub unsafe fn clear(err: *mut yo_error) {
    if err.is_null() {
        return;
    }
    // SAFETY: the caller promises the struct is valid and initialised.
    let size = unsafe { (*err).size };
    if size < end_of::<u32>(offset_of!(yo_error, code)) {
        return;
    }
    // SAFETY: same, and every write is guarded by the declared size.
    unsafe {
        (*err).code = Code::Ok.as_u32();
        if size >= end_of::<u8>(offset_of!(yo_error, retryable)) {
            (*err).retryable = 0;
        }
        if size >= end_of::<*const c_char>(offset_of!(yo_error, message)) {
            (*err).message = core::ptr::null();
        }
        if size >= end_of::<*const c_char>(offset_of!(yo_error, position)) {
            (*err).position = core::ptr::null();
        }
        if size >= end_of::<*const c_char>(offset_of!(yo_error, url)) {
            (*err).url = core::ptr::null();
        }
        if size >= end_of::<*const c_char>(offset_of!(yo_error, detail)) {
            (*err).detail = core::ptr::null();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blank() -> yo_error {
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

    fn read(p: *const c_char) -> Option<String> {
        if p.is_null() {
            return None;
        }
        // SAFETY: every pointer this crate puts in the struct is a NUL
        // terminated literal with static lifetime.
        Some(
            unsafe { core::ffi::CStr::from_ptr(p) }
                .to_string_lossy()
                .into_owned(),
        )
    }

    #[test]
    fn the_layout_is_the_one_the_header_declares() {
        // The header lays these out in this order, and a mismatch here is an
        // ABI break that no compiler would catch.
        assert_eq!(offset_of!(yo_error, size), 0);
        assert_eq!(offset_of!(yo_error, code), 4);
        assert_eq!(offset_of!(yo_error, retryable), 8);
        assert!(offset_of!(yo_error, message) >= 16);
        assert!(offset_of!(yo_error, position) > offset_of!(yo_error, message));
        assert!(offset_of!(yo_error, url) > offset_of!(yo_error, position));
        assert!(offset_of!(yo_error, detail) > offset_of!(yo_error, url));
    }

    #[test]
    fn every_field_arrives() {
        let mut e = blank();
        // SAFETY: `e` is a real struct with its size set.
        unsafe {
            Report::new(Code::WrongType, "the key holds a hash, not a string\0")
                .at("user:42\0")
                .with_detail("expected: string\nfound: hash\0")
                .emit(&mut e, None);
        }
        assert_eq!(e.code, Code::WrongType.as_u32());
        assert_eq!(e.retryable, 0);
        assert_eq!(
            read(e.message).unwrap(),
            "the key holds a hash, not a string"
        );
        assert_eq!(read(e.position).unwrap(), "user:42");
        assert_eq!(read(e.detail).unwrap(), "expected: string\nfound: hash");
        assert_eq!(read(e.url).as_deref(), Code::WrongType.url());
    }

    #[test]
    fn retryable_comes_from_the_table_and_not_from_the_call_site() {
        let mut e = blank();
        // SAFETY: as above.
        unsafe {
            Report::new(Code::Locked, "another process holds the writer lock\0").emit(&mut e, None)
        };
        assert_eq!(e.retryable, 1, "Locked is retryable in errors.toml");
    }

    #[test]
    fn an_older_caller_struct_gets_only_the_fields_it_has() {
        let mut e = blank();
        // A build that predates `detail` would declare a struct ending after
        // `url`. Nothing may be written past that.
        e.size = offset_of!(yo_error, detail) as u32;
        // A sentinel, never dereferenced, only compared. `without_provenance`
        // rather than a plain `as` cast because a cast from an integer is an
        // unsupported operation under `-Zmiri-strict-provenance` and this test
        // is one of the ones Miri runs.
        let sentinel = core::ptr::without_provenance::<c_char>(0xdead_beef);
        e.detail = sentinel;
        // SAFETY: `e` is valid and its size says how much of it exists.
        unsafe {
            Report::new(Code::Corrupt, "checksum mismatch\0")
                .with_detail("segment 4\0")
                .emit(&mut e, None);
        }
        assert_eq!(e.code, Code::Corrupt.as_u32());
        assert!(
            !e.url.is_null(),
            "url is inside the declared size and should be written"
        );
        assert_eq!(
            e.detail, sentinel,
            "detail was written past the caller's struct"
        );
    }

    #[test]
    fn an_uninitialised_struct_is_left_alone() {
        let mut e = blank();
        e.size = 0;
        e.code = 12345;
        // SAFETY: `e` is a real allocation; the point is that size zero means
        // the engine cannot tell what is there and writes nothing.
        unsafe { Report::new(Code::Invalid, "no\0").emit(&mut e, None) };
        assert_eq!(e.code, 12345);
    }

    #[test]
    fn null_is_a_legal_error_out_param() {
        // SAFETY: null is explicitly allowed and means the caller does not want
        // the detail.
        unsafe {
            Report::new(Code::NotFound, "no such key\0").emit(core::ptr::null_mut(), None);
            clear(core::ptr::null_mut());
        }
    }

    #[test]
    fn clear_wipes_a_reused_struct() {
        let mut e = blank();
        // SAFETY: `e` is valid with its size set.
        unsafe {
            Report::new(Code::Io, "read failed\0")
                .at("offset 0\0")
                .emit(&mut e, None);
            clear(&mut e);
        }
        assert_eq!(e.code, 0);
        assert!(e.message.is_null());
        assert!(e.position.is_null());
        assert!(e.url.is_null());
    }
}
