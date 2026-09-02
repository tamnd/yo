//! How much processor time this server has used, for `INFO cpu`.
//!
//! It is one `getrusage` call and it is here rather than in a platform crate
//! because `INFO` is the only thing that asks. A monitoring tool graphs this
//! against wall clock to see whether a server is busy or waiting, so a made up
//! zero would be worse than nothing: a flat line reads as an idle server.
//!
//! Redis reports six numbers and this reports four. The two that are missing
//! are `used_cpu_sys_main_thread` and `used_cpu_user_main_thread`, which need
//! `RUSAGE_THREAD`, and that is Linux only. Reporting the process totals under
//! a name that says main thread would be right on a single threaded server and
//! wrong on the one this becomes, so they are left out.
//!
//! Windows has no `getrusage` and it does have `GetProcessTimes`, which is the
//! same two numbers for this process. What it has no equivalent of is the child
//! totals, so those two are zero there and that is true rather than made up:
//! this server starts no children.
//!
//! On a platform with neither there is no `# CPU` section at all, and a client
//! that does not find the field falls back, where a client that finds a zero
//! believes it.

/// Processor time used since this process started, in seconds.
///
/// The first pair is this process and the second is every child it has waited
/// for, which is the split Redis reports.
pub struct Usage {
    /// Time in the kernel on this process's behalf.
    pub sys: f64,
    /// Time running this process's own instructions.
    pub user: f64,
    /// The same, for children that have been waited for.
    pub sys_children: f64,
    /// The same, for children that have been waited for.
    pub user_children: f64,
}

/// Read the counters, or `None` where the platform has no way to.
#[cfg(unix)]
pub fn usage() -> Option<Usage> {
    let (user, sys) = read(libc::RUSAGE_SELF);
    let (user_children, sys_children) = read(libc::RUSAGE_CHILDREN);
    Some(Usage {
        sys,
        user,
        sys_children,
        user_children,
    })
}

/// The user and system totals for one `RUSAGE_` target, in seconds.
///
/// A failed call is zero rather than a refusal, because the only documented
/// failure is an argument that is not one of the constants and both of the ones
/// passed here are.
#[cfg(unix)]
fn read(who: libc::c_int) -> (f64, f64) {
    // SAFETY: `getrusage` writes into the struct and reads nothing else, and
    // the struct is a local that lives across the call.
    let ru = unsafe {
        let mut ru: libc::rusage = core::mem::zeroed();
        if libc::getrusage(who, &raw mut ru) != 0 {
            return (0.0, 0.0);
        }
        ru
    };
    let secs = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 / 1_000_000.0;
    (secs(ru.ru_utime), secs(ru.ru_stime))
}

/// Read the counters, or `None` where the platform has no way to.
///
/// `GetProcessTimes` is the Windows equivalent of `getrusage` for this process
/// and it is exact rather than sampled. There is no equivalent for children:
/// Windows does not keep a running total for processes this one has waited for,
/// because it has no `wait` in that sense and no parent child accounting to hang
/// one off. This server starts no children, so the honest total for them is
/// zero, and that is what those two fields say.
#[cfg(windows)]
pub fn usage() -> Option<Usage> {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};

    let mut created = FILETIME::default();
    let mut exited = FILETIME::default();
    let mut sys = FILETIME::default();
    let mut user = FILETIME::default();
    // SAFETY: all four are locals that live across the call, the handle from
    // `GetCurrentProcess` is a pseudo handle that is always valid and never
    // needs closing, and the call only writes into what it is given.
    let ok = unsafe {
        GetProcessTimes(
            GetCurrentProcess(),
            &raw mut created,
            &raw mut exited,
            &raw mut sys,
            &raw mut user,
        )
    };
    if ok == 0 {
        return None;
    }
    Some(Usage {
        sys: secs(sys),
        user: secs(user),
        sys_children: 0.0,
        user_children: 0.0,
    })
}

/// A `FILETIME` as seconds.
///
/// It is a count of hundred nanosecond ticks split across two 32 bit halves,
/// and it is not aligned well enough to be read as a `u64` in place, which is
/// why Windows hands it over in halves in the first place.
#[cfg(windows)]
fn secs(t: windows_sys::Win32::Foundation::FILETIME) -> f64 {
    let ticks = (u64::from(t.dwHighDateTime) << 32) | u64::from(t.dwLowDateTime);
    ticks as f64 / 1e7
}

/// Read the counters, or `None` where the platform has no way to.
#[cfg(not(any(unix, windows)))]
pub fn usage() -> Option<Usage> {
    None
}
