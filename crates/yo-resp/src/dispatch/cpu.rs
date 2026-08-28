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
//! On a platform with no `getrusage` there is no `# CPU` section at all, and a
//! client that does not find the field falls back, where a client that finds a
//! zero believes it.

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
#[cfg(not(unix))]
pub fn usage() -> Option<Usage> {
    None
}
