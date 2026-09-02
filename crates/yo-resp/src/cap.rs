//! How much memory this process actually gets, and what to do with that number.
//!
//! A server that sizes itself from the machine is wrong on every machine where
//! it is not the only thing running, and in 2026 that is most of them. Inside a
//! container the kernel will happily tell you the host has 512 GB while the
//! cgroup this process lives in will kill it at 2. So the question is not what
//! the machine has, it is what this process is allowed to use, and the answer
//! is the smaller of the two.
//!
//! # The over 4 rule
//!
//! [`Cap::budget`] is a quarter of the limit, not the whole of it, and that
//! quarter is not a hedge. `16-implementation-plan.md` M5 records what happened
//! in aki when the pools were sized straight from `memory.max`: the same bytes
//! were charged twice, once to the pool and once to the pages the pool was
//! sitting on, and a result that should have been 1.58x came out at 5.14x and
//! 5.77x. Dividing by four fixed it. That is an empirical number rather than a
//! derived one, which is exactly why both the limit and the budget are reported
//! rather than only the one that gets used. The next person to be surprised by
//! this should be able to see both numbers without reading the source.
//!
//! # Where the numbers come from
//!
//! Under cgroup v2 the limit is in `memory.max`, in the directory named by the
//! `0::` line of `/proc/self/cgroup` under the cgroup mount. A limit can be set
//! on any ancestor and the tightest one wins, so this walks up to the mount
//! point taking the smallest number it finds. Under cgroup v1 the same idea
//! lives in `memory/memory.limit_in_bytes`, where "no limit" is a very large
//! number rather than a word. Neither file exists anywhere but Linux, and on a
//! machine without them there is no limit to find, which is the honest answer
//! and not an error.
//!
//! The parsing and the walk are separate from the paths they normally read, so
//! the tests build a directory tree and check the walk against it on every
//! platform rather than only on the one where it matters.

use std::path::{Path, PathBuf};

/// Where cgroup v2 is mounted on any system that has it.
const CGROUP_ROOT: &str = "/sys/fs/cgroup";

/// Anything at or above this in `memory.limit_in_bytes` means no limit.
///
/// cgroup v1 has no word for unlimited, so it writes `PAGE_COUNTER_MAX` scaled
/// by the page size, which comes out as a number near `i64::MAX` that differs
/// between kernels and page sizes. Treating anything within a factor of two of
/// the top as no limit is what every other reader of this file does.
const V1_UNLIMITED: u64 = u64::MAX / 4;

/// What this process is allowed to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Cap {
    /// The cgroup limit in bytes, or `None` where there is no cgroup or no
    /// limit set on it.
    pub cgroup: Option<u64>,
    /// What the machine has in bytes, or `None` where it could not be asked.
    pub host: Option<u64>,
}

/// The reading, taken once and kept.
///
/// None of it changes while the process runs, and the files behind it are on a
/// virtual filesystem that is cheap to read but not free, so everything that
/// wants the numbers asks here rather than going back to `/sys` for each one.
#[must_use]
pub fn cap() -> Cap {
    static ONCE: std::sync::OnceLock<Cap> = std::sync::OnceLock::new();
    *ONCE.get_or_init(Cap::read)
}

impl Cap {
    /// Ask the running system.
    #[must_use]
    pub fn read() -> Cap {
        Cap {
            cgroup: cgroup_limit(Path::new(CGROUP_ROOT), Path::new("/proc/self/cgroup")),
            host: host_memory(),
        }
    }

    /// The smaller of the two, which is what this process actually gets.
    ///
    /// A cgroup limit larger than the machine is not a limit, so the host
    /// number wins there, and a machine with no cgroup limit is capped only by
    /// itself.
    #[must_use]
    pub fn limit(&self) -> Option<u64> {
        match (self.cgroup, self.host) {
            (Some(c), Some(h)) => Some(c.min(h)),
            (Some(c), None) => Some(c),
            (None, h) => h,
        }
    }

    /// A quarter of the limit, which is what a pool should be sized from.
    ///
    /// Zero when there is no limit to take a quarter of, which is the same
    /// thing `maxmemory` means by zero and is why it is zero and not `None`.
    #[must_use]
    pub fn budget(&self) -> u64 {
        self.limit().map_or(0, |b| b / 4)
    }
}

/// Read a `memory.max` or `memory.limit_in_bytes` file.
///
/// `max` is cgroup v2's word for no limit. A number near the top of the range
/// is cgroup v1's. Anything else is bytes.
fn parse_limit(text: &str) -> Option<u64> {
    let text = text.trim();
    if text == "max" {
        return None;
    }
    let n: u64 = text.parse().ok()?;
    if n >= V1_UNLIMITED { None } else { Some(n) }
}

/// Pull the cgroup v2 path out of `/proc/self/cgroup`.
///
/// The v2 line is the one with an empty hierarchy id and an empty controller
/// list, which is written `0::`. What follows is a path relative to the mount
/// point, and inside a cgroup namespace it is just `/`.
fn parse_self_cgroup(text: &str) -> Option<&str> {
    text.lines()
        .find_map(|line| line.strip_prefix("0::"))
        .map(str::trim)
}

/// The tightest limit on this cgroup or any of its ancestors.
///
/// Split out from [`Cap::read`] so the tests can point it at a directory tree
/// they built, which is the only way to check the walk on a machine that has no
/// cgroups.
fn cgroup_limit(root: &Path, self_cgroup: &Path) -> Option<u64> {
    // Only if v2 said nothing at all, because a machine running both has the v2
    // answer as the real one.
    v2_limit(root, self_cgroup).or_else(|| {
        let text = std::fs::read_to_string(root.join("memory/memory.limit_in_bytes")).ok()?;
        parse_limit(&text)
    })
}

/// The v2 half of [`cgroup_limit`]: walk from this process's directory up to
/// the mount point and keep the smallest limit written down anywhere on the way.
fn v2_limit(root: &Path, self_cgroup: &Path) -> Option<u64> {
    let rel = std::fs::read_to_string(self_cgroup).ok()?;
    let rel = parse_self_cgroup(&rel)?.trim_start_matches('/');

    let mut dir: PathBuf = root.join(rel);
    let mut best: Option<u64> = None;
    loop {
        if let Ok(text) = std::fs::read_to_string(dir.join("memory.max"))
            && let Some(n) = parse_limit(&text)
        {
            best = Some(best.map_or(n, |b: u64| b.min(n)));
        }
        if dir == root {
            break;
        }
        // Stop at the mount point rather than walking off the top of it, and
        // stop anyway if the path was not under the root to begin with.
        match dir.parent() {
            Some(p) if p.starts_with(root) || p == root => dir = p.to_path_buf(),
            _ => break,
        }
    }
    best
}

/// What the machine has.
#[cfg(target_os = "linux")]
fn host_memory() -> Option<u64> {
    // SAFETY: two reads of process independent configuration, neither of which
    // takes a pointer or leaves anything behind.
    let (pages, size) = unsafe {
        (
            libc::sysconf(libc::_SC_PHYS_PAGES),
            libc::sysconf(libc::_SC_PAGESIZE),
        )
    };
    if pages > 0 && size > 0 {
        Some(pages as u64 * size as u64)
    } else {
        None
    }
}

/// What the machine has.
#[cfg(target_vendor = "apple")]
fn host_memory() -> Option<u64> {
    let mut out: u64 = 0;
    let mut len = size_of::<u64>();
    // SAFETY: the name is a C string literal, the buffer is one `u64` and `len`
    // says so, and the two null pointers are the documented way to say there is
    // no new value to set.
    let rc = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&raw mut out).cast(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 && out > 0 { Some(out) } else { None }
}

/// What the machine has, on a system with no way to ask that is worth linking.
#[cfg(not(any(target_os = "linux", target_vendor = "apple")))]
fn host_memory() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cgroup tree under a temporary directory, so the walk can be tested on
    /// a machine that has never heard of cgroups.
    struct Tree(PathBuf);

    impl Tree {
        fn new(name: &str) -> Tree {
            let dir = std::env::temp_dir().join(format!("yo-cap-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("could not make a temporary directory");
            Tree(dir)
        }

        fn write(&self, rel: &str, text: &str) -> PathBuf {
            let at = self.0.join(rel);
            std::fs::create_dir_all(at.parent().expect("a file has a parent"))
                .expect("could not make a directory");
            std::fs::write(&at, text).expect("could not write");
            at
        }

        fn root(&self) -> PathBuf {
            self.0.join("cgroup")
        }
    }

    impl Drop for Tree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn max_means_there_is_no_limit() {
        assert_eq!(parse_limit("max\n"), None);
        assert_eq!(parse_limit("  max  "), None);
    }

    #[test]
    fn a_number_near_the_top_is_cgroup_v1_saying_no_limit() {
        assert_eq!(parse_limit("9223372036854771712"), None);
        assert_eq!(parse_limit(&u64::MAX.to_string()), None);
    }

    #[test]
    fn a_real_number_is_bytes() {
        assert_eq!(parse_limit("2147483648\n"), Some(2 * 1024 * 1024 * 1024));
    }

    #[test]
    fn nonsense_is_no_limit_rather_than_a_panic() {
        assert_eq!(parse_limit(""), None);
        assert_eq!(parse_limit("-1"), None);
        assert_eq!(parse_limit("2gb"), None);
    }

    #[test]
    fn the_v2_line_is_the_one_with_no_controllers() {
        let text = "12:pids:/user.slice\n1:name=systemd:/user.slice\n0::/user.slice/app.scope\n";
        assert_eq!(parse_self_cgroup(text), Some("/user.slice/app.scope"));
    }

    #[test]
    fn inside_a_cgroup_namespace_the_path_is_just_the_root() {
        assert_eq!(parse_self_cgroup("0::/\n"), Some("/"));
    }

    #[test]
    fn a_v1_only_machine_has_no_v2_line() {
        assert_eq!(parse_self_cgroup("6:memory:/\n3:cpu:/\n"), None);
    }

    #[test]
    fn the_limit_is_read_from_the_leaf() {
        let t = Tree::new("leaf");
        let me = t.write("proc", "0::/a/b\n");
        t.write("cgroup/a/b/memory.max", "1073741824\n");
        assert_eq!(cgroup_limit(&t.root(), &me), Some(1024 * 1024 * 1024));
    }

    #[test]
    fn an_ancestor_with_a_tighter_limit_wins() {
        // This is the case worth having a test for. A pod gets a generous
        // limit and the namespace it lives in gets a mean one, and the process
        // is held to the mean one even though nothing in its own directory
        // says so.
        let t = Tree::new("ancestor");
        let me = t.write("proc", "0::/pods/one\n");
        t.write("cgroup/memory.max", "max\n");
        t.write("cgroup/pods/memory.max", "536870912\n");
        t.write("cgroup/pods/one/memory.max", "4294967296\n");
        assert_eq!(cgroup_limit(&t.root(), &me), Some(512 * 1024 * 1024));
    }

    #[test]
    fn a_tree_that_says_max_all_the_way_up_has_no_limit() {
        let t = Tree::new("nolimit");
        let me = t.write("proc", "0::/a\n");
        t.write("cgroup/memory.max", "max\n");
        t.write("cgroup/a/memory.max", "max\n");
        assert_eq!(cgroup_limit(&t.root(), &me), None);
    }

    #[test]
    fn cgroup_v1_is_read_when_v2_has_nothing_to_say() {
        let t = Tree::new("v1");
        let me = t.write("proc", "6:memory:/\n");
        t.write("cgroup/memory/memory.limit_in_bytes", "268435456\n");
        assert_eq!(cgroup_limit(&t.root(), &me), Some(256 * 1024 * 1024));
    }

    #[test]
    fn a_machine_with_no_cgroups_at_all_reports_none() {
        let t = Tree::new("nothing");
        assert_eq!(
            cgroup_limit(&t.root(), &t.0.join("not-here")),
            None,
            "a missing file is a machine without cgroups, not an error"
        );
    }

    #[test]
    fn the_tighter_of_the_two_is_the_one_that_counts() {
        let big = 64 * 1024 * 1024 * 1024;
        let small = 2 * 1024 * 1024 * 1024;
        assert_eq!(
            Cap {
                cgroup: Some(small),
                host: Some(big)
            }
            .limit(),
            Some(small)
        );
        // A container told it may have more than the machine holds has not
        // been given more than the machine holds.
        assert_eq!(
            Cap {
                cgroup: Some(big),
                host: Some(small)
            }
            .limit(),
            Some(small)
        );
    }

    #[test]
    fn the_budget_is_a_quarter_and_zero_when_there_is_nothing_to_take_a_quarter_of() {
        let cap = Cap {
            cgroup: Some(4 * 1024 * 1024 * 1024),
            host: None,
        };
        assert_eq!(cap.budget(), 1024 * 1024 * 1024);
        assert_eq!(Cap::default().budget(), 0);
    }

    #[test]
    fn asking_the_real_machine_answers_something_sensible() {
        let cap = Cap::read();
        // Not an assertion about this machine's size, only that a number that
        // came back is a number a machine could have.
        if let Some(h) = cap.host {
            assert!(h >= 64 * 1024 * 1024, "a host with {h} bytes is not real");
        }
        if let Some(l) = cap.limit() {
            assert_eq!(cap.budget(), l / 4);
        }
    }
}
