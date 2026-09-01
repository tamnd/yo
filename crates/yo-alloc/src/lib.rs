//! A global allocator that turns an accidental heap allocation on a shard path
//! into a crash.
//!
//! Y7 says there is no global allocator call on a command path. That is easy to
//! write down and impossible to keep by review once more than one person is in
//! the codebase, because the allocating constructs in Rust are the comfortable
//! ones: `format!`, `to_vec`, `collect`, `Box::new`, a `Vec` that grows, a
//! `String` built to make an error message. Each costs tens of nanoseconds
//! against a 150 ns budget, and none of them looks wrong in a diff.
//!
//! So the rule is enforced instead of reviewed. A shard thread marks itself
//! [`enter_no_alloc`] before the command loop, and from that point any
//! allocation aborts the process with a message naming the size. Setup, arena
//! growth and anything else that legitimately needs the heap wraps itself in
//! [`allow`], which is a visible, greppable, deliberate act.
//!
//! # Cost when it is off
//!
//! The check is one thread local load and a branch, on a path that already
//! calls into the system allocator. It is not measurable next to `malloc`.
//! Non shard threads never set the flag and pay the same single branch.
//!
//! The claims are cheaper still. [`allow`] and the three named forms of it come
//! down to a relaxed load of a static, false in any process that has never armed
//! a thread, which is every shipped binary. That is what lets a claim sit inside
//! a loop where the growth actually happens rather than being hoisted to the top
//! of a function it does not describe.
//!
//! # Three modes, and why an abort is not the only one
//!
//! An abort tells you about one violation per run, which is the wrong tool for
//! finding out how many there are. Nothing in this project had been checked
//! against Y7 since the rule was written down, so the first question is not
//! "stop on the first one" but "what is the list".
//!
//! [`Mode::Report`] answers that. It suspends the check, captures a backtrace,
//! prints each distinct site once and counts the repeats, and lets the
//! allocation through. It allocates while it does this, on purpose and with the
//! check turned off around it, because a debugging mode that cannot use the heap
//! cannot tell you where you are.
//!
//! [`Mode::Abort`] is the rule as written, for a build that is expected to be
//! clean. [`Mode::Off`] is the default, so installing the allocator does not
//! change what a shipped binary does until somebody asks for it.
//!
//! Off costs nothing rather than costing a branch, because nothing arms the
//! thread: [`guard`] is where the mode is read, and when it is off the thread
//! flag is never set and the allocator's check is the same false it would be on
//! any other thread.
//!
//! # Getting the list
//!
//! `yodb` installs this and `pump` wraps its dispatch in [`guard`], so
//! `YO_ALLOC=report yodb serve` answers the question, and `cargo xtask alloc`
//! is that with a server, a workload and a parser around it. It builds a debug
//! `yodb`, drives it with about nine thousand commands covering every type and
//! prints one line per distinct site. Debug on purpose: release inlines the
//! interesting frames into each other and the report comes back naming
//! `serve_command` for everything.
//!
//! Run it before touching anything here, and believe it over anything written
//! down. Twice during this work the list contradicted what was already recorded
//! about what was left, and both times the list was right. A report mode exists
//! so that this is measured rather than argued about, and that only works if it
//! is the list somebody actually looks at.
//!
//! It is also the gate. An empty list exits 0 and anything else exits 1, and
//! `ci.yml` runs it on every push, so a new allocation on a command path fails
//! the pull request that added it rather than being found a release later.
//!
//! # What it found, and the four piles it sorted into
//!
//! The first arming reported 31 distinct sites, and they were never one problem.
//! The list is empty now, and it got there four different ways.
//!
//! Most of them were the first touch of a key. Creating a set, hash, list or
//! zset allocates the body, and the slab that holds bodies of that type doubles
//! when it fills. That is real allocation on a command path and it is also the
//! only sensible place for it, so those sites wanted a claim written down and a
//! [`first_touch`] around them rather than a fix. They have one now.
//!
//! Then the ones worth having, which were per command and in steady state, which
//! is exactly what Y7 is about. A `to_vec` of the value in `APPEND`, `SETRANGE`,
//! `EXPIRE`, `GETEX`, `INCRBYFLOAT` and the string arm of `COPY`. A `Vec` built
//! per call to hold the operands of a set operation. A `Vec` of indices in
//! `LREM`. An owned key out of `RANDOMKEY`. The record copy in `RENAME`. The
//! engine's own list of free decoder slots. The hash table a `SUNION` walked
//! everything into, which was the largest of them and is now a table the
//! database keeps. And the old value out of `SET ... GET`, `GETSET` and
//! `GETDEL`, which looked like a signature question and was not: the owning
//! method stayed for the embedded caller and the wire took a `_with` form that
//! hands the value over where it lies, because the wire writes it into the reply
//! and never looks at it again. Every one of those is gone.
//!
//! Third, memory that is proportional to what is stored rather than to what is
//! served. An intset run gets longer as members go into it and there is no
//! arrangement of that code which stores ten thousand integers in the room it
//! had for eight. That is [`for_the_data`], and the test of the claim is that a
//! workload which stops adding data stops allocating.
//!
//! Fourth, scratch buffers the database keeps and refills: the `ZRANDMEMBER`
//! permutation and the set algebra tables. Those allocate when a call is bigger
//! than every call before it and never otherwise. That is [`high_water`], and
//! the test of the claim is a unit test making the same call twice and counting
//! zero the second time. Every site wrapped in it has one.
//!
//! So the order is: report first, sort the list into the piles, fix the ones
//! that are per command and put the right claim on the ones that are not. The
//! three claims are all [`allow`] underneath and differ only in what they say,
//! which is the entire point of having three of them: `git grep first_touch` is
//! a list of the places a key is created, and it stays one.
//!
//! # Using it
//!
//! ```no_run
//! # use yo_alloc::YoAlloc;
//! #[global_allocator]
//! static ALLOC: YoAlloc = YoAlloc::new();
//! ```
//!
//! The engine installs this in `yodb`. A library consumer of `yodb` does not get
//! it, because choosing a global allocator is the application's call and never a
//! library's.

#![deny(missing_docs)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

thread_local! {
    /// Zero means allocation is allowed. Anything above zero forbids it.
    ///
    /// A counter rather than a flag so that [`allow`] nests correctly, which
    /// matters because arena growth can be reached from more than one depth.
    static FORBID: Cell<u32> = const { Cell::new(0) };
}

/// What happens when a marked thread allocates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    /// Nothing. [`guard`] does not mark the thread and the check never fires.
    ///
    /// The default, so that installing the allocator in a binary is not on its
    /// own a change to what that binary does.
    #[default]
    Off,
    /// Print each distinct site once, count the rest, and carry on.
    Report,
    /// Abort the process on the first one. Y7 as written.
    Abort,
}

/// The mode, as a number, because a static has to be something an atomic holds.
static MODE: AtomicU8 = AtomicU8::new(0);

/// Whether any thread in this process has ever been marked.
///
/// Not the mode. [`enter_no_alloc`] is public and a test calls it without going
/// near [`set_mode`], so the mode being off does not mean no thread is marked.
/// This is the question [`allow`] actually needs answered, and it is one
/// relaxed load of a static rather than a thread local.
static ARMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// How many violations have been seen in [`Mode::Report`].
static SEEN_TOTAL: AtomicU64 = AtomicU64::new(0);

/// One entry per distinct backtrace, so a site in a loop prints once.
static SITES: Mutex<BTreeMap<u64, Site>> = Mutex::new(BTreeMap::new());

/// What was seen at one site.
#[derive(Debug)]
struct Site {
    /// How many allocations landed here.
    count: u64,
    /// The largest one, which is usually the one worth looking at first.
    largest: usize,
}

/// What a marked thread does when it allocates.
#[must_use]
pub fn mode() -> Mode {
    match MODE.load(Ordering::Relaxed) {
        1 => Mode::Report,
        2 => Mode::Abort,
        _ => Mode::Off,
    }
}

/// Set what a marked thread does when it allocates.
///
/// Meant to be called once, from `main`, before any thread is marked. It is an
/// atomic store rather than a `OnceLock` so that a test can set it and put it
/// back, which is the only reason it is allowed to happen twice.
pub fn set_mode(m: Mode) {
    MODE.store(
        match m {
            Mode::Off => 0,
            Mode::Report => 1,
            Mode::Abort => 2,
        },
        Ordering::Relaxed,
    );
}

/// Set the mode from `YO_ALLOC`, which is `off`, `report` or `abort`.
///
/// Answers `None` when the variable is set to something else, and the caller is
/// expected to refuse to start rather than carry on. A typo that silently turns
/// the check off is precisely the failure this module exists to prevent, so it
/// is not treated as an unset variable.
///
/// An unset variable is [`Mode::Off`] and is not an error.
#[must_use]
pub fn set_mode_from_env() -> Option<Mode> {
    let m = parse_mode(std::env::var("YO_ALLOC").ok().as_deref())?;
    set_mode(m);
    Some(m)
}

/// The reading half of [`set_mode_from_env`], split out so it can be tested
/// without a process wide environment change.
fn parse_mode(v: Option<&str>) -> Option<Mode> {
    match v {
        None | Some("" | "off") => Some(Mode::Off),
        Some("report") => Some(Mode::Report),
        Some("abort") => Some(Mode::Abort),
        Some(_) => None,
    }
}

/// Mark this thread for the length of the returned value, if the mode says so.
///
/// This is what a command loop wraps its dispatch in. It is a guard rather than
/// a pair of calls because a panic in the middle of a batch would otherwise
/// leave the thread marked for the rest of the process, and a thread that can
/// never allocate again is a worse failure than the one being looked for.
///
/// A no-op under [`Mode::Off`], down to not touching the thread local, so the
/// cost of having this in the loop when nobody asked for it is one relaxed load
/// per batch.
#[must_use = "the mark lasts as long as the guard, so dropping it here does nothing"]
pub fn guard() -> Guard {
    let on = mode() != Mode::Off;
    if on {
        enter_no_alloc();
    }
    Guard(on)
}

/// The mark from [`guard`], undone when it goes out of scope.
#[derive(Debug)]
pub struct Guard(bool);

impl Drop for Guard {
    #[inline]
    fn drop(&mut self) {
        if self.0 {
            exit_no_alloc();
        }
    }
}

/// Everything [`Mode::Report`] collected, as `(sites, allocations)`.
///
/// Both are zero in the other two modes, which is what makes this worth calling
/// unconditionally at shutdown.
#[must_use]
pub fn seen() -> (usize, u64) {
    let n = SITES.lock().map_or(0, |s| s.len());
    (n, SEEN_TOTAL.load(Ordering::Relaxed))
}

/// Mark this thread as a shard thread: from here on, allocating aborts.
///
/// Called once by each shard as it enters its loop. There is no matching exit
/// in normal operation because a shard thread never stops being one. A loop that
/// is not a shard's, and so does want the mark to end, wants [`guard`].
#[inline]
pub fn enter_no_alloc() {
    // Latched here so that [`allow`] can be free in a process where nothing is
    // ever marked, which is every shipped binary and every benchmark. It is
    // never cleared: a process that has armed one thread once pays the full
    // path for the rest of its life, which is the right way round because that
    // process is the one being measured for violations rather than for speed.
    ARMED.store(true, Ordering::Relaxed);
    FORBID.with(|f| f.set(f.get().saturating_add(1)));
}

/// Undo one [`enter_no_alloc`].
///
/// Exists for tests and for the embedded single thread mode (`15` section 7),
/// where the caller's thread is temporarily the shard and then goes back to
/// being the caller's thread.
#[inline]
pub fn exit_no_alloc() {
    FORBID.with(|f| f.set(f.get().saturating_sub(1)));
}

/// Whether allocation is currently forbidden on this thread.
#[inline]
pub fn is_forbidden() -> bool {
    FORBID.with(|f| f.get()) > 0
}

/// Run `f` with allocation permitted, then restore the previous state.
///
/// Every call to this is a claim that the work inside is off the command path.
/// Wrapping a command path in it to silence an abort is the one way to misuse
/// this module, so the calls are meant to be few and easy to find.
///
/// # Cost
///
/// One relaxed load of a static when no thread in the process has ever been
/// marked, and a thread local read and two writes when one has.
///
/// The load is in front of the rest because these calls are not all at the top
/// of a function any more. The set algebra makes its claim around the insert
/// that grows its table, which is once per member, because the alternative is
/// one guard around the whole walk and that hides whatever the caller's closure
/// does. A thread local read and two writes per member of a union is not
/// obviously free, and a laptop with other work on it could not resolve the
/// difference either way: two runs of `setops_small` with the same code in place
/// disagreed by more than the effect being looked for. So this stopped being an
/// argument about how cheap a thread local is and became a load of a static that
/// is false in every shipped binary and in every benchmark.
#[inline]
pub fn allow<T>(f: impl FnOnce() -> T) -> T {
    if !ARMED.load(Ordering::Relaxed) {
        return f();
    }
    let saved = FORBID.with(|c| c.replace(0));
    let guard = Restore(saved);
    let out = f();
    drop(guard);
    out
}

/// Run `f`, which is a key coming into existence for the first time.
///
/// [`allow`] with a name on it, and the name is the claim. Y7 says no allocation
/// on a command path, and the first `SADD` to a key that was not there has to
/// make a set somewhere. There is no arrangement of this code that avoids it and
/// no reason to want one: it happens once per key rather than once per command,
/// and a workload that creates a key on every command is one where the
/// allocation is the smallest thing it is paying for.
///
/// So the rule this module enforces is the one that is actually true. Nothing on
/// a command path allocates except a key being created, and every place that
/// does is this call, which makes the list of them a grep rather than an
/// argument.
///
/// This is the one way to misuse the module. Wrapping steady state work in it to
/// stop an abort would leave the check passing and the rule broken, which is
/// worse than not having the check.
#[inline]
pub fn first_touch<T>(f: impl FnOnce() -> T) -> T {
    allow(f)
}

/// Run `f`, which is a collection getting bigger because more was put in it.
///
/// [`allow`] with a name on it, and the name is the claim. The rule is that
/// nothing is proportional to the number of commands served, not that nothing
/// ever calls the allocator: an intset that has taken its ten thousandth member
/// has to have grown nine times along the way, and there is no arrangement of
/// that code which stores ten thousand integers in the room it had for eight.
///
/// The test is whether a workload that stops adding data stops allocating.
/// [`first_touch`] is the same argument for a key that was not there at all,
/// and this is the argument for the key that was.
#[inline]
pub fn for_the_data<T>(f: impl FnOnce() -> T) -> T {
    allow(f)
}

/// Run `f`, which is a buffer the database keeps reaching a size it has never
/// been asked for before.
///
/// [`allow`] with a name on it, and the name is the claim. A `ZRANDMEMBER` needs
/// somewhere to shuffle and a `SUNION` needs somewhere to check for duplicates,
/// and both of those are cleared and refilled rather than built and dropped. So
/// the allocation is not per command, it is per high water mark: a database that
/// has answered a union over a million members holds a million member table, and
/// every union after it that is no larger pays the allocator nothing.
///
/// The test is whether the same call made twice allocates the second time. That
/// is a unit test rather than an argument, and every site wrapped in this has
/// one. [`for_the_data`] is the neighbouring claim, and the difference is that
/// this memory is not holding anything between commands: it is scratch that has
/// grown to fit the largest question asked so far.
#[inline]
pub fn high_water<T>(f: impl FnOnce() -> T) -> T {
    allow(f)
}

struct Restore(u32);

impl Drop for Restore {
    #[inline]
    fn drop(&mut self) {
        FORBID.with(|c| c.set(self.0));
    }
}

/// The allocator. Delegates to the system allocator and checks the flag first.
#[derive(Debug, Default, Clone, Copy)]
pub struct YoAlloc;

impl YoAlloc {
    /// A new allocator.
    pub const fn new() -> YoAlloc {
        YoAlloc
    }
}

/// A marked thread allocated. Report it or stop the process.
///
/// [`Mode::Off`] lands here too and aborts, because a thread is only ever marked
/// because something asked for it. [`guard`] does not mark under `Off`, so the
/// only way to reach this with the mode off is a direct [`enter_no_alloc`], and
/// that call means what it has always meant.
#[cold]
#[inline(never)]
fn violation(layout: Layout, what: &str) {
    if mode() == Mode::Report {
        report(layout, what);
        return;
    }
    abort_now(layout, what)
}

/// Note the site and let the allocation through.
///
/// Runs with the check suspended, because everything here allocates: capturing a
/// backtrace, rendering it, and keeping it in a map. That is the deal in this
/// mode. Without the suspension the first violation would recurse until the
/// stack ran out, which is a worse way to learn about a `format!` in a hot loop
/// than being told where it is.
fn report(layout: Layout, what: &str) {
    SEEN_TOTAL.fetch_add(1, Ordering::Relaxed);
    allow(|| {
        let trace = std::backtrace::Backtrace::force_capture().to_string();
        // The whole rendered trace is the identity of the site. Two allocations
        // from the same line reached by different callers are different entries,
        // which is what you want when the line is inside a helper.
        let key = fnv1a(trace.as_bytes());
        let Ok(mut sites) = SITES.lock() else {
            return;
        };
        let size = layout.size();
        match sites.entry(key) {
            std::collections::btree_map::Entry::Occupied(mut e) => {
                let s = e.get_mut();
                s.count += 1;
                s.largest = s.largest.max(size);
            }
            std::collections::btree_map::Entry::Vacant(e) => {
                e.insert(Site {
                    count: 1,
                    largest: size,
                });
                // Printed once, on the way in, so that a run that ends in a
                // crash still leaves the list behind.
                eprintln!("yo: allocation on a marked thread: {what} of {size} bytes\n{trace}");
            }
        }
    });
}

/// 64 bit FNV-1a. Enough to tell two backtraces apart and short enough to write
/// out rather than take a dependency for.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cold]
#[inline(never)]
fn abort_now(layout: Layout, what: &str) -> ! {
    // No formatting machinery here on purpose. `format!` allocates, and this is
    // the one place in the process where allocating is known to be unavailable.
    // Two `write_str` calls and an integer written by hand cost nothing and
    // cannot recurse.
    use std::io::Write as _;
    let mut buf = [0u8; 32];
    let n = write_usize(&mut buf, layout.size());
    let mut err = std::io::stderr().lock();
    let _ = err.write_all(b"yo: allocation on a shard thread: ");
    let _ = err.write_all(what.as_bytes());
    let _ = err.write_all(b" of ");
    let _ = err.write_all(&buf[..n]);
    let _ = err.write_all(
        b" bytes.\nThis is Y7: no global allocator call on a command path.\n\
          Move the allocation to setup, or wrap it in yo_alloc::allow if it is\n\
          genuinely off the command path.\n",
    );
    let _ = err.flush();
    std::process::abort()
}

fn write_usize(buf: &mut [u8; 32], mut v: usize) -> usize {
    if v == 0 {
        buf[0] = b'0';
        return 1;
    }
    let mut tmp = [0u8; 32];
    let mut n = 0;
    while v > 0 {
        tmp[n] = b'0' + (v % 10) as u8;
        v /= 10;
        n += 1;
    }
    for i in 0..n {
        buf[i] = tmp[n - 1 - i];
    }
    n
}

// SAFETY: every method forwards to `System`, which upholds the `GlobalAlloc`
// contract. The added check only ever diverges before calling through, so no
// pointer is created, invalidated or leaked by it.
unsafe impl GlobalAlloc for YoAlloc {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if is_forbidden() {
            violation(layout, "alloc");
        }
        // SAFETY: forwarding the caller's own valid layout.
        unsafe { System.alloc(layout) }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if is_forbidden() {
            violation(layout, "alloc_zeroed");
        }
        // SAFETY: forwarding the caller's own valid layout.
        unsafe { System.alloc_zeroed(layout) }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if is_forbidden() {
            violation(layout, "realloc");
        }
        // SAFETY: forwarding the caller's own valid pointer and layout.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // Deallocation is deliberately not checked. A value allocated during
        // setup and dropped on the shard thread is normal and harmless, and
        // aborting on it would make the rule unusable. What costs time is the
        // allocation, and that is what is caught.
        //
        // SAFETY: forwarding the caller's own valid pointer and layout.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_permitted() {
        assert!(!is_forbidden());
    }

    #[test]
    fn enter_and_exit_are_balanced() {
        assert!(!is_forbidden());
        enter_no_alloc();
        assert!(is_forbidden());
        enter_no_alloc();
        assert!(is_forbidden());
        exit_no_alloc();
        assert!(is_forbidden(), "one exit must not undo two enters");
        exit_no_alloc();
        assert!(!is_forbidden());
    }

    #[test]
    fn allow_permits_and_restores() {
        enter_no_alloc();
        assert!(is_forbidden());
        let v = allow(|| {
            assert!(!is_forbidden());
            vec![1u8, 2, 3]
        });
        assert_eq!(v.len(), 3);
        assert!(is_forbidden(), "allow must restore the previous state");
        exit_no_alloc();
    }

    #[test]
    fn allow_nests() {
        enter_no_alloc();
        allow(|| {
            allow(|| assert!(!is_forbidden()));
            assert!(!is_forbidden());
        });
        assert!(is_forbidden());
        exit_no_alloc();
    }

    #[test]
    fn allow_restores_when_the_body_panics() {
        enter_no_alloc();
        let r = std::panic::catch_unwind(|| {
            allow(|| panic!("boom"));
        });
        assert!(r.is_err());
        assert!(
            is_forbidden(),
            "a panic inside allow must not leave the thread permitted"
        );
        exit_no_alloc();
    }

    /// The flag is per thread. A shard marking itself must not affect the
    /// accept loop or a test harness thread.
    #[test]
    fn the_flag_does_not_cross_threads() {
        enter_no_alloc();
        let other = std::thread::spawn(is_forbidden).join().unwrap();
        assert!(!other, "another thread saw this thread's flag");
        exit_no_alloc();
    }

    /// The mode is one static for the whole process, so the tests that move it
    /// take turns. Without this they would race with each other rather than with
    /// anything real.
    static MODE_TESTS: Mutex<()> = Mutex::new(());

    fn one_at_a_time() -> std::sync::MutexGuard<'static, ()> {
        MODE_TESTS.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn the_mode_starts_off_and_survives_a_round_trip() {
        let _turn = one_at_a_time();
        assert_eq!(Mode::default(), Mode::Off, "off is the default");
        for m in [Mode::Report, Mode::Abort, Mode::Off] {
            set_mode(m);
            assert_eq!(mode(), m);
        }
    }

    #[test]
    fn the_env_variable_reads_three_words_and_refuses_the_rest() {
        assert_eq!(parse_mode(None), Some(Mode::Off));
        assert_eq!(parse_mode(Some("")), Some(Mode::Off));
        assert_eq!(parse_mode(Some("off")), Some(Mode::Off));
        assert_eq!(parse_mode(Some("report")), Some(Mode::Report));
        assert_eq!(parse_mode(Some("abort")), Some(Mode::Abort));
        // A typo has to be an error rather than a quiet off, because a quiet off
        // is the check not running while somebody believes it is.
        assert_eq!(parse_mode(Some("abrot")), None);
        assert_eq!(parse_mode(Some("Report")), None);
        assert_eq!(parse_mode(Some("1")), None);
    }

    /// Both halves of the guard, in a thread of its own so that setting the mode
    /// cannot be seen by another test's assertion about the flag.
    #[test]
    fn the_guard_marks_only_when_the_mode_asks() {
        let _turn = one_at_a_time();
        std::thread::spawn(|| {
            set_mode(Mode::Off);
            {
                let _g = guard();
                assert!(!is_forbidden(), "off must not mark the thread at all");
            }
            for m in [Mode::Report, Mode::Abort] {
                set_mode(m);
                {
                    let _g = guard();
                    assert!(is_forbidden(), "{m:?} must mark it");
                }
                assert!(!is_forbidden(), "and the guard must undo it");
            }
            set_mode(Mode::Off);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn the_guard_unmarks_when_the_body_panics() {
        let _turn = one_at_a_time();
        std::thread::spawn(|| {
            set_mode(Mode::Report);
            let r = std::panic::catch_unwind(|| {
                let _g = guard();
                assert!(is_forbidden());
                panic!("boom");
            });
            assert!(r.is_err());
            assert!(
                !is_forbidden(),
                "a panic inside the guard must not leave the thread marked forever"
            );
            set_mode(Mode::Off);
        })
        .join()
        .unwrap();
    }

    /// Report mode has to survive the thing it is reporting on, because the
    /// reporting itself allocates on a thread where allocating is what set it
    /// off. It is driven directly here rather than through a real allocation,
    /// since this crate's own test binary deliberately does not install the
    /// allocator: several of the tests above spawn threads and panic while the
    /// flag is up, which is exactly what an installed one would abort on.
    #[test]
    fn report_mode_records_instead_of_aborting() {
        let _turn = one_at_a_time();
        std::thread::spawn(|| {
            let (sites_before, total_before) = seen();
            set_mode(Mode::Report);
            {
                let _g = guard();
                assert!(is_forbidden());
                // Three from one line, so the total moves by three and the site
                // is only recorded, and printed, once.
                for size in [8usize, 64, 4096] {
                    let layout = Layout::from_size_align(size, 8).unwrap();
                    violation(layout, "alloc");
                }
                assert!(is_forbidden(), "reporting must put the mark back");
            }
            set_mode(Mode::Off);

            let (sites, total) = seen();
            assert_eq!(total - total_before, 3, "every violation is counted");
            assert_eq!(sites - sites_before, 1, "one line is one site");
        })
        .join()
        .unwrap();
    }

    #[test]
    fn distinct_traces_hash_apart() {
        assert_ne!(fnv1a(b"one"), fnv1a(b"two"));
        assert_eq!(fnv1a(b"same"), fnv1a(b"same"));
    }

    #[test]
    fn integers_render_without_allocating() {
        let mut buf = [0u8; 32];
        for (v, want) in [
            (0usize, "0"),
            (7, "7"),
            (1024, "1024"),
            (2097152, "2097152"),
        ] {
            let n = write_usize(&mut buf, v);
            assert_eq!(std::str::from_utf8(&buf[..n]).unwrap(), want);
        }
    }
}
