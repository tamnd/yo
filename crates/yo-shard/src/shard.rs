//! The shard runtime: one thread per core, one owner per piece of state.
//!
//! `05` section 1 in code. A shard is a thread, pinned to a core, that owns its
//! data outright. Nothing else can reach that data, which is enforced by the
//! type system rather than by convention: [`ShardLocal`] is neither `Send` nor
//! `Sync`, so a value inside one cannot leave the thread that built it and
//! cannot be borrowed from another thread. There is no lock around it because
//! there is nothing to lock against.
//!
//! Work reaches a shard as a job on an SPSC lane. Each submitter owns its own
//! lane into every shard, so a shard's inbox is N single producer queues rather
//! than one multi producer queue, and no compare and swap is involved in
//! getting work across.
//!
//! A shard spins for a while when it runs dry and then parks. Spinning first is
//! what keeps a busy system off the futex path, and parking eventually is what
//! keeps an idle system off the power meter.

use crate::epoch::Epochs;
use crate::spsc::{self, Receiver, Sender};
use core::marker::PhantomData;
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{JoinHandle, Thread};
use yo_common::ShardId;

/// How many times a shard checks its lanes before it gives up and parks.
///
/// Tuned against the cost of a futex round trip, which is a few microseconds
/// when it goes badly. Spinning this long is cheaper than parking once if work
/// is about to arrive, and this is the last spin before a park, not a spin
/// under a lock, so it never blocks anyone else.
const SPIN_LIMIT: u32 = 512;

/// Default depth of one lane.
pub const LANE_CAPACITY: usize = 1024;

/// State a shard owns and nobody else can touch.
///
/// The entire concurrency argument for this engine is that the compiler will
/// not let you get this wrong. `ShardLocal<T>` is `!Send` and `!Sync`, so it
/// cannot be moved to another thread and cannot be shared with one. The only
/// way to reach a shard's map is to send that shard a job.
pub struct ShardLocal<T> {
    value: T,
    /// A raw pointer field is the standard way to opt out of both auto traits.
    /// It holds nothing and is never read.
    _pin: PhantomData<*mut ()>,
}

impl<T> ShardLocal<T> {
    /// Wrap a value as shard local. Call this on the shard thread.
    pub fn new(value: T) -> ShardLocal<T> {
        ShardLocal {
            value,
            _pin: PhantomData,
        }
    }

    /// Take the value back out. Only reachable on the owning thread, because
    /// that is the only place the wrapper exists.
    pub fn into_inner(self) -> T {
        self.value
    }
}

impl<T> core::ops::Deref for ShardLocal<T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T> core::ops::DerefMut for ShardLocal<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

/// What a job sees when it runs.
pub struct ShardCtx<T> {
    /// Which shard this is.
    pub id: ShardId,
    /// The state this shard owns.
    pub state: ShardLocal<T>,
    epochs: Arc<Epochs>,
}

impl<T> ShardCtx<T> {
    /// Every shard's epoch counters, for retirement decisions.
    pub fn epochs(&self) -> &Epochs {
        &self.epochs
    }

    /// This shard's index as a `usize`.
    #[inline]
    pub fn index(&self) -> usize {
        self.id.0 as usize
    }
}

type Job<T> = Box<dyn FnOnce(&mut ShardCtx<T>) + Send + 'static>;

const RUNNING: u32 = 0;
const PARKED: u32 = 1;

struct Signal {
    state: AtomicU32,
    thread: OnceLock<Thread>,
}

impl Signal {
    fn wake(&self) {
        if self.state.swap(RUNNING, Ordering::AcqRel) == PARKED
            && let Some(t) = self.thread.get()
        {
            t.unpark();
        }
    }
}

/// A handle that can put work on any shard, using lanes it owns exclusively.
///
/// One of these belongs to one thread. Clone it and you get a different set of
/// lanes, which is what keeps every lane single producer. Hand the same one to
/// two threads and it will not compile, because it is `!Sync`.
pub struct Submitter<T: 'static> {
    senders: Vec<Sender<Job<T>>>,
    signals: Vec<Arc<Signal>>,
    /// `Cell` rather than a raw pointer, because a submitter is allowed to move
    /// to another thread. What it is not allowed to do is be in two at once.
    _not_sync: PhantomData<core::cell::Cell<()>>,
}

impl<T: 'static> Submitter<T> {
    /// How many shards this submitter can reach.
    pub fn shards(&self) -> usize {
        self.senders.len()
    }

    /// Put a job on shard `shard`, handing it back if that shard's lane is
    /// full.
    ///
    /// Full means the shard is behind. The caller decides what to do about it,
    /// because only the caller knows whether the right answer is to retry, to
    /// shed the request, or to tell the client to slow down. The job comes back
    /// intact rather than being dropped, so retrying costs nothing extra.
    pub fn try_send<F>(&self, shard: usize, job: F) -> Result<(), Rejected<T>>
    where
        F: FnOnce(&mut ShardCtx<T>) + Send + 'static,
    {
        self.push(shard, Box::new(job))
    }

    /// Try again with a job that was handed back.
    pub fn retry(&self, shard: usize, job: Rejected<T>) -> Result<(), Rejected<T>> {
        self.push(shard, job.0)
    }

    fn push(&self, shard: usize, job: Job<T>) -> Result<(), Rejected<T>> {
        match self.senders[shard].push(job) {
            Ok(()) => {
                self.signals[shard].wake();
                Ok(())
            }
            Err(back) => Err(Rejected(back)),
        }
    }

    /// Put a job on shard `shard`, spinning until there is room.
    pub fn send<F>(&self, shard: usize, job: F)
    where
        F: FnOnce(&mut ShardCtx<T>) + Send + 'static,
    {
        let mut job: Job<T> = Box::new(job);
        loop {
            match self.senders[shard].push(job) {
                Ok(()) => {
                    self.signals[shard].wake();
                    return;
                }
                Err(back) => {
                    job = back;
                    std::hint::spin_loop();
                }
            }
        }
    }

    /// Run a job on shard `shard` and wait for its answer.
    ///
    /// The control plane path. It allocates and it blocks, both of which are
    /// fine for setup, stats and tests, and neither of which belongs anywhere
    /// near a command being served.
    pub fn call<F, R>(&self, shard: usize, job: F) -> R
    where
        F: FnOnce(&mut ShardCtx<T>) -> R + Send + 'static,
        R: Send + 'static,
    {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        self.send(shard, move |ctx| {
            let _ = tx.send(job(ctx));
        });
        rx.recv().expect("the shard died holding a reply")
    }
}

/// A job that would not fit, handed back whole.
pub struct Rejected<T: 'static>(Job<T>);

impl<T: 'static> core::fmt::Debug for Rejected<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("Rejected(lane full)")
    }
}

struct ShardHandle {
    signal: Arc<Signal>,
    join: Option<JoinHandle<()>>,
}

/// A set of shards, their threads, and their epochs.
pub struct Runtime<T: 'static> {
    handles: Vec<ShardHandle>,
    epochs: Arc<Epochs>,
    free: Mutex<Vec<Submitter<T>>>,
    stop: Arc<AtomicBool>,
}

/// How to build a [`Runtime`].
pub struct Builder {
    shards: usize,
    submitters: usize,
    lane_capacity: usize,
    pin: bool,
}

impl Builder {
    /// One shard per available core, four submitters, pinning on.
    pub fn new() -> Builder {
        Builder {
            shards: std::thread::available_parallelism().map_or(1, |n| n.get()),
            submitters: 4,
            lane_capacity: LANE_CAPACITY,
            pin: true,
        }
    }

    /// How many shard threads to run.
    pub fn shards(mut self, n: usize) -> Builder {
        assert!(n > 0, "a runtime needs at least one shard");
        assert!(n <= u16::MAX as usize, "shard ids are sixteen bits");
        self.shards = n;
        self
    }

    /// How many independent submitters to hand out. Each one owns a lane into
    /// every shard, so this is the number of threads that may submit work.
    pub fn submitters(mut self, n: usize) -> Builder {
        assert!(n > 0, "somebody has to be able to submit work");
        self.submitters = n;
        self
    }

    /// Depth of one lane.
    pub fn lane_capacity(mut self, n: usize) -> Builder {
        self.lane_capacity = n;
        self
    }

    /// Whether to pin each shard thread to a core. On by default, and worth
    /// turning off when several runtimes share a machine or when running under
    /// a tool that does its own placement.
    pub fn pin(mut self, yes: bool) -> Builder {
        self.pin = yes;
        self
    }

    /// Start the threads. `init` runs on each shard thread and builds the state
    /// that shard will own.
    pub fn build<T, F>(self, init: F) -> Runtime<T>
    where
        T: 'static,
        F: Fn(ShardId) -> T + Send + Clone + 'static,
    {
        let epochs = Epochs::new(self.shards);
        let stop = Arc::new(AtomicBool::new(false));

        // lanes[shard][submitter]
        let mut receivers: Vec<Vec<Receiver<Job<T>>>> = Vec::with_capacity(self.shards);
        let mut senders: Vec<Vec<Sender<Job<T>>>> = (0..self.submitters)
            .map(|_| Vec::with_capacity(self.shards))
            .collect();
        for _ in 0..self.shards {
            let mut rxs = Vec::with_capacity(self.submitters);
            for (s, per_submitter) in senders.iter_mut().enumerate() {
                let _ = s;
                let (tx, rx) = spsc::lane::<Job<T>>(self.lane_capacity);
                per_submitter.push(tx);
                rxs.push(rx);
            }
            receivers.push(rxs);
        }

        let signals: Vec<Arc<Signal>> = (0..self.shards)
            .map(|_| {
                Arc::new(Signal {
                    state: AtomicU32::new(RUNNING),
                    thread: OnceLock::new(),
                })
            })
            .collect();

        let mut handles = Vec::with_capacity(self.shards);
        for (id, rxs) in receivers.into_iter().enumerate() {
            let signal = Arc::clone(&signals[id]);
            let epochs = Arc::clone(&epochs);
            let stop = Arc::clone(&stop);
            let init = init.clone();
            let pin = self.pin;
            let join = std::thread::Builder::new()
                .name(format!("yo-shard-{id}"))
                .spawn(move || {
                    if pin {
                        pin_to_core(id);
                    }
                    let _ = signal.thread.set(std::thread::current());
                    let mut ctx = ShardCtx {
                        id: ShardId(id as u16),
                        state: ShardLocal::new(init(ShardId(id as u16))),
                        epochs,
                    };
                    run(&mut ctx, &rxs, &signal, &stop);
                })
                .expect("could not start a shard thread");
            handles.push(ShardHandle {
                signal: Arc::clone(&signals[id]),
                join: Some(join),
            });
        }

        let free = senders
            .into_iter()
            .map(|s| Submitter {
                senders: s,
                signals: signals.clone(),
                _not_sync: PhantomData,
            })
            .collect();

        Runtime {
            handles,
            epochs,
            free: Mutex::new(free),
            stop,
        }
    }
}

impl Default for Builder {
    fn default() -> Builder {
        Builder::new()
    }
}

fn run<T>(ctx: &mut ShardCtx<T>, rxs: &[Receiver<Job<T>>], signal: &Signal, stop: &AtomicBool) {
    let me = ctx.index();
    let mut spins = 0u32;
    loop {
        let mut did = 0usize;
        // One epoch bump around the whole sweep, not around each job. A batch
        // is the unit that can hold an address, and the sweep is the batch.
        ctx.epochs.enter(me);
        for rx in rxs {
            while let Some(job) = rx.pop() {
                job(ctx);
                did += 1;
            }
        }
        ctx.epochs.leave(me);

        if did > 0 {
            spins = 0;
            continue;
        }
        if stop.load(Ordering::Acquire) {
            // Drain anything that raced in behind the flag, then finish.
            let mut left = 0;
            ctx.epochs.enter(me);
            for rx in rxs {
                while let Some(job) = rx.pop() {
                    job(ctx);
                    left += 1;
                }
            }
            ctx.epochs.leave(me);
            if left == 0 {
                return;
            }
            continue;
        }

        spins += 1;
        if spins < SPIN_LIMIT {
            std::hint::spin_loop();
            continue;
        }

        signal.state.store(PARKED, Ordering::Release);
        // Check once more after announcing the park. A sender that pushed
        // before our store will find PARKED and unpark us; one that pushed
        // after it is visible here. Either way we do not sleep on work.
        let pending = rxs.iter().any(|rx| !rx.is_empty()) || stop.load(Ordering::Acquire);
        if pending {
            signal.state.store(RUNNING, Ordering::Release);
        } else {
            std::thread::park();
            signal.state.store(RUNNING, Ordering::Release);
        }
        spins = 0;
    }
}

impl<T: 'static> Runtime<T> {
    /// A runtime with default settings.
    pub fn new<F>(init: F) -> Runtime<T>
    where
        F: Fn(ShardId) -> T + Send + Clone + 'static,
    {
        Builder::new().build(init)
    }

    /// Start configuring a runtime.
    pub fn builder() -> Builder {
        Builder::new()
    }

    /// How many shards are running.
    pub fn shards(&self) -> usize {
        self.handles.len()
    }

    /// The epoch counters.
    pub fn epochs(&self) -> &Arc<Epochs> {
        &self.epochs
    }

    /// Take one of the submitters.
    ///
    /// Panics once they run out, rather than handing back something that would
    /// quietly share a lane. Ask the builder for more if you need more.
    pub fn submitter(&self) -> Submitter<T> {
        self.free
            .lock()
            .unwrap()
            .pop()
            .expect("all submitters are handed out; build with more")
    }

    /// Give a submitter back so another thread can take it.
    pub fn release(&self, s: Submitter<T>) {
        self.free.lock().unwrap().push(s);
    }

    /// Which shard a hash belongs to.
    ///
    /// The top bits, because the low bits already pick the bucket inside a
    /// shard's index and reusing them would correlate the two.
    #[inline]
    pub fn shard_of(&self, hash: u64) -> usize {
        ((hash >> 32) as usize) % self.handles.len()
    }

    /// Stop every shard and wait for it.
    ///
    /// Jobs already on a lane run before the shard exits. Jobs submitted after
    /// this returns go nowhere.
    pub fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Release);
        for h in &self.handles {
            h.signal.wake();
            if let Some(t) = h.signal.thread.get() {
                t.unpark();
            }
        }
        for h in &mut self.handles {
            if let Some(j) = h.join.take() {
                let _ = j.join();
            }
        }
    }
}

impl<T: 'static> Drop for Runtime<T> {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Pin the calling thread to a core.
///
/// Linux only for now. macOS has no real affinity call, only a hint that the
/// scheduler is free to ignore, so there is nothing honest to do there and this
/// says so instead of pretending.
#[cfg(target_os = "linux")]
fn pin_to_core(core: usize) {
    // SAFETY: `cpu_set_t` is a plain bitmask that libc is happy to receive
    // zeroed, and the two macros below only touch the set we just made.
    unsafe {
        let mut set: libc::cpu_set_t = core::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        let n = libc::sysconf(libc::_SC_NPROCESSORS_ONLN).max(1) as usize;
        libc::CPU_SET(core % n, &mut set);
        libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &set);
    }
}

#[cfg(not(target_os = "linux"))]
fn pin_to_core(_core: usize) {}

#[cfg(test)]
mod tests {
    use super::*;

    // Miri runs these on interpreted threads, so the counts shrink under it.
    // Every path still gets taken, including the park and the wakeup.
    #[cfg(miri)]
    const JOBS: usize = 400;
    #[cfg(not(miri))]
    const JOBS: usize = 100_000;

    #[cfg(miri)]
    const PER_WORKER: usize = 100;
    #[cfg(not(miri))]
    const PER_WORKER: usize = 20_000;

    #[cfg(miri)]
    const QUEUED: u32 = 100;
    #[cfg(not(miri))]
    const QUEUED: u32 = 1000;

    #[cfg(miri)]
    const SPREAD: u64 = 4_000;
    #[cfg(not(miri))]
    const SPREAD: u64 = 100_000;

    struct Probe<T>(PhantomData<T>);
    trait Auto {
        fn sendable(&self) -> bool {
            false
        }
        fn shareable(&self) -> bool {
            false
        }
    }
    impl<T> Auto for Probe<T> {}
    impl<T: Send> Probe<T> {
        fn sendable(&self) -> bool {
            true
        }
    }
    impl<T: Sync> Probe<T> {
        fn shareable(&self) -> bool {
            true
        }
    }

    #[test]
    fn the_ownership_rules_are_in_the_type_system() {
        // If any of these flipped, the no locks argument would be gone, so it
        // is worth a test rather than a comment. An inherent method wins over a
        // trait method when it applies, so the answer is true only when the
        // bound really holds.
        let control = Probe::<Vec<u8>>(PhantomData);
        assert!(control.sendable() && control.shareable(), "probe is broken");

        // Shard state cannot leave its thread and cannot be borrowed from
        // another one.
        assert!(!Probe::<ShardLocal<Vec<u8>>>(PhantomData).sendable());
        assert!(!Probe::<ShardLocal<Vec<u8>>>(PhantomData).shareable());

        // A submitter may move to another thread. It may not be used by two at
        // once, which is what keeps every lane single producer.
        assert!(Probe::<Submitter<u8>>(PhantomData).sendable());
        assert!(!Probe::<Submitter<u8>>(PhantomData).shareable());

        // The runtime itself is shareable, because everything reachable through
        // it is either atomic or behind a lock on the control plane.
        assert!(Probe::<Runtime<u8>>(PhantomData).sendable());
        assert!(Probe::<Runtime<u8>>(PhantomData).shareable());
    }

    #[test]
    fn jobs_land_on_the_shard_they_were_addressed_to() {
        let rt: Runtime<Vec<usize>> = Builder::new().shards(4).pin(false).build(|_| Vec::new());
        let sub = rt.submitter();
        for i in 0..4 {
            sub.send(i, move |ctx| {
                assert_eq!(ctx.index(), i);
                ctx.state.push(i);
            });
        }
        for i in 0..4 {
            let seen = sub.call(i, |ctx| ctx.state.clone());
            assert_eq!(seen, vec![i]);
        }
    }

    #[test]
    fn every_submitted_job_runs_exactly_once() {
        const N: usize = JOBS;
        let rt: Runtime<usize> = Builder::new().shards(4).pin(false).build(|_| 0usize);
        let sub = rt.submitter();
        for i in 0..N {
            sub.send(i % 4, |ctx| {
                *ctx.state += 1;
            });
        }
        let total: usize = (0..4).map(|s| sub.call(s, |ctx| *ctx.state)).sum();
        assert_eq!(total, N);
    }

    #[test]
    fn several_threads_submit_on_their_own_lanes() {
        const PER: usize = PER_WORKER;
        let rt: Arc<Runtime<usize>> = Arc::new(
            Builder::new()
                .shards(4)
                .submitters(4)
                .pin(false)
                .build(|_| 0usize),
        );
        let workers: Vec<_> = (0..4)
            .map(|w| {
                let rt = Arc::clone(&rt);
                std::thread::spawn(move || {
                    let sub = rt.submitter();
                    for i in 0..PER {
                        sub.send((w + i) % 4, |ctx| {
                            *ctx.state += 1;
                        });
                    }
                    // Sending only puts the job on a lane. Ordering is per lane
                    // and a shard polls its lanes in turn, so a job this worker
                    // sent can still be sitting there after the send returns,
                    // and a count read through anybody else's lane would be
                    // short. A call on this worker's own lane comes back only
                    // once everything it put on that lane has run, so doing one
                    // per shard before letting go is what makes the total below
                    // an exact number rather than a race the test happens to
                    // win on a quiet machine.
                    for s in 0..4 {
                        sub.call(s, |_| ());
                    }
                    rt.release(sub);
                })
            })
            .collect();
        for w in workers {
            w.join().unwrap();
        }
        let sub = rt.submitter();
        let total: usize = (0..4).map(|s| sub.call(s, |ctx| *ctx.state)).sum();
        assert_eq!(total, PER * 4);
    }

    #[test]
    fn shutdown_drains_what_is_already_queued() {
        let mut rt: Runtime<usize> = Builder::new().shards(2).pin(false).build(|_| 0usize);
        let sub = rt.submitter();
        let done = Arc::new(AtomicU32::new(0));
        for _ in 0..QUEUED {
            let done = Arc::clone(&done);
            sub.send(0, move |_| {
                done.fetch_add(1, Ordering::Relaxed);
            });
        }
        rt.shutdown();
        assert_eq!(done.load(Ordering::Relaxed), QUEUED);
    }

    #[test]
    fn a_parked_shard_wakes_up() {
        let rt: Runtime<usize> = Builder::new().shards(2).pin(false).build(|_| 0usize);
        let sub = rt.submitter();
        // Long enough that the shard is certainly past its spin limit and
        // asleep in the futex.
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!(sub.call(1, |ctx| *ctx.state), 0);
        sub.send(1, |ctx| *ctx.state = 42);
        assert_eq!(sub.call(1, |ctx| *ctx.state), 42);
    }

    #[test]
    fn shard_of_spreads_hashes() {
        let rt: Runtime<()> = Builder::new().shards(8).pin(false).build(|_| ());
        let mut counts = [0usize; 8];
        for i in 0..SPREAD {
            counts[rt.shard_of(yo_common::wyhash(&i.to_le_bytes(), 0))] += 1;
        }
        let each = SPREAD as usize / 8;
        for (s, &c) in counts.iter().enumerate() {
            assert!(
                c > each * 4 / 5 && c < each * 6 / 5,
                "shard {s} got {c}, which is not an even spread"
            );
        }
    }
}
