//! The six stages, and the state that survives between two turns of them.

use crate::budget::Budget;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use yo_common::Result;
use yo_shard::Epochs;
use yo_shard::spsc::Receiver;

/// Commands taken out of the intake lanes in one turn.
///
/// `04` section 3 fixes this at 64 and gives the reason: the prefetch distance
/// is the whole batch, and Y1 removes what usually keeps that window short,
/// which is a lock held across it and other threads able to invalidate a line
/// inside it. Valkey ships 16 and Redis 8.4 ships 16 because they have both.
pub const BATCH_MAX: usize = 64;

/// What the loop does after a command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    /// Carry on with the rest of the batch.
    Next,
    /// End the batch here.
    ///
    /// For a command whose key set is not known until something earlier in the
    /// same batch has finished: a `MULTI` body, a `WAIT`, a blocking form.
    /// Prefetching those keys was not possible, so running them in this batch
    /// would be running them cold, and whatever was drained behind this command
    /// waits for the next turn instead of being thrown away.
    Break,
}

/// The shard behind the loop.
///
/// One implementation per kind of shard, which for now means one: the string
/// plane. The loop makes these calls in the order `04` section 2 lists and
/// never in another order, so an implementation can rely on `prefetch` for a
/// piece of work coming before `run` for it, on `flush` coming after every
/// `run` in the batch, and on `maintain` coming last.
pub trait Engine {
    /// One command, however the layer above chose to represent it.
    type Work;

    /// The hash of the key this work touches, or `None` when it touches none.
    ///
    /// Called once per command, on the first walk. Whatever comes back is
    /// handed to [`Engine::run`] on the second walk, so a command's key is
    /// hashed once per batch rather than once per walk.
    fn key_hash(&self, work: &Self::Work) -> Option<u64>;

    /// Ask the cache for whatever `run` is about to load for this work.
    ///
    /// Usually one call to the index's own prefetch. It has to be cheap and it
    /// has to read nothing, because it runs for all 64 commands before the
    /// first one executes.
    ///
    /// The work comes with the hash because a hash on its own does not say
    /// which structure to warm. Two commands in one batch can carry the same
    /// key into different databases, and later into different types, so the
    /// engine needs the command to know which index the bucket is in.
    fn prefetch(&self, work: &Self::Work, hash: u64);

    /// Execute one command.
    ///
    /// `hash` is what `key_hash` returned for this work, so a lookup takes the
    /// hashed form rather than hashing the key a second time.
    fn run(&mut self, work: Self::Work, hash: Option<u64>) -> Flow;

    /// Write out the replies the batch produced.
    ///
    /// One `writev` per connection touched, never one per reply. aki's
    /// `HGETALL` profile spent 69.7 percent of its time in write syscalls, and
    /// this is the call that exists so that does not happen again.
    fn flush(&mut self);

    /// Hand the submission queue to the kernel.
    ///
    /// The first stage. One syscall when there is something queued, and none at
    /// all under SQPoll. Does nothing by default, which is right for a shard
    /// with no ring under it.
    ///
    /// # Errors
    ///
    /// Whatever the ring says. The turn stops there and reports it.
    fn submit_io(&mut self) -> Result<()> {
        Ok(())
    }

    /// Pick up completions that have arrived.
    ///
    /// Where a parked command finds out its write landed. Does nothing by
    /// default.
    ///
    /// # Errors
    ///
    /// Whatever the ring says, including a failure an earlier submission left
    /// behind.
    fn drain_io(&mut self) -> Result<()> {
        Ok(())
    }

    /// Spend up to `budget` on background work.
    ///
    /// Expiry sampling, eviction, the compaction handshake, partition
    /// rebalance and tier demotion, in that order of priority, per `04` section
    /// 6. Does nothing by default.
    fn maintain(&mut self, budget: &mut Budget) {
        let _ = budget;
    }
}

/// What one turn of the loop did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Turn {
    /// Commands executed.
    pub commands: usize,
    /// Whether a command ended the batch early.
    pub broke: bool,
    /// Commands drained but not executed, waiting for the next turn.
    pub carried: usize,
    /// Maintenance units spent.
    pub maintained: u32,
}

impl Turn {
    /// Whether the turn found nothing to do.
    ///
    /// Maintenance does not count. A shard sampling for expired keys with no
    /// commands in front of it is idle, and a caller spinning on this is right
    /// to back off.
    #[must_use]
    pub const fn is_idle(&self) -> bool {
        self.commands == 0 && self.carried == 0
    }
}

/// The loop, and what it keeps between turns.
///
/// Owns the engine outright. A shard is one thread, one engine and one of
/// these, and none of the three is shared with anything.
pub struct Reactor<E: Engine> {
    engine: E,
    id: usize,
    epochs: Arc<Epochs>,
    lanes: Vec<Receiver<E::Work>>,
    /// Drained and not yet executed. Empty at the end of most turns, and
    /// holding the tail of a broken batch otherwise.
    pending: VecDeque<E::Work>,
    /// The first walk's answers, positional against `pending`.
    hashes: Vec<Option<u64>>,
    /// Which lane the next drain starts from, so a busy lane cannot starve a
    /// quiet one.
    lane: usize,
    budget: u32,
    turns: u64,
    commands: u64,
    batches: u64,
    full: u64,
    breaks: u64,
    idle: u64,
}

impl<E: Engine> Reactor<E> {
    /// A reactor for shard `id`, taking work from `lanes`.
    ///
    /// One lane per submitter, which is one per network reactor and one per
    /// embedded caller thread, and each is single producer single consumer, so
    /// no queue here ever has two writers.
    ///
    /// # Panics
    ///
    /// If `id` is not a shard `epochs` has a slot for. That is a wiring mistake
    /// at startup and there is nothing sensible to do with it later.
    pub fn new(engine: E, id: usize, epochs: Arc<Epochs>, lanes: Vec<Receiver<E::Work>>) -> Self {
        assert!(id < epochs.len(), "shard {id} has no epoch slot");
        Reactor {
            engine,
            id,
            epochs,
            lanes,
            pending: VecDeque::with_capacity(BATCH_MAX),
            hashes: Vec::with_capacity(BATCH_MAX),
            lane: 0,
            budget: crate::MAINTENANCE_UNITS,
            turns: 0,
            commands: 0,
            batches: 0,
            full: 0,
            breaks: 0,
            idle: 0,
        }
    }

    /// A reactor with no lanes, for the caller who is the shard.
    ///
    /// `15` section 7's embedded mode. There is no queue to cross and no thread
    /// to hand to, so the loop stops being a loop and becomes
    /// [`Reactor::execute`], which runs the same dispatch the server path runs.
    /// Y23 asks for the same code rather than the same idea, and this is what
    /// that means in practice.
    pub fn inline(engine: E) -> Self {
        Reactor::new(engine, 0, Epochs::new(1), Vec::new())
    }

    /// Change the maintenance allowance per turn.
    ///
    /// Zero means no maintenance at all, which is what a benchmark measuring
    /// the command path alone wants and what nothing in production wants.
    #[must_use]
    pub fn with_maintenance(mut self, units: u32) -> Self {
        self.budget = units;
        self
    }

    /// The engine.
    pub const fn engine(&self) -> &E {
        &self.engine
    }

    /// The engine, for a caller that owns both ends of it.
    pub const fn engine_mut(&mut self) -> &mut E {
        &mut self.engine
    }

    /// The shard this reactor is.
    #[must_use]
    pub const fn id(&self) -> usize {
        self.id
    }

    /// Turns taken.
    #[must_use]
    pub const fn turns(&self) -> u64 {
        self.turns
    }

    /// Commands executed.
    #[must_use]
    pub const fn commands(&self) -> u64 {
        self.commands
    }

    /// Batches drained.
    #[must_use]
    pub const fn batches(&self) -> u64 {
        self.batches
    }

    /// Batches that came out full, which is the number that says whether the
    /// batch size is doing anything.
    ///
    /// A shard whose batches are never full is latency bound and the prefetch
    /// walk is buying it very little. One whose batches are always full is
    /// throughput bound, and the window is either the right size or too small.
    #[must_use]
    pub const fn full_batches(&self) -> u64 {
        self.full
    }

    /// Batches ended early by a command that could not be prefetched with the
    /// rest of them.
    #[must_use]
    pub const fn breaks(&self) -> u64 {
        self.breaks
    }

    /// Turns that found nothing to do.
    #[must_use]
    pub const fn idle_turns(&self) -> u64 {
        self.idle
    }

    /// Commands drained and waiting, which is only ever the tail of a broken
    /// batch.
    #[must_use]
    pub fn carried(&self) -> usize {
        self.pending.len()
    }

    /// One turn of the six stages.
    ///
    /// # Errors
    ///
    /// From the ring, at either of the two stages that touch it. The turn stops
    /// at the failure rather than carrying on with half a batch, and nothing is
    /// lost: whatever was drained is still held for the next turn.
    pub fn tick(&mut self) -> Result<Turn> {
        self.turns += 1;

        // 1. Submit. One syscall, or zero under SQPoll.
        self.engine.submit_io()?;

        // 2. Intake. Only when the last batch finished, because a broken
        // batch's tail is already a batch, and drawing more in on top of it
        // would push the window past 64.
        if self.pending.is_empty() {
            self.fill();
            if !self.pending.is_empty() {
                self.batches += 1;
                if self.pending.len() == BATCH_MAX {
                    self.full += 1;
                }
            }
        }

        let mut turn = Turn::default();
        if self.pending.is_empty() {
            // An idle turn skips the epoch and the flush. Skipping the epoch is
            // safe rather than merely cheap: an even counter is what `all_past`
            // reads as holding nothing, so a shard that never enters again does
            // not hold reclamation up. Skipping the flush is safe because a
            // turn with no commands in it touched no connection.
            self.idle += 1;
        } else {
            // 3. Enter, once per batch and not once per command.
            self.epochs.enter(self.id);

            // 4a. The first walk: hash, and ask for the line.
            self.hashes.clear();
            for w in &self.pending {
                let h = self.engine.key_hash(w);
                if let Some(h) = h {
                    self.engine.prefetch(w, h);
                }
                self.hashes.push(h);
            }

            // 4b. The second walk: execute what the first walk warmed.
            let mut n = 0;
            while let Some(w) = self.pending.pop_front() {
                let h = self.hashes[n];
                n += 1;
                if self.engine.run(w, h) == Flow::Break {
                    turn.broke = true;
                    self.breaks += 1;
                    break;
                }
            }

            // 5. Leave, then write the replies out.
            self.epochs.leave(self.id);
            self.engine.flush();

            self.commands += n as u64;
            turn.commands = n;
            turn.carried = self.pending.len();
        }

        // 6. Completions, then a bounded slice of background work.
        self.engine.drain_io()?;
        if self.budget > 0 {
            let mut budget = Budget::new(self.budget);
            self.engine.maintain(&mut budget);
            turn.maintained = budget.spent();
        }
        Ok(turn)
    }

    /// Turns until `stop` is set and there is nothing left to run.
    ///
    /// The backoff is deliberately plain: spin for a while, then yield. A shard
    /// that is expected to be busy runs on a pinned thread and never gets here,
    /// and one that is not busy should give the core back rather than burn it.
    ///
    /// # Errors
    ///
    /// The first failure any turn reports. Whatever was drained stays drained,
    /// so a caller that decides to carry on can call [`Reactor::tick`] again.
    pub fn run_until(&mut self, stop: &AtomicBool) -> Result<()> {
        const SPINS: u32 = 128;
        let mut idle = 0u32;
        loop {
            if !self.tick()?.is_idle() {
                idle = 0;
                continue;
            }
            if stop.load(Ordering::Acquire) {
                // One more turn, in case something raced in behind the flag.
                if self.tick()?.is_idle() {
                    return Ok(());
                }
                continue;
            }
            idle += 1;
            if idle < SPINS {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
                idle = 0;
            }
        }
    }

    /// Execute one command directly, with no queue in the way.
    ///
    /// The embedded path. The same `prefetch` and the same `run` the loop
    /// calls, so a command has one implementation rather than an inline one and
    /// a server one that drift apart.
    ///
    /// The epoch is entered and left around the call, which is two stores and a
    /// fence. That is what a caller pays for being allowed to hold on to what a
    /// command returned, and [`Reactor::execute_all`] is how to pay it once for
    /// many commands instead of once each.
    pub fn execute(&mut self, work: E::Work) -> Flow {
        self.turns += 1;
        self.commands += 1;
        self.epochs.enter(self.id);
        let hash = self.engine.key_hash(&work);
        if let Some(h) = hash {
            self.engine.prefetch(&work, h);
        }
        let flow = self.engine.run(work, hash);
        self.epochs.leave(self.id);
        flow
    }

    /// Execute a batch directly, in the same two walks the loop uses.
    ///
    /// Returns how many ran, which is short of what went in when one of them
    /// broke the batch. The rest are dropped rather than queued, because an
    /// inline caller is the one holding the work and a queue here would be a
    /// second place it can live.
    ///
    /// The batch goes through the same buffer the loop drains into, so a caller
    /// doing this in a hot loop allocates on the first call and never again.
    pub fn execute_all<I>(&mut self, work: I) -> usize
    where
        I: IntoIterator<Item = E::Work>,
    {
        self.pending.extend(work);
        if self.pending.is_empty() {
            return 0;
        }
        self.turns += 1;
        self.epochs.enter(self.id);

        self.hashes.clear();
        for w in &self.pending {
            let h = self.engine.key_hash(w);
            if let Some(h) = h {
                self.engine.prefetch(w, h);
            }
            self.hashes.push(h);
        }

        let mut n = 0;
        while let Some(w) = self.pending.pop_front() {
            let h = self.hashes[n];
            n += 1;
            if self.engine.run(w, h) == Flow::Break {
                self.breaks += 1;
                break;
            }
        }
        self.pending.clear();

        self.epochs.leave(self.id);
        self.commands += n as u64;
        n
    }

    /// Draw up to [`BATCH_MAX`] commands, round robin from the lane after the
    /// one the last drain finished on.
    fn fill(&mut self) {
        if self.lanes.is_empty() {
            return;
        }
        let n = self.lanes.len();
        let mut room = BATCH_MAX;
        let mut at = self.lane;
        loop {
            let mut took = 0;
            for _ in 0..n {
                if room == 0 {
                    self.lane = at;
                    return;
                }
                if let Some(w) = self.lanes[at].pop() {
                    self.pending.push_back(w);
                    room -= 1;
                    took += 1;
                }
                at = (at + 1) % n;
            }
            if took == 0 {
                self.lane = at;
                return;
            }
        }
    }
}

impl<E: Engine> std::fmt::Debug for Reactor<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reactor")
            .field("id", &self.id)
            .field("lanes", &self.lanes.len())
            .field("turns", &self.turns)
            .field("commands", &self.commands)
            .field("batches", &self.batches)
            .field("full_batches", &self.full)
            .field("breaks", &self.breaks)
            .field("idle_turns", &self.idle)
            .field("carried", &self.pending.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use yo_common::{Code, Error};
    use yo_shard::spsc::{Sender, lane};

    /// Every call the loop made, in the order it made it.
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Step {
        Submit,
        Prefetch(u64),
        Run(u64),
        Flush,
        Drain,
        Maintain,
    }

    /// The work value that stands for a command with no key, since a command
    /// with no key is the one case the two walks treat differently.
    const KEYLESS: u64 = u64::MAX;

    /// An engine that records rather than does, which is the only way to assert
    /// about an order.
    ///
    /// The steps sit behind a `RefCell` because `prefetch` is handed `&self`,
    /// same as a real engine's is, and a test double is not a reason to reach
    /// for unsafe.
    struct Recorder {
        steps: RefCell<Vec<Step>>,
        break_on: Option<u64>,
        fail_submit: bool,
        fail_drain: bool,
        maintenance_item: u32,
    }

    impl Recorder {
        fn new() -> Recorder {
            Recorder {
                steps: RefCell::new(Vec::new()),
                break_on: None,
                fail_submit: false,
                fail_drain: false,
                maintenance_item: 0,
            }
        }

        fn push(&self, step: Step) {
            self.steps.borrow_mut().push(step);
        }

        fn steps(&self) -> Vec<Step> {
            self.steps.borrow().clone()
        }

        fn runs(&self) -> Vec<u64> {
            self.steps
                .borrow()
                .iter()
                .filter_map(|s| match s {
                    Step::Run(v) => Some(*v),
                    _ => None,
                })
                .collect()
        }

        fn count(&self, want: &Step) -> usize {
            self.steps.borrow().iter().filter(|s| *s == want).count()
        }
    }

    impl Engine for Recorder {
        type Work = u64;

        fn key_hash(&self, work: &u64) -> Option<u64> {
            if *work == KEYLESS { None } else { Some(*work) }
        }

        fn prefetch(&self, _work: &u64, hash: u64) {
            self.push(Step::Prefetch(hash));
        }

        fn run(&mut self, work: u64, hash: Option<u64>) -> Flow {
            assert_eq!(
                hash,
                if work == KEYLESS { None } else { Some(work) },
                "the second walk gets the hash the first walk took"
            );
            self.push(Step::Run(work));
            if self.break_on == Some(work) {
                return Flow::Break;
            }
            Flow::Next
        }

        fn flush(&mut self) {
            self.push(Step::Flush);
        }

        fn submit_io(&mut self) -> Result<()> {
            self.push(Step::Submit);
            if self.fail_submit {
                return Err(Error::new(Code::Io, "submit said no"));
            }
            Ok(())
        }

        fn drain_io(&mut self) -> Result<()> {
            self.push(Step::Drain);
            if self.fail_drain {
                return Err(Error::new(Code::Io, "drain said no"));
            }
            Ok(())
        }

        fn maintain(&mut self, budget: &mut Budget) {
            self.push(Step::Maintain);
            if self.maintenance_item == 0 {
                return;
            }
            while budget.spend(self.maintenance_item) {}
        }
    }

    /// A reactor on `lanes` lanes, with the sending ends and the epochs handed
    /// back.
    fn wired(lanes: usize) -> (Reactor<Recorder>, Vec<Sender<u64>>, Arc<Epochs>) {
        let mut rxs = Vec::new();
        let mut txs = Vec::new();
        for _ in 0..lanes {
            let (tx, rx) = lane(1024);
            txs.push(tx);
            rxs.push(rx);
        }
        let epochs = Epochs::new(1);
        let r = Reactor::new(Recorder::new(), 0, Arc::clone(&epochs), rxs);
        (r, txs, epochs)
    }

    /// Where the last prefetch and the first run landed, which is the whole
    /// claim the two walk shape makes.
    fn walk_boundary(steps: &[Step]) -> (usize, usize) {
        let last_prefetch = steps
            .iter()
            .rposition(|s| matches!(s, Step::Prefetch(_)))
            .expect("nothing was prefetched");
        let first_run = steps
            .iter()
            .position(|s| matches!(s, Step::Run(_)))
            .expect("nothing ran");
        (last_prefetch, first_run)
    }

    #[test]
    fn the_stages_run_in_the_order_the_spec_lists_them() {
        let (mut r, tx, _e) = wired(1);
        tx[0].push(7).unwrap();
        let turn = r.tick().unwrap();
        assert_eq!(turn.commands, 1);
        assert_eq!(
            r.engine().steps(),
            vec![
                Step::Submit,
                Step::Prefetch(7),
                Step::Run(7),
                Step::Flush,
                Step::Drain,
                Step::Maintain,
            ]
        );
    }

    #[test]
    fn every_command_is_prefetched_before_any_of_them_runs() {
        let (mut r, tx, _e) = wired(1);
        for i in 0..8 {
            tx[0].push(i).unwrap();
        }
        r.tick().unwrap();
        let (last_prefetch, first_run) = walk_boundary(&r.engine().steps());
        assert!(
            last_prefetch < first_run,
            "the two walks overlapped, which makes the prefetch distance one"
        );
        assert_eq!(r.engine().runs(), (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn a_batch_stops_at_sixty_four() {
        let (mut r, tx, _e) = wired(1);
        for i in 0..200 {
            tx[0].push(i).unwrap();
        }
        assert_eq!(r.tick().unwrap().commands, BATCH_MAX);
        assert_eq!(r.full_batches(), 1);
        assert_eq!(r.tick().unwrap().commands, BATCH_MAX);
        assert_eq!(r.tick().unwrap().commands, BATCH_MAX);
        assert_eq!(r.tick().unwrap().commands, 200 - 3 * BATCH_MAX);
        assert_eq!(r.batches(), 4);
        assert_eq!(r.full_batches(), 3);
        assert_eq!(r.commands(), 200);
        assert_eq!(r.engine().runs(), (0..200).collect::<Vec<_>>());
    }

    #[test]
    fn a_break_leaves_the_rest_of_the_batch_for_the_next_turn() {
        let (mut r, tx, _e) = wired(1);
        for i in 0..10 {
            tx[0].push(i).unwrap();
        }
        r.engine_mut().break_on = Some(3);
        let turn = r.tick().unwrap();
        assert!(turn.broke);
        assert_eq!(turn.commands, 4, "the command that broke it still ran");
        assert_eq!(turn.carried, 6);
        assert_eq!(r.engine().count(&Step::Flush), 1, "the replies still went");

        // The next turn takes the carried tail and nothing new, so the window
        // never grows past one batch.
        r.engine_mut().break_on = None;
        let turn = r.tick().unwrap();
        assert_eq!(turn.commands, 6);
        assert_eq!(turn.carried, 0);
        assert_eq!(r.engine().runs(), (0..10).collect::<Vec<_>>());
        assert_eq!(r.breaks(), 1);
        assert_eq!(r.batches(), 1, "a broken batch is one batch, not two");
    }

    #[test]
    fn the_epoch_moves_once_per_batch_and_not_once_per_command() {
        let (mut r, tx, epochs) = wired(1);
        for i in 0..10 {
            tx[0].push(i).unwrap();
        }
        let before = epochs.get(0);
        r.tick().unwrap();
        assert_eq!(
            epochs.get(0),
            before + 2,
            "one enter and one leave for ten commands"
        );
        assert_eq!(r.commands(), 10);
    }

    #[test]
    fn an_idle_turn_does_not_touch_the_epoch_or_the_replies() {
        let (mut r, _tx, epochs) = wired(1);
        let before = epochs.get(0);
        let turn = r.tick().unwrap();
        assert!(turn.is_idle());
        assert_eq!(epochs.get(0), before, "an idle shard holds nothing");
        assert_eq!(r.engine().count(&Step::Flush), 0);
        assert_eq!(
            r.engine().count(&Step::Drain),
            1,
            "completions still get picked up"
        );
        assert_eq!(
            r.engine().count(&Step::Maintain),
            1,
            "and background work still runs"
        );
        assert_eq!(r.idle_turns(), 1);
        assert_eq!(r.batches(), 0);
    }

    #[test]
    fn work_comes_off_every_lane_rather_than_the_first_one() {
        let (mut r, tx, _e) = wired(4);
        for (i, t) in tx.iter().enumerate() {
            for j in 0..4u64 {
                t.push(i as u64 * 10 + j).unwrap();
            }
        }
        let turn = r.tick().unwrap();
        assert_eq!(turn.commands, 16);
        let runs = r.engine().runs();
        for lane in 0..4u64 {
            let from_lane = runs.iter().filter(|v| **v / 10 == lane).count();
            assert_eq!(from_lane, 4, "lane {lane} was skipped or drained twice");
        }
        assert_eq!(
            runs[..4].iter().map(|v| v / 10).collect::<Vec<_>>(),
            vec![0, 1, 2, 3],
            "round robin, so the first four are one from each lane"
        );
    }

    #[test]
    fn a_lane_that_never_stops_cannot_starve_the_others() {
        let (mut r, tx, _e) = wired(2);
        // Lane 0 has more than a batch on its own. Lane 1 has one command and
        // has to get out in the first turn all the same.
        for i in 0..200 {
            tx[0].push(i).unwrap();
        }
        tx[1].push(9_999).unwrap();
        r.tick().unwrap();
        assert!(
            r.engine().runs().contains(&9_999),
            "the quiet lane waited behind a whole batch of the busy one"
        );
    }

    #[test]
    fn maintenance_gets_a_budget_and_stops_when_it_is_spent() {
        let (mut r, _tx, _e) = wired(1);
        r.engine_mut().maintenance_item = 100;
        let turn = r.tick().unwrap();
        // The slice stops on the item that takes it past the end rather than
        // before it, so the overshoot is one item and never more.
        assert!(
            (crate::MAINTENANCE_UNITS..crate::MAINTENANCE_UNITS + 100).contains(&turn.maintained),
            "spent {}",
            turn.maintained
        );

        let mut r = r.with_maintenance(0);
        let before = r.engine().count(&Step::Maintain);
        let turn = r.tick().unwrap();
        assert_eq!(turn.maintained, 0);
        assert_eq!(
            r.engine().count(&Step::Maintain),
            before,
            "a zero budget is no slice at all rather than an empty one"
        );
    }

    #[test]
    fn a_failed_submit_stops_the_turn_before_the_batch() {
        let (mut r, tx, _e) = wired(1);
        tx[0].push(1).unwrap();
        r.engine_mut().fail_submit = true;
        assert_eq!(r.tick().unwrap_err().code(), Code::Io);
        assert_eq!(r.engine().runs(), Vec::<u64>::new());
        assert_eq!(r.carried(), 0, "nothing was drained, so nothing is held");

        // And the work is still in the lane afterwards.
        r.engine_mut().fail_submit = false;
        assert_eq!(r.tick().unwrap().commands, 1);
    }

    #[test]
    fn a_failed_drain_still_ran_the_batch_and_flushed_it() {
        let (mut r, tx, _e) = wired(1);
        tx[0].push(1).unwrap();
        r.engine_mut().fail_drain = true;
        assert_eq!(r.tick().unwrap_err().code(), Code::Io);
        assert_eq!(r.engine().runs(), vec![1]);
        assert_eq!(r.engine().count(&Step::Flush), 1);
    }

    #[test]
    fn a_command_with_no_key_is_run_without_a_hash() {
        let (mut r, tx, _e) = wired(1);
        tx[0].push(KEYLESS).unwrap();
        let turn = r.tick().unwrap();
        assert_eq!(turn.commands, 1);
        assert_eq!(
            r.engine().count(&Step::Prefetch(KEYLESS)),
            0,
            "there is nothing to warm for a command with no key"
        );
    }

    #[test]
    fn inline_execution_takes_the_same_walk_as_the_loop() {
        let mut r = Reactor::inline(Recorder::new());
        assert_eq!(r.execute(9), Flow::Next);
        assert_eq!(
            r.engine().steps(),
            vec![Step::Prefetch(9), Step::Run(9)],
            "no submit, no flush and no maintenance, and the same two calls"
        );
        assert_eq!(r.commands(), 1);
    }

    #[test]
    fn inline_batches_pay_for_the_epoch_once() {
        let mut r = Reactor::inline(Recorder::new());
        assert_eq!(r.execute_all(0..20), 20);
        assert_eq!(r.engine().runs(), (0..20).collect::<Vec<_>>());
        let (last_prefetch, first_run) = walk_boundary(&r.engine().steps());
        assert!(last_prefetch < first_run, "inline gets the two walks too");
    }

    #[test]
    fn an_inline_batch_stops_at_a_break() {
        let mut r = Reactor::inline(Recorder::new());
        r.engine_mut().break_on = Some(2);
        assert_eq!(r.execute_all(0..10), 3);
        assert_eq!(r.engine().runs(), vec![0, 1, 2]);
        assert_eq!(r.carried(), 0, "inline holds nothing over to next time");

        // And the next batch is not the last one's leftovers.
        r.engine_mut().break_on = None;
        assert_eq!(r.execute_all(100..103), 3);
        assert_eq!(r.engine().runs(), vec![0, 1, 2, 100, 101, 102]);
    }

    #[test]
    fn an_empty_inline_batch_is_not_a_turn() {
        let mut r = Reactor::inline(Recorder::new());
        assert_eq!(r.execute_all(Vec::new()), 0);
        assert_eq!(r.turns(), 0);
        assert!(r.engine().steps().is_empty());
    }

    #[test]
    fn run_until_returns_when_the_flag_is_set_and_the_lanes_are_dry() {
        let (mut r, tx, _e) = wired(1);
        for i in 0..300 {
            tx[0].push(i).unwrap();
        }
        let stop = AtomicBool::new(true);
        r.run_until(&stop).unwrap();
        assert_eq!(r.commands(), 300, "the flag does not drop queued work");
        assert!(r.turns() >= 5);
    }

    #[test]
    fn a_reactor_says_what_it_has_been_doing() {
        let (mut r, tx, _e) = wired(1);
        tx[0].push(1).unwrap();
        r.tick().unwrap();
        let said = format!("{r:?}");
        assert!(said.contains("commands: 1"), "{said}");
        assert!(said.contains("lanes: 1"), "{said}");
    }
}
