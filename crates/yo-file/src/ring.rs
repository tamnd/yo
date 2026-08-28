//! The write half of a log, done through the ring instead of through `pwrite`.
//!
//! `06` section 3 asks for two hundred thousand durable commits a second in
//! group mode. A synchronous `pwrite` followed by a synchronous `fdatasync`
//! cannot get there, not because either call is slow but because the shard is
//! stopped for the whole of both of them. The ring's answer is that the shard is
//! never stopped: it hands the bytes over, keeps running, and finds out later.
//!
//! Three things make that safe rather than merely fast, and all three are the
//! reason this is a state machine and not a wrapper.
//!
//! **The bytes have to outlive the call.** [`yo_record::PageWrite`] gives up its
//! borrow when `write` returns, and the kernel reads the buffer long after that,
//! so a submission straight from the borrow would be a dangling read. The bytes
//! are staged into a buffer this type owns and the buffer goes back on the free
//! list when the completion arrives. The copy is a memcpy against a syscall, and
//! it is also what makes registered buffers free later, since a registered set
//! has to be stable addresses and these already are.
//!
//! **Writes to one page cannot be in flight together.** io_uring does not order
//! submissions against each other. A page is flushed again every time its dirty
//! part grows, and every one of those flushes rewrites the page header, so two
//! in flight at once could land with the older header last and leave a page
//! claiming fewer records than it has. So a second write to a page waits for the
//! first. In group mode this never waits, because the sync boundary has already
//! drained everything.
//!
//! **A sync means nothing until the writes it covers have landed.** io_uring
//! runs a queued `fsync` in parallel with the writes queued before it. So
//! [`RingWriter::sync`] does not submit anything if writes are still out. It
//! records what the sync will have to cover and [`RingWriter::poll`] submits it
//! the moment the last write comes back. This is the single most common way to
//! get an io_uring durability bug and it is worth the extra state.

use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;

use yo_common::{Code, Error, Result};
use yo_uring::{Fd, Kind, Pending, Ring, RingConfig};

/// A submission that has not come back.
struct Inflight {
    /// Which staged buffer holds the bytes.
    buf: usize,
    /// The page this belongs to, so the one write per page rule can be lifted
    /// when it lands.
    page_addr: u64,
    /// Where in the file the next unwritten byte goes. Moves on a short write.
    off: u64,
    /// How much of the staged buffer has been written. Only ever non zero after
    /// a short write.
    done: usize,
    /// How much there is in total.
    len: usize,
    /// The log address this write covers up to, which is what a sync will make
    /// durable.
    covers_upto: u64,
}

/// The asynchronous write path for one shard's log.
///
/// One of these per [`crate::LogFile`] in ring mode. Not `Sync`, like everything
/// else on the data path.
pub struct RingWriter {
    ring: Ring,
    pending: Pending<Inflight>,
    /// Staged bytes. The outer `Vec` reallocating moves these `Vec` headers and
    /// not the heap blocks they point at, which is the whole reason a pointer
    /// handed to the kernel stays good across a push.
    bufs: Vec<Vec<u8>>,
    free: Vec<usize>,
    /// Pages with a write in flight, and how many.
    busy: HashMap<u64, u32>,
    file: Arc<File>,
    writes_inflight: u32,
    written_upto: u64,
    landed_upto: u64,
    durable_upto: u64,
    /// A sync that has been asked for and cannot be submitted yet, and the
    /// address it will cover.
    sync_wanted: Option<u64>,
    /// A sync that is out, and the address it covers.
    sync_inflight: Option<u64>,
    writes: u64,
    syncs: u64,
    waits: u64,
    /// The first failure since the last time anybody asked, kept because a
    /// completion arrives long after the call that caused it has returned and
    /// there is nowhere else to put it.
    failed: Option<Error>,
}

#[cfg(unix)]
fn raw(f: &File) -> Fd {
    use std::os::fd::AsRawFd;
    f.as_raw_fd()
}

#[cfg(windows)]
fn raw(f: &File) -> Fd {
    use std::os::windows::io::AsRawHandle;
    f.as_raw_handle()
}

impl RingWriter {
    /// Builds a writer over an already open file.
    ///
    /// The `Arc<File>` is held rather than the descriptor, because a submission
    /// outlives the call that made it and a descriptor closed underneath the
    /// kernel is a write into whatever got the number next.
    ///
    /// # Errors
    ///
    /// [`Code::Invalid`] for a configuration that cannot work, or
    /// [`Code::Unsupported`] when the kernel refuses the ring.
    pub fn new(file: Arc<File>, config: &RingConfig) -> Result<RingWriter> {
        let ring = Ring::new(config)?;
        let depth = ring.entries();
        Ok(RingWriter {
            pending: Pending::with_capacity(depth),
            ring,
            bufs: Vec::new(),
            free: Vec::new(),
            busy: HashMap::new(),
            file,
            writes_inflight: 0,
            written_upto: 0,
            landed_upto: 0,
            durable_upto: 0,
            sync_wanted: None,
            sync_inflight: None,
            writes: 0,
            syncs: 0,
            waits: 0,
            failed: None,
        })
    }

    /// Whether this is a real ring, which off Linux it is not.
    #[must_use]
    pub fn is_uring(&self) -> bool {
        self.ring.is_uring()
    }

    /// The log address below which everything is durable.
    #[must_use]
    pub const fn durable_upto(&self) -> u64 {
        self.durable_upto
    }

    /// The log address below which everything has been handed over, durable or
    /// not.
    #[must_use]
    pub const fn written_upto(&self) -> u64 {
        self.written_upto
    }

    /// How many writes were submitted.
    #[must_use]
    pub const fn writes(&self) -> u64 {
        self.writes
    }

    /// How many syncs were submitted. Group commit is the claim that this stays
    /// far below the commit count.
    #[must_use]
    pub const fn syncs(&self) -> u64 {
        self.syncs
    }

    /// How many staging buffers exist. This is the high water mark of writes in
    /// flight at once, since a buffer is only ever allocated when the free list
    /// is empty.
    #[must_use]
    pub fn buffers(&self) -> usize {
        self.bufs.len()
    }

    /// How many bytes the staging buffers are holding on to.
    #[must_use]
    pub fn staged_bytes(&self) -> usize {
        self.bufs.iter().map(Vec::capacity).sum()
    }

    /// How many times a caller had to be stopped to wait for the ring.
    ///
    /// The number that says whether the asynchronous path is actually
    /// asynchronous. In group mode it should stay at zero, because the sync
    /// boundary drains everything before the next write is staged.
    #[must_use]
    pub const fn stalls(&self) -> u64 {
        self.waits
    }

    /// Submissions that have not come back.
    #[must_use]
    pub fn in_flight(&self) -> u32 {
        self.pending.len()
    }

    /// Hands bytes over, without waiting for them to land.
    ///
    /// # Errors
    ///
    /// [`Code::Io`] for a failure this call or an earlier completion found out
    /// about. An error does not lose the page: the log still has it and can
    /// write it again.
    pub fn write(
        &mut self,
        page_addr: u64,
        off: u64,
        bytes: &[u8],
        covers_upto: u64,
    ) -> Result<()> {
        self.take_failure()?;
        // One write per page at a time, because the ring does not order them and
        // a page header landing out of order is a page that lies about how much
        // of it is real.
        while self.busy.contains_key(&page_addr) {
            self.wait_for_one()?;
        }
        // Room in the table, which is the same thing as room in the ring, since
        // the table is sized from it.
        while self.pending.room() == 0 {
            self.wait_for_one()?;
        }

        let buf = self.stage(bytes);
        self.submit_write(
            Inflight {
                buf,
                page_addr,
                off,
                done: 0,
                len: bytes.len(),
                covers_upto,
            },
            true,
        )
    }

    /// Asks for everything handed over so far to be made durable.
    ///
    /// Returns immediately and usually without submitting anything. What it
    /// does is record the address the next sync has to cover;
    /// [`RingWriter::poll`] issues it when the writes it covers have landed.
    ///
    /// # Errors
    ///
    /// [`Code::Io`] for a failure an earlier completion found out about.
    pub fn sync(&mut self) -> Result<()> {
        self.take_failure()?;
        // Nothing has been handed over since the last sync covered everything,
        // so there is nothing for this one to do. Same reasoning as the
        // synchronous path: group commit runs off a timer and most ticks find an
        // idle shard.
        if self.written_upto == self.durable_upto
            && self.sync_wanted.is_none()
            && self.sync_inflight.is_none()
        {
            return Ok(());
        }
        let want = self.written_upto;
        self.sync_wanted = Some(self.sync_wanted.map_or(want, |w| w.max(want)));
        self.pump()
    }

    /// Walks the completions and moves the state machine on.
    ///
    /// Once a turn of the shard loop. This is the only place [`RingWriter::durable_upto`]
    /// moves.
    ///
    /// # Errors
    ///
    /// The first failure found since the last call.
    pub fn poll(&mut self) -> Result<()> {
        self.pump()?;
        self.take_failure()
    }

    /// Runs until everything handed over is durable.
    ///
    /// Not on the hot path. This is shutdown, and the checkpoint, and a test
    /// that wants to assert about a file rather than about a state machine.
    ///
    /// # Errors
    ///
    /// The first failure found on the way.
    pub fn drain(&mut self) -> Result<()> {
        self.settle();
        self.take_failure()
    }

    /// Waits until every write handed over has reached the file, and asks for
    /// nothing to be made durable.
    ///
    /// This is what a read needs. A page read back through `pread` sees the
    /// bytes as soon as the write completes, whether or not anything has been
    /// synced, because durability is a question about power cuts and not about
    /// what the next read returns.
    ///
    /// A failure is kept for the next [`RingWriter::poll`] rather than returned,
    /// because the caller of this is a read path that has nowhere to put one.
    pub fn quiesce(&mut self) {
        while self.writes_inflight > 0 {
            if let Err(e) = self.wait_for_one() {
                self.remember(e);
                return;
            }
        }
    }

    /// Runs the state machine to a stop, durability included, keeping any
    /// failure rather than returning it.
    fn settle(&mut self) {
        if self.written_upto > self.durable_upto {
            let want = self.written_upto;
            self.sync_wanted = Some(self.sync_wanted.map_or(want, |w| w.max(want)));
        }
        while !self.pending.is_empty() || self.sync_wanted.is_some() {
            if let Err(e) = self.wait_for_one() {
                self.remember(e);
                return;
            }
        }
    }

    /// A staging buffer holding a copy of `bytes`.
    fn stage(&mut self, bytes: &[u8]) -> usize {
        let i = match self.free.pop() {
            Some(i) => i,
            None => {
                self.bufs.push(Vec::new());
                self.bufs.len() - 1
            }
        };
        let b = &mut self.bufs[i];
        b.clear();
        b.extend_from_slice(bytes);
        i
    }

    /// Queues one write and books it in. `fresh` is false for the remainder of a
    /// short write, which is already counted.
    fn submit_write(&mut self, w: Inflight, fresh: bool) -> Result<()> {
        let page_addr = w.page_addr;
        let covers_upto = w.covers_upto;
        let off = w.off;
        let ptr = self.bufs[w.buf][w.done..].as_ptr();
        let len = u32::try_from(w.len - w.done).unwrap_or(u32::MAX);
        let token = self.pending.park(Kind::Write, w)?;
        // SAFETY: the bytes live in `self.bufs`, whose heap block is not freed
        // or reallocated until the completion returns the buffer to the free
        // list, and `self.file` is held for at least as long as this writer. The
        // `Drop` at the bottom of this file is what makes the second half true
        // even when the writer goes away with submissions still out.
        let r = unsafe { self.ring.write_at(raw(&self.file), ptr, len, off, token) };
        if let Err(e) = r {
            // It never reached the kernel, so the slot and the buffer come
            // straight back rather than waiting for a completion that is not
            // coming.
            if let Some(w) = self.pending.take(token) {
                self.free.push(w.buf);
            }
            return Err(e);
        }
        self.writes += 1;
        if fresh {
            self.writes_inflight += 1;
            *self.busy.entry(page_addr).or_insert(0) += 1;
            self.written_upto = self.written_upto.max(covers_upto);
        }
        Ok(())
    }

    /// Hands the queue over, drains what is there, and issues a deferred sync if
    /// the moment has come. Never blocks.
    fn pump(&mut self) -> Result<()> {
        self.ring.submit()?;
        self.reap();
        self.maybe_sync()?;
        Ok(())
    }

    /// Blocks until at least one submission comes back, then does what `pump`
    /// does. The only thing in here that waits.
    fn wait_for_one(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            // Nothing is out. If a sync is waiting for writes that have all
            // landed, this is what issues it, and otherwise there is genuinely
            // nothing to wait for and waiting would be forever.
            self.maybe_sync()?;
            if self.pending.is_empty() {
                return Ok(());
            }
        }
        self.waits += 1;
        self.ring.submit_and_wait(1)?;
        self.reap();
        self.maybe_sync()?;
        Ok(())
    }

    /// Walks every completion that has arrived.
    fn reap(&mut self) {
        // Collected first because resubmitting a short write from inside the
        // drain would want the ring twice over.
        let mut done: Vec<(yo_uring::Completion, Option<Inflight>)> = Vec::new();
        let pending = &mut self.pending;
        self.ring.drain(|c| {
            let state = pending.take(c.token);
            done.push((c, state));
        });
        for (c, state) in done {
            match c.kind() {
                Kind::Write => self.finish_write(&c, state),
                Kind::Fsync => self.finish_sync(&c),
                // Nothing else is ever submitted here, and a tag that says
                // otherwise arrived corrupted. Dropping it is the only thing
                // that does not make it worse.
                _ => {}
            }
        }
    }

    fn finish_write(&mut self, c: &yo_uring::Completion, state: Option<Inflight>) {
        let Some(mut w) = state else {
            // A stale tag, which the generation caught. There is nothing to
            // return and nothing to book in.
            return;
        };
        if let Some(e) = c.error() {
            self.retire_write(&w);
            self.free.push(w.buf);
            self.remember(e);
            return;
        }
        let n = c.bytes() as usize;
        w.done += n;
        w.off += n as u64;
        if w.done < w.len && n > 0 {
            // A short write. The rest goes out again on the same buffer, and
            // the page stays busy, so nothing else can get in front of it.
            if let Err(e) = self.submit_write(w, false) {
                self.remember(e);
            }
            return;
        }
        if w.done < w.len {
            // Zero bytes and no error, which means the device is not going to
            // finish this. Reporting it beats looping forever.
            self.retire_write(&w);
            self.free.push(w.buf);
            self.remember(
                Error::new(Code::Io, "a log page write stopped short").with_detail(format!(
                    "page_addr={} wrote={} of={}",
                    w.page_addr, w.done, w.len
                )),
            );
            return;
        }
        self.landed_upto = self.landed_upto.max(w.covers_upto);
        self.retire_write(&w);
        self.free.push(w.buf);
    }

    /// Books a write out, whether it worked or not.
    fn retire_write(&mut self, w: &Inflight) {
        self.writes_inflight = self.writes_inflight.saturating_sub(1);
        if let Some(n) = self.busy.get_mut(&w.page_addr) {
            *n -= 1;
            if *n == 0 {
                self.busy.remove(&w.page_addr);
            }
        }
    }

    fn finish_sync(&mut self, c: &yo_uring::Completion) {
        let covered = self.sync_inflight.take();
        if let Some(e) = c.error() {
            // Whatever it covered is not durable, and the address does not move.
            // The next sync will cover the same range again.
            if let Some(covered) = covered {
                self.sync_wanted = Some(self.sync_wanted.map_or(covered, |w| w.max(covered)));
            }
            self.remember(e);
            return;
        }
        if let Some(covered) = covered {
            // Monotonic by construction, and clamped anyway, because an address
            // that goes backwards unparks nobody and confuses everybody.
            self.durable_upto = self.durable_upto.max(covered);
        }
    }

    /// Issues the deferred sync if there is one and the writes it covers have
    /// all landed.
    fn maybe_sync(&mut self) -> Result<()> {
        if self.sync_inflight.is_some() || self.writes_inflight > 0 {
            return Ok(());
        }
        let Some(want) = self.sync_wanted else {
            return Ok(());
        };
        // A sync that landed while this one was waiting already covered it.
        // Issuing it anyway would be a device round trip for a range that is
        // already durable, which is the same waste the synchronous path skips on
        // an idle tick.
        if want <= self.durable_upto {
            self.sync_wanted = None;
            return Ok(());
        }
        // Everything the sync is meant to cover has actually reached the file.
        // If it has not, something failed, and syncing would claim durability
        // for bytes that are not there.
        if self.landed_upto < want {
            self.sync_wanted = None;
            return Ok(());
        }
        let token = self.pending.park(
            Kind::Fsync,
            Inflight {
                buf: usize::MAX,
                page_addr: u64::MAX,
                off: 0,
                done: 0,
                len: 0,
                covers_upto: want,
            },
        )?;
        self.ring.fsync_data(raw(&self.file), token)?;
        self.ring.submit()?;
        self.sync_wanted = None;
        self.sync_inflight = Some(want);
        self.syncs += 1;
        Ok(())
    }

    fn remember(&mut self, e: Error) {
        if self.failed.is_none() {
            self.failed = Some(e);
        }
    }

    fn take_failure(&mut self) -> Result<()> {
        match self.failed.take() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

impl std::fmt::Debug for RingWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RingWriter")
            .field("uring", &self.ring.is_uring())
            .field("in_flight", &self.pending.len())
            .field("written_upto", &self.written_upto)
            .field("durable_upto", &self.durable_upto)
            .field("buffers", &self.bufs.len())
            .finish()
    }
}

impl Drop for RingWriter {
    fn drop(&mut self) {
        // The kernel is reading buffers that are about to be freed. Leaving
        // without them back is a use after free in another process's timeline,
        // which is the worst kind to debug, so this waits however long it takes.
        while !self.pending.is_empty() {
            if self.ring.submit_and_wait(1).is_err() {
                // The ring is gone. Nothing further can arrive, and continuing
                // to wait would be a hang on the shutdown path.
                break;
            }
            let pending = &mut self.pending;
            self.ring.drain(|c| {
                pending.take(c.token);
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read as _;
    use std::path::PathBuf;

    struct Tmp(PathBuf);

    impl Tmp {
        fn new(name: &str) -> Tmp {
            let mut p = std::env::temp_dir();
            p.push(format!("yo-ring-{name}-{}.bin", std::process::id()));
            let _ = std::fs::remove_file(&p);
            Tmp(p)
        }

        fn create(&self) -> Arc<File> {
            Arc::new(
                std::fs::OpenOptions::new()
                    .create(true)
                    .truncate(true)
                    .read(true)
                    .write(true)
                    .open(&self.0)
                    .expect("a file in the temp directory"),
            )
        }

        fn contents(&self) -> Vec<u8> {
            let mut v = Vec::new();
            File::open(&self.0)
                .expect("the file")
                .read_to_end(&mut v)
                .expect("a read");
            v
        }
    }

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn writer(t: &Tmp) -> RingWriter {
        RingWriter::new(t.create(), &RingConfig::plain().with_entries(64)).expect("a ring")
    }

    #[cfg_attr(
        miri,
        ignore = "a real ring and a real file, neither of which Miri has"
    )]
    #[test]
    fn bytes_land_where_they_were_sent() {
        let t = Tmp::new("lands");
        let mut w = writer(&t);
        w.write(0, 0, b"hello", 5).unwrap();
        w.write(0, 5, b" ring", 10).unwrap();
        w.drain().unwrap();
        assert_eq!(t.contents(), b"hello ring");
    }

    /// The reason `sync` does not submit anything. A sync queued behind writes
    /// runs in parallel with them, so it has to wait until they are back, and
    /// the address only moves when the sync itself lands.
    #[cfg_attr(
        miri,
        ignore = "a real ring and a real file, neither of which Miri has"
    )]
    #[test]
    fn nothing_is_durable_until_the_sync_that_covers_it_comes_back() {
        let t = Tmp::new("durable");
        let mut w = writer(&t);
        assert_eq!(w.durable_upto(), 0);

        w.write(0, 0, b"aaaa", 100).unwrap();
        assert_eq!(w.written_upto(), 100);
        w.sync().unwrap();
        assert_eq!(w.durable_upto(), 0, "a sync that has not landed is not one");

        w.drain().unwrap();
        assert_eq!(w.durable_upto(), 100);
        assert_eq!(w.syncs(), 1);

        w.write(0, 4, b"bbbb", 200).unwrap();
        w.sync().unwrap();
        assert_eq!(
            w.durable_upto(),
            100,
            "`sync` moved the address by itself, which is `poll`'s job"
        );
        w.drain().unwrap();
        assert_eq!(w.durable_upto(), 200);
        assert_eq!(t.contents(), b"aaaabbbb");
    }

    #[cfg_attr(
        miri,
        ignore = "a real ring and a real file, neither of which Miri has"
    )]
    #[test]
    fn an_idle_tick_costs_nothing() {
        let t = Tmp::new("idle");
        let mut w = writer(&t);
        w.sync().unwrap();
        w.sync().unwrap();
        w.drain().unwrap();
        assert_eq!(w.syncs(), 0, "an empty log has nothing to sync");

        w.write(0, 0, b"x", 1).unwrap();
        w.drain().unwrap();
        assert_eq!(w.syncs(), 1);
        for _ in 0..10 {
            w.sync().unwrap();
            w.poll().unwrap();
        }
        assert_eq!(w.syncs(), 1, "ten idle ticks and no device touched");
    }

    /// The rule that makes this correct rather than merely fast. Two writes to
    /// one page cannot be out at once, or the older page header lands last.
    #[cfg_attr(
        miri,
        ignore = "a real ring and a real file, neither of which Miri has"
    )]
    #[test]
    fn two_writes_to_one_page_are_never_in_flight_together() {
        let t = Tmp::new("ordered");
        let mut w = writer(&t);
        for i in 0..32u64 {
            // Every one of these rewrites the first four bytes, which is what a
            // page header does on every flush.
            w.write(0, 0, &[i as u8; 4], i + 1).unwrap();
            assert!(
                w.in_flight() <= 1,
                "two writes to one page were out at once"
            );
        }
        w.drain().unwrap();
        assert_eq!(
            t.contents(),
            vec![31u8; 4],
            "the last write did not land last"
        );
    }

    /// Different pages are the case that is allowed to overlap, and the one the
    /// whole thing is for.
    #[cfg_attr(
        miri,
        ignore = "a real ring and a real file, neither of which Miri has"
    )]
    #[test]
    fn writes_to_different_pages_go_out_together() {
        let t = Tmp::new("parallel");
        let mut w = writer(&t);
        for i in 0..16u64 {
            w.write(i, i * 4, &[i as u8; 4], (i + 1) * 4).unwrap();
        }
        assert_eq!(w.in_flight(), 16, "the writes were serialised");
        assert_eq!(w.stalls(), 0, "somebody waited who did not have to");
        w.drain().unwrap();

        let got = t.contents();
        assert_eq!(got.len(), 64);
        for i in 0..16usize {
            assert_eq!(&got[i * 4..i * 4 + 4], &[i as u8; 4]);
        }
    }

    /// The buffer pool is the thing that keeps the copy from being an
    /// allocation. Sixteen writes at once need sixteen buffers, and the next
    /// sixteen need none.
    #[cfg_attr(
        miri,
        ignore = "a real ring and a real file, neither of which Miri has"
    )]
    #[test]
    fn staging_buffers_are_reused_rather_than_allocated() {
        let t = Tmp::new("pool");
        let mut w = writer(&t);
        let page = vec![9u8; 4096];
        for i in 0..16u64 {
            w.write(i, i * 4096, &page, (i + 1) * 4096).unwrap();
        }
        w.drain().unwrap();
        let high = w.buffers();
        assert!(high <= 16, "more buffers than writes in flight: {high}");

        for round in 1..8u64 {
            for i in 0..16u64 {
                w.write(i, i * 4096, &page, (round * 16 + i + 1) * 4096)
                    .unwrap();
            }
            w.drain().unwrap();
        }
        assert_eq!(w.buffers(), high, "the pool grew after it was warm");
    }

    /// Backpressure rather than an unbounded table. The ring is sixty four deep,
    /// so the sixty fifth write waits instead of queueing behind nothing.
    #[cfg_attr(
        miri,
        ignore = "a real ring and a real file, neither of which Miri has"
    )]
    #[test]
    fn a_full_ring_stalls_the_caller_instead_of_growing() {
        let t = Tmp::new("backpressure");
        let mut w = writer(&t);
        for i in 0..200u64 {
            w.write(i, i * 8, b"12345678", (i + 1) * 8).unwrap();
        }
        assert!(w.in_flight() <= 64, "more in flight than the ring is deep");
        assert!(
            w.stalls() > 0,
            "two hundred writes into a ring of sixty four never waited"
        );
        w.drain().unwrap();
        assert_eq!(t.contents().len(), 1600);
    }

    #[cfg_attr(
        miri,
        ignore = "a real ring and a real file, neither of which Miri has"
    )]
    #[test]
    fn a_writer_that_goes_away_with_submissions_out_waits_for_them() {
        let t = Tmp::new("drop");
        {
            let mut w = writer(&t);
            for i in 0..16u64 {
                w.write(i, i * 4, b"abcd", (i + 1) * 4).unwrap();
            }
            assert!(w.in_flight() > 0);
            // No drain, no sync. The `Drop` is what has to see this through.
        }
        assert_eq!(t.contents().len(), 64);
    }
}
