# Changelog

What each release changed, why, and what it costs you. The versioning rules and what a section has to contain are in [RELEASING.md](RELEASING.md).

While the major is 0, a minor release may break anything, including the on-disk format. The format is frozen at `M6`, not before.

## Unreleased

Nothing yet.

## 0.2.0 — 2026-08-28

**Milestone: M1, the record plane and the file.** M0 was a workspace and the pieces every milestone stands on. This is the first release that writes anything down. A `.yo` file, a hybrid log over it, replay, compaction, an independent reader that shares no code with the engine, a checker built on that reader, and a crash injection harness with an exact oracle.

### Added

- **`yo-format`.** The byte layouts and nothing that touches a file: superblock, checkpoint entry, page header, record framing, and the encode and decode for each. Both the engine and the independent reader are written against the same document rather than against each other, which is the only arrangement where a disagreement means something.
- **`yo-record`.** The hybrid log. Mutable, read only and stable regions over a ring of resident pages, per shard epochs, group commit with four durability modes, replay, and F2 style lookup based compaction with dead byte accounting deciding when it runs. `Log<S>` is not `Send`, so installing a record is a store rather than a compare and swap.
- **`yo-file`.** The `.yo` file. Two superblocks at 0 and 16384 with the root flip done as sync the data, write the slot that is not live, sync again. Log addresses count payload bytes rather than file bytes, so every page address is an exact multiple of the payload length and asking whether a segment is where it says it is is arithmetic rather than a guess.
- **`yo-reader`.** A second reader that shares no code with the engine. It is not a test double. It is the only thing that can catch the format drifting away from what is written down, and it is what makes a yanked release still readable.
- **`yo check`.** The tool you run when a database will not start. It separates a usage failure from a verdict, so a file that is not ours exits 2 and a file that is ours and damaged exits 1, and it never writes to the file it is checking.
- **`yo-crash`.** Crash injection with an exact oracle: lose everything, lose a prefix, reorder, tear at sector granularity, scatter, and flip a bit, against an image that only holds what a sync covered. Every damaged image is read twice, by recovery and by the independent reader, and compared record by record and on where the walk stops. Seeded with splitmix64, so a failure is a seed somebody can type in.
- **A `crash` CI job**, running a fifth of the gate across four shapes on fixed seeds. Fixed rather than drawn from the clock, because a job that picks its own seeds fails once a month on a seed nobody wrote down.

### Changed

- **The miri CI job runs its two borrow models as two jobs rather than one job doing both in sequence.** Miri had become the long pole on every pull request by a wide margin, over an hour while the other eleven jobs finished inside twenty minutes. Stacked borrows and tree borrows have nothing to say to each other, so a pull request now waits for the slower of the two rather than for their sum. On the first run under the split the two came in at 43 and 40 minutes side by side. The check name changes from `miri` to `miri (stacked borrows)` and `miri (tree borrows)`, and `fail-fast` is off so one model failing still leaves the other model's answer on the same run.

### Fixed

- **A `yo-format` test never finished under Miri.** Flipping every bit of a 16 KiB superblock slot and decoding after each one is sixteen thousand iterations, each copying and checksumming 16 KiB. Natively that is a few milliseconds. Interpreted it did not finish inside a CI job's six hour ceiling, and it took a nightly Miri run with it. Under Miri the walk is now a stride of 601 bytes plus nine offsets visited by hand, which is 37 seconds; every ordinary run still flips every byte.
- **The log handed the store the previous tenant of a page, and it was a silent corruption.** Writes go out a block at a time, so a flush part way through a page sends the end of log sentinel and then the rest of the block behind it, and those trailing bytes were whatever the previous tenant of the ring slot left there. The same block goes out again on the next flush with more records in it and nothing syncs in between, so a device may take one sector from the first write and the neighbouring sector from the second. The page then comes back with a used mark from the later write and a sentinel sector from the earlier one, and the earlier one has a stale record where the sentinel should be. It parses, because a record's checksum covers its own bytes and says nothing about the address they belong at, and replay walked into it and handed back a record that was never written there. The tail of the block past the sentinel is now zeroed on the way out, which costs under a block per flush. Found by `yo-crash` at seed 26281, 400 records into an 8192 byte page, after a hundred thousand trials at the default shape had run clean.
- **`Reader::records` capped its walk at the page header's `used` field.** A crash can take the header write and leave the record writes, and on that file the reader reported fewer records than the file holds while recovery found all of them. `used` now sizes the first read and nothing else, and the walk runs to the sentinel, which is what replay does.
- **The commit benchmark was measuring a tmpfs** and was wrong by four orders of magnitude. It now takes its directory from `YO_BENCH_DIR` and prints the device's `fdatasync` floor next to the result, so the two can be read together. Any commit rate published before this should be discarded.

### Format

The `.yo` layout is defined by this release and is not frozen. `MAGIC` is `tamndyo fmt001`, superblocks are 16 KiB at 0 and 16384, the data area starts at 32768, and a region is 32 MiB. A minor release may still change any of it before `M6`.

Not changed but worth writing down, because it is the shape of the bug above: a record cannot be tied to its address. Any future path that puts old bytes under a new address ends the same silent way. Mixing the address into the record checksum would close that off and it is a format change rather than a fix, so it is not in this release.

### Performance

Development measurements. Not one of the machines available meets the gate box definition in `bench/00` section 7, so nothing here is a gate number.

- Durable commits, group mode, on gamingpc, an i9-13900K under WSL2, release profile with fat LTO and one codegen unit: the `commit_batch` curve crosses 200 thousand per second at a batch of roughly 1024, on a device whose `fdatasync` costs 2.3 milliseconds. Development measurement, and the batch size and the device floor are part of the number rather than footnotes to it.

### Known gaps

- **The two performance rows of the M1 gate are open**, and the reason is the box rather than the code: 200 thousand or more durable commits per second in group mode on NVMe with SQPoll, and a 10 GB file opening in under 100 ms. Neither gamingpc nor server1, server2 or server3 qualifies under `bench/00` section 7. No row from those machines is a gate row.
- `yo-file` still bump allocates regions and never reuses a freed one.

## 0.1.1 — 2026-08-28

A CI release. Nothing in the library changed, and a `0.1.0` file, API and ABI are all still exactly what they were. What changed is how long you wait to find out whether a change is good.

### Changed

- **CI is split into a fast path and a deep path, and the fast path is held to three minutes.** `ci.yml` runs on every push and every pull request: style, the debug test matrix on Linux, macOS and Windows, the MSRV floor, loom on the lane, and the docs build. `deep.yml` runs nightly, on demand against any ref, and on a pull request carrying the `deep` label: Miri under both borrow models, the fuzzers, the release test matrix, the benchmark smoke run, and the cold build from source. The reason is that a check slower than a context switch is a check people learn to route around, and the slow half was reaching six hours while the fast half was done in ninety seconds. Nothing was deleted. A milestone pull request carries the `deep` label and waits for all of it, which is now written into RELEASING.md.
- **Debug info is off in CI and the cache is only written from main.** `CARGO_PROFILE_DEV_DEBUG=0` cuts what rustc emits and what the linker then has to chew through, which is worth about a third of the Windows wall clock and most of the cache size everywhere, and no CI job here reads a backtrace line number. `save-if` on the cache means a pull request reads main's cache and skips the upload, which was twenty to forty seconds a job spent saving something the next pull request would not have used.
- **The debug and release test matrices are separate.** The test job used to build the whole workspace twice. The release half moved to `deep.yml`. A bug that only shows up under optimisation is real, it is just not worth paying for on every commit.
- **The two Miri borrow models run as two jobs rather than one job doing both in sequence.** They have nothing to say to each other, so the wait is the slower of the two rather than their sum. The check name changes from `miri` to `miri (stacked borrows)` and `miri (tree borrows)`, and `fail-fast` is off so one model failing still leaves the other model's answer on the same run.
- **Every job has a timeout.** A Miri job was killed by the runner's own six hour ceiling after sitting on a single test, which is a long time to wait to be told nothing. Miri gets 120 minutes and everything else less.

### Known gaps

- Miri and the fuzzers no longer gate an ordinary pull request. A patch that introduces undefined behaviour can now land and be caught the following morning by the nightly rather than before the merge. That is the trade the three minute budget is bought with, and the `deep` label is the way to pay it back on a change that deserves it.

## 0.1.0 — 2026-08-26

**Milestone: M0, Skeleton.** The parts every later milestone stands on, built first and on their own so that a mistake in them is found now rather than underneath a hundred thousand lines. Nothing here is a database yet. It is a workspace, a hash, an allocator, a bucket, a shard runtime and the CI that keeps them honest.

### Added

- **Workspace.** Rust edition 2024, resolver 3, stable 1.98.0, MSRV 1.94. The MSRV is held about six months behind stable and is a CI job, not a note in a file.
- **`yo-common`.** Hashing, log addresses, Redis slot arithmetic, and the error type. Errors are generated at build time from `errors.toml`, so a code, its message and its documentation cannot drift apart: there is one place to change and three outputs.
- **`yo-alloc`.** A global allocator that aborts. Not a fallback and not a measurement helper: a test that runs the hot path under it either does not allocate or does not finish. The claim that the statement path is allocation free is worth nothing unless something enforces it.
- **`yo-arena`.** Bump allocation in 2 MiB segments, with `MADV_HUGEPAGE` on Linux. One pointer add per allocation and a whole segment released at once.
- **`yo-index`.** The 64 byte bucket, which is one cache line, holding a word of tags matched with SWAR so a probe touches one line and compares eight tags in a few instructions. Growth is a dashtable split, so it is per bucket rather than a stop the world rehash of everything.
- **`yo-shard`.** One thread per core, one owner per thing. `ShardLocal<T>` is neither `Send` nor `Sync` by construction, which is the property that removes the locks rather than a comment asking people not to share.
- **Fuzz targets** for the bucket, the arena, the map and the lane.
- **CI, nine jobs**, and the ones that matter are the slow ones: Miri over every unsafe block under both stacked and tree borrows, loom on the lane, cargo-fuzz, a three OS test matrix in debug and release, the MSRV floor, a docs build with warnings denied, a benchmark smoke run, and a build from a source tarball with no git history.

### Performance

Development measurements, all of them. Not one of the machines available meets the gate box definition in `bench/00` section 7, so nothing here is a gate number and none of it should be quoted as one. Taken on gamingpc, an i9-13900K inside a WSL2 VM, pinned to core 0 with `taskset`, release profile with fat LTO and one codegen unit.

- Bucket tag match: **0.80 ns** on a hit, **0.53 ns** on a miss. The M0 gate is 4 ns or under, so this clears it with room, on a box that does not qualify.
- Arena bump allocation: **1.23 to 1.27 ns**, against **28.2 to 30.9 ns** for the system allocator on the same shapes. The gate is 2 ns or under.
- Hashing, wyhash against fnv1a, by input length: 3.08 against 2.32 ns at 8 bytes, 4.18 against 25.6 ns at 64 bytes, 12.1 against 153 ns at 256 bytes. fnv1a wins at 8 bytes and loses by an order of magnitude everywhere else, which is why wyhash is the default.
- Map get, hit, at a thousand keys: **12.3 ns**. This one is in cache and is **not** comparable to aki's 46.5 ns at a million keys. The comparable row needs a quiet machine with a working set that does not fit in L2, and it is not being reported until there is one.

### Fixed

- The shard benchmark was measuring `format!` rather than the map. Any shard number published before this should be discarded.
- Undefined behaviour in `Bucket::tag_word`, found by Miri, fixed by taking the pointer from the bucket rather than from the tag array.
- Two build failures that were invisible on the arm dev machine and only showed up on x86: an `unused_unsafe` warning around the crc32 intrinsics, and the `chunks_exact` to `as_chunks` migration. Every commit is now built on x86 before it lands.
- Windows CI was failing on line endings. Fixed with `.gitattributes` and a normalizer in the tooling rather than by turning the check off.
- A `yo-shard` test counted jobs across four lanes through a fifth one and expected an exact total. Ordering is per lane and a shard polls its lanes in turn, so jobs sent on one lane can still be queued when a call on another lane comes back, and the count was short. It passed on a quiet machine for weeks and failed once on a loaded macOS runner. The test now waits on each lane it used. Nothing in the runtime changed: no job was ever lost, `send` spins until the lane takes it.

### Known gaps

- **There is no qualified benchmark box.** server1, server2 and server3 are all below the kernel 6.12 io_uring floor and all oversubscribed. gamingpc clears the kernel floor but is a hybrid P and E core CPU inside a WSL2 VM sharing the machine with other work. Every number in this release says which it is. Rival comparisons start at `M2` and need a box that qualifies first.
- Nothing is stored. There is no file, no log, no records and no protocol. Those are `M1` and after.
