# Changelog

What each release changed, why, and what it costs you. The versioning rules and what a section has to contain are in [RELEASING.md](RELEASING.md).

While the major is 0, a minor release may break anything, including the on-disk format. The format is frozen at `M6`, not before.

## Unreleased

Nothing yet.

## 0.1.0 — 2026-08-25

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

### Known gaps

- **There is no qualified benchmark box.** server1, server2 and server3 are all below the kernel 6.12 io_uring floor and all oversubscribed. gamingpc clears the kernel floor but is a hybrid P and E core CPU inside a WSL2 VM sharing the machine with other work. Every number in this release says which it is. Rival comparisons start at `M2` and need a box that qualifies first.
- Nothing is stored. There is no file, no log, no records and no protocol. Those are `M1` and after.
