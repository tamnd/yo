# Changelog

What each release changed, why, and what it costs you. The versioning rules and what a section has to contain are in [RELEASING.md](RELEASING.md).

While the major is 0, a minor release may break anything, including the on-disk format. The format is frozen at `M6`, not before.

## Unreleased

### Changed

- **The executable is `yodb`, not `yo`.** `yo` on PATH is Yeoman's and has been for over a decade, and it is Yeoman's on Homebrew core, Chocolatey and the AUR, so shipping a binary called `yo` means shipping a collision into every channel we would want to publish through. Only the executable moves. The crate is `yodb`, the library is `libyo`, the header is `yo.h`, the symbols are `yo_*` and the file extension is still `.yo`.

### Added

- **`yo-kv`, where the data structures actually live.** One method per Redis command, taking and returning ordinary Rust values, with no protocol anywhere near it. The wire calls into this and the embedded API calls into this, so there is one `INCR` rather than two that drift apart (Y23). An embedded program calls `Strings::incr` and gets back an `i64` or a `yo_common::Error`, without serialising a command, crossing a socket or parsing a reply.
- **The string type, which is the first row of M2.** All 26 commands: `SET` with `NX`, `XX`, `GET`, `KEEPTTL`, `EX`, `PX`, `EXAT`, `PXAT` and Redis 8.4's `IFEQ`, `IFNE`, `IFDEQ` and `IFDNE`, then `GET`, `GETSET`, `GETDEL`, `GETEX`, `SETNX`, `SETEX`, `PSETEX`, `MSET`, `MSETNX`, `MGET`, `APPEND`, `STRLEN`, `SETRANGE`, `GETRANGE`, `SUBSTR`, `INCR`, `DECR`, `INCRBY`, `DECRBY`, `INCRBYFLOAT`, `LCS`, and Redis 8.4's `MSETEX`, `DELEX` and `DIGEST` and 8.8's `INCREX`.
- **`LCS`, ported rather than reimplemented.** The backtrack is Redis's, quirks included, so the ranges `LCS IDX` reports land on the same bytes in the same order, `MINMATCHLEN` filters the same runs, and `len` is still the length of the whole subsequence and not the sum of what survived the filter. The documented `ohmytext` and `mynewtext` example comes out byte for byte.
- **`INCREX`, which is a counter with a policy attached.** Redis 8.8 put four ideas in one command: add an integer or a float, refuse or clamp a result that leaves a range, and set, keep, clear or conditionally set the deadline, all in the round trip that used to be `INCR` followed by `EXPIRE`. The reply is the value and the amount applied, so a client can tell a refused increment from one that added nothing without comparing against a value it did not have, and a refused increment does not create the key. The windowed rate limiter is one setting of it, `EX window ENX`, which starts the window only when there was no deadline, so a burst inside one window expires together rather than each call pushing the deadline out.
- **`DELEX`, compare and delete.** The other half of `SET ... IFEQ`, and the read modify write nobody was doing correctly: a client that reads a value, decides it is stale and deletes it can be beaten to the key in between, and `WATCH` plus `MULTI` costs a round trip to avoid that. It carries the same four conditions `SET` does, `IFEQ`, `IFNE`, `IFDEQ` and `IFDNE`.
- **`DIGEST`, and an XXH3 that is bit for bit Redis's.** Redis 8.4 added `DIGEST` so the digest conditions are worth using: a client holding a megabyte value sends sixteen hex characters instead of the megabyte. That only works if our number is the same number, so the implementation uses the real XXH3 secret rather than a hash that is merely good, and it is checked against digests read out of a running Redis at every length the algorithm branches on, which is both sides of 16, 128, 240 and the 1024 byte block.
- **A one byte header in front of every string.** Encoding in two bits, a deadline flag in a third, and eight bytes of deadline only for the keys that have one, which most do not. The alternative is a side table for the TTL, and a side table is a second cache miss on a path whose whole budget is one miss.
- **`INCR` that does not touch the arena.** An int encoded value stores the eight bytes of the integer rather than its digits, so incrementing it is a probe, an add and an eight byte store back into the record the probe landed on. No allocation, no free, no second record and no rehash. There is a test that increments a key a thousand times and asserts the arena's live byte count did not move.
- **`RawMap::set_with` and `RawMap::value_mut`.** The first lets a caller write straight into the record instead of building the value in a scratch buffer and having it copied in again, which is one memcpy per `SET` rather than two. The second is what makes the `INCR` path above possible.
- **`OBJECT ENCODING` that matches Redis, including the parts that look like bugs.** `SET k 007` is `embstr` and gives back `007`, `SET k 42` is `int`, the `embstr` boundary is 44 bytes, `APPEND` onto an existing key leaves `raw` even when the result reads as a number, and `APPEND` onto a key that is not there does not, because Redis runs the new value through `tryObjectEncoding` on create.
- **`commands.toml`, the command audit as a file rather than a document.** Every command carries a storage plan, a bound or materialise verdict, the Redis it came from, its arity, and how far up the stack it reaches. `cargo xtask check` enforces the rule from `12` section 3 that a command with no storage plan does not ship, and it enforces the group budgets too, so a group that claims 25 commands and holds 24 fails the same job a stale generated header fails. A document that says the same thing goes stale and nobody notices.
- **`divergences.toml`, the register of every place yo is knowingly not Redis.** Ten rows today, each one saying what diverges, why, and whether it is the design or goes away at a named milestone. A command marked divergent has to name a row that exists, and the check fails if it does not, so a divergence cannot be added in one commit and forgotten in the next. A compatibility claim with unlisted exceptions is how every Redis compatible product loses its users' trust exactly once.
- **`MSETEX`, `DELEX` and `INCREX`, checked against a running Redis 8.8 rather than written from memory.** The register shipped them marked `wire = "unverified"` because writing an argument order down from memory would make it look verified simply because it is in the code. Asking a real server settled it, and all three were wrong in substance and not only in argument order. `MSETEX` takes a key count and its condition is over the whole set, so `NX` needs every key missing and a partial match writes nothing. `DELEX` has four conditions rather than one. `INCREX` is the counter above and not the rate limiter it looked like. The string group is 26 commands and not 25, because `DIGEST` was missing.
- **A clock that only moves when it is told to.** `04` section 5 says there is no global clock read on the data path, so the shard reads the clock once per turn of the loop and all 64 commands in that batch compare against the same number. A clock read is tens of nanoseconds against a budget of a hundred and fifty, and paying it per command means paying it 64 times for an answer that did not change. Every expiry test in the crate drives a fixed clock instead of sleeping.
- **`yo-reactor`, the shard loop.** `04` section 2 is nine lines of pseudocode and this is those nine lines with the bookkeeping filled in. Six stages in one order forever: submit, drain a batch, enter the epoch, walk the batch twice, leave and flush, then completions and a bounded maintenance slice. There is no executor under it, no future, no waker and no work stealing. The loop is the scheduler.
- **The two walk batch.** A batch is 64 commands, and it is walked once to hash every key and ask the cache for the bucket that hash selects, then once to run them. Valkey and Redis 8.4 both use a window of 16 because they hold a lock across it and other threads can invalidate a line inside it, and Y1 removes both of those reasons. On a ten million key map the second walk is worth 13 percent on aarch64 and 20 percent on x86-64, and on a hundred thousand key map it is worth nothing on either because there was never a miss to hide.
- **A break that does not throw work away.** A command whose key set is not known until something earlier in the batch has finished, which is a `MULTI` body, a `WAIT` or a blocking form, ends the batch where it stands. What was already drained behind it waits for the next turn rather than being executed cold or dropped.
- **Inline execution, which is the embedded path.** A caller who is already on the shard thread calls `execute` or `execute_all` and gets the same prefetch and the same dispatch the server path gets, so a command has one implementation rather than two that drift. Y23 asks for the same code and not the same idea.
- **A maintenance budget in instructions rather than in wall clock.** Reading a clock is a syscall or at best a serialising instruction, and a wall clock slice does a different amount of work on a busy machine than on an idle one, so the tail it produces cannot be reproduced. A unit budget is a subtraction, and a pass that overshoots does so by one item instead of by however long that item took.
- **A software prefetch hint in `yo-common`.** One instruction on x86-64 and one on aarch64, nothing at all anywhere else, and nothing under Miri. It reads no memory and has no effect a program can observe, which is why the tests for it can only check that.
- **`Index::prefetch` and `RawMap::{hash_of, prefetch, get_hashed}`.** The first walk hashes a key and warms its bucket, the second walk looks it up with the hash it already has. A key in a batch is hashed once rather than once per walk.
- **`yo-shape`, the shape tag.** A C ABI erases types and a file outlives every process that opens it, so a Python program has no way to find out that the collection it just opened as a map of strings was written as a map of integers by a Go program in March. The tag is what finds out: a canonical byte description of the element type, in a grammar no language owns, hashed to 128 bits. Same type, same tag, in every language and on every machine. Field order is part of it, because order is layout, and so are integer widths, because "int" is a different number of bytes in every language that will open the file.
- **A shape mismatch that tells you what moved.** Comparing two tags answers "the same or not", which is the worst message a database can give: the reader knows something changed and nothing about what. So the description is stored beside the tag and a mismatch renders both shapes, underlines the part that differs, names the difference in a sentence, and says whether it is additive or breaking. Adding a variant to an enum reads as `enum Status gained variant "Cancelled" at position 3` with `change=additive` in the error's detail, and swapping two fields reads as a reorder rather than as two renames.
- **BLAKE3 in `yo-common`, written rather than depended on.** The same call the other three hashes in that crate got: the tag is a number six bindings have to agree on forever, so the algorithm is part of the format and belongs next to `crc32c` and `xxh3`. The published crate would also pull a C toolchain into every build in the workspace for a hash that runs once when a collection is opened. All 35 of the BLAKE3 team's own test vectors pass, from the empty input to a hundred kilobytes, which is what pins the chunk chaining and the tree.
- **`yodb`, the embedded API, which is the crate a person actually types.** Two lines get you a database and there is no third line: `yo::open(yo::MEMORY)` then `db.map::<String, u64>("hits")`. No server to start, no connection string, no pool, no schema to declare somewhere else first. `Map<K, V>` has `get`, `with`, `set`, `del`, `contains`, `len` and `is_empty`, and a lookup takes a borrowed key the way `HashMap` does, so a `Map<String, u64>` is read with `map.get("home")` and not with a `String` built for the length of one call. The package on crates.io is `yodb` because `yo` is taken, and the library inside it is `yo`, so `cargo add yodb` is followed by `use yo::open`.
- **The type parameters are the schema, and they are checked.** Opening a name a second time with a different type is refused before a single byte is read as the wrong thing, with the message from `yo-shape` saying what moved and whether the change is additive or breaking. That is what a query language usually buys you with a runtime error in production, bought here with a compile error on a laptop and one check when the collection is opened.
- **Zero copy per call rather than per database.** `get` hands back an owned value because that is what most code wants, and `with` hands the bytes over where they lie, which allocates nothing. Both read the same collection and the caller picks per call (Y29). On an Apple M4 a `with` on a million key map is 11.9 ns against the raw map's 10.5, so the shape, the handle and the decode together cost about 1.4 ns, and a `get` of a `u64` is 11.8 ns because it decodes out of the arena rather than copying out and decoding after.
- **A database that returns errors where it could have panicked.** A closure passed to `with` that calls back into the same database gets a sentence telling it to read first and write after the closure returns, rather than a `RefCell` panic from inside a dependency. A record larger than a segment is `YO_ERR_FULL` naming the limit, rather than an assert further down.
- **The Redis keyspace, reached without the socket.** `db.strings()` is the string keyspace and every method on it calls the same `yo-kv` method a command off the wire calls, so `SET`, `GET`, `INCR`, `MSET`, `GETSET`, `GETDEL`, `SETNX`, `APPEND`, `STRLEN` and the rest are one implementation with two doors rather than an embedded copy that agrees with the server for now (Y23). Keys are byte strings the way Redis's are, so a `&str`, a `String`, a `&[u8]` and a `Vec<u8>` all work and none of them are converted on the way in.
- **Inline execution mode, which is where the point read budget is actually spent.** The calling thread is the shard, so a command is a function call with no queue, no message and no wakeup (`15` section 7). On an Apple M4 the embedded path costs almost exactly what the store costs on its own: `GET` through `with` is 27.0 ns against the store's 24.5, `SET` is 36.3 against 33.9, `INCR` is 10.1 against 10.8 and an eight pair `MSET` is 162 ns against 156, so the handle, the borrow and the `AsRef` together are worth two or three nanoseconds against a budget of a hundred and fifty. `get` costs 54.7 ns because it hands back an owned `Vec`, which is the copy `with` is there to avoid.
- **A clock that stays off the data path even with no loop to refresh it in.** `04` section 5 keeps the clock read out of the command path and the shard loop refreshes it once per turn, but inline mode has no loop, and reading the clock per call costs 10 ns on top of an 18 ns `GET`. So the clock is read only when its answer can be observed: a keyspace that has never been given a deadline cannot hold an expired key, and the first call that sets one turns the reads on for good. `Db::reads_the_clock` says which state a database is in, and there is a benchmark for both.
- **`db.counter("hits")`, the handle that holds the key.** Counting is the commonest thing a string key is used for and spelling the key at every call site is one more place for a typo, so `Counter` has `get`, `incr`, `decr`, `add`, `set` and `reset` and reads as zero until something writes it, the way Redis's does. It is the same key the keyspace and a Redis client see, and `set` writes the digits without allocating.

- **Building an error message would have aborted the server.** An error carries a `String`, a shard thread that allocates aborts, and every error message in `yo-kv` is built on a shard thread, so the first client to send `INCR` at a key holding a word would have taken the process down with it. It has not happened yet only because the reactor is not wired to the string type, which is the next pull request. Constructing an error is now allowed to touch the heap, since an error is by definition off the path the hundred and fifty nanosecond budget is about, and `Error::fmt` exists for the messages that need formatting because `format!` allocates before the constructor ever sees it.
- **`INCREX` obeyed a range it should have refused.** `LBOUND 10 UBOUND 5` is a range nothing fits in, and refusing every increment forever is a hard bug to find in a caller that has its arguments the wrong way round. A real 8.8 answers `ERR LBOUND can't be greater than UBOUND` and now so does this.
- **`INCRBYFLOAT` stored `raw` where Redis stores `embstr`.** A float counter that lands on a short result gets `tryObjectEncoding` run over it on the way in, the same as `SET` does, so `SET k 5` followed by `INCRBYFLOAT k 1` leaves `embstr` and not `raw`. `APPEND` and `SETRANGE` really do leave `raw` even for four bytes, which is what made the wrong behaviour look right. The old test asserted the bug, so it was locked in until `INCREX BYFLOAT` was written from what a real server reports and the two disagreed.
- **`parse_i64` accepted `-0`, which Redis does not.** Redis's `string2ll` tests its zero case against the length of the whole string, so the minus sign pushes `-0` past it and into the one to nine gate, which it fails. That is not only a parsing detail, because the same rules decide whether a string is stored int encoded. Left alone, `SET k -0` would have been stored as the integer zero and `GET k` would have handed back `0` for a value the client wrote as `-0`.

### Known divergences

Four new ones, all in the string type, all listed here rather than left to be discovered. The full register, including the seven that came out of the spec, is `divergences.toml` at the root.

- A string is capped just under 2 MiB rather than at Redis's 512 MiB, because a value lives in one arena segment. The band above that is the log region in `06` section 2 and lands with tiering in M5, at which point the cap goes up to Redis's. A value past the cap is a `YO_ERR_FULL` carrying Redis's own message and not a panic.
- Expiry is lazy only. A key past its deadline is dropped when something touches it, and it is dropped at the exact millisecond, since a deadline equal to now has passed. The active cycle that would reclaim a key nobody ever touches again is maintenance slice work and lands in M5.
- `LCS` refuses a table over 64 million cells, which is 256 MiB of counters and two strings of about eight thousand bytes each. Redis has no explicit limit and fails on the allocation instead, which on a server that has overcommitted is a kill rather than an error, and `LCS` on two large strings is the easiest accidental denial of service in the string group.
- The float counters count in `f64` where Redis counts in the C `long double`, which is eighty bit on x86-64 and a hundred and twenty eight bit on aarch64. Redis therefore does not agree with itself across machines and we agree with ourselves everywhere. A value that needs more than seventeen significant digits differs and nothing else does.

### Internal

- **`MSET`, `MSETNX` and `MSETEX` take an iterator of pairs rather than a slice.** The wire layer has its pairs as positions in the connection's read buffer, so a slice would mean collecting them into a `Vec` before the call. `MSET` is one of the four commands M2 is measured on and a shard thread that allocates aborts, so an API that forces an allocation to call it is the wrong API. The iterator is walked twice, which is why it has to be `Clone`, and an iterator over borrowed slices is two words to copy.
- **`parse_f64` moved from `yo-kv` to `yo-common`, next to `parse_i64`.** Same reason the integer one moved out of `yo-resp`: whether an argument is a float and whether a stored value is a float are the same question, and the answer should not live in a layer the other one cannot reach.
- **`num.rs` moved from `yo-resp` to `yo-common`.** Whether a bulk length parses and whether a string is stored int encoded are the same `string2ll` question, and the answer should not live in the wire layer where the storage layer cannot reach it without depending on the protocol. `yo_resp::num` still resolves and still means what it meant.

### Performance

Release profile with fat LTO, criterion at 20 samples. The map is `RawMap` with 32 byte values and the commands are lookups, so these are loop costs and not command costs.

M4 MacBook Pro, aarch64:

| what | 100 thousand keys | 10 million keys |
| --- | --- | --- |
| `execute`, one command inline | 12.3 ns | 17.3 ns |
| `execute_all`, batch of 64, prefetch on | 11.5 ns per command | 15.7 ns per command |
| `execute_all`, batch of 64, prefetch off | 11.8 ns per command | 18.1 ns per command |
| `tick`, the same batch through the lane and the epoch | 17.8 ns per command | 21.7 ns per command |

gamingpc, x86-64 on Windows with the gnu toolchain:

| what | 100 thousand keys | 10 million keys |
| --- | --- | --- |
| `execute`, one command inline | 31.1 ns | 53.6 ns |
| `execute_all`, batch of 64, prefetch on | 28.4 ns per command | 38.9 ns per command |
| `execute_all`, batch of 64, prefetch off | 23.5 ns per command | 48.5 ns per command |
| `tick`, the same batch through the lane and the epoch | 35.3 ns per command | 51.8 ns per command |

The gap between the last two rows is what the server path pays over the embedded one, and it includes the 64 pushes into the intake lane, which on a real server somebody else does on another thread. The hundred thousand key row on Windows has the cold arm apparently ahead of the warm one, and that is noise rather than a finding, since the two intervals overlap across most of their range.

## 0.3.1 — 2026-08-28

Four pull requests and no milestone. `yo-resp` and `yo-uring` are M2 pieces, the fuzz and crash work and the ring wiring are M1 evidence, and none of the four closes the milestone it belongs to, so this is a patch. The on-disk format is untouched.

### Added

- **`yo-resp`, the RESP2 and RESP3 codec.** The first piece of M2. Requests decode into ranges over the connection's own read buffer, so a command sees the bytes the kernel delivered and nothing is copied on the way in. Replies are written straight out as wire bytes with no intermediate value, which is Y18. Multibulk and inline requests, every RESP2 and RESP3 type, Redis's own limits, and Redis's own protocol error text character for character.
- **One reply path for both protocols.** A command writes a map, a set or a push and the RESP2 downgrade happens in the codec rather than in the command. That is one place to test instead of three hundred, and it is what stops a command from working on one protocol and not the other.
- **A resume state on the request decoder.** A value that arrives in ten thousand pieces is scanned once rather than ten thousand times. Without it a slow link makes the parser quadratic in the size of the value, which is a real denial of service on a real network rather than a theoretical one.
- **A protocol fuzz target.** `12` section 11 point 3. It checks that no input panics or allocates from a count off the wire, that a decode reports exactly what it consumed, and that feeding the same bytes one at a time reaches the same verdict as feeding them all at once.
- **A file fuzz target, which is the M1 exit gate the project did not have.** The gate is that the independent reader agrees with the engine on every file the fuzzer produces, and until now there was no fuzzer producing files. It builds a real `.yo` file from fuzzer chosen record shapes across up to four shards, walks it with the engine's own replay and with the independent reader, and compares both against what went in, then compares the superblock and the checkpoint entries field for field between the two sides.
- **The hundred thousand fault crash gate, as a nightly job.** Every push runs twenty thousand faults, which catches a regression and fits the three minute budget. The gate number is a hundred thousand and now runs in `deep` across five shapes, so the number in the milestone is a number CI produces rather than one somebody ran once.
- **`yo-uring`, the per shard submission ring.** The second piece of M2 and the thing the last open M1 exit gate row is waiting on, because that row asks for SQPoll and SQPoll is io_uring. One ring per shard, four thousand and ninety six entries, storage and network submissions on the same ring told apart by the user data tag, which is `04` section 7 exactly.
- **Parked state instead of an async runtime.** A submission that has not finished has its caller's state in a slot, and the next turn of the loop picks it up when the completion arrives. There is no future here, no waker, no executor and no `.await`, which is what `16` section 0 means when it disqualifies anything with an executor in it. The VLDB 2026 io_uring ladder is 16.5 thousand transactions a second when every submission is waited on and 183 thousand when the execution is restructured this way, so the parking is the design rather than an optimisation.
- **A tag that makes a stale completion detectable.** Eight bits of kind, twenty four of slot, thirty two of generation. The kind routes storage against network with no lookup. The generation is the correctness part: a slot is reused the moment it is freed, so without it the completion for a cancelled read lands on an unrelated connection and nobody ever finds out why.
- **The same state machine on macOS and Windows, not a gate and not a stub.** `04` section 7 asks for synchronous storage off Linux with the shape kept, so that is what is there. `Features::is_uring` says which of the two produced a row, because a benchmark number that does not say which kernel path it took is a number that will eventually be wrong.
- **The log writes through the ring.** `LogFile::use_ring` puts a shard's log on the submission ring, and from there the shard hands bytes over and keeps going rather than stopping for a `pwrite` and an `fdatasync`. This is the piece the 200 thousand durable commits a second row in M1 has been waiting on: `yo-uring` existed, nothing used it, and a ring nothing writes through is not a measurement.
- **A staging buffer pool, because a borrow cannot be handed to the kernel.** `PageWrite<'_>` gives its borrow up when the call returns and the kernel reads the buffer long after that, so the bytes are copied into a buffer the writer owns and the buffer goes back on the free list when the completion arrives. The pool stops growing once it is as deep as the writes in flight, and the copy is a memcpy against a syscall.
- **At most one write in flight per page.** io_uring does not order two submissions against each other, and every flush of a page rewrites that page's header, so two in flight at once could land with the older header last and leave a page claiming fewer records than it has. A second write to a page waits for the first. In group mode this never waits, because the sync boundary has already drained everything.
- **A sync that waits for the writes it covers.** A queued `fsync` runs in parallel with the writes queued ahead of it, so `sync` records the address it has to cover and returns, and the `fsync` goes out from `poll` once the last write has landed. That is the most common way to get an io_uring durability bug and it is why `durable_upto` moves in exactly one place.
- **`PageSink::poll` and `Log::poll`.** Where an asynchronous sink runs its state machine, once a turn of the shard loop, and therefore where a caller parked on `CommitAction::WaitFor` stops being parked. A synchronous sink takes the default and does nothing.
- **A `commit_ring` benchmark row.** The synchronous path against the ring, in group mode, at batches of 64, 512 and 4096, waiting for durability on both sides so the two columns are the same measurement. It prints whether the ring was really io_uring, because off Linux it is the portable backend and that number is not a gate row.

### Changed

- **`sync` durability mode parks the caller when the sink is asynchronous.** It used to sync and reply, which is right when the sync has finished by the time the call returns and is a reply before the fsync when it has not. It now replies immediately if the sink is already there and otherwise waits on the address, exactly like group mode. Same guarantee, reached later, and no change at all for a synchronous sink.
- **`os` durability mode submits at a page boundary.** That mode promises the operating system has the bytes, and with a ring under it the bytes sit in the submission queue until something submits them. A page boundary is once per 32 MiB, so the call costs nothing anybody can measure.

### Fixed

- **Two hundred megabytes of build artifacts were committed to the repository.** The fuzz directory is excluded from the workspace and has a build directory of its own, and `.gitignore` said `/target`, which is anchored to the root and did not cover it. 895 object files were tracked. They are untracked now and the pattern covers any depth. The history has been rewritten to drop them as well, so a fresh clone is 824 KB rather than 53 MB. Every file in every commit is byte for byte what it was, only the commit hashes moved, and the four release tags were rewritten and force pushed along with `main`. A commit hash written down before 2026-08-28 will not resolve, which is the price of the rewrite and is worth it once.
- **Two ring tests passed everywhere except on the platform they were about.** They polled once after a sync and asserted durability had moved, which is true of the portable backend, where the write and the fsync both happen inside the call that submits them, and false of io_uring, where the first poll picks the write completion up and only then lets the fsync go out. They now turn the loop until the address moves, with a cap so a real stall is a panic rather than a hang. Cross compiling had said the Linux path was fine and running it said otherwise, which is the whole argument for running on the machines rather than checking against them.

### Performance

Both tables are development measurements and neither is a gate number. Group commit, both columns waiting for durability so they are the same measurement, `commit_ring` at three batch sizes, throughput in commits a second.

macOS aarch64 on APFS, the portable backend rather than io_uring: 17.1 thousand against 17.8 with the ring at a batch of 64, 131.3 against 129.6 at 512, and 866.6 against 835.1 at 4096. The ring column pays for a memcpy and some bookkeeping and gets no asynchrony back, because off Linux there is none to get, so being a few percent behind is the expected result and not a regression.

Linux x86_64 on server3, kernel 6.8, ext4 on a virtio disk, real io_uring: 4.9 thousand against 3.9 at 64, 47.9 against 22.9 at 512, and 132.4 against 142.0 at 4096. The 512 row was re-run on its own with more samples and came back the other way at 37.0 against 42.1, so that gap was the box. Error bars on this machine are around plus or minus 30 percent, which is what a shared VPS on a QEMU disk gives, and at that spread the two columns cannot be told apart in either direction.

### Known gaps

- The 200 thousand durable commits a second row in M1 is still open, and as of this release the reason is hardware rather than missing code. The path runs end to end on real io_uring and lands the same bytes as the synchronous path, but the row asks for NVMe with SQPoll on the box described in `bench/00` section 7, and none of the machines available is that box.
- Registered buffers are unimplemented and `Features::registered_buffers` reports false. The staging pool is what they will attach to.
- `IORING_OP_URING_CMD` is deliberately out of scope for v1, per `04` section 7.

## 0.3.0 — 2026-08-28

**Milestone: DX0, the ABI contract.** No engine work. This is the surface every binding will sit on, described once and generated from that description, so that the header, the error codes and the machine readable model cannot drift apart. RELEASING.md gives a minor to every engine or DX milestone, and this is the first DX one.

### Added

- **`yo-capi`, the C ABI, and `include/yo.h` generated from one model.** Every binding this project will ever have sits on this surface, so it is described once in `xtask/src/model.rs` and `crates/yo-common/errors.toml` and the header, the error code spellings and the machine readable `api.model.json` are all emitted from it. A generated file that somebody edited by hand is a header that no longer matches the library, so `cargo xtask check` regenerates all of it and CI fails on a diff.
- **A `capi` CI job that compiles the header the way a caller would.** Two compilers, gcc and clang, every dialect they support, warnings as errors, and then the example is linked and run. A header that compiles and a library that works are two different claims and this checks both.
- **`examples/c`, a hello world and the build script that produces it.** Not a snippet in a README. Something a caller can run, which is the only version of an example that stays true.
- **`cargo xtask` as an alias, so the command is the same everywhere.** In CI, in the contributing guide and in muscle memory. `cargo run -p xtask --` in one place and `cargo xtask` in another is how a documented command stops matching the one people type.

### Fixed

- **The fuzz job built against the wrong target and stopped compiling.** `cargo fuzz build` defaults `--target` to the platform the cargo-fuzz binary itself was built for. install-action does not carry cargo-fuzz and falls back to cargo-binstall, which fetches the musl build, so on a gnu runner the default came out as `x86_64-unknown-linux-musl` and the build died on a missing musl std with a sanitizer error on top. The target is now whatever `rustc -vV` reports as the host, which stays right however the toolchain was assembled.
- **A workflow file that does not parse produced a run with no jobs in it.** A failure with nothing to click on, no log, and no indication of which file broke, which is the worst failure mode this CI has because the thing that broke is invisible in the check list. One unquoted colon in a `run:` line was enough to do it. The style job now parses every workflow file before anything else runs.

### Known gaps

- The ABI is at `YO_ABI_VERSION_MINOR` 1 and nothing depends on it yet. It is a contract with no callers, which is the point of doing it before the callers exist, but it also means none of it has been exercised by a real binding.

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

- **The deep workflow went from 88 minutes to 6, and no check was dropped to get there.** The slowest job is now `miri (tree, yo-record-rest)` at 5 minutes 42 seconds, and the run finishes 20 seconds behind it. Miri was the whole of it. `cargo miri test --workspace` walks the crates one after another, so the wait was every crate added together; it is now one job per crate, taken from `cargo metadata` rather than from a list in the workflow, so a crate added next month gets interpreted without anybody remembering. Inside a job it runs through nextest instead of the built in harness, which matters because `cargo miri test` is a single process and libtest's threads do not help: Miri interprets threads on one core. nextest gives each test its own process, so a four core runner interprets four tests at once. `yo-record` locally went from 322.7 seconds to 121.2 at 359 percent CPU, and four interpreters on the heaviest crate peaked at 865 MB between them, so the parallelism costs megabytes and buys minutes. `yo-record` is split again by hand into its compaction tests and everything else, because those are half the crate and there is a floor to how far they shrink. The five fuzz targets run at the same time rather than one after another, which is five minutes down to one, and they share nothing so there was never a reason for the queue.
- **`format!` is gone from the test loops that run under Miri.** Miri charges per operation rather than per instruction, so the formatting machinery costs a couple of milliseconds a call where the same bytes built by hand cost microseconds. `yo-index` calls it once per set, get, delete and contains, which is ten thousand calls in one test, and taking it out took that crate's shard from 102.8 seconds to 57.7 locally. `yo-common` went from 16.7 to 9.1 the same way. The keys and values are byte for byte what `format!` produced, so no test is checking anything different.
- **Slice equality instead of a byte at a time loop in the arena tests.** Both read all six megabytes, but `==` on `[u8]` bottoms out in `memcmp`, which Miri runs as a shim rather than interpreting, and one test went from 265 seconds to 0.062. Nothing is skipped to get that.
- **Two dead ends, written down so nobody pays for them twice.** `--release` does not help Miri: `yo-record` measured 309.8 seconds against 309.7 for debug, because the cost is interpreted MIR operations and optimisation does not remove them. Neither does a slicing by eight CRC, which is the same eight table lookups per eight bytes however it is written. Both are real on hardware and neither is a Miri lever.
- **The `deep` job timeouts came down from 45 minutes to 25 for Miri and 15 for the rest.** A timeout well above the expected time is a job that hangs for three quarters of an hour before telling you.

### Known trade

nextest does not run doctests, so the `yo-crash` doctest is no longer interpreted under Miri. It still runs in `ci.yml` on every push, so what was lost is the doctest being interpreted, not the doctest.

### Fixed

- **A `yo-format` test never finished under Miri.** Flipping every bit of a 16 KiB superblock slot and decoding after each one is sixteen thousand iterations, each copying and checksumming 16 KiB. Natively that is a few milliseconds. Interpreted it did not finish inside a CI job's six hour ceiling, and it took a nightly Miri run with it. Under Miri the walk is now a stride of 2039 bytes plus nine offsets visited by hand, which is about 22 seconds; every ordinary run still flips every byte.
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
