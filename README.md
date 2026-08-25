# yo

Redis-compatible database in Rust, in one .yo file. Every Redis data structure plus documents, vectors and graphs, with your own language's types in place of a query language. RESP3 on the wire, embedded in six languages.

## Status

Early. `M0` has landed, which is the workspace, the hash, the allocator, the index bucket, the shard runtime and the CI that keeps them honest. Nothing is stored yet: the record plane and the file are `M1` and are in progress. See the [milestones](https://github.com/tamnd/yo/milestones) for what each one is gated on.

The version is `0.x` and it means what it says. A minor release may break the API, the ABI or the on-disk format, and the format is not frozen until `M6`. What changed in each release is in [CHANGELOG.md](CHANGELOG.md); the rules are in [RELEASING.md](RELEASING.md).

Benchmark numbers published so far are development measurements taken on a machine that does not meet the gate box definition, and each one says so where it appears. Comparisons against Redis and Valkey start at `M2`, on a box that qualifies.
