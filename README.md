# yo

Redis-compatible database in Rust, in one .yo file. Every Redis data structure plus documents, vectors and graphs, with your own language's types in place of a query language. RESP3 on the wire, embedded in six languages.

## Status

Early. `M0` has landed, which is the workspace, the hash, the allocator, the index bucket, the shard runtime and the CI that keeps them honest. Nothing is stored yet: the record plane and the file are `M1` and are in progress. See the [milestones](https://github.com/tamnd/yo/milestones) for what each one is gated on.

The version is `0.x` and it means what it says. A minor release may break the API, the ABI or the on-disk format, and the format is not frozen until `M6`. What changed in each release is in [CHANGELOG.md](CHANGELOG.md); the rules are in [RELEASING.md](RELEASING.md).

Benchmark numbers published so far are development measurements taken on a machine that does not meet the gate box definition, and each one says so where it appears. Comparisons against Redis and Valkey start at `M2`, on a box that qualifies.

## Packages

There is nothing to install. Every package below is published at `0.0.1`, holds a name, and raises the moment you call anything in it.

| | Command | Repository |
|---|---|---|
| Rust | `cargo add yodb` | here |
| Python | `pip install yodb` | [yo-python](https://github.com/tamnd/yo-python) |
| Node | `npm i @yodb/core` | [yo-node](https://github.com/tamnd/yo-node) |
| Go | `go get github.com/tamnd/yo-go` | [yo-go](https://github.com/tamnd/yo-go) |
| Java | `com.tamnd:yodb` | [yo-java](https://github.com/tamnd/yo-java) |
| .NET | `dotnet add package Yodb` | [yo-dotnet](https://github.com/tamnd/yo-dotnet) |
| Swift | `.package(url: "https://github.com/tamnd/yo-swift", from: "0.0.1")` | [yo-swift](https://github.com/tamnd/yo-swift) |
| Dart | `dart pub add yodb` | [yo-dart](https://github.com/tamnd/yo-dart) |
| C, C++ | nothing published; the library ships as `libyo` | [yo-c](https://github.com/tamnd/yo-c) |
| Docker | `docker run tamnd87/yo` | here |

The command is `yodb` and not `yo`, because `yo` on `PATH` belongs to Yeoman. Only the executable and the environment variables take that prefix. The C library stays `libyo`, the header stays `yo.h`, the symbols stay `yo_*` and the file extension stays `.yo`.

Two registries are not plain `yodb`. npm refused the unscoped name under its similarity filter, so the package is `@yodb/core`. Docker Hub refused the account name `tamnd` at signup, so the image is under `tamnd87` and is `tamnd87/yo`. Everywhere else is `yodb`.

Each of these was installed on a machine that had never seen it before this table was written, rather than being copied out of a build script.
