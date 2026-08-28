# Releasing

`yo` releases early and often, from `0.1.0`, well before anything here is worth depending on. The reason is not ceremony. A milestone that has not been tagged has no fixed meaning, so a benchmark number, a bug report or a spec amendment cannot be pinned to anything, and six months from now nobody will be able to tell which version of the format a file was written by. A tag is what makes all three possible.

## What the numbers mean

While the major is 0, the guarantees are these and only these.

**Minor** goes up when a milestone lands. `M0` is `0.1.0`, and every engine or DX milestone that merges takes the next minor. A minor may break anything: the Rust API, the C ABI, the on-disk format, the command set. That is what a zero major is for, and pretending otherwise would be worse than saying it plainly.

**Patch** goes up for anything that lands between milestones. Fixes, benchmark corrections, documentation, a dependency bump, CI. A patch never changes the on-disk format and never removes a public item. It is cut after a handful of pull requests rather than after each one, so a patch is usually two to five changes with one set of notes covering them.

Put another way: a minor is only ever cut when a milestone is done, and everything in between is a patch.

**The on-disk format is not frozen until `M6`.** Until then a minor may change it, and when it does the release notes say so under its own heading with the migration, if there is one. From `M6` on, a file written by any later version is readable by every version at or above its `min_reader_version`, and that is checked in CI by the independent reader.

`1.0.0` is `M9`. It means the format is frozen, the six language bindings are real, and the gate numbers in `bench/` are met on a qualified box rather than on whatever machine was free.

## Cutting one

1. The PR is green on `ci`, and for a minor it is also green on `deep`. `deep` does not run on a pull request unless somebody asks for it, so put the `deep` label on the PR and wait for Miri, the fuzzers, the release matrix and the benchmark smoke run. A patch can go in on `ci` alone as long as the nightly `deep` run before it was green. Either way the tree has been built and tested on x86 as well as arm, because the arm dev machine cannot see the crc intrinsic paths or the target specific lints.
2. Bump `workspace.package.version` in the root `Cargo.toml`, and the `version` on every path dependency in `[workspace.dependencies]` with it, then run `cargo update --workspace` so `Cargo.lock` agrees. The release workflow refuses a tag whose version does not match the manifest, so a half-done bump fails before anything is published rather than after.
3. Add the release's section to `CHANGELOG.md`. Write it for somebody who was not in the room: what changed, why it changed, and what it costs them. Numbers get their machine and their profile next to them or they do not go in.
4. Merge the PR.
5. Tag the merge commit `vX.Y.Z` and push the tag. The `release` workflow takes it from there: it re-runs the gate jobs against the tag, then publishes the GitHub release with the changelog section as its body.

A tag is never moved and never deleted. If a release is wrong, the fix is the next patch.

## Signing

Two public keys are checked in under `keys/`, and both are published at <https://yo.tamnd.dev> as well. They are here before there is anything signed with them, which is the point: a key that appears in the same release as the first signature is a key nobody has had a chance to disagree with.

`keys/minisign.pub`, key id `9551442DEC1CD552`, signs `SHA256SUMS` for the script installers. `keys/tamnd-signing-key.asc`, `tamnd <dev@tamnd.com>`, fingerprint `F737 055C 3ACD 3956 2FE2  6163 46D1 5643 1C21 8272`, expires 2029-08-27 and signs the Maven Central artifacts.

The same fingerprints are on `keyserver.ubuntu.com` and `keys.openpgp.org`, neither of which we run. A key checked against a copy of itself on the same host proves nothing, so the check worth doing is that three places agree.

## Names

`cargo xtask reserve verify` asks every registry whether the names in `names.toml` are still ours, and `cargo xtask reserve docs` asks whether that file and `dx/12` section 2 still say the same thing. Both run weekly in `names.yml` and again on every release, and a release does not publish if either fails. The reasoning, the six-state machine and the reason the probes are Python rather than Rust are in `dx/16` section 10.

`verify` uses three exit codes and the difference matters: 0 is held, 1 is a name lost or transferred, 2 is a probe that could not get an answer. Both non-zero codes stop a release, because a gate that cannot see a name must not wave it through, but only the first one means something is actually wrong.

## What a release note has to contain

Every section carries whichever of these apply, in this order. Headings that have nothing under them are left out rather than filled with "none".

- **Milestone.** Which one, and a sentence on what it was for.
- **Added**, **Changed**, **Fixed.** Ordinary changes.
- **Format.** Any change to the `.yo` layout, with the offset or field, and whether a file written by the previous version still opens.
- **Performance.** Numbers, each with the machine, the profile, and whether it is a gate number or a development measurement. A development measurement says so in the same sentence as the number, not in a footnote.
- **Known gaps.** What the milestone was gated on and did not reach, and what is deliberately unimplemented so far. This is the section that keeps the rest honest.

## Style

Prose is not hard wrapped. One line per paragraph and one line per bullet, however long it runs, in every markdown file, pull request body, issue body and commit message in this repository. A sentence broken across two lines makes a one word edit show up as a two line diff, and it makes a paragraph impossible to grep for.
