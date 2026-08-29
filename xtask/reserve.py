#!/usr/bin/env python3
"""reserve — the name-reservation tool for the `tamnd` publishing identity.

Implements `dx/16` §10. Five subcommands, all reached through `cargo xtask`:

    cargo xtask reserve audit    probe every row, rewrite state/probed
    cargo xtask reserve plan     print what would be published, per registry
    cargo xtask reserve apply    publish one placeholder   (needs credentials)
    cargo xtask reserve verify   assert every released name is still ours
    cargo xtask reserve docs     check `dx/12` §2's table against names.toml

`audit`, `plan`, `docs` and `verify` need no credentials and run anonymously.
`apply` reads the environment loaded by `yoenv` (`dx/16` §8).

This is Python and not Rust on purpose. `cargo xtask reserve` shells out to it,
so the engine's dependency graph never grows an HTTP client, a TLS stack and a
JSON parser for the sake of a tool that runs weekly in CI. The workspace's
dependency list is a thing users read (`dx/00` P9), and `ureq` + `rustls` +
`serde_json` is a lot of supply chain to carry for this. Nothing here imports
anything outside the standard library, and that is the property to keep.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import functools
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

UA = "yo-name-audit (+https://github.com/tamnd/yo)"
TIMEOUT = 20
IDENTITY = "tamnd"

# Docker Hub is the one registry where the account is not `tamnd`. The name was
# refused at signup on 2026-08-29 and it is not visible through any read API, so
# there is nothing to point at and no way to find out who has it. The account is
# `tamnd87` and the image is `tamnd87/yo`, which is the second exception to
# one-name-everywhere after npm's `@yodb/core`. It lives here rather than being
# derived from a row's `owner`, because a probe that trusted the file to say who
# owns a name could never report that a name had been taken from us.
DOCKER_IDENTITY = "tamnd87"

# Named so that a machine which already has one does not get a second, and so
# that a stale one can be found and deleted by hand. `docker buildx create` with
# no name invents one, which makes both of those harder than they need to be.
BUILDX_BUILDER = "yo-placeholder"

# The version every placeholder is published at, in one place because it moves.
# It was 0.0.0 until a registry that will not take the same version twice had to
# be republished, and it will move again for the same reason. Three probes used
# to compare against a literal 0.0.0 to tell "our placeholder" from "a real
# release", and after the republish all three quietly started reading our own
# empty packages as shipped software.
PLACEHOLDER = os.environ.get("YODB_PLACEHOLDER_VERSION", "0.0.1")

# ---------------------------------------------------------------------------
# states
# ---------------------------------------------------------------------------
# free      nothing is published under this name and we do not hold it
# reserved  we hold the name, placeholder published (§6)
# released  we hold the name and a real version is published
# blocked   someone else holds it; the fallback applies
# fallback  we are using the fallback name because the primary is blocked
# unknown   the probe could not reach the registry — never treated as free

FREE, RESERVED, RELEASED, BLOCKED, FALLBACK, UNKNOWN = (
    "free", "reserved", "released", "blocked", "fallback", "unknown"
)

# A regression is a state moving in the wrong direction. `audit` exits non-zero
# on any of these, because each one means a name we were counting on is gone.
REGRESSIONS = {
    (RESERVED, FREE), (RELEASED, FREE),      # our package vanished
    (RESERVED, BLOCKED), (RELEASED, BLOCKED),  # transferred out from under us
    (FREE, BLOCKED),                          # someone took it first
}


def get(url: str, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:  # noqa: BLE001 — a network failure is `unknown`, not `free`
        print(f"    ! {type(e).__name__}: {e}", file=sys.stderr)
        return 0, b""


def get_json(url: str, headers: dict[str, str] | None = None):
    code, body = get(url, headers)
    if code == 0:
        return None, None
    try:
        return code, json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return code, None


# ---------------------------------------------------------------------------
# probes — each returns (state, note)
# ---------------------------------------------------------------------------

def held_state(version: str) -> str:
    """What a name we hold is in, given the version published under it.

    A placeholder is not a release. The package exists so that nobody else can
    take the name and there is no yo inside it, so the row says `reserved` until
    a real version replaces it, at which point it says `released` on its own
    without anybody editing this file.
    """
    return RESERVED if version == PLACEHOLDER else RELEASED


def p_crates(name: str, _ns: str):
    code, d = get_json(f"https://crates.io/api/v1/crates/{name}")
    if code is None:
        return UNKNOWN, "unreachable"
    if code == 404 or (d and d.get("errors")):
        return FREE, ""
    if d and "crate" in d:
        owners_code, owners = get_json(f"https://crates.io/api/v1/crates/{name}/owners")
        who = ""
        if owners and owners.get("users"):
            who = ",".join(u.get("login", "?") for u in owners["users"])
        ours = IDENTITY in who
        v = d["crate"].get("max_version", "?")
        return (held_state(v) if ours else BLOCKED), f"v{v} owner={who or '?'}"
    return UNKNOWN, f"http {code}"


# crates.io and npm publish an owner list, so `is this ours` is a lookup there.
# PyPI and pub.dev publish neither owners nor uploaders, so the only public
# evidence of authorship is the metadata the package itself declares. That is
# weaker, and it is worth being exact about how weak: someone else could put
# our repository URL in their package. What they could not do is take a name we
# already hold, and this check only ever runs on names we already hold or on
# names that are free. It distinguishes "our placeholder" from "a stranger's
# package that happens to share the name", which is the question `audit` asks.
def _ours_by_repo(url: str) -> bool:
    return bool(url) and url.rstrip("/").lower() == REPO.lower()


def p_pypi(name: str, _ns: str):
    code, d = get_json(f"https://pypi.org/pypi/{name}/json")
    if code is None:
        return UNKNOWN, "unreachable"
    if code == 404:
        return FREE, ""
    if d:
        info = d.get("info", {})
        v = info.get("version", "?")
        urls = info.get("project_urls") or {}
        if _ours_by_repo(urls.get("Repository") or info.get("home_page") or ""):
            return held_state(v), f"v{v}, repository is ours"
        return BLOCKED, f"v{v} by {info.get('author') or info.get('maintainer') or '?'}"
    return UNKNOWN, f"http {code}"


def p_npm(name: str, ns: str):
    # The namespace is part of the name on npm, and dropping it does not fail
    # loudly: `@yodb/core` probed as `core` returns somebody else's package,
    # with their maintainers, and the row reads as a confident wrong answer.
    # Caught on 2026-08-28 only because the regression check fired on it.
    full = f"{ns}/{name}" if ns else name
    code, d = get_json(f"https://registry.npmjs.org/{full.replace('/', '%2f')}")
    if code is None:
        return UNKNOWN, "unreachable"
    if code == 404:
        return FREE, ""
    if d is not None:
        versions = d.get("versions") or {}
        maint = d.get("maintainers") or []
        if not versions and not maint:
            # HTTP 200, no versions, no maintainers: every version was
            # unpublished. `dx/16` §3.1 assumed the tombstone was the thing
            # standing in the way and that only a real `npm publish` would
            # settle it. The publish ran on 2026-08-28 and the tombstone was
            # never the constraint: npm refused with `Package name too similar
            # to existing packages yo,idb,nedb,lowdb,code,node,zod`, its
            # similarity filter, which applies to any unscoped new name and
            # would refuse this one on a registry where nobody had ever
            # published it.
            unpub = (d.get("time") or {}).get("unpublished", {})
            when = unpub.get("time", "?")[:10] if isinstance(unpub, dict) else "?"
            return BLOCKED, (
                f"unpublished {when}; publish refused by npm's name-similarity "
                f"filter, not by the tombstone"
            )
        who = ",".join(m.get("name", "?") for m in maint)
        ours = IDENTITY in who
        return (RELEASED if ours else BLOCKED), f"maintainers={who or '?'}"
    return UNKNOWN, f"http {code}"


_NPMRC = None


def _npm(args):
    """Run npm authenticated by NPM_TOKEN.

    The token goes in a userconfig file rather than an environment variable.
    `NPM_CONFIG_TOKEN` was the obvious way to do this and it has not worked
    since npm 9 removed the `token` config; npm warns `Unknown env config
    "token"` and then runs the command unauthenticated, so every probe came
    back as a permissions error that looked like a real answer.
    """
    global _NPMRC
    if _NPMRC is None:
        fd, _NPMRC = tempfile.mkstemp(prefix="npmrc-reserve-")
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(
                "//registry.npmjs.org/:_authToken=%s\n"
                % os.environ["NPM_TOKEN"]
            )
        atexit.register(lambda p=_NPMRC: os.path.exists(p) and os.unlink(p))
    return subprocess.run(
        ["npm", *args], capture_output=True, text=True,
        env={**os.environ, "npm_config_userconfig": _NPMRC},
    )


def _npm_err(r):
    """npm writes warnings and the error to stderr, so take the error line."""
    lines = [l for l in r.stderr.splitlines() if "npm error" in l]
    return (lines[0] if lines else r.stderr.strip())[:70]


@functools.lru_cache(maxsize=1)
def _npm_whoami():
    r = _npm(["whoami"])
    return r.stdout.strip() if r.returncode == 0 else None


def p_npm_scope(_name: str, ns: str):
    """npm organisation existence is NOT anonymously probeable.

    Three endpoints were tried on 2026-08-28 and all three fail closed:
      - www.npmjs.com/org/<org>        403 for every org, including ones that
                                       exist (`expressjs` also 403)
      - registry -/user/org.couchdb.user:<n>   401, needs auth
      - -/v1/search?text=scope:<org>   returns 0 for an org with no packages,
                                       so it cannot distinguish free from empty

    So this returns `unknown` rather than guessing. An authenticated check is
    `npm org ls <scope>` with NPM_TOKEN set, which `verify` uses once the
    credential exists. Reporting `free` here would be a claim we cannot
    support, and the whole point of the audit is that its rows are observed.
    """
    scope = ns.lstrip("@")
    if not os.environ.get("NPM_TOKEN"):
        return UNKNOWN, "not anonymously probeable; needs NPM_TOKEN"

    # A scope matching the account name is that account's own user scope. npm
    # gives every user one and it cannot be an organisation, because npm
    # refuses an org name a user already holds. The org endpoints answer 403
    # for it, which reads like a permissions problem and is not one, so this
    # case is settled before `npm org ls` is asked anything.
    if _npm_whoami() == scope:
        return RESERVED, f"user scope of npm account {scope}, not an org"

    r = _npm(["org", "ls", scope, "--json"])
    if r.returncode == 0:
        return RESERVED, f"org exists, {len(json.loads(r.stdout or '{}'))} member(s)"
    if "404" in r.stderr or "Not Found" in r.stderr:
        return FREE, "org does not exist (authenticated)"
    return UNKNOWN, f"npm org ls failed: {_npm_err(r)}"


def p_pub(name: str, _ns: str):
    code, d = get_json(f"https://pub.dev/api/packages/{name}")
    if code is None:
        return UNKNOWN, "unreachable"
    if code == 404:
        return FREE, ""
    if d:
        latest = d.get("latest") or {}
        v = latest.get("version", "?")
        spec = latest.get("pubspec") or {}
        # The publisher is the stronger signal and it is separately fetchable,
        # so ask for it first. A package published by the account but not yet
        # transferred to the verified publisher answers null, which is ours and
        # is also a thing to go and fix, so the note says which it is.
        _, pub = get_json(f"https://pub.dev/api/packages/{name}/publisher")
        pid = (pub or {}).get("publisherId")
        if pid:
            ours = pid == "tamnd.com"
            note = f"v{v}, publisher {pid}"
        else:
            ours = _ours_by_repo(spec.get("repository") or spec.get("homepage") or "")
            note = f"v{v}, no publisher set, repository is ours" if ours else f"v{v}"
        if ours:
            return held_state(v), note
        return BLOCKED, note
    return UNKNOWN, f"http {code}"


def p_nuget(name: str, _ns: str):
    # This probe used to return BLOCKED for any 200, which meant that the
    # moment our own package appeared the row would flip to "somebody else has
    # it" — the same shape as the npm and PyPI ownership bugs, and the third
    # time this exact mistake has been made in this file. A 200 says the name
    # is taken and says nothing about by whom, so ask.
    code, d = get_json(
        f"https://api.nuget.org/v3/registration5-semver1/{name.lower()}/index.json"
    )
    if code is None or code == 0:
        return UNKNOWN, "unreachable"
    if code == 404:
        # NuGet runs new packages through a validation pipeline — a malware
        # scan and a signing check — and publishes nothing until it finishes.
        # Every public read 404s in the meantime, so a package pushed minutes
        # ago is indistinguishable from a name nobody has ever taken. The push
        # itself is the only thing that can tell them apart: it answers 409
        # Conflict for an id+version already in the pipeline. That is a write,
        # so this probe cannot do it and must not pretend otherwise. `apply`
        # writing `reserved` and the next `audit` reading `free` is the
        # regression check firing on a real ambiguity, not a false alarm.
        return FREE, ""
    if code != 200 or not d:
        return UNKNOWN, f"http {code}"

    entry, version = None, "?"
    for page in d.get("items") or []:
        for leaf in page.get("items") or []:
            ce = leaf.get("catalogEntry") or {}
            if ce:
                entry, version = ce, ce.get("version", "?")
    if entry is None:
        # A paged registration index links to its pages instead of inlining
        # them. Only large packages hit this, and following the link is one
        # more request rather than a different answer.
        pages = [p.get("@id") for p in (d.get("items") or []) if p.get("@id")]
        if pages:
            _, pd = get_json(pages[-1])
            for leaf in (pd or {}).get("items") or []:
                ce = leaf.get("catalogEntry") or {}
                if ce:
                    entry, version = ce, ce.get("version", "?")
    if entry is None:
        return UNKNOWN, "registration index has no catalog entry"

    authors = entry.get("authors") or ""
    if isinstance(authors, list):
        authors = ",".join(authors)
    if _ours_by_repo(entry.get("projectUrl") or "") or IDENTITY in authors:
        return held_state(version), f"v{version} by {authors}"
    return BLOCKED, f"v{version} by {authors or '?'}"


def p_maven(name: str, ns: str):
    # Ask the repository before asking the search index. `repo1` is the thing
    # Maven itself resolves against and it is authoritative the moment a
    # deployment publishes; `search.maven.org` is a separate index that trails
    # it by hours. On 2026-08-28 `com.tamnd:yodb:0.0.0` was downloadable from
    # repo1, signature and all, while solr still reported `numFound: 0` — the
    # same read-replica lag npm showed, in a registry where it lasts longer.
    code, _ = get(
        f"https://repo1.maven.org/maven2/{ns.replace('.', '/')}/{name}/maven-metadata.xml"
    )
    if code == 200:
        return RESERVED, "in repo1 (authoritative)"

    code, d = get_json(
        f"https://search.maven.org/solrsearch/select?q=g:%22{ns}%22&rows=1&wt=json"
    )
    if code is None or d is None:
        return UNKNOWN, "unreachable"
    n = (d.get("response") or {}).get("numFound", -1)
    # Central publishes artifacts, not namespace ownership. Searching for a
    # groupId with no artifacts under it cannot tell "nobody has this" from
    # "we verified it on 2026-08-28 and have not published yet", and those are
    # opposite answers. It said `free` for the second one, which is the same
    # mistake the npm scope rows were written to avoid.
    #
    # So an empty result is `unknown` and says why. A non-empty result is ours
    # rather than someone else's, because the namespace is verified to this
    # identity and Central will not accept a publish under it from anyone else.
    if n == 0:
        return UNKNOWN, "no artifacts; Central does not expose namespace ownership"
    return RELEASED, f"numFound: {n}"


def p_brew(name: str, _ns: str):
    code, d = get_json(f"https://formulae.brew.sh/api/formula/{name}.json")
    if code == 0:
        return UNKNOWN, "unreachable"
    if code == 404:
        return FREE, ""
    if d:
        return BLOCKED, f"{d.get('desc','?')}"
    return UNKNOWN, f"http {code}"


def p_choco(name: str, _ns: str):
    code, body = get(
        "https://community.chocolatey.org/api/v2/Packages()"
        f"?$filter=Id%20eq%20'{name}'"
    )
    if code == 0:
        return UNKNOWN, "unreachable"
    n = body.count(b"<entry>")
    return (FREE, "") if n == 0 else (BLOCKED, f"{n} package versions")


def p_aur(name: str, _ns: str):
    code, d = get_json(f"https://aur.archlinux.org/rpc/v5/info?arg[]={name}")
    if code is None or d is None:
        return UNKNOWN, "unreachable"
    if d.get("resultcount", 0) == 0:
        return FREE, ""
    return BLOCKED, d["results"][0].get("Description", "")


def p_snap(name: str, _ns: str):
    code, d = get_json(
        f"https://api.snapcraft.io/v2/snaps/info/{name}",
        {"Snap-Device-Series": "16"},
    )
    if code is None:
        return UNKNOWN, "unreachable"
    if d and d.get("error-list"):
        if d["error-list"][0].get("code") == "resource-not-found":
            return FREE, ""
    return (BLOCKED, "exists") if code == 200 else (UNKNOWN, f"http {code}")


def p_scoop(name: str, _ns: str):
    code, _ = get(
        f"https://raw.githubusercontent.com/ScoopInstaller/Main/master/bucket/{name}.json"
    )
    if code == 0:
        return UNKNOWN, "unreachable"
    return (FREE, "Main bucket") if code == 404 else (BLOCKED, "in Main")


def p_dockerhub(name: str, ns: str):
    """Docker Hub, where a namespace is an account and cannot be probed.

    This used to read a 404 from `v2/users/{ns}/` as FREE, and it reported
    `tamnd` free on 2026-08-29. Signup then refused `tamnd`. Docker Hub has no
    public endpoint that separates "nobody has this" from "you may not have
    this": `v2/users/`, `v2/orgs/` and the publisher API all 404 for a refused
    name and for a name nobody has ever asked for, and `v2/repositories/{ns}/`
    and the registry auth service answer 200 for every string on earth. The only
    oracle is the signup form, which is a write.

    So a 404 is now UNKNOWN and not FREE. It is the weaker claim, and it is the
    true one. The cost of the old answer was a name in the audit table that read
    as ours for the taking for as long as nobody tried, which is `dx/16` §7.3
    again: a reading that looks like a measurement and is not one.

    A 200 is the one direction the endpoint can actually answer: somebody holds
    it, and it says who. If that somebody is us the name is held, and whether it
    is `reserved` or `released` follows from the image under it, not from the
    account.
    """
    code, d = get_json(f"https://hub.docker.com/v2/users/{ns}/")
    if code == 0:
        return UNKNOWN, "unreachable"
    if code == 404:
        return UNKNOWN, "no public account; only signup can tell free from refused"
    if code != 200 or not d:
        return UNKNOWN, f"http {code}"
    who = d.get("username") or "?"
    if who != DOCKER_IDENTITY:
        return BLOCKED, f"account {who}"
    code, tags = get_json(
        f"https://hub.docker.com/v2/repositories/{ns}/{name}/tags/?page_size=100"
    )
    if code == 404:
        # The account is ours, so every repository under it is ours to create
        # and nobody else can take this name. That is the whole reservation on
        # Docker Hub; the image is only there so `docker run` says the same
        # sentence the other bindings raise.
        return RESERVED, f"account {who}, no image yet"
    if code != 200 or not tags:
        return UNKNOWN, f"account {who}, tags http {code}"
    names = {t.get("name") for t in tags.get("results") or []}
    real = sorted(n for n in names if n and n not in {"latest", PLACEHOLDER})
    if real:
        return RELEASED, f"account {who}, tags {', '.join(real[:3])}"
    return RESERVED, f"account {who}, {PLACEHOLDER}"


def p_rubygems(name: str, _ns: str):
    code, d = get_json(f"https://rubygems.org/api/v1/gems/{name}.json")
    if code == 0:
        return UNKNOWN, "unreachable"
    if code == 404:
        return FREE, ""
    if d:
        return BLOCKED, f"v{d.get('version','?')}"
    return UNKNOWN, f"http {code}"


def p_hex(name: str, _ns: str):
    code, _ = get(f"https://hex.pm/api/packages/{name}")
    if code == 0:
        return UNKNOWN, "unreachable"
    return (FREE, "") if code == 404 else (BLOCKED, f"http {code}")


def p_packagist(name: str, ns: str):
    code, _ = get(f"https://repo.packagist.org/p2/{ns}/{name}.json")
    if code == 0:
        return UNKNOWN, "unreachable"
    return (FREE, "") if code == 404 else (BLOCKED, f"http {code}")


def p_conda(name: str, _ns: str):
    code, _ = get(f"https://api.anaconda.org/package/conda-forge/{name}")
    if code == 0:
        return UNKNOWN, "unreachable"
    return (FREE, "") if code == 404 else (BLOCKED, f"http {code}")


def p_cocoapods(name: str, _ns: str):
    code, _ = get(f"https://trunk.cocoapods.org/api/v1/pods/{name}")
    if code == 0:
        return UNKNOWN, "unreachable"
    return (FREE, "") if code == 404 else (BLOCKED, f"http {code}")


def p_github_repo(name: str, ns: str):
    # Authenticate when there is a token. The anonymous API allows 60 requests
    # an hour per address, this file probes six repositories per run, and a
    # weekly CI job sharing a runner pool burns that on somebody else's build.
    # A token raises it to 5000 and costs nothing here: the repositories are
    # public, so the token changes the rate limit and not the answer.
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_DEPLOY_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    code, _ = get(f"https://api.github.com/repos/{ns}/{name}", headers)
    if code == 0:
        return UNKNOWN, "unreachable"
    if code == 404:
        return FREE, "does not exist"
    if code == 200:
        return RESERVED, "exists"
    # 403 and 429 are the rate limiter, not an answer about the repository.
    # This came back as "LOST OR TRANSFERRED, this blocks the release" for all
    # six repositories at once, which is what a rate limit looks like when a
    # failed probe is allowed to mean something.
    if code in (403, 429):
        return UNKNOWN, f"http {code} (rate limited, not an answer)"
    return UNKNOWN, f"http {code}"


def p_dns(name: str, _ns: str):
    """A domain is `released` when it resolves, `free` when it has no SOA."""
    try:
        out = subprocess.run(
            ["dig", "+short", "SOA", name],
            capture_output=True, text=True, timeout=15,
        )
    except Exception:  # noqa: BLE001
        return UNKNOWN, "dig unavailable"
    soa = out.stdout.strip()
    return (RELEASED, soa.split()[0]) if soa else (FREE, "no SOA")


PROBES: dict[str, Callable[[str, str], tuple[str, str]]] = {
    "crates.io": p_crates,
    "pypi": p_pypi,
    "npm": p_npm,
    "npm-scope": p_npm_scope,
    "pub.dev": p_pub,
    "nuget": p_nuget,
    "maven-central": p_maven,
    "homebrew-core": p_brew,
    "chocolatey": p_choco,
    "aur": p_aur,
    "snap": p_snap,
    "scoop-main": p_scoop,
    "docker-hub": p_dockerhub,
    "rubygems": p_rubygems,
    "hex": p_hex,
    "packagist": p_packagist,
    "conda-forge": p_conda,
    "cocoapods": p_cocoapods,
    "github": p_github_repo,
    "dns": p_dns,
}

# Registries whose probe actually reads the namespace. Every other probe takes
# `_ns` and throws it away, which is correct for a registry with no namespaces
# and silently wrong for a row that has one: the probe answers about a
# different package and the row looks observed. `npm` was in the second
# category for one audit run and reported a stranger's maintainer list as this
# project's. A row that names a namespace a probe cannot use is a bug in the
# file, so it fails the run rather than producing a confident wrong answer.
NAMESPACED = {"npm", "npm-scope", "maven-central", "docker-hub", "packagist", "github"}


def check_namespaces(rows) -> list[str]:
    return [
        f"{r.registry}:{r.namespace}/{r.name} has a namespace but "
        f"{r.registry}'s probe ignores it"
        for r in rows
        if r.namespace and r.registry not in NAMESPACED
    ]


# ---------------------------------------------------------------------------
# names.toml — read and write without a TOML writer dependency
# ---------------------------------------------------------------------------

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    print("python 3.11+ required (tomllib)", file=sys.stderr)
    raise SystemExit(2)

FIELDS = ["registry", "name", "namespace", "state", "probed", "owner", "fallback", "note"]


@dataclass
class Row:
    registry: str
    name: str
    namespace: str = ""
    state: str = FREE
    probed: str = ""
    owner: str = IDENTITY
    fallback: str = ""
    note: str = ""
    reserve: bool = True  # false = audited but never published to (§11)
    # A state established by hand, for a name whose registry has no endpoint
    # that can answer the question. It is honoured only where the probe comes
    # back `unknown`; anything a probe can actually see still wins, so a stale
    # verdict cannot hide a name being taken out from under us.
    verdict: str = ""

    def key(self) -> str:
        return f"{self.registry}:{self.namespace or '-'}/{self.name}"


def load(path: str) -> list[Row]:
    with open(path, "rb") as f:
        doc = tomllib.load(f)
    return [Row(**r) for r in doc.get("name", [])]


def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def dump(path: str, rows: list[Row], header: str) -> None:
    out = [header.rstrip(), ""]
    w = max(len(f) for f in FIELDS)
    for r in rows:
        out.append("[[name]]")
        for f in FIELDS:
            out.append(f'{f.ljust(w)} = "{esc(getattr(r, f))}"')
        if not r.reserve:
            out.append(f'{"reserve".ljust(w)} = false')
        if r.verdict:
            out.append(f'{"verdict".ljust(w)} = "{esc(r.verdict)}"')
        out.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(out).rstrip() + "\n")


HEADER = """\
# names.toml — every name the `tamnd` publishing identity holds or intends to
# hold. The declarative source for `dx/16` §4's table, `dx/12` §2's table, and
# the weekly CI ownership check (`17` §6 check 5).
#
# GENERATED FIELDS: `state`, `probed` and `note` are rewritten by
# `reserve.py audit`. Everything else is written by hand.
#
# state:  free      nothing published, we do not hold it
#         reserved  we hold it, placeholder published (dx/16 §6)
#         released  we hold it, a real version is published
#         blocked   someone else holds it; use `fallback`
#         fallback  we are on the fallback because the primary is blocked
#         unknown   the probe failed — never treated as free
#
# reserve = false means the row is audited but deliberately never published to
# (dx/16 §11): port-PR channels, and names we refuse to squat.
#
# verdict = "<state>" pins a state a probe cannot reach, for a registry with no
# endpoint that answers the question. It is honoured only where the probe says
# `unknown`, so it can never hide a name being taken. `note` is then written by
# hand as well and says how the verdict was established.\
"""


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

def cmd_audit(args) -> int:
    rows = load(args.file)
    bad = check_namespaces(rows)
    if bad:
        for b in bad:
            print(f"  {b}", file=sys.stderr)
        return 2
    today = date.today().isoformat()
    changed, regressed = [], []

    width = max(len(r.key()) for r in rows)
    for r in rows:
        probe = PROBES.get(r.registry)
        if probe is None:
            print(f"  {r.key().ljust(width)}  no probe for registry {r.registry!r}")
            continue
        state, note = probe(r.name, r.namespace)
        if r.verdict and state == UNKNOWN:
            # A probe that could not see is not evidence against something a
            # person established some other way. Docker Hub refusing `tamnd` at
            # signup is a fact; `v2/users/tamnd/` returning 404 afterwards is
            # not a contradiction of it, it is the same silence that a name
            # nobody has ever asked for gives back.
            print(f"= {r.key().ljust(width)}  {r.verdict.ljust(8)}  "
                  f"{r.note}  (by hand; probe says {state})")
            r.state, r.probed = r.verdict, today
            continue
        old = r.state
        flag = " "
        if old and old != state:
            if (old, state) in REGRESSIONS:
                flag, regressed = "!", regressed + [(r, old, state)]
            else:
                flag = "~"
            changed.append((r, old, state))
        print(f"{flag} {r.key().ljust(width)}  {state.ljust(8)}  {note}")
        r.state, r.probed, r.note = state, today, note

    dump(args.file, rows, HEADER)

    print()
    tally: dict[str, int] = {}
    for r in rows:
        tally[r.state] = tally.get(r.state, 0) + 1
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    print(f"  {len(rows)} rows, {len(changed)} changed, {len(regressed)} regressed")

    if regressed:
        print("\nREGRESSED — a name we were counting on is gone:", file=sys.stderr)
        for r, old, new in regressed:
            print(f"  {r.key()}: {old} -> {new}  {r.note}", file=sys.stderr)
        return 1
    return 0


# §6 rule 2 fixes the description to one form, and the wording is not
# decorative: it says the project exists, says what the package will be, and
# says it is not usable yet. Those three claims are what separate a placeholder
# from squatting under every registry policy in §6's first paragraph, so the
# form is built here rather than written out per package, where one of them
# would eventually go missing.
DESC_FORM = (
    "Placeholder. yo is an embedded multi-model database in Rust; "
    "this package will hold {what}. Not yet usable."
)

# What each name will hold. Keyed by `registry:name` because three of the
# crates.io rows are the same registry and three different things, and because
# "its <language> binding" is wrong for the engine's own crate.
WHAT = {
    "crates.io:yodb": "the engine itself",
    "crates.io:yodb-sys": "the raw declarations for its C library",
    "crates.io:yodb-macros": "its procedural macros",
    "pypi:yodb": "its Python binding",
    "pub.dev:yodb": "its Dart binding",
    "maven-central:yodb": "its Java binding",
    "nuget:Yodb": "its .NET binding",
    # Keyed on the bare name, so npm's row is `core` and not `@yodb/core`, and
    # Docker Hub's is `yo` and not `tamnd87/yo`. Both namespaces are exceptions
    # to one-name-everywhere and both live in the row's `namespace` field;
    # putting either one in this key would hide the exception in a second place.
    "npm:core": "its Node.js binding",
    "docker-hub:yo": "the server, as a container image",
    "cocoapods:Yodb": "its Swift binding",
    "rubygems:yodb": "its Ruby binding",
    "hex:yodb": "its Elixir binding",
    "packagist:yodb": "its PHP binding",
}

REPO = "https://github.com/tamnd/yo"

# §6 rule 5: the README's second line is the milestone the real release lands
# at, so a reader who finds the placeholder learns when it stops being one.
MILESTONE = "The real release lands at milestone DX6 (M7). Until then this package does nothing."

# §6 rule 6: one symbol, and it raises this. Same sentence in every language,
# because a user who hits it in two ecosystems should not have to work out
# whether they are two different problems.
def raises_for(version: str) -> str:
    """The sentence, carrying the version of the package it is compiled into.

    It used to name 0.0.0 in a constant, which stopped being true the moment the
    placeholders were republished at 0.0.1 and left every binding in every
    language telling users the version of a package that was no longer on the
    registry. The version is the one useful thing in the sentence, since it is
    how a reader works out whether what they installed is the empty one.
    """
    return f"yo is not usable yet. This is a reserved placeholder at {version}; see {REPO}"


def desc_for(row) -> str:
    what = WHAT.get(f"{row.registry}:{row.name}")
    if what is None:
        raise SystemExit(
            f"no description phrase for {row.registry}:{row.name}. §6 rule 2 "
            f"fixes the wording, so add it to WHAT rather than improvising one."
        )
    return DESC_FORM.format(what=what)


PLACEHOLDER_DESC = DESC_FORM.format(what="its <language> binding")


def cmd_plan(args) -> int:
    rows = load(args.file)
    version = PLACEHOLDER
    todo = [r for r in rows if r.reserve and r.state == FREE]
    skip = [r for r in rows if r.reserve and r.state in (RESERVED, RELEASED)]
    block = [r for r in rows if r.reserve and r.state == BLOCKED]

    print(f"placeholder version: {version}")
    print(f"description:         {PLACEHOLDER_DESC}\n")
    print(f"WOULD PUBLISH ({len(todo)}):")
    for r in todo:
        print(f"  {r.registry:<16} {r.namespace + '/' if r.namespace else ''}{r.name}")
    print(f"\nALREADY HELD ({len(skip)}) — skipped, not republished:")
    for r in skip:
        print(f"  {r.registry:<16} {r.name}  [{r.state}]")
    if block:
        print(f"\nBLOCKED ({len(block)}) — fallback applies:")
        for r in block:
            print(f"  {r.registry:<16} {r.name} -> {r.fallback or 'NO FALLBACK SET'}  ({r.note})")
    nofb = [r for r in block if not r.fallback]
    if nofb:
        print(f"\n{len(nofb)} blocked row(s) have no fallback. Decide before apply.", file=sys.stderr)
        return 1
    return 0


def cmd_verify(args) -> int:
    """Assert every name we claim to hold is still ours. A release gate."""
    rows = load(args.file)
    held = [r for r in rows if r.state in (RESERVED, RELEASED)]
    bad, unsure = [], []
    width = max((len(r.key()) for r in held), default=10)
    for r in held:
        probe = PROBES.get(r.registry)
        if probe is None:
            continue
        state, note = probe(r.name, r.namespace)
        ok = state in (RESERVED, RELEASED)
        mark = "  " if ok else ("? " if state == UNKNOWN else "! ")
        print(f"{mark}{r.key().ljust(width)}  {state.ljust(8)}  {note}")
        if state == UNKNOWN:
            unsure.append((r, state, note))
        elif not ok:
            bad.append((r, state, note))

    print(f"\n  {len(held)} held, {len(bad)} lost, {len(unsure)} could not be checked")

    # `unknown` is not `lost`, and keeping them apart is the whole point of
    # having six states rather than three. GitHub's anonymous rate limit once
    # turned six reachable, intact, correctly-owned repositories into "LOST OR
    # TRANSFERRED, this blocks the release" — a broken measurement reported as
    # a finding, which is worse than no measurement. Both still fail, because a
    # release gate that cannot see a name must not wave it through, but they
    # fail with different words and different exit codes so that whoever reads
    # the CI log knows whether to panic or to retry.
    if bad:
        print("\nLOST OR TRANSFERRED — this blocks the release (dx/12 §5 step 2):", file=sys.stderr)
        for r, state, note in bad:
            print(f"  {r.key()}: expected held, got {state}  {note}", file=sys.stderr)
    if unsure:
        print("\nCOULD NOT CHECK — no verdict, and no evidence anything is wrong:", file=sys.stderr)
        for r, state, note in unsure:
            print(f"  {r.key()}: {note}", file=sys.stderr)
        print("  Retry. If it persists it is the probe or the network, not the name.",
              file=sys.stderr)
    return 1 if bad else (2 if unsure else 0)


# ---------------------------------------------------------------------------
# apply — build a placeholder, prove it builds, then publish it
# ---------------------------------------------------------------------------
#
# One registry per invocation, and a dry run unless `--yes` is passed. Both of
# those are deliberate. A registry publish is permanent: crates.io and PyPI
# have no delete, only yank and delete-the-file-but-keep-the-name, and a loop
# that publishes to twelve registries has twelve chances to make the same
# mistake permanently before anyone reads the first line of output.
#
# Every builder writes a complete package into a temporary directory and then
# runs that ecosystem's own packaging step over it. That step is the check: it
# is the same code the registry runs, so a package that survives it is a
# package that will not arrive broken (§6 rule 6).


def _w(root: str, rel: str, text: str) -> str:
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(text if text.endswith("\n") else text + "\n")
    return p


def _readme(desc: str, body: str) -> str:
    # §6 rule 5 fixes the first two lines: the description, then the milestone.
    # Not a title. A reader who lands on the registry page sees what this is
    # and when it stops being that, above the fold and before any prose.
    return f"{desc}\n\n{MILESTONE}\n\n{body.strip()}\n"


PLACEHOLDER_BODY = f"""
This name is reserved for a project that exists and is being built, not held
against one that might be. The engine is at {REPO}, it has a changelog and a
milestone track, and this package is a signpost to it.

Nothing here works yet. There is one symbol and calling it raises. That is the
whole package, on purpose: a placeholder that fails at import or resolution
time is a cost imposed on a stranger's build, and a placeholder that pretends
to work is worse.
"""

LICENSE_LINE = "MIT OR Apache-2.0"


class Step:
    """A named command. `check` steps run always; `live` steps only with --yes.

    `loud` steps print their output even when they succeed. Every publish is
    loud, because the registry's own words are the only account of what
    happened and a quiet success hides the difference between "uploaded" and
    "skipped, it was already there" — `dotnet nuget push --skip-duplicate`
    exits 0 for both, and finding that out afterwards cost an hour.
    """

    def __init__(self, label, argv, cwd=None, env=None, loud=False):
        self.label, self.argv, self.cwd, self.env = label, argv, cwd, env
        self.loud = loud

    def run(self) -> int:
        print(f"    $ {redact(' '.join(self.argv))}")
        r = subprocess.run(
            self.argv, cwd=self.cwd, env={**os.environ, **(self.env or {})},
            capture_output=True, text=True,
        )
        if r.returncode != 0 or self.loud:
            for line in redact(r.stdout + r.stderr).splitlines():
                print(f"      | {line}")
        return r.returncode


# NuGet is the one publisher here that takes its credential as a command line
# argument rather than an environment variable, so the whole key was printed
# into the run log by the line that echoes the command. It went into a
# transcript before this existed. Every other secret in the environment goes
# through the same filter rather than just that one, because the next tool to
# do this will not announce itself either.
def redact(text: str) -> str:
    for name, value in os.environ.items():
        if len(value) < 12:
            continue  # too short to be a credential, too likely to be a word
        if any(k in name for k in ("TOKEN", "KEY", "PASSWORD", "PASSPHRASE",
                                   "SECRET", "CREDENTIALS")):
            text = text.replace(value, f"${{{name}}}")
    return text


def b_crates(row, root, version, desc):
    _w(root, "Cargo.toml", f"""
[package]
name = "{row.name}"
version = "{version}"
edition = "2021"
rust-version = "1.94"
description = "{desc}"
license = "{LICENSE_LINE}"
repository = "{REPO}"
homepage = "{REPO}"
readme = "README.md"
keywords = ["database", "embedded", "placeholder"]
categories = ["database"]

[dependencies]
""".lstrip())
    _w(root, "README.md", _readme(desc, PLACEHOLDER_BODY))
    _w(root, "src/lib.rs", f'''
//! {desc}
//!
//! {MILESTONE}

#![forbid(unsafe_code)]
#![deny(missing_docs)]

/// The message every placeholder in every ecosystem raises with.
pub const NOT_YET: &str = "{raises_for(version)}";

/// Opens a database. Always panics: this version is a reserved placeholder.
///
/// It panics rather than returning an error because there is no error a caller
/// could handle. Nothing about the argument is wrong; the package is empty.
pub fn open(_path: &str) -> ! {{
    panic!("{{NOT_YET}}");
}}

#[cfg(test)]
mod tests {{
    #[test]
    #[should_panic(expected = "not usable yet")]
    fn open_panics() {{
        super::open("x.yo");
    }}
}}
'''.lstrip())
    return (
        [Step("package and verify", ["cargo", "publish", "--dry-run", "--allow-dirty"], root)],
        [Step("publish", ["cargo", "publish", "--allow-dirty"], root, loud=True)],
    )


def b_pypi(row, root, version, desc):
    _w(root, "pyproject.toml", f"""
[project]
name = "{row.name}"
version = "{version}"
description = "{desc}"
readme = "README.md"
requires-python = ">=3.9"
license = "{LICENSE_LINE}"
keywords = ["database", "embedded", "placeholder"]
classifiers = [
    "Development Status :: 1 - Planning",
    "Intended Audience :: Developers",
    "Topic :: Database :: Database Engines/Servers",
]

[project.urls]
Homepage = "{REPO}"
Repository = "{REPO}"
Issues = "{REPO}/issues"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{row.name}"]
""".lstrip())
    _w(root, "README.md", _readme(desc, PLACEHOLDER_BODY))
    _w(root, f"src/{row.name}/__init__.py", f'''
"""{desc}

{MILESTONE}
"""

__all__ = ["NOT_YET", "open"]
__version__ = "{version}"

#: The message every placeholder in every ecosystem raises with.
NOT_YET = "{raises_for(version)}"


def open(path):  # noqa: A001 — matches the API this will eventually have
    """Open a database. Always raises: this version is a reserved placeholder.

    It raises rather than returning None because a caller cannot tell a None
    from a database that failed to open, and a placeholder that is quiet is a
    placeholder that reaches production.
    """
    raise NotImplementedError(NOT_YET)
'''.lstrip())
    return (
        [Step("build sdist and wheel", ["uv", "build"], root)],
        # No argument: uv's default is the glob `dist/*`. Passing `dist/` looks
        # equivalent and is not — uv globs the argument itself rather than
        # walking a directory, so a directory matches no files and it exits
        # with "No files found to publish" after a successful build.
        [Step("publish", ["uv", "publish"], root, loud=True)],
    )


def b_pub(row, root, version, desc):
    _w(root, "pubspec.yaml", f"""
name: {row.name}
version: {version}
description: "{desc}"
homepage: {REPO}
repository: {REPO}
issue_tracker: {REPO}/issues

environment:
  sdk: ^3.0.0
""".lstrip())
    _w(root, "README.md", _readme(desc, PLACEHOLDER_BODY))
    _w(root, "CHANGELOG.md", f"## {version}\n\n- Name reservation. No functionality.\n")
    _w(root, "LICENSE", LICENSE_TEXT)
    _w(root, "analysis_options.yaml", "include: package:lints/recommended.yaml\n")
    _w(root, f"lib/{row.name}.dart", f'''
/// {desc}
///
/// {MILESTONE}
library;

/// The message every placeholder in every ecosystem raises with.
const String notYet = '{raises_for(version)}';

/// Opens a database. Always throws: this version is a reserved placeholder.
///
/// It throws [UnsupportedError] rather than returning null because a null here
/// would be indistinguishable from a database that failed to open.
Never open(String path) {{
  throw UnsupportedError(notYet);
}}
'''.lstrip())
    return (
        [Step("dry run", ["dart", "pub", "publish", "--dry-run"], root)],
        [Step("publish", ["dart", "pub", "publish", "--force"], root, loud=True)],
    )


LICENSE_TEXT = """\
This package is licensed under either of

  Apache License, Version 2.0   http://www.apache.org/licenses/LICENSE-2.0
  MIT license                   http://opensource.org/licenses/MIT

at your option. The full texts are in the repository at
https://github.com/tamnd/yo.
"""


def b_maven(row, root, version, desc):
    """Maven Central, through the Central Portal.

    This is the only builder that needs a settings file, a signature and a JDK,
    and the only one where the packaging step is not also the upload step. It
    is also the only registry here that will not take an artifact without a
    detached OpenPGP signature over every file, which is why §7 step 6's key
    exists at all.
    """
    gid, aid = row.namespace, row.name
    pkg = f"{gid}.{aid}".replace("-", "")
    _w(root, "pom.xml", f"""
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>

  <groupId>{gid}</groupId>
  <artifactId>{aid}</artifactId>
  <version>{version}</version>
  <packaging>jar</packaging>

  <name>{aid}</name>
  <description>{desc}</description>
  <url>{REPO}</url>

  <licenses>
    <license>
      <name>MIT</name>
      <url>https://opensource.org/licenses/MIT</url>
    </license>
    <license>
      <name>Apache License, Version 2.0</name>
      <url>https://www.apache.org/licenses/LICENSE-2.0</url>
    </license>
  </licenses>

  <developers>
    <developer>
      <id>tamnd</id>
      <name>tamnd</name>
      <url>https://github.com/tamnd</url>
    </developer>
  </developers>

  <scm>
    <url>{REPO}</url>
    <connection>scm:git:{REPO}.git</connection>
    <developerConnection>scm:git:{REPO}.git</developerConnection>
  </scm>

  <properties>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    <maven.compiler.release>11</maven.compiler.release>
  </properties>

  <build>
    <plugins>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-source-plugin</artifactId>
        <version>3.3.1</version>
        <executions><execution>
          <id>attach-sources</id><goals><goal>jar-no-fork</goal></goals>
        </execution></executions>
      </plugin>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-javadoc-plugin</artifactId>
        <version>3.12.0</version>
        <executions><execution>
          <id>attach-javadocs</id><goals><goal>jar</goal></goals>
        </execution></executions>
      </plugin>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-gpg-plugin</artifactId>
        <version>3.2.8</version>
        <executions><execution>
          <id>sign-artifacts</id>
          <phase>verify</phase>
          <goals><goal>sign</goal></goals>
          <configuration>
            <!-- gpg 2 wants a pinentry by default and there is no terminal in
                 CI. Loopback mode makes it take the passphrase from the
                 settings file instead, which is the only way this runs
                 unattended. -->
            <gpgArguments>
              <arg>--pinentry-mode</arg>
              <arg>loopback</arg>
            </gpgArguments>
          </configuration>
        </execution></executions>
      </plugin>
      <plugin>
        <groupId>org.sonatype.central</groupId>
        <artifactId>central-publishing-maven-plugin</artifactId>
        <version>0.11.0</version>
        <extensions>true</extensions>
        <configuration>
          <publishingServerId>central</publishingServerId>
          <autoPublish>true</autoPublish>
          <waitUntil>published</waitUntil>
        </configuration>
      </plugin>
    </plugins>
  </build>
</project>
""".lstrip())
    _w(root, "README.md", _readme(desc, PLACEHOLDER_BODY))
    _w(root, f"src/main/java/{pkg.replace('.', '/')}/Yo.java", f"""
package {pkg};

/**
 * {desc}
 *
 * <p>{MILESTONE}
 */
public final class Yo {{

  /** The message every placeholder in every ecosystem raises with. */
  public static final String NOT_YET = "{raises_for(version)}";

  private Yo() {{}}

  /**
   * Opens a database. Always throws: this version is a reserved placeholder.
   *
   * <p>It throws rather than returning null because a null here would be
   * indistinguishable from a database that failed to open.
   *
   * @param path where the database would live
   * @return never returns
   * @throws UnsupportedOperationException always
   */
  public static Object open(String path) {{
    throw new UnsupportedOperationException(NOT_YET);
  }}
}}
""".lstrip())

    # A settings file per run, in the package's own temporary directory, so the
    # token and the key passphrase never touch ~/.m2 and never outlive the
    # build. Mode 600 before anything is written into it.
    settings = os.path.join(root, "settings.xml")
    fd = os.open(settings, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<settings>
  <servers>
    <server>
      <id>central</id>
      <username>{os.environ['MAVEN_CENTRAL_USERNAME']}</username>
      <password>{os.environ['MAVEN_CENTRAL_PASSWORD']}</password>
    </server>
  </servers>
  <profiles>
    <profile>
      <id>signing</id>
      <properties>
        <gpg.keyname>{os.environ['MAVEN_GPG_KEY_ID']}</gpg.keyname>
        <gpg.passphrase>{os.environ['MAVEN_GPG_PASSPHRASE']}</gpg.passphrase>
      </properties>
    </profile>
  </profiles>
  <activeProfiles><activeProfile>signing</activeProfile></activeProfiles>
</settings>
""")

    # gpg-agent caches a passphrase for ten minutes by default, and that cache
    # is shared with anything else that used the key. A run with a deliberately
    # wrong passphrase signed successfully because of it, which made the
    # signing check look like it was verifying something it was not. Anyone
    # re-testing this path has to `gpgconf --kill gpg-agent` first; CI has no
    # such cache and needs the real value.
    mvn = ["mvn", "-B", "-ntp", "-s", settings]
    env = {"JAVA_HOME": JAVA_HOME} if JAVA_HOME else {}
    return (
        [Step("package, javadoc, sources, sign", mvn + ["verify"], root, env)],
        [Step("deploy to Central", mvn + ["deploy"], root, env)],
    )


def b_nuget(row, root, version, desc):
    """NuGet, through `dotnet pack` and `dotnet nuget push`.

    The package id is `Yodb`, capitalised, because NuGet ids are conventionally
    PascalCase and the .NET binding is the one place in this project where the
    house style loses to the ecosystem's. Ids are case-insensitive for
    resolution but the casing shown on the site is whatever the first publish
    used, so it is worth getting right once.
    """
    # `dotnet pack` reads its metadata from the csproj, so there is no separate
    # nuspec. A nuspec alongside a csproj is also legal and silently wins, which
    # is a good way to publish metadata you thought you had edited.
    _w(root, f"{row.name}.csproj", f"""
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <!-- netstandard2.0 rather than a net10.0 target: this package will one day
         carry a native binding that anything from .NET Framework upwards can
         consume, and the floor is easier to raise later than to lower. -->
    <TargetFramework>netstandard2.0</TargetFramework>
    <LangVersion>latest</LangVersion>
    <Nullable>enable</Nullable>
    <TreatWarningsAsErrors>true</TreatWarningsAsErrors>
    <GenerateDocumentationFile>true</GenerateDocumentationFile>

    <PackageId>{row.name}</PackageId>
    <Version>{version}</Version>
    <Authors>tamnd</Authors>
    <Description>{desc}</Description>
    <PackageLicenseExpression>MIT OR Apache-2.0</PackageLicenseExpression>
    <PackageProjectUrl>{REPO}</PackageProjectUrl>
    <RepositoryUrl>{REPO}</RepositoryUrl>
    <PackageReadmeFile>README.md</PackageReadmeFile>
    <!-- Off by default, and the warning it produces on a package with no
         assembly documentation is not the failure it looks like. -->
    <IncludeSymbols>false</IncludeSymbols>
  </PropertyGroup>

  <ItemGroup>
    <None Include="README.md" Pack="true" PackagePath="\\" />
  </ItemGroup>

</Project>
""".lstrip())
    _w(root, "README.md", _readme(desc, PLACEHOLDER_BODY))
    _w(root, "Yo.cs", f'''
// `using System;` is not redundant here. ImplicitUsings is a net6.0-and-later
// SDK default and this targets netstandard2.0, so without it even
// NotSupportedException does not resolve.
using System;

namespace {row.name};

/// <summary>{desc}</summary>
/// <remarks>{MILESTONE}</remarks>
public static class Yo
{{
    /// <summary>The message every placeholder in every ecosystem raises with.</summary>
    public const string NotYet = "{raises_for(version)}";

    /// <summary>Opens a database. Always throws: this version is a reserved placeholder.</summary>
    /// <param name="path">Ignored.</param>
    /// <exception cref="NotSupportedException">Always.</exception>
    public static object Open(string path) => throw new NotSupportedException(NotYet);
}}
'''.lstrip())
    nupkg = os.path.join(root, "bin", "Release", f"{row.name}.{version}.nupkg")
    return (
        [Step("pack", ["dotnet", "pack", "-c", "Release", "--nologo"], root,
                       DOTNET_ENV)],
        # --skip-duplicate so a re-run after a partial failure is not itself a
        # failure. NuGet answers 409 for an id+version that already exists and
        # the push command treats that as fatal without it.
        [Step("push", ["dotnet", "nuget", "push", nupkg,
                       "--source", "https://api.nuget.org/v3/index.json",
                       "--api-key", os.environ.get("NUGET_API_KEY", ""),
                       "--skip-duplicate"], root, DOTNET_ENV, loud=True)],
    )


# The SDK installs to ~/.dotnet and puts nothing on PATH, because the official
# install script deliberately does not edit shell profiles. Same class of
# problem as JAVA_HOME below: the tool is present, the subprocess cannot see it,
# and the error is "command not found" rather than anything about .NET.
DOTNET_ENV = {}
for _d in (os.path.expanduser("~/.dotnet"), "/usr/local/share/dotnet"):
    if os.path.isfile(os.path.join(_d, "dotnet")):
        DOTNET_ENV = {
            "PATH": _d + os.pathsep + os.environ.get("PATH", ""),
            "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
            "DOTNET_NOLOGO": "1",
        }
        break


# Homebrew installs a JDK where `java` on PATH cannot find it, so maven runs
# with no runtime at all and reports it as its own failure. Pointing JAVA_HOME
# at the cellar is less fragile than asking the user to have run `brew link`.
JAVA_HOME = next(
    (p for p in ("/opt/homebrew/opt/openjdk@21", "/opt/homebrew/opt/openjdk")
     if os.path.isdir(p)),
    os.environ.get("JAVA_HOME", ""),
)


def b_npm(row, root, version, desc):
    """npm, through `npm publish`.

    This builder exists because the npm placeholder was published by hand and
    ended up carrying a different sentence from the other five. Nobody noticed
    until the package was installed on a machine that had never seen it, which
    is exactly the check `dx/12` §1 is for and exactly the reason a builder that
    generates the text beats a person retyping it.

    The already-published `@yodb/core@0.0.0` is not corrected by this. npm never
    lets a version number be reused, not even after an unpublish inside the
    72-hour window, so the choice was between burning the number and leaving one
    divergent string in one ecosystem. The divergence is recorded in `dx/16` §6
    and this builder is what stops it recurring at the next publish.

    The package name is `@yodb/core`, the one exception to one-name-everywhere,
    so it is assembled from the row's namespace rather than from its name.
    """
    pkg = f"{row.namespace}/{row.name}"

    _w(root, "package.json", json.dumps({
        "name": pkg,
        "version": version,
        "description": desc,
        "license": LICENSE_LINE,
        "homepage": REPO,
        "repository": {"type": "git", "url": f"git+{REPO}.git"},
        "bugs": {"url": f"{REPO}/issues"},
        "author": row.owner,
        "type": "module",
        "main": "./index.js",
        "exports": {".": "./index.js"},
        "files": ["index.js", "README.md"],
        "engines": {"node": ">=20"},
        "keywords": ["database", "embedded", "placeholder"],
        "publishConfig": {"access": "public"},
    }, indent=2) + "\n")
    _w(root, "README.md", _readme(desc, PLACEHOLDER_BODY))
    _w(root, "index.js", f'''
// {desc}
//
// {MILESTONE}
//
// Importing this module succeeds and does nothing, on purpose. A placeholder
// that throws on import is a broken artifact in a stranger's dependency tree,
// and that is a real cost to impose on somebody for the sake of holding a name.
// Calling something is what tells you where you are.

/** The message every placeholder in every ecosystem raises with. */
export const NOT_YET = "{raises_for(version)}";

/**
 * Opens a database. Always throws: this version is a reserved placeholder.
 *
 * It throws rather than returning undefined because an undefined here is
 * indistinguishable from a database that failed to open.
 */
export function open(path) {{
  throw new Error(NOT_YET);
}}

export default {{ NOT_YET, open }};
'''.lstrip())

    # npm reads the token from an .npmrc and not from a bare environment
    # variable, and it expands ${NPM_TOKEN} inside one. Written into the build
    # root rather than into ~/.npmrc so a publish run cannot leave a credential
    # behind in the home directory of whatever machine it happened on.
    _w(root, ".npmrc", "//registry.npmjs.org/:_authToken=${NPM_TOKEN}\n")

    return (
        [Step("pack and verify", ["npm", "pack", "--dry-run"], root)],
        [Step("publish", ["npm", "publish", "--access", "public"], root,
              loud=True)],
    )


def b_docker(row, root, version, desc):
    """Docker Hub, through `docker buildx build --push`.

    The image is not what holds the name. Owning the account holds it: every
    repository under `tamnd87` is ours to create and nobody else can make one
    there. The image exists so that `docker run tamnd87/yo` answers with the
    same sentence the other seven bindings raise, instead of "not found", which
    is what somebody who read the README and tried it would otherwise get.

    It is `scratch` plus one static binary, about 1.2 MB, because a placeholder
    that pulls a 70 MB base image to print one line is charging a stranger real
    bandwidth for our name. The binary is cross-compiled in the builder stage
    rather than emulated, so `linux/amd64` and `linux/arm64` both come out of
    one pass on either kind of machine and neither needs QEMU.

    It exits 1. Everything else in this file raises, and an image that printed
    the sentence and exited 0 would pass a health check.
    """
    tag = f"{row.namespace}/{row.name}"
    raises = raises_for(version)

    _w(root, "main.go", f'''
// {desc}
//
// {MILESTONE}
package main

import (
	"fmt"
	"os"
)

// notYet is the message every placeholder in every yo ecosystem carries.
const notYet = "{raises}"

func main() {{
	fmt.Fprintln(os.Stderr, notYet)
	// Non-zero, for the same reason the library bindings raise instead of
	// returning nil: a caller that does not check is told anyway.
	os.Exit(1)
}}
'''.lstrip())

    _w(root, "go.mod", f"module {row.name}\n\ngo 1.25\n")

    # CGO off and the symbol table stripped is what makes the binary runnable on
    # `scratch`, which has no libc and no dynamic loader to find one with.
    _w(root, "Dockerfile", f'''
# syntax=docker/dockerfile:1

FROM --platform=$BUILDPLATFORM golang:1.25-alpine AS build
ARG TARGETOS
ARG TARGETARCH
WORKDIR /src
COPY go.mod main.go ./
RUN CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH \\
    go build -trimpath -ldflags="-s -w" -o /yodb .

FROM scratch
COPY --from=build /yodb /yodb
LABEL org.opencontainers.image.title="{tag}" \\
      org.opencontainers.image.description="{desc}" \\
      org.opencontainers.image.version="{version}" \\
      org.opencontainers.image.source="{REPO}" \\
      org.opencontainers.image.licenses="{LICENSE_LINE}"
ENTRYPOINT ["/yodb"]
'''.lstrip())

    _w(root, "README.md", _readme(desc, PLACEHOLDER_BODY))

    # Docker reads credentials from a config.json and does not expand
    # environment variables inside one, so unlike the .npmrc above this file
    # holds the real thing. It is written 0600 into the build root, which is a
    # temporary directory, for the same reason the .npmrc is: a publish run must
    # not leave a credential in the home directory of the machine it ran on.
    auth = base64.b64encode(
        f"{os.environ['DOCKERHUB_USERNAME']}:{os.environ['DOCKERHUB_TOKEN']}".encode()
    ).decode()
    cfg = _w(root, ".docker/config.json", json.dumps(
        {"auths": {"https://index.docker.io/v1/": {"auth": auth}}}, indent=2) + "\n")
    os.chmod(cfg, 0o600)
    dcfg = os.path.join(root, ".docker")
    env = {"DOCKER_CONFIG": dcfg}

    # DOCKER_CONFIG is not just where the credential lives. It is also where
    # docker finds per-user CLI plugins and where buildx keeps the list of
    # builder instances, so pointing it at a directory holding only a
    # config.json takes buildx away and then takes its builders away. Both
    # failures land on the push step alone, after the same flags worked twice in
    # the checks: first `unknown flag: --builder`, then `no builder found`.
    # Neither error mentions DOCKER_CONFIG.
    #
    # So everything in the real directory is linked across and only config.json
    # is replaced. The isolation that is wanted here is of the credential, not
    # of the tool.
    real = os.environ.get("DOCKER_CONFIG") or os.path.expanduser("~/.docker")
    if os.path.isdir(real):
        for entry in os.listdir(real):
            if entry != "config.json":
                os.symlink(os.path.join(real, entry), os.path.join(dcfg, entry))

    plat = "linux/amd64,linux/arm64"
    # A manifest list needs the container driver. The default `docker` driver
    # builds one platform and refuses to export two, with an error that reads
    # like a flag problem rather than a builder problem. Created here if it is
    # missing so the first run on a new machine works.
    ensure = ["sh", "-c",
              f"docker buildx inspect {BUILDX_BUILDER} >/dev/null 2>&1 || "
              f"docker buildx create --name {BUILDX_BUILDER} "
              f"--driver docker-container --bootstrap"]

    return (
        [
            Step("builder", ensure, root),
            # Builds both platforms and throws the result away. It is the whole
            # publish minus the push, which is the only check worth having here.
            Step("build both platforms", [
                "docker", "buildx", "build", "--builder", BUILDX_BUILDER,
                "--platform", plat, ".",
            ], root),
            # Builds one platform, runs it, and requires the whole line back.
            # Every other builder here checks that an artifact packs; none of
            # them check what it says, which is how npm shipped a different
            # sentence from the other five for a day. An image can be run, so
            # this one is checked, and `-x` means a sentence with something
            # appended to it fails too.
            Step("run it and read the message", ["sh", "-c",
                 f"docker buildx build --builder {BUILDX_BUILDER} --load "
                 f"-t {row.name}-placeholder-check:{version} . >/dev/null 2>&1 "
                 f"&& docker run --rm {row.name}-placeholder-check:{version} "
                 f"2>&1 | grep -qxF {shlex.quote(raises)}"], root),
        ],
        [Step("build and push", [
            "docker", "buildx", "build", "--builder", BUILDX_BUILDER,
            "--platform", plat, "--push",
            "-t", f"{tag}:{version}", "-t", f"{tag}:latest", ".",
        ], root, env=env, loud=True)],
    )


BUILDERS = {
    "crates.io": b_crates,
    "pypi": b_pypi,
    "npm": b_npm,
    "pub.dev": b_pub,
    "maven-central": b_maven,
    "nuget": b_nuget,
    "docker-hub": b_docker,
}

# The credential each builder needs before it is worth starting.
NEEDS = {
    "crates.io": ["CARGO_REGISTRY_TOKEN"],
    "pypi": ["UV_PUBLISH_TOKEN"],
    "npm": ["NPM_TOKEN"],
    "pub.dev": ["PUB_CREDENTIALS"],
    "maven-central": [
        "MAVEN_CENTRAL_USERNAME", "MAVEN_CENTRAL_PASSWORD",
        "MAVEN_GPG_KEY_ID", "MAVEN_GPG_PASSPHRASE",
    ],
    "nuget": ["NUGET_API_KEY"],
    "docker-hub": ["DOCKERHUB_USERNAME", "DOCKERHUB_TOKEN"],
}


def cmd_apply(args) -> int:
    rows = load(args.file)
    version = PLACEHOLDER
    reg = args.registry

    if reg not in BUILDERS:
        print(
            f"no builder for {reg!r}. Have: {', '.join(sorted(BUILDERS))}.\n"
            f"A registry with no builder is a registry this tool will not "
            f"publish to by hand-waving, which is the point.",
            file=sys.stderr,
        )
        return 2

    missing = [v for v in NEEDS[reg] if not os.environ.get(v)]
    if missing:
        print(f"missing credential(s): {', '.join(missing)}. Run `yoenv`.", file=sys.stderr)
        return 2

    # `free` is the normal precondition. Maven Central is the exception and it
    # is a documented one: Central publishes artifacts and does not expose
    # namespace ownership, so a verified namespace with nothing in it probes as
    # `unknown` forever (see p_maven). Waiting for it to turn `free` would wait
    # for something that cannot happen. Every other registry keeps the strict
    # rule, because there `unknown` means the probe could not ask.
    ok_states = {FREE, UNKNOWN} if reg == "maven-central" else {FREE}
    if reg == "docker-hub":
        # The second documented exception, and it is structural rather than a
        # gap in a probe. A Docker Hub namespace is an account, so once the
        # account exists every repository under it is already ours and this row
        # can never read `free` again. Waiting for `free` would wait for the
        # reservation to be undone.
        ok_states = {RESERVED, UNKNOWN}
    if args.republish:
        # A seatbelt, not a formality. This command publishes placeholders and
        # nothing else, so it may only ever push a 0.0.x. Without the check a
        # stale YODB_PLACEHOLDER_VERSION could be pushed over a real release,
        # and on npm, pub.dev and Docker Hub that moves `latest` to it.
        if not version.startswith("0.0."):
            print(
                f"refusing to republish at {version}: this command publishes "
                f"placeholders and a placeholder is a 0.0.x. Cut a real "
                f"release with the release workflow, not with this.",
                file=sys.stderr,
            )
            return 2
        ok_states = ok_states | {RESERVED, RELEASED}
    todo = [r for r in rows if r.reserve and r.registry == reg and r.state in ok_states]
    if not todo:
        print(
            f"nothing to do for {reg}: no rows in state "
            f"{', '.join(sorted(ok_states))}."
        )
        return 0

    print(f"registry:  {reg}")
    print(f"version:   {version}")
    print(f"mode:      {'PUBLISH (permanent)' if args.yes else 'dry run'}\n")

    failed = []
    for row in todo:
        desc = desc_for(row)
        print(f"  {row.name}")
        print(f"    {desc}")
        with tempfile.TemporaryDirectory(prefix=f"reserve-{reg}-") as root:
            check, live = BUILDERS[reg](row, root, version, desc)
            for step in check:
                if step.run() != 0:
                    print(f"    FAILED at: {step.label}", file=sys.stderr)
                    failed.append(row)
                    break
            else:
                if not args.yes:
                    print("    ok, not published (pass --yes)")
                    continue
                for step in live:
                    if step.run() != 0:
                        print(f"    FAILED at: {step.label}", file=sys.stderr)
                        failed.append(row)
                        break
                else:
                    row.state = RESERVED
                    row.probed = date.today().isoformat()
                    row.note = f"placeholder {version}"
                    print("    published")
        print()

    if args.yes:
        dump(args.file, rows, HEADER)
        print("names.toml updated. Run `audit` to confirm against the registry.")
    if failed:
        print(f"{len(failed)} failed: {', '.join(r.name for r in failed)}", file=sys.stderr)
        return 1
    return 0


def cmd_docs(args) -> int:
    """Check `dx/12` §2's name table against `names.toml`.

    §10 originally said this table would be *generated*, and generating it was
    tried first. It loses too much: the Why column is an argument, one row per
    name, and moving that prose into a TOML string makes it unreadable in both
    places. What the generation was actually for is the property that the two
    cannot disagree, and a check gets that property without moving anything.

    So the direction is inverted. The document keeps the prose, this asserts
    the facts in it, and a name that appears in one and not the other fails.
    """
    rows = [r for r in load(args.file) if r.reserve]
    try:
        doc = open(args.doc).read()
    except OSError as e:
        # Exit 2, not 0 and not 1. "The table disagrees" and "there was no table
        # to read" are different answers and the second one is not a pass, which
        # matters most in CI, where the spec tree is a second checkout and the
        # thing most likely to go missing.
        print(f"cannot read {args.doc}: {e}", file=sys.stderr)
        print("pass the path to dx/12-packaging-and-release.md as an argument.",
              file=sys.stderr)
        return 2

    # The name table only. The install matrix above it and the prose below it
    # both mention names in passing, and a check that accepted a name because
    # it appeared anywhere in the file would accept almost anything.
    m = re.search(r"\n## 2\. Names\n(.*?)\n## ", doc, re.S)
    if not m:
        print(f"no '## 2. Names' section in {args.doc}", file=sys.stderr)
        return 2
    section = m.group(1)
    mentioned = set(re.findall(r"`([A-Za-z0-9_.:@/-]+)`", section))

    problems = []
    for r in rows:
        if r.state not in (RESERVED, RELEASED) or r.registry not in DOC_REGISTRIES:
            continue
        # The forms a name legitimately takes in prose: bare, namespaced, and
        # for Maven the coordinate rather than the artifact id on its own.
        forms = {r.name, f"{r.namespace}/{r.name}", f"{r.namespace}:{r.name}",
                 f"{r.namespace}"}
        if not (forms & mentioned):
            problems.append(f"{r.key()} is held and dx/12 §2 does not mention it")

    known = {r.name for r in rows} | {r.namespace for r in rows if r.namespace}
    known |= {f"{r.namespace}/{r.name}" for r in rows if r.namespace}
    known |= {f"{r.namespace}:{r.name}" for r in rows if r.namespace}
    # Names in the table that no row backs. Anything that is not plausibly a
    # package name is skipped rather than guessed at: the column also carries
    # commands, paths and file extensions.
    for tok in sorted(mentioned):
        if tok in known or tok in DOC_NOT_A_NAME or "." in tok.split("/")[-1]:
            continue
        if tok.startswith((".", "/")) or " " in tok:
            continue
        if re.fullmatch(r"(dx|bench)/\d+", tok):  # a cross-reference, not a name
            continue
        problems.append(f"dx/12 §2 names `{tok}` and names.toml has no row for it")

    for p in problems:
        print(f"  {p}", file=sys.stderr)
    print(f"\n  {len(rows)} rows, {len(mentioned)} names in the table, {len(problems)} disagreement(s)")
    return 1 if problems else 0


# The registries `dx/12` §2 is a table of. GitHub repositories, DNS names and
# npm scopes are all held and none of them belongs in a table of package names:
# a repository is a URL, a domain is infrastructure, and a scope is a prefix on
# the names already listed. They are audited in `dx/16` §2.2 instead, and a
# check that demanded them here would be enforcing a table nobody wants.
DOC_REGISTRIES = {
    "crates.io", "pypi", "npm", "pub.dev", "nuget", "maven-central",
    "cocoapods", "homebrew-core", "chocolatey", "aur", "snap", "scoop-main",
    "docker-hub", "rubygems", "hex", "packagist", "conda-forge",
}

# Tokens in dx/12 §2 that are deliberately not registry names: commands, paths,
# symbols and the two channels that are pull requests rather than publishes.
DOC_NOT_A_NAME = {
    "PATH", "yo_*", "libyo", "libyo-dev", "libyo_debug", "yo-go", "yo-swift",
    "yo-web", "yo-kit", "cargo", "npm", "pip", "brew", "scoop", "apt", "dnf",
    "tamnd", "tamnd/tap/yodb", "tamnd.yodb", "yodb-bin", "com.tamnd:yodb",
    "github.com/tamnd/yo-go", "github.com/tamnd/yo-swift", "Yodb", "yo",
    "yo.tamnd.dev", "install.sh", "0.0.0",
}


# The repository root, found from this file rather than from the working
# directory, so `cargo xtask reserve` behaves the same from anywhere in the
# tree. Same reasoning as `xtask::root()` next door, and the same one level up.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The spec tree is a separate repository (`tamnd/Wiki-Notes`) and is not
# checked out beside this one on most machines, so `docs` takes the path as an
# argument. The default is where it sits on a workstation; CI passes its own
# checkout. There is deliberately no fallback that skips the check when the
# file is absent: a check that quietly passes because it could not find what it
# was checking is the failure mode this whole file exists to argue against.
SPEC_DOC = os.path.expanduser("~/notes/Spec/2064yo/dx/12-packaging-and-release.md")


def main() -> int:
    ap = argparse.ArgumentParser(prog="cargo xtask reserve")
    ap.add_argument("--file", default=os.path.join(ROOT, "names.toml"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n, fn in (("audit", cmd_audit), ("plan", cmd_plan), ("verify", cmd_verify)):
        sub.add_parser(n).set_defaults(fn=fn)
    ap_apply = sub.add_parser("apply")
    ap_apply.add_argument("registry", help="one registry per invocation, on purpose")
    ap_apply.add_argument(
        "--yes", action="store_true",
        help="actually publish. Without it, everything is built and checked "
             "and nothing leaves the machine.",
    )
    ap_apply.add_argument(
        "--republish", action="store_true",
        help="also act on names already held, to move them to a new "
             "placeholder version. Refuses anything that is not a 0.0.x.",
    )
    ap_apply.set_defaults(fn=cmd_apply)
    ap_docs = sub.add_parser("docs")
    ap_docs.add_argument("doc", nargs="?", default=SPEC_DOC)
    ap_docs.set_defaults(fn=cmd_docs)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
