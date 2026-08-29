#!/usr/bin/env bash
# Install every published yo placeholder on a machine that has never seen it.
#
# Run it against a Docker host that has none of this project's caches on it:
#
#     ssh <host> 'bash -s' < xtask/blank-machine-install.sh
#
# It lives in the repository because the first two versions of it did not. Both
# were written at a prompt, both grew the same fix for the same Maven bug, and
# the second grew it because the first had been thrown away with the terminal
# it was typed into. `dx/12` section 1 says every install path is exercised in
# CI on a clean machine per release; until that exists this is what stands in
# for it, and a stand-in nobody can re-run is not one.
#
# Each case is a fresh container with only its language toolchain in it, so a
# pass means the registry copy works for a stranger, not that a leftover cache
# on this host happens to contain the right bytes.
#
# `bash -c` and not `bash -lc`. A login shell re-reads /etc/profile in these
# images and replaces PATH with the distro default, which drops /usr/local/cargo
# and /usr/lib/dart/bin. The first run of this script did that and reported
# "cargo: command not found" as if the crate were broken.

WANT='yo is not usable yet. This is a reserved placeholder at 0.0.1; see https://github.com/tamnd/yo'

pass=0; fail=0

run() {   # run <label> <image> <script>
  local label=$1 image=$2 script=$3 out rc
  echo
  echo "=============================================================="
  echo "== $label   ($image)"
  echo "=============================================================="
  out=$(docker run --rm --network host -w /w "$image" bash -c "$script" 2>&1)
  rc=$?
  echo "$out"
  echo "-- exit $rc"
  if printf '%s' "$out" | grep -qF "$WANT"; then
    echo "== $label: PASS (canonical message)"
    pass=$((pass+1))
  else
    echo "== $label: FAIL (canonical message absent)"
    fail=$((fail+1))
  fi
}

run rust rust:slim '
  cargo new --quiet t && cd t &&
  cargo add yodb 2>&1 | tail -3 &&
  printf "fn main(){ yodb::open(\"x.yo\"); }\n" > src/main.rs &&
  cargo run --quiet 2>&1 | tail -5
'

run python python:3.13-slim '
  pip install --quiet yodb &&
  pip show yodb | head -2 &&
  python -c "import yodb; yodb.open(\"x.yo\")" 2>&1 | tail -3
'

run node node:22-slim '
  npm install --silent @yodb/core &&
  node --input-type=module -e "
    import m from \"@yodb/core\";
    try { m.open(\"x.yo\") } catch (e) { console.log(e.message) }
  "
'

run dotnet mcr.microsoft.com/dotnet/sdk:9.0 '
  dotnet new console -o t --force >/dev/null 2>&1 && cd t &&
  dotnet add package Yodb >/dev/null 2>&1 &&
  printf "try { Yodb.Yo.Open(\"x.yo\"); } catch (Exception e) { Console.WriteLine(e.Message); }\n" > Program.cs &&
  dotnet run 2>&1 | tail -3
'

run dart dart:stable '
  mkdir -p t/lib && cd t &&
  printf "name: t\nversion: 0.0.1\nenvironment:\n  sdk: ^3.0.0\n" > pubspec.yaml &&
  dart pub add yodb >/dev/null 2>&1 &&
  mkdir -p bin &&
  printf "import \"package:yodb/yodb.dart\" as yodb;\nvoid main(){ try { yodb.open(\"x.yo\"); } catch (e) { print(e); } }\n" > bin/t.dart &&
  dart run bin/t.dart 2>&1 | tail -3
'

run java maven:3-eclipse-temurin-21 '
  mkdir -p t/src/main/java && cd t &&
  cat > pom.xml <<EOF
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>t</groupId><artifactId>t</artifactId><version>1</version>
  <properties><maven.compiler.release>21</maven.compiler.release></properties>
  <dependencies>
    <dependency><groupId>com.tamnd</groupId><artifactId>yodb</artifactId><version>0.0.1</version></dependency>
  </dependencies>
</project>
EOF
  printf "public class T { public static void main(String[] a){ try { com.tamnd.yodb.Yo.open(\"x.yo\"); } catch (Throwable e) { System.out.println(e.getMessage()); } } }\n" > src/main/java/T.java &&
  # Not `-o=false`. `-o` IS offline mode and takes no value, so that spelling
  # ran Maven offline, which could not fetch the exec plugin and failed with
  # NoPluginFoundForPrefixException, which is an error that looks nothing like
  # "you asked for offline mode". It cost an hour on 2026-08-28, was fixed at
  # the prompt rather than in this file, and so came back on 2026-08-29.
  #
  # No exec plugin either. Resolve the classpath, compile, run it with java.
  # One less plugin to download is one less thing that can fail for a reason
  # that has nothing to do with the artifact under test.
  mvn -q dependency:build-classpath -Dmdep.outputFile=cp.txt 2>&1 | tail -5
  javac -cp "$(cat cp.txt)" -d out src/main/java/T.java &&
  java -cp "out:$(cat cp.txt)" T 2>&1 | tail -5
'

# Go and Swift have no registry and no placeholder release, so there is no
# canonical message to grep for. What is being asked of them is narrower and is
# the whole of what dx/12 section 1 promises: can a stranger with only the
# toolchain resolve the dependency the install matrix tells them to write.
probe() {   # probe <label> <image> <script>
  local label=$1 image=$2 script=$3
  echo
  echo "=============================================================="
  echo "== $label   ($image)   [resolution probe, no message expected]"
  echo "=============================================================="
  docker run --rm --network host -w /w "$image" bash -c "$script" 2>&1
  echo "-- exit $?"
}

# Promoted from a resolution probe to a real verdict on 2026-08-29. On the
# 08-28 run this could only be a probe, because the module had no package in it
# and so had no message to check. It has both now.
run go golang:1.26 '
  go mod init t >/dev/null 2>&1
  GOFLAGS=-mod=mod go get github.com/tamnd/yo-go@v0.0.1 2>&1 | tail -3
  cat > main.go <<EOF
package main

import (
  "fmt"
  yo "github.com/tamnd/yo-go"
)

func main() {
  _, err := yo.Open("app.yo")
  fmt.Println(err)
}
EOF
  go run .
'

# Docker has no toolchain container to run inside, because the artifact under
# test is the container. So this one runs on the host and deletes the image
# first, which is what makes it a blank-machine check rather than a check that
# this host still has the bytes the push put there twenty minutes ago.
#
# No tag. `docker run tamnd87/yo` is what the README tells a reader to type, so
# it is what gets tested, and it also checks that `latest` points somewhere.
echo
echo "=============================================================="
echo "== docker   (tamnd87/yo, on the host)"
echo "=============================================================="
docker rmi -f tamnd87/yo:latest tamnd87/yo:0.0.1 >/dev/null 2>&1
out=$(docker run --rm tamnd87/yo 2>&1); rc=$?
echo "$out"
echo "-- exit $rc"
if printf '%s' "$out" | grep -qF "$WANT" && [ "$rc" -ne 0 ]; then
  echo "== docker: PASS (canonical message, non-zero exit)"
  pass=$((pass+1))
else
  echo "== docker: FAIL"
  fail=$((fail+1))
fi

# Two Package.swift files, because they answer two different questions. The
# first is what dx/12 section 1 tells a user to write and it is the one that
# matters. The second says whether the failure is the missing tag or the code.
probe swift swift:6.2 '
  mkdir -p t/Sources/t && cd t && printf "" > Sources/t/main.swift
  write() {
    cat > Package.swift <<EOF
// swift-tools-version:6.0
import PackageDescription
let package = Package(
  name: "t",
  dependencies: [.package(url: "https://github.com/tamnd/yo-swift", $1)],
  targets: [.executableTarget(name: "t", dependencies: [.product(name: "Yodb", package: "yo-swift")])])
EOF
  }
  echo "--- as the install matrix documents it: from: \"0.0.1\""
  write "from: \"0.0.1\"" ; swift build 2>&1 | tail -4
  echo "--- same package pinned to the branch instead"
  write "branch: \"main\"" ; swift build 2>&1 | tail -4
'

echo
echo "=============================================================="
echo "pass=$pass fail=$fail"
