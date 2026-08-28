#!/usr/bin/env bash
#
# Builds and runs the C example against the static library.
#
# The point is not that the example works. The point is that the header compiles
# under every dialect a caller might be in, with warnings as errors, because a
# header that only builds in the author's dialect is not a portable ABI. So this
# compiles the same file once per standard and only then links and runs it.
#
# Usage: examples/c/build.sh [cc]

set -euo pipefail

CC="${1:-${CC:-cc}}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/target/c-example"
mkdir -p "$OUT"

echo "building the static library"
cargo build --release --package yo-capi --manifest-path "$ROOT/Cargo.toml"

LIB="$ROOT/target/release/libyo_capi.a"
if [ ! -f "$LIB" ]; then
  echo "expected $LIB to exist" >&2
  exit 1
fi

WARN="-Wall -Wextra -Wpedantic -Werror -Wconversion -Wshadow -Wcast-qual"

# Every dialect the header claims to support. A warning in any of them is a
# failure, because the caller who hits it will be someone else.
for std in c99 c11 c17 c23; do
  if echo 'int main(void){return 0;}' | "$CC" -std="$std" -x c - -o /dev/null 2>/dev/null; then
    echo "compiling as $std"
    # shellcheck disable=SC2086
    "$CC" -std="$std" $WARN -DYO_STATIC -I "$ROOT/include" \
      -c "$ROOT/examples/c/hello.c" -o "$OUT/hello.$std.o"
  else
    echo "skipping $std, this compiler does not have it"
  fi
done

# And as C++, because a C++ caller including a C header is the most common way
# an extern "C" block turns out to be missing.
CXX="${CXX:-c++}"
for std in c++17 c++20; do
  if echo 'int main(){return 0;}' | "$CXX" -std="$std" -x c++ - -o /dev/null 2>/dev/null; then
    echo "compiling as $std"
    # shellcheck disable=SC2086
    "$CXX" -std="$std" -Wall -Wextra -Wpedantic -Werror -DYO_STATIC \
      -I "$ROOT/include" -x c++ -c "$ROOT/examples/c/hello.c" \
      -o "$OUT/hello.$std.o"
  else
    echo "skipping $std, this compiler does not have it"
  fi
done

echo "linking"
LINK_EXTRA=""
case "$(uname -s)" in
  Linux) LINK_EXTRA="-lpthread -ldl -lm" ;;
  Darwin) LINK_EXTRA="-framework CoreFoundation" ;;
esac
# shellcheck disable=SC2086
"$CC" -std=c11 $WARN -DYO_STATIC -I "$ROOT/include" \
  "$ROOT/examples/c/hello.c" "$LIB" $LINK_EXTRA -o "$OUT/hello"

echo "running"
"$OUT/hello"
