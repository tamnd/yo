/* The sixty second C program.
 *
 * Open a database, write a key, read it back, read a batch of keys through an
 * arena, close everything. That is the whole ABI at this milestone, and if this
 * file stops compiling or stops printing what it printed yesterday then the ABI
 * has moved and somebody should know before a binding finds out.
 *
 * Build it with examples/c/build.sh, which is also what CI runs.
 *
 * SPDX-License-Identifier: MIT OR Apache-2.0
 */

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "yo.h"

/* A slice over a C string literal, which is what most keys are in practice. */
static yo_slice s(const char *text) {
  yo_slice out;
  out.ptr = (const uint8_t *)text;
  out.len = (uint64_t)strlen(text);
  return out;
}

static void print_error(const char *what, const yo_error *err) {
  fprintf(stderr, "%s failed: %s (%s, retryable=%u)\n", what,
          err->message ? err->message : "no message", yo_code_name(err->code),
          (unsigned)err->retryable);
  if (err->url) {
    fprintf(stderr, "  see %s\n", err->url);
  }
  if (err->detail) {
    fprintf(stderr, "%s\n", err->detail);
  }
}

int main(void) {
  yo_error err = YO_ERROR_INIT;
  yo_open_options opts = YO_OPEN_OPTIONS_INIT;
  yo_db *db;
  yo_arena *arena;
  yo_slice value;
  yo_slice keys[3];
  yo_slice results[3];
  int32_t rc;
  int i;

  /* Step one, always: check that the library you loaded is the one this file
   * was compiled against. A major mismatch means the structs above have moved
   * and every call after this point is reading someone else's memory. */
  if ((yo_abi_version() >> 16) != YO_ABI_VERSION_MAJOR) {
    fprintf(stderr, "abi major mismatch: header %d, library %u\n",
            YO_ABI_VERSION_MAJOR, yo_abi_version() >> 16);
    return 1;
  }
  printf("yo %s, abi %u.%u\n", yo_version_string(), yo_abi_version() >> 16,
         yo_abi_version() & 0xffff);

  /* Inline mode: this thread is the shard. It is the default and it is what
   * makes a point read cost a probe rather than a queue. */
  opts.shards = 1;
  db = yo_open(NULL, &opts, &err);
  if (db == NULL) {
    print_error("yo_open", &err);
    return 1;
  }

  if (yo_set(db, s("user:42"), s("tam"), &err) != 0) {
    print_error("yo_set", &err);
    return 1;
  }

  /* The borrowed read. No copy at all: the slice points into the engine and is
   * good until the next write. */
  rc = yo_get(db, s("user:42"), &value, &err);
  if (rc < 0) {
    print_error("yo_get", &err);
    return 1;
  }
  assert(rc == 1);
  printf("user:42 = %.*s\n", (int)value.len, (const char *)value.ptr);

  /* A miss is a return value, not an error. Nothing is written to err. */
  rc = yo_get(db, s("user:99"), &value, &err);
  assert(rc == 0);
  assert(value.ptr == NULL);
  assert(err.code == YO_OK);
  printf("user:99 is not there, and that is not an error\n");

  /* The batch. One crossing, one arena, three results. This is the shape every
   * binding uses on the hot path, and the reason a loop over ten thousand rows
   * does not become ten thousand allocations. */
  arena = yo_arena_new(db, &err);
  if (arena == NULL) {
    print_error("yo_arena_new", &err);
    return 1;
  }

  yo_set(db, s("a"), s("alpha"), &err);
  yo_set(db, s("c"), s("gamma"), &err);
  keys[0] = s("a");
  keys[1] = s("b");
  keys[2] = s("c");

  rc = yo_get_many(db, keys, 3, arena, results, &err);
  if (rc < 0) {
    print_error("yo_get_many", &err);
    return 1;
  }
  printf("batch found %d of 3 in %llu arena bytes\n", (int)rc,
         (unsigned long long)yo_arena_used(arena));
  for (i = 0; i < 3; i++) {
    if (results[i].ptr == NULL) {
      printf("  %.*s is missing\n", (int)keys[i].len, (const char *)keys[i].ptr);
    } else {
      printf("  %.*s = %.*s\n", (int)keys[i].len, (const char *)keys[i].ptr,
             (int)results[i].len, (const char *)results[i].ptr);
    }
  }

  /* Reset rather than free, if there is another batch coming. Every pointer in
   * results is dead the moment this returns. */
  yo_arena_reset(arena);
  assert(yo_arena_used(arena) == 0);

  /* An error on purpose, so the example shows what one looks like rather than
   * only describing it. Owned mode is not implemented yet and says so. */
  opts.shards = 8;
  if (yo_open(NULL, &opts, &err) == NULL) {
    printf("asking for 8 shards: %s (%s)\n", err.message,
           yo_code_name(err.code));
  }

  yo_arena_free(arena);
  yo_close(db);
  printf("ok\n");
  return 0;
}
