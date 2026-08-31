"""Drive a yodb under YO_ALLOC=report and print what allocated on a command path.

Y7 says nothing allocates on a command path. yo-alloc can enforce that by
aborting, but an abort tells you about one violation per run, which is the wrong
tool for finding out how many there are. Report mode captures a backtrace for
each distinct site instead, and this is the thing that produces the workload
those backtraces come from.

It builds yodb, starts it on a free port with the check armed, sends about four
and a half thousand commands covering every type, stops the server and prints
one line per distinct site with the innermost frame that is in this repository.

Two passes over the same commands. The first one creates every key it touches,
so it sees a key coming into existence, which is the one allocation a command is
allowed to make and is annotated as yo_alloc::first_touch where it happens. The
second pass runs against keys that are already there, which is the steady state
Y7 is actually about.

A debug build on purpose. Release inlines the interesting frames into each other
and drops the line numbers, so the report comes back pointing at serve_command
for everything and you learn nothing.

    cargo xtask alloc            build, run and print the list
    cargo xtask alloc --keep     the same, and leave the raw log behind
"""

import os
import re
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KEEP = "--keep" in sys.argv[1:]


def enc(*parts):
    out = b"*%d\r\n" % len(parts)
    for p in parts:
        b = p.encode() if isinstance(p, str) else p
        out += b"$%d\r\n%s\r\n" % (len(b), b)
    return out


def workload(tag):
    def k(name):
        return "%s:%s" % (tag, name)

    cmds = []
    a = cmds.append
    for i in range(50):
        a(("SET", k("s%d" % i), "value-%d" % i))
        a(("GET", k("s%d" % i)))
        a(("APPEND", k("s%d" % i), "-more"))
        a(("SETRANGE", k("s%d" % i), "2", "XY"))
        a(("STRLEN", k("s%d" % i)))
        a(("GETRANGE", k("s%d" % i), "0", "5"))
        a(("EXPIRE", k("s%d" % i), "10000"))
        a(("TTL", k("s%d" % i)))
        a(("PERSIST", k("s%d" % i)))
        a(("INCR", k("n%d" % i)))
        a(("INCRBY", k("n%d" % i), "7"))
        a(("INCRBYFLOAT", k("f%d" % i), "1.5"))
        a(("DECR", k("n%d" % i)))
        a(("SETEX", k("e%d" % i), "100", "v"))
        a(("GETSET", k("s%d" % i), "reset"))
        a(("SETNX", k("s%d" % i), "no"))
        a(("EXISTS", k("s%d" % i)))
        a(("TYPE", k("s%d" % i)))

        a(("SADD", k("set%d" % i), "1", "2", "3", "4"))
        a(("SADD", k("txt%d" % i), "alpha", "beta", "gamma"))
        a(("SISMEMBER", k("set%d" % i), "2"))
        a(("SMISMEMBER", k("set%d" % i), "2", "9"))
        a(("SCARD", k("set%d" % i)))
        a(("SMEMBERS", k("set%d" % i)))
        a(("SRANDMEMBER", k("set%d" % i), "2"))
        a(("SINTER", k("set%d" % i), k("set%d" % ((i + 1) % 50))))
        a(("SUNION", k("set%d" % i), k("set%d" % ((i + 1) % 50))))
        a(("SDIFF", k("set%d" % i), k("set%d" % ((i + 1) % 50))))
        a(("SINTER", k("txt%d" % i), k("txt%d" % ((i + 1) % 50))))
        a(("SUNION", k("txt%d" % i), k("txt%d" % ((i + 1) % 50))))
        a(("SREM", k("set%d" % i), "4"))
        a(("SMOVE", k("set%d" % i), k("set%d" % ((i + 1) % 50)), "3"))

        a(("HSET", k("h%d" % i), "a", "1", "b", "two"))
        a(("HGET", k("h%d" % i), "a"))
        a(("HMGET", k("h%d" % i), "a", "b", "c"))
        a(("HINCRBY", k("h%d" % i), "a", "3"))
        a(("HINCRBYFLOAT", k("h%d" % i), "g", "2.25"))
        a(("HGETALL", k("h%d" % i)))
        a(("HKEYS", k("h%d" % i)))
        a(("HVALS", k("h%d" % i)))
        a(("HLEN", k("h%d" % i)))
        a(("HEXISTS", k("h%d" % i), "a"))
        a(("HRANDFIELD", k("h%d" % i), "2"))
        a(("HDEL", k("h%d" % i), "b"))

        a(("RPUSH", k("l%d" % i), "x", "y", "z"))
        a(("LPUSH", k("l%d" % i), "w"))
        a(("LRANGE", k("l%d" % i), "0", "-1"))
        a(("LINDEX", k("l%d" % i), "1"))
        a(("LLEN", k("l%d" % i)))
        a(("LSET", k("l%d" % i), "0", "q"))
        a(("LINSERT", k("l%d" % i), "BEFORE", "y", "mid"))
        a(("LPOS", k("l%d" % i), "y"))
        a(("LMOVE", k("l%d" % i), k("l2%d" % i), "LEFT", "RIGHT"))
        a(("RPOPLPUSH", k("l%d" % i), k("l2%d" % i)))
        a(("LREM", k("l%d" % i), "1", "z"))
        a(("LPOP", k("l%d" % i)))
        a(("RPOP", k("l%d" % i)))
        a(("LTRIM", k("l2%d" % i), "0", "5"))

        a(("ZADD", k("z%d" % i), "1", "one", "2", "two", "3.5", "three"))
        a(("ZSCORE", k("z%d" % i), "one"))
        a(("ZINCRBY", k("z%d" % i), "2.5", "one"))
        a(("ZRANGE", k("z%d" % i), "0", "-1"))
        a(("ZRANGE", k("z%d" % i), "0", "-1", "WITHSCORES"))
        a(("ZRANGEBYSCORE", k("z%d" % i), "0", "10"))
        a(("ZRANK", k("z%d" % i), "two"))
        a(("ZREVRANK", k("z%d" % i), "two"))
        a(("ZCARD", k("z%d" % i)))
        a(("ZCOUNT", k("z%d" % i), "0", "10"))
        a(("ZRANDMEMBER", k("z%d" % i), "2"))
        a(("ZREM", k("z%d" % i), "three"))

        a(("SETBIT", k("bits%d" % i), "100", "1"))
        a(("GETBIT", k("bits%d" % i), "100"))
        a(("BITCOUNT", k("bits%d" % i)))
        a(("BITPOS", k("bits%d" % i), "1"))
        a(("PFADD", k("hll%d" % i), "a", "b", "c"))
        a(("PFCOUNT", k("hll%d" % i)))

        a(("MSET", k("m1%d" % i), "a", k("m2%d" % i), "b"))
        a(("MGET", k("m1%d" % i), k("m2%d" % i)))
        a(("RENAME", k("m1%d" % i), k("m3%d" % i)))
        a(("COPY", k("m3%d" % i), k("m4%d" % i)))
        a(("DEL", k("m4%d" % i)))
        a(("RANDOMKEY",))
        a(("SCAN", "0", "COUNT", "10"))
        a(("DBSIZE",))
        a(("MULTI",))
        a(("SET", k("tx%d" % i), "in-a-transaction"))
        a(("INCR", k("txn%d" % i)))
        a(("EXEC",))
    a(("PING",))
    a(("INFO",))
    a(("COMMAND", "COUNT"))
    a(("CONFIG", "GET", "maxmemory"))
    a(("SUBSCRIBE", "chan"))
    a(("UNSUBSCRIBE", "chan"))
    return cmds


def free_port():
    """A port nothing is on right now.

    Bound and released rather than picked out of a range, so a second copy of
    this running at the same time does not land on the same number. The server
    refuses to start onto a port something is already answering on, so the
    remaining race turns into a clear error rather than a wrong answer.
    """
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_ready(port, proc):
    deadline = time.time() + 30
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=1)
            s.sendall(enc("PING"))
            if b"PONG" in s.recv(64):
                s.close()
                return True
            s.close()
        except OSError:
            time.sleep(0.05)
    return False


def drive(port, passes=2):
    sock = socket.create_connection(("127.0.0.1", port))
    sock.settimeout(60)
    sock.sendall(enc("HELLO", "3"))
    sock.recv(1 << 16)
    total = 0
    for p in range(passes):
        cmds = workload("p0")
        sock.sendall(b"".join(enc(*c) for c in cmds))
        total += len(cmds)
        # Drained by sending a PING and reading until it comes back. The replies
        # are not parsed, because all that matters is that the server has run
        # everything before the next pass starts.
        sock.sendall(enc("PING"))
        buf = b""
        while b"+PONG\r\n" not in buf[-64:]:
            chunk = sock.recv(1 << 20)
            if not chunk:
                break
            buf = buf[-64:] + chunk
    sock.close()
    return total


def innermost(trace):
    """The first frame in the trace that is code in this repository.

    Everything above it is the allocator, the reporting machinery and the
    standard library's vector growth, which is the same for every site and says
    nothing about which line asked for the memory.
    """
    lines = trace.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"\s*\d+: (.*)", line)
        if not m or i + 1 >= len(lines):
            continue
        below = lines[i + 1].strip()
        if not below.startswith("at "):
            continue
        path = below[3:]
        if "/crates/" not in path:
            continue
        if "/yo-alloc/" in path or "/yo-cli/src/main.rs" in path:
            continue
        rel = path.split("/crates/", 1)[1]
        return "crates/" + rel, m.group(1)
    return None, None


def main():
    print("building yodb, debug, so the backtraces keep their line numbers")
    build = subprocess.run(
        ["cargo", "build", "-p", "yo-cli", "--bin", "yodb"], cwd=ROOT
    )
    if build.returncode != 0:
        return build.returncode

    binary = os.path.join(ROOT, "target", "debug", "yodb")
    port = free_port()
    log = tempfile.NamedTemporaryFile(
        prefix="yo-alloc-", suffix=".log", delete=False, mode="w+"
    )
    env = dict(os.environ, YO_ALLOC="report", RUST_BACKTRACE="full")
    proc = subprocess.Popen(
        [binary, "serve", "--port", str(port)],
        cwd=tempfile.gettempdir(),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        if not wait_ready(port, proc):
            log.flush()
            print("the server never came up. Its output was:")
            print(open(log.name).read())
            return 1
        sent = drive(port)
        print("sent %d commands on port %d" % (sent, port))
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        log.flush()

    text = open(log.name).read()
    if not KEEP:
        os.unlink(log.name)

    blocks = text.split("yo: allocation on a marked thread: ")[1:]
    print("")
    if not blocks:
        print("nothing allocated on a command path.")
        return 0
    print("%d distinct site(s):" % len(blocks))
    for b in blocks:
        head = b.split("\n")[0]
        where, what = innermost(b)
        print("  %-22s %s" % (head, where or "outside this repository"))
        if what:
            print("  %-22s   %s" % ("", what))
    if KEEP:
        print("")
        print("raw log at %s" % log.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
