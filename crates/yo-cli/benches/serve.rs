//! What a thread count is worth over a real socket.
//!
//! The number the MT milestone is gated on comes from tamnd/cache-bench, which
//! drives eight servers through `memtier_benchmark` on a pinned Linux box and
//! takes a day and a half to answer once. That is the right way to publish a
//! comparison and the wrong way to find out whether a change helped, because a
//! loop that takes a day and a half is a loop nobody turns.
//!
//! This asks the same shape of question in a minute on whatever machine is in
//! front of you. A real `yodb serve` in its own process, a real unix socket,
//! real clients on their own threads, and the only thing varied is `--threads`.
//! It compares yo against nothing and it publishes nothing. It answers one
//! question: what does the next thread buy.
//!
//! # Reading it
//!
//! The column that matters is `vs 1`, which is throughput at N threads over
//! throughput at one. A server that scales gets a number near N and a server
//! that does not gets a number near one, and no amount of absolute throughput
//! makes up for the second one. The first published sweep has yo going from
//! 1890 to 2980 Kops between one thread and eight where Pogocache goes from
//! 1696 to 13244, which is 1.6 against 7.8, and that ratio is the whole problem
//! this bench exists to shorten the loop on.
//!
//! The column to read before that one is `cv`. Every cell is measured three
//! times and what is reported is the median of the three and how far apart they
//! were, because the first thing a bench has to be able to say is that the
//! machine it is on cannot answer. A laptop with a virtual machine and two
//! builds running answered 36 Kops one run and 133 the next for the same
//! server. A cell above 0.05 is marked noisy and there is a line at the end
//! saying how many were, and a run with any of those in it is a run to throw
//! away rather than a run to read carefully.
//!
//! Absolute throughput here is not comparable with anything from the harness.
//! The clients are on the same machine and are not pinned away from the server,
//! so at high thread counts the load generator competes with the thing it is
//! measuring. That flattens the curve rather than steepening it, so a good
//! ratio here is believable and a bad one is worth checking on a real box
//! before believing it.
//!
//! # Running it
//!
//! ```text
//! cargo bench -p yo-cli --bench serve
//! YO_BENCH_THREADS=1,2,4,8 YO_BENCH_PIPELINE=1,50 cargo bench -p yo-cli --bench serve
//! YO_BENCH_CLIENTS=2 YO_BENCH_CONNS=8 cargo bench -p yo-cli --bench serve
//! YO_BENCH_REPEATS=7 cargo bench -p yo-cli --bench serve
//! YO_BENCH_OP=set cargo bench -p yo-cli --bench serve
//! ```
//!
//! `YO_BENCH_SMOKE` cuts it to one short cell, which is what CI runs to find
//! out that it still builds and still runs.
//!
//! # Reads or writes
//!
//! `YO_BENCH_OP` picks what the measured window sends, `get` by default and
//! `set` for the other one. Both run against the same filled keyspace, so a
//! `set` is an overwrite of a key that is already there rather than an insert,
//! and what is measured is the write path rather than the map growing.
//!
//! Writes are worth measuring separately because they are not the same shape of
//! work. A read takes a stripe's lock and copies a value out. A write takes the
//! same lock, replaces the value, and then owes the rest of the server an
//! account of what it did: keyspace notifications, the expiry sweep, and the
//! replication feed. The published sweep has yo answering three to five reads
//! for every write it answers where the engines beside it are nearer one for
//! one, and that gap is what this mode exists to shorten the loop on.
//!
//! `YO_BENCH_OP_AGAINST` sets the far side of a pair, so the two halves differ
//! by the operation and by nothing else and the ratio is what a write costs
//! over a read. That is the form to use on a machine that will not hold still,
//! for the same reason the paired build comparison is.
//!
//! ```text
//! YO_BENCH_OP=set YO_BENCH_OP_AGAINST=get YO_BENCH_REPEATS=11 \
//!   YO_BENCH_AGAINST=target/release/yodb cargo bench -p yo-cli --bench serve
//! ```
//!
//! `YO_BENCH_SIZE` is how long a value is, `1` by default, and it takes a range
//! as well as a number. `YO_BENCH_SIZE=1-1024` is the harness's
//! `--data-size-range` and means the same thing: every value gets a length
//! drawn from it, and the fill draws separately from the measured pass, so a
//! write is over a value of some other length than the one already there.
//!
//! That is a different setting from `YO_BENCH_SIZE=1024` rather than a longer
//! one, and the difference is the point. A `SET` over a value the length of the
//! old one can reuse what is already allocated and a `SET` over a value of some
//! other length cannot, so a bench that writes one length every time measures
//! the first of those and calls it the write path. Anything being blamed on the
//! write path wants checking at both.
//!
//! `YO_BENCH_KEYS` is how many keys each client thread cycles through, a
//! hundred thousand by default against the two and a half million the harness
//! holds. Anything the server does that walks the keyspace rather than one key
//! of it costs nothing at the default and something at the harness's size, so
//! a gap that only opens when this is turned up is a gap with a cause.
//!
//! # Comparing two builds on a machine that is not quiet
//!
//! `YO_BENCH_AGAINST` points at a second `yodb` binary and turns the sweep into
//! an A/B. Every cell is then measured twice per repeat, once with each binary,
//! back to back and in alternating order, and what is reported is the ratio
//! between them.
//!
//! This is the mode to reach for when the box will not hold still, and what
//! makes it work is the median rather than the pairing. A pair is measured
//! minutes apart at most, which cancels the slow drift, and it does not cancel
//! a compiler that had half the cores for six seconds of one half of it. So an
//! individual pair on a busy machine is still a coin toss and the median of
//! enough of them is not.
//!
//! That is measured rather than assumed. On a laptop at a load average of ten,
//! with a virtual machine and two builds on it, this binary against a copy of
//! itself read as follows.
//!
//! ```text
//! three pairs      1689 Kops against 1527      1.16x
//! eleven pairs     3680 Kops against 3242      1.02x
//! ```
//!
//! The true answer is 1.00x, the absolute throughput moved by a factor of two
//! between those two runs, and eleven pairs got within two percent of the
//! truth anyway. Three did not.
//!
//! So the column to read is `agree`, which is how many of the pairs came out on
//! the same side of 1.00 as the median did. Ten of eleven is a result and six
//! of eleven is a coin toss, whatever the median says, and a cell where fewer
//! than three quarters agree is marked. The `cv` beside it is the spread of the
//! ratio, which on a busy box is large and is not by itself a reason to throw
//! the cell away. The way to find out what this machine needs is to run it
//! against a copy of the same binary and turn the repeats up until it says
//! 1.00x, which is a calibration nothing else here can give you.
//!
//! Both binaries have to be built the same way, since half of a ratio built at
//! a different optimisation level is not a comparison of anything. The binary
//! this drives by default is the one cargo built for the bench, which is the
//! `bench` profile, so the other one is built with `--profile bench` too and
//! lands in `target/release`.
//!
//! ```text
//! git worktree add /tmp/before <commit>
//! (cd /tmp/before && cargo build --profile bench -p yo-cli)
//! YO_BENCH_REPEATS=11 YO_BENCH_AGAINST=/tmp/before/target/release/yodb \
//!   cargo bench -p yo-cli --bench serve
//! ```
//!
//! What it cannot do is tell you the machine was quiet. A ratio measured while
//! a compiler had half the cores is a true ratio under that load, and a change
//! that only pays off on an idle box will look smaller than it is. It answers
//! whether this build beats that one on this machine, which is the question a
//! change is judged on, and not what either of them is worth, which is the
//! question the published sweep answers.
//!
//! `YO_BENCH_DEBUG` is sent to the server as `DEBUG` subcommands before the
//! clients start, separated by semicolons, and it is how a cost gets pinned on
//! a part of the server rather than guessed at. Turning a job off and measuring
//! again says what that job was costing, which no profile of a server doing
//! four things at once will tell you as plainly. Its first use was this, at
//! pipeline 50 on a ten core laptop with two client threads:
//!
//! ```text
//!                                  1 thread   4 threads
//! everything on                        6417        3283
//! DEBUG PAUSE-CRON 1                   6217        5573
//! DEBUG DICT-RESIZING 0                5176        4518
//! DEBUG SET-ACTIVE-EXPIRE 0            6506        5661
//! ```
//!
//! One thread pays nothing for the maintenance slice and four threads pay
//! nearly half their throughput for it, and turning off either of the two jobs
//! recovers most of that. So it is not the work either job does, it is the two
//! of them walking every stripe of every database on every worker and taking
//! each stripe's lock to find nothing.
//!
//! Those four numbers are four separate runs, which is why they were taken on a
//! quiet machine and why they could not be taken again on a busy one.
//! `YO_BENCH_DEBUG_AGAINST` sets the far side of a pair instead, so the same
//! attribution becomes one paired run of the same build against itself with one
//! job turned off, and the ratio is what that job costs.
//!
//! ```text
//! YO_BENCH_DEBUG="PAUSE-CRON 1" YO_BENCH_DEBUG_AGAINST= YO_BENCH_REPEATS=11 \
//!   YO_BENCH_AGAINST=target/release/yodb cargo bench -p yo-cli --bench serve
//! ```
//!
//! Unset, the far side is set up the same way this one is, which is what a
//! comparison of two builds wants. Set, including set to nothing as above, the
//! two sides differ by that and by nothing else.
//!
//! A number measured with any of this set is a number about a server that is
//! not doing its job, and belongs in an argument about where time goes rather
//! than in a claim about how fast yo is.

/// Everything here is a unix socket and a forked server, and Windows has
/// neither in the shape this uses. The bench still has to build there, because
/// the test matrix builds every target on all three, so on Windows it is a main
/// that says why it did nothing.
#[cfg(not(unix))]
fn main() {
    println!("this bench drives a unix socket, so there is nothing to measure here");
}

#[cfg(unix)]
fn main() {
    unix::main();
}

#[cfg(unix)]
mod unix {
    use std::io::{ErrorKind, Read, Write};
    use std::os::unix::net::UnixStream;
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command, Stdio};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Arc, Barrier};
    use std::time::{Duration, Instant};

    /// Thread counts to sweep, unless the environment says otherwise.
    ///
    /// Powers of two up to eight, because the interesting part of the curve is
    /// the start of it. A server that has stopped scaling by four threads does
    /// not start again at sixteen, and a laptop cannot say anything honest
    /// about sixteen anyway.
    const THREADS: &[usize] = &[1, 2, 4, 8];

    /// Pipeline depths to sweep.
    ///
    /// One and fifty, the two ends of the harness's range. At one the cost is a
    /// read and a write per operation and every engine lands near the same
    /// number. At fifty the syscalls are amortised away and what is left is the
    /// engine, which is where a thread count either shows up or does not.
    const PIPELINE: &[usize] = &[1, 50];

    /// How many client threads offer load, and how many connections each holds.
    ///
    /// Sixty four connections in all, enough that a server with eight threads
    /// has something for each of them and few enough that one machine can drive
    /// them without the client becoming the measurement.
    ///
    /// The thread count is worth overriding on a small machine, and this is why.
    /// A worker that has run out of work spins before it sleeps, so N server
    /// threads occupy N cores whether or not there is anything for them to do.
    /// Add the client threads and a box with fewer cores than that total is
    /// measuring the scheduler: on a ten core laptop, four client threads
    /// against eight server threads produced 36 Kops one run and 133 the next
    /// for the same build. Keep the two totals under the core count and the
    /// numbers settle down.
    const CLIENT_THREADS: usize = 4;
    const CONNS_PER_CLIENT: usize = 16;

    /// How long a cell is counted for, and how long it runs first without being
    /// counted.
    const MEASURE: Duration = Duration::from_secs(3);
    const WARMUP: Duration = Duration::from_secs(1);

    /// How many distinct keys each client thread cycles through, unless the
    /// environment says otherwise.
    ///
    /// Enough that this is a lookup rather than one cache line read over and
    /// over, and few enough that filling them is a second rather than a minute.
    ///
    /// It is worth turning up, because a server does not necessarily do the
    /// same work per operation at every size. Anything that walks the keyspace
    /// rather than one key of it, a rehash, an expiry sweep, a memory
    /// accounting pass, costs nothing here and costs something at the two and a
    /// half million keys the harness holds. Filling is the price: this many
    /// keys of this many bytes get written down every connection's first
    /// socket before the clock starts.
    const KEYS: usize = 100_000;

    /// How long a value is, unless the environment says otherwise.
    ///
    /// One byte, which is the cheapest a value can be and is therefore the
    /// setting that measures the server rather than `memcpy`. The harness draws
    /// its lengths uniformly from `1-1024`, which is a different thing and not
    /// only a longer one, so that is the setting to read beside a published
    /// number.
    const SIZE: &str = "1";

    /// How many times a cell is measured before one of its numbers is reported.
    ///
    /// Three, which is the fewest that gives a median and a spread. The median
    /// is what gets reported and the spread is what says whether reporting it
    /// means anything.
    const REPEATS: usize = 3;

    /// The coefficient of variation above which a cell is not worth quoting.
    ///
    /// A ten core laptop with a virtual machine and two builds on it answered
    /// 36 Kops one run and 133 the next for the same server, so the first thing
    /// this has to be able to say is that the box it is on cannot answer. Five
    /// percent is well inside what a quiet machine does and well outside what a
    /// busy one does, and the published sweeps use the same figure as their
    /// exit condition.
    const NOISY: f64 = 0.05;

    pub fn main() {
        let smoke = std::env::var_os("YO_BENCH_SMOKE").is_some();
        let threads = counts("YO_BENCH_THREADS", if smoke { &[1] } else { THREADS });
        let pipelines = counts("YO_BENCH_PIPELINE", if smoke { &[50] } else { PIPELINE });
        let clients = one("YO_BENCH_CLIENTS", CLIENT_THREADS);
        let conns = one("YO_BENCH_CONNS", CONNS_PER_CLIENT);
        let repeats = one("YO_BENCH_REPEATS", if smoke { 1 } else { REPEATS });
        let op = Op::parse(&text("YO_BENCH_OP"));
        let sizes = Sizes::parse(&match std::env::var("YO_BENCH_SIZE") {
            Ok(text) => text,
            Err(_) => SIZE.to_owned(),
        });
        let keys = one("YO_BENCH_KEYS", KEYS);
        // Same rule as the debug setup below: unset means the far side does
        // what this one does, and set means the pair differs by that.
        let op_against = match std::env::var("YO_BENCH_OP_AGAINST") {
            Ok(name) => Op::parse(&name),
            Err(_) => op,
        };
        let (warmup, measure) = if smoke {
            (Duration::from_millis(100), Duration::from_millis(400))
        } else {
            (WARMUP, MEASURE)
        };

        let mine = PathBuf::from(env!("CARGO_BIN_EXE_yodb"));
        let against = std::env::var_os("YO_BENCH_AGAINST").map(PathBuf::from);
        let debug = text("YO_BENCH_DEBUG");
        // Unset means the far side is set up the same way this one is, which is
        // what a comparison of two builds wants. Set, including set to nothing,
        // means the two sides differ by that and by nothing else, which is what
        // an attribution wants.
        let debug_against = std::env::var_os("YO_BENCH_DEBUG_AGAINST")
            .map(|t| t.to_string_lossy().into_owned())
            .unwrap_or_else(|| debug.clone());
        if let Some(that) = &against {
            println!("this {}", mine.display());
            println!("that {}", that.display());
            if debug != debug_against {
                println!("this DEBUG {debug:?}");
                println!("that DEBUG {debug_against:?}");
            }
        }
        let ops = if op == op_against {
            op.as_str().to_owned()
        } else {
            format!("{} against {}", op.as_str(), op_against.as_str())
        };

        let mut bad = 0;
        for pipeline in pipelines {
            println!(
                "\n{ops} at pipeline {pipeline}, {clients} client threads of {conns} connections"
            );
            if against.is_some() {
                println!(
                    "{:>8}  {:>12}  {:>12}  {:>9}  {:>6}  {:>5}",
                    "threads", "Kops/sec", "that", "this/that", "cv", "agree"
                );
            } else {
                println!(
                    "{:>8}  {:>12}  {:>8}  {:>6}",
                    "threads", "Kops/sec", "vs 1", "cv"
                );
            }
            let mut first = 0.0_f64;
            for (at, &count) in threads.iter().enumerate() {
                let spec = Cell {
                    threads: count,
                    pipeline,
                    clients,
                    conns,
                    keys,
                    sizes,
                    warmup,
                    measure,
                };
                let this = Side {
                    bin: &mine,
                    at: 0,
                    debug: &debug,
                    op,
                };
                let mut rates = Vec::with_capacity(repeats);
                let mut theirs = Vec::with_capacity(repeats);
                let mut ratios = Vec::with_capacity(repeats);
                for round in 0..repeats {
                    let Some(that) = &against else {
                        rates.push(cell(&spec, &this));
                        continue;
                    };
                    let that = Side {
                        bin: that,
                        at: 1,
                        debug: &debug_against,
                        op: op_against,
                    };
                    // Alternating, so that whichever binary goes first is not
                    // the same one every time. The first of a pair starts on a
                    // machine that has just been left alone and the second on
                    // one that has just had a server killed on it, and that is
                    // a difference worth cancelling rather than measuring.
                    let (a, b) = if round % 2 == 0 {
                        let a = cell(&spec, &this);
                        (a, cell(&spec, &that))
                    } else {
                        let b = cell(&spec, &that);
                        (cell(&spec, &this), b)
                    };
                    rates.push(a);
                    theirs.push(b);
                    ratios.push(if b > 0.0 { a / b } else { 0.0 });
                }

                let (rate, spread) = middle(&mut rates);
                if against.is_some() {
                    let that = middle(&mut theirs).0;
                    // The ratio's spread rather than the throughput's, because
                    // the throughput either side of a pair is allowed to move
                    // with the machine. It is printed and it is not what the
                    // cell is judged on, since on a busy box it stays large
                    // while the median beside it converges anyway.
                    let (ratio, cv) = middle(&mut ratios);
                    let with = agreeing(&ratios, ratio);
                    let flag = if with * 4 < repeats * 3 {
                        bad += 1;
                        " toss"
                    } else {
                        ""
                    };
                    println!(
                        "{count:>8}  {:>12.0}  {:>12.0}  {ratio:>8.2}x  {cv:>6.2}  {:>5}{flag}",
                        rate / 1000.0,
                        that / 1000.0,
                        format!("{with}/{repeats}")
                    );
                    continue;
                }
                if at == 0 {
                    first = rate;
                }
                let ratio = if first > 0.0 { rate / first } else { 0.0 };
                let flag = if spread > NOISY {
                    bad += 1;
                    " noisy"
                } else {
                    ""
                };
                println!(
                    "{count:>8}  {:>12.0}  {ratio:>7.2}x  {spread:>6.2}{flag}",
                    rate / 1000.0
                );
            }
        }
        if bad > 0 && against.is_some() {
            println!(
                "\n{bad} cells had fewer than three quarters of their pairs on the same side of \
                 1.00 as the median, so what those cells report is which run won. Turn \
                 YO_BENCH_REPEATS up, because the median of enough pairs converges where a single \
                 pair does not: on a laptop at a load average of ten, this binary against a copy \
                 of itself read 1.16x over three pairs and 1.02x over eleven, and the true answer \
                 is 1.00x."
            );
        } else if bad > 0 {
            println!(
                "\n{bad} cells came out above a coefficient of variation of {NOISY:.2}, so this \
                 machine is not quiet enough to be measuring anything. Nothing here is worth \
                 quoting until it runs clean. If what you need is whether one build beats \
                 another rather than what either is worth, point YO_BENCH_AGAINST at the other \
                 one, which is the question a busy box can still answer."
            );
        }
    }

    /// How many pairs came out on the same side of parity as the median did.
    ///
    /// The sign of a pair rather than its size, because a pair that lost the
    /// machine for a second has a size that means nothing and a side that still
    /// usually means something. Ten of eleven pairs agreeing is a result and six
    /// of eleven is a coin toss, whatever the median between them says.
    ///
    /// A pair that came out exactly at parity counts for neither side, which is
    /// the honest reading of it and is rare enough not to matter.
    fn agreeing(ratios: &[f64], median: f64) -> usize {
        if median > 1.0 {
            return ratios.iter().filter(|r| **r > 1.0).count();
        }
        if median < 1.0 {
            return ratios.iter().filter(|r| **r < 1.0).count();
        }
        ratios.len()
    }

    /// The median of a cell's runs, and how far apart they were.
    ///
    /// The median rather than the mean, because a run that lost the machine to
    /// something else is an outlier rather than evidence, and the whole point of
    /// the spread beside it is that it is reported instead of being averaged
    /// away. The spread is the coefficient of variation, which is what the
    /// published sweeps report per cell and what the exit condition on the
    /// milestone is written in.
    fn middle(rates: &mut [f64]) -> (f64, f64) {
        rates.sort_by(f64::total_cmp);
        let median = rates[rates.len() / 2];
        if rates.len() < 2 {
            return (median, 0.0);
        }
        let mean = rates.iter().sum::<f64>() / rates.len() as f64;
        let spread = rates.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>();
        let sd = (spread / (rates.len() - 1) as f64).sqrt();
        let cv = if mean > 0.0 { sd / mean } else { 0.0 };
        (median, cv)
    }

    /// What the measured window sends.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Op {
        Get,
        Set,
    }

    impl Op {
        fn parse(name: &str) -> Op {
            match name {
                "" | "get" | "GET" => Op::Get,
                "set" | "SET" => Op::Set,
                other => panic!("YO_BENCH_OP is get or set, not {other:?}"),
            }
        }

        /// How many line terminators one reply to this arrives in.
        ///
        /// A `GET` answers with a bulk string, which is a header line and a
        /// body line. A `SET` answers `+OK`, which is one line.
        fn lines(self) -> usize {
            match self {
                Op::Get => 2,
                Op::Set => 1,
            }
        }

        fn as_str(self) -> &'static str {
            match self {
                Op::Get => "get",
                Op::Set => "set",
            }
        }
    }

    /// One cell of the sweep: a server at one thread count, driven one way.
    struct Cell {
        threads: usize,
        pipeline: usize,
        clients: usize,
        conns: usize,
        keys: usize,
        sizes: Sizes,
        warmup: Duration,
        measure: Duration,
    }

    /// One half of a pair, which unpaired means the only half there is.
    ///
    /// Everything here is allowed to differ between the two halves, and that is
    /// the point of it. A pair whose sides differ by the binary compares two
    /// builds, a pair whose sides differ by a `DEBUG` subcommand says what that
    /// job costs, and a pair whose sides differ by the operation says what a
    /// write costs over a read. All three are questions a busy machine can
    /// answer paired and cannot answer one number at a time.
    struct Side<'a> {
        bin: &'a Path,
        /// Only keeps the two off one socket path, so that a server taking its
        /// time to die cannot be found by the run after it.
        at: usize,
        debug: &'a str,
        op: Op,
    }

    /// One server at one thread count, driven at one pipeline depth.
    fn cell(cell: &Cell, side: &Side<'_>) -> f64 {
        let Cell {
            threads,
            pipeline,
            clients,
            conns: per_client,
            keys,
            sizes,
            warmup,
            measure,
        } = *cell;
        let Side { bin, at, debug, op } = *side;
        let socket = socket_path(threads, pipeline, at);
        let mut server = Server::start(bin, &socket, threads);
        server.wait_until_listening();

        let go = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicU64::new(0));
        // Every client has to be connected and its keys written before the
        // clock starts, or the first thread ready spends the window racing
        // threads that are still opening sockets.
        let gate = Arc::new(Barrier::new(clients + 1));

        let mut hands = Vec::with_capacity(clients);
        for id in 0..clients {
            let socket = socket.clone();
            let go = Arc::clone(&go);
            let stop = Arc::clone(&stop);
            let done = Arc::clone(&done);
            let gate = Arc::clone(&gate);
            let debug = debug.to_owned();
            hands.push(std::thread::spawn(move || {
                let mut conns: Vec<UnixStream> =
                    (0..per_client).map(|_| connect(&socket)).collect();
                let batches = Batches::new(id, pipeline, op, keys, sizes);
                // One client sends them and the rest do not, because these
                // change the server rather than the connection and sending
                // them four times only means saying the same thing four times.
                if id == 0 {
                    debug_setup(&mut conns[0], &debug);
                }
                fill(&mut conns[0], id, keys, sizes);
                gate.wait();
                let mut at = 0;
                // The warmup runs the same loop and counts none of it, so what
                // it buys is a window where the maps have grown and the
                // connections have settled.
                while !go.load(Ordering::Relaxed) {
                    at = round(&mut conns, &batches, at);
                }
                let mut count = 0_u64;
                while !stop.load(Ordering::Relaxed) {
                    at = round(&mut conns, &batches, at);
                    count += (per_client * pipeline) as u64;
                }
                done.fetch_add(count, Ordering::Relaxed);
            }));
        }

        gate.wait();
        std::thread::sleep(warmup);
        go.store(true, Ordering::Relaxed);
        let counted_from = Instant::now();
        std::thread::sleep(measure);
        stop.store(true, Ordering::Relaxed);
        // Read before the join, so that a thread taking a moment to notice the
        // flag does not lengthen the window it is divided by.
        let elapsed = counted_from.elapsed().as_secs_f64();
        for hand in hands {
            let _ = hand.join();
        }
        let total = done.load(Ordering::Relaxed);

        server.stop();
        let _ = std::fs::remove_file(&socket);
        if elapsed > 0.0 {
            total as f64 / elapsed
        } else {
            0.0
        }
    }

    /// One pipeline down each connection in turn, and the replies to it.
    fn round(conns: &mut [UnixStream], batches: &Batches, mut at: usize) -> usize {
        for conn in conns {
            let batch = batches.at(at);
            at = batches.next(at);
            if conn.write_all(batch).is_err() {
                continue;
            }
            drain(conn, batches.pipeline * batches.lines);
        }
        at
    }

    /// Every command this client will ever send, encoded once.
    ///
    /// A load generator that formats a key on the hot path is measuring itself,
    /// so the whole key space goes into one buffer up front and a batch is a
    /// slice of it. Batches start on a multiple of the pipeline depth, which is
    /// what makes every one of them contiguous.
    struct Batches {
        buf: Vec<u8>,
        /// Where each batch starts, with a final entry for where the last ends.
        marks: Vec<usize>,
        pipeline: usize,
        /// How many line terminators one reply arrives in.
        lines: usize,
    }

    impl Batches {
        fn new(client: usize, pipeline: usize, op: Op, keys: usize, sizes: Sizes) -> Batches {
            let value = value_of(sizes);
            // Not the seed the fill used, so that a write is over a value of
            // some other length than the one already there. That is what the
            // harness does and it is the case an in place overwrite misses.
            let mut seed = (client as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let count = (keys / pipeline).max(1);
            let mut buf = Vec::with_capacity(count * pipeline * 32);
            let mut marks = Vec::with_capacity(count + 1);
            let mut k = 0;
            for _ in 0..count {
                marks.push(buf.len());
                for _ in 0..pipeline {
                    let key = key_of(client, k);
                    match op {
                        Op::Get => encode(&mut buf, &[b"GET", key.as_bytes()]),
                        Op::Set => {
                            let n = sizes.draw(&mut seed);
                            encode(&mut buf, &[b"SET", key.as_bytes(), &value[..n]]);
                        }
                    }
                    k += 1;
                }
            }
            marks.push(buf.len());
            Batches {
                buf,
                marks,
                pipeline,
                lines: op.lines(),
            }
        }

        fn at(&self, batch: usize) -> &[u8] {
            &self.buf[self.marks[batch]..self.marks[batch + 1]]
        }

        fn next(&self, batch: usize) -> usize {
            if batch + 2 == self.marks.len() {
                0
            } else {
                batch + 1
            }
        }
    }

    /// A running `yodb serve`, killed however this bench ends.
    struct Server {
        child: Child,
        socket: PathBuf,
    }

    impl Server {
        fn start(bin: &Path, socket: &Path, threads: usize) -> Server {
            let _ = std::fs::remove_file(socket);
            let child = Command::new(bin)
                .arg("serve")
                .arg("--no-port")
                .arg("--unixsocket")
                .arg(socket)
                .arg("--threads")
                .arg(threads.to_string())
                // The banner would land in the middle of the table. Its stderr
                // is left alone, because a server that fails to start has
                // something to say and the panic below only says that it did.
                .stdout(Stdio::null())
                .spawn()
                .unwrap_or_else(|e| panic!("{} would not start: {e}", bin.display()));
            Server {
                child,
                socket: socket.to_path_buf(),
            }
        }

        /// Wait for the socket to answer rather than for a fixed time, because
        /// a cold start on a loaded machine is slower than any number worth
        /// writing down.
        fn wait_until_listening(&mut self) {
            let deadline = Instant::now() + Duration::from_secs(10);
            while Instant::now() < deadline {
                if UnixStream::connect(&self.socket).is_ok() {
                    return;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            panic!("yodb did not start listening on {}", self.socket.display());
        }

        fn stop(&mut self) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }

    impl Drop for Server {
        fn drop(&mut self) {
            self.stop();
        }
    }

    fn connect(socket: &Path) -> UnixStream {
        let stream = UnixStream::connect(socket).expect("the server is listening");
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .expect("a timeout the platform accepts");
        stream
    }

    /// Send whatever this side was asked to send, before anything is measured.
    ///
    /// Semicolons between subcommands, spaces inside one, and `DEBUG` is added
    /// rather than typed. Nothing checks the replies beyond reading them, since
    /// a subcommand this build does not have answers an error and the run that
    /// follows is then a run with that job still on, which the number will say.
    fn debug_setup(conn: &mut UnixStream, text: &str) {
        for one in text.split(';').map(str::trim).filter(|s| !s.is_empty()) {
            let parts: Vec<&[u8]> = std::iter::once(b"DEBUG".as_slice())
                .chain(one.split_whitespace().map(str::as_bytes))
                .collect();
            let mut out = Vec::new();
            encode(&mut out, &parts);
            let _ = conn.write_all(&out);
            drain(conn, 1);
        }
    }

    /// Write the keys this client will use, before anything is measured.
    ///
    /// A hit and a miss are different amounts of work, so a bench over a
    /// mixture of the two would be measuring the mixture. That is why a `set`
    /// run fills too: an overwrite and an insert are different amounts of work
    /// as well, and a run that starts empty spends its window growing the map.
    fn fill(conn: &mut UnixStream, client: usize, keys: usize, sizes: Sizes) {
        const AT_ONCE: usize = 256;
        let value = value_of(sizes);
        let mut seed = client as u64 + 1;
        let mut out = Vec::with_capacity(AT_ONCE * (48 + sizes.high));
        let mut owed = 0;
        for k in 0..keys {
            let n = sizes.draw(&mut seed);
            encode(
                &mut out,
                &[b"SET", key_of(client, k).as_bytes(), &value[..n]],
            );
            owed += 1;
            if owed == AT_ONCE || k + 1 == keys {
                let _ = conn.write_all(&out);
                // One terminator per reply here, because `SET` answers `+OK`.
                drain(conn, owed);
                out.clear();
                owed = 0;
            }
        }
    }

    /// Read until `owed` line terminators have arrived.
    ///
    /// Counting terminators rather than parsing, because every reply this asks
    /// for has a shape known at the call site and counting is what a load
    /// generator can afford on the hot path.
    fn drain(conn: &mut UnixStream, owed: usize) -> bool {
        let mut buf = [0_u8; 64 * 1024];
        let mut seen = 0;
        let mut last = 0_u8;
        while seen < owed {
            match conn.read(&mut buf) {
                Ok(0) => return false,
                Ok(n) => {
                    for &b in &buf[..n] {
                        if last == b'\r' && b == b'\n' {
                            seen += 1;
                        }
                        last = b;
                    }
                }
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(_) => return false,
            }
        }
        true
    }

    /// A RESP array of bulk strings, which is what a client sends.
    fn encode(out: &mut Vec<u8>, parts: &[&[u8]]) {
        out.extend_from_slice(format!("*{}\r\n", parts.len()).as_bytes());
        for part in parts {
            out.extend_from_slice(format!("${}\r\n", part.len()).as_bytes());
            out.extend_from_slice(part);
            out.extend_from_slice(b"\r\n");
        }
    }

    /// Keys belong to a client thread, so two of them never read the same key
    /// and the sharing being measured is the server's rather than the bench's.
    fn key_of(client: usize, k: usize) -> String {
        format!("bench:{client}:{k}")
    }

    /// How long a value is, which the harness draws from a range rather than
    /// fixing.
    ///
    /// The difference matters more than it looks. A `SET` over a value that is
    /// the length the old one was can reuse what is already allocated, and a
    /// `SET` over a value that is some other length cannot. A bench that writes
    /// one length every time is measuring the first of those and calling it the
    /// write path.
    #[derive(Clone, Copy)]
    struct Sizes {
        low: usize,
        high: usize,
    }

    impl Sizes {
        /// `N` for one length, `LOW-HIGH` for a length drawn from a range.
        ///
        /// The second spelling is `memtier`'s `--data-size-range` and means the
        /// same thing it means there.
        fn parse(text: &str) -> Sizes {
            let (low, high) = match text.split_once('-') {
                Some((low, high)) => (low, high),
                None => (text, text),
            };
            let read = |part: &str| {
                part.trim()
                    .parse::<usize>()
                    .unwrap_or_else(|_| panic!("YO_BENCH_SIZE is N or LOW-HIGH, not {text:?}"))
                    .max(1)
            };
            let (low, high) = (read(low), read(high));
            assert!(low <= high, "YO_BENCH_SIZE has {low} above {high}");
            Sizes { low, high }
        }

        /// One length, and the seed moved on.
        fn draw(self, seed: &mut u64) -> usize {
            if self.low == self.high {
                return self.low;
            }
            let span = (self.high - self.low + 1) as u64;
            self.low + (roll(seed) % span) as usize
        }
    }

    /// A buffer to cut values out of, as long as the longest one asked for.
    ///
    /// One byte repeated rather than anything random, because a value with a
    /// carriage return in it would end a reply early and the reply reader here
    /// counts line terminators rather than parsing.
    fn value_of(sizes: Sizes) -> Vec<u8> {
        vec![b'v'; sizes.high]
    }

    /// The next number out of a seed, and the seed moved on.
    ///
    /// Xorshift, because what this needs is lengths that do not repeat and a
    /// run that does the same thing twice, and not randomness anybody would
    /// defend.
    fn roll(seed: &mut u64) -> u64 {
        let mut x = *seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *seed = x;
        x
    }

    /// A socket path this run owns, so two benches at once do not fight.
    fn socket_path(threads: usize, pipeline: usize, side: usize) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "yodb-bench-{}-t{threads}-p{pipeline}-s{side}.sock",
            std::process::id()
        ));
        path
    }

    /// A string out of the environment, empty if it is not there.
    fn text(name: &str) -> String {
        std::env::var_os(name)
            .map(|t| t.to_string_lossy().into_owned())
            .unwrap_or_default()
    }

    /// One number out of the environment, or the default.
    fn one(name: &str, fallback: usize) -> usize {
        std::env::var_os(name)
            .and_then(|text| text.to_string_lossy().trim().parse().ok())
            .filter(|n| *n > 0)
            .unwrap_or(fallback)
    }

    /// A comma separated list out of the environment, or the default.
    fn counts(name: &str, fallback: &[usize]) -> Vec<usize> {
        let Some(text) = std::env::var_os(name) else {
            return fallback.to_vec();
        };
        let parsed: Vec<usize> = text
            .to_string_lossy()
            .split(',')
            .filter_map(|part| part.trim().parse().ok())
            .filter(|n| *n > 0)
            .collect();
        if parsed.is_empty() {
            fallback.to_vec()
        } else {
            parsed
        }
    }
}
