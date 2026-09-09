//! Users, the rules that describe what each of them may do, and the gate that
//! holds every command to them.
//!
//! # What an ACL actually is
//!
//! A user is a name, a switch saying whether it is usable at all, a list of
//! password hashes, and one or more selectors. A selector is the interesting
//! part: a set of commands, a set of key patterns and a set of channel patterns.
//! A command is allowed if any one selector allows the command and every key and
//! channel it touches. Selectors are tried in order and the first that says yes
//! wins, which is how a user can be given two unrelated jobs without either of
//! them leaking into the other. Without them the only way to say "read anything
//! under `cache:` and write anything under `job:`" is to allow reads and writes
//! over both.
//!
//! Every user starts with one selector, called the root selector here and in
//! Redis, which is the one the bare rules land in: `ACL SETUSER u +get ~k:*`
//! writes into it, and `ACL SETUSER u (+get ~k:*)` adds another one beside it.
//!
//! # Why the rules are kept as text as well as bits
//!
//! Command permission is a bitmap, one bit a command, because the gate reads it
//! on every command a guarded server runs and a set of names would mean a lookup
//! there. But `ACL LIST` and `ACL GETUSER` have to hand back something that can
//! be fed to `ACL SETUSER` and produce the same user, and a bitmap cannot say
//! whether `+@read -get` or the eleven names that leaves was what the operator
//! wrote. Worse, the two are not the same user: the first allows a command added
//! to the `@read` category later and the second does not.
//!
//! So a selector keeps both. The bitmap is what the gate reads and the rule list
//! is what describes it, and every rule that touches commands appends itself to
//! the list after removing whatever it overrides. That is Redis's design and it
//! is the reason `ACL GETUSER` on a user built with categories reads back in
//! categories. The removal is the fiddly half and it is spelled out at
//! [`Selector::note`].
//!
//! # The one thing here that is not Redis's
//!
//! Redis gives every subcommand of a container command its own identity and its
//! own bit, so `+config|get` sets a bit and `+config|nope` is refused because
//! there is no such command. There is no subcommand table here yet, which is
//! D-114, so both go through the mechanism Redis keeps for the other case: a
//! list of first arguments a command is allowed with. The gate then reaches the
//! same answer for every rule an operator would actually write, and the
//! difference is that `+config|nope` is accepted here and refused there. It is
//! written down as D-137 rather than papered over, and it goes away with the
//! subcommand table.
//!
//! # Where the gate sits
//!
//! Right after the password and the transaction refusal and right before the
//! memory limit, which is where `processCommand` puts it. That ordering is
//! visible: a command a user is not allowed to run is refused before the server
//! decides whether it is out of memory, and a command with the wrong number of
//! arguments is told that rather than told it is not allowed.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use std::sync::atomic::{AtomicBool, AtomicU64};

use yo_common::{Code, Error, Result, glob_matches, sha256};

use super::args::{self, Args, is};
use super::keyspec::{self, Access};
use super::table::{self, Spec};
use super::{Server, Session};
use crate::reply::Out;

/// The user every connection is on until it says otherwise.
pub(super) const DEFAULT: &[u8] = b"default";

/// The categories `ACL CAT` lists, in the order it lists them.
///
/// The first twenty two are Redis's own list in Redis's own order, which is the
/// order they are declared in `server.h` and not alphabetical. The nine after
/// them are the module surfaces this engine has built in. A real server with
/// those modules loaded reports them here too, so the shape is right even though
/// a bare server answers twenty two.
const CATEGORIES: [&str; 31] = [
    "keyspace",
    "read",
    "write",
    "set",
    "sortedset",
    "list",
    "hash",
    "string",
    "array",
    "bitmap",
    "hyperloglog",
    "geo",
    "stream",
    "pubsub",
    "admin",
    "fast",
    "slow",
    "blocking",
    "dangerous",
    "connection",
    "transaction",
    "scripting",
    "bloom",
    "cms",
    "cuckoo",
    "graph",
    "json",
    "search",
    "tdigest",
    "timeseries",
    "topk",
];

/// How many words of bitmap it takes to give every command a bit.
const WORDS: usize = table::count().div_ceil(64);

/// The read bit on a key pattern, which is what `%R~` sets.
const READ: u8 = 1;
/// The write bit, which is what `%W~` sets.
const WRITE: u8 = 2;

/// A key pattern and what it may be used for.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Pattern {
    /// [`READ`], [`WRITE`] or both.
    flags: u8,
    /// The glob itself, without the sigil.
    glob: Vec<u8>,
}

impl Pattern {
    /// The pattern written the way `ACL LIST` writes it.
    ///
    /// A pattern good for both is written bare, because `%RW~x` and `~x` are the
    /// same thing and the bare form is the one everybody types. A pattern good
    /// for one keeps its sigil, because there is no other way to say it.
    fn describe(&self, into: &mut Vec<u8>) {
        match self.flags {
            READ => into.extend_from_slice(b"%R~"),
            WRITE => into.extend_from_slice(b"%W~"),
            _ => into.push(b'~'),
        }
        into.extend_from_slice(&self.glob);
    }
}

/// One set of commands, keys and channels, which is what a permission check runs
/// against.
#[derive(Debug, Clone)]
struct Selector {
    /// One bit a command, indexed by its row in the table.
    allowed: [u64; WORDS],
    /// Whether every command is allowed without looking, which `+@all` sets and
    /// taking any command away clears.
    all_commands: bool,
    /// Whether a command that does not exist yet would be allowed.
    ///
    /// Redis keeps this as a reserved bit past the end of the bitmap, set by
    /// `+@all` and cleared by `-@all` and by nothing else, so that `+@all -get`
    /// and `-@all +everything-but-get` can be told apart. They describe back
    /// differently and they behave differently the day a command is added.
    future: bool,
    /// Whether every key is allowed.
    all_keys: bool,
    /// Whether every channel is allowed.
    all_channels: bool,
    /// The command rules in the order they were given, space separated.
    ///
    /// Not the bitmap written out. See the module note for why the two are both
    /// kept.
    rules: Vec<u8>,
    /// The first arguments a command is allowed with, when it is not allowed
    /// outright. Sorted by command index.
    firstargs: Vec<(u16, Vec<Vec<u8>>)>,
    /// The first arguments a container command is refused with, when it is
    /// allowed outright. Sorted by command index.
    ///
    /// This is how `+config -config|get` is held. A server with a row a
    /// subcommand has does it by clearing that row's bit, and the day yo has one
    /// too this goes away with the rest of D-114.
    deniedfirst: Vec<(u16, Vec<Vec<u8>>)>,
    /// The key patterns, in the order they were added.
    patterns: Vec<Pattern>,
    /// The channel patterns, in the order they were added.
    channels: Vec<Vec<u8>>,
}

impl Selector {
    /// A selector that allows nothing, which is where every user starts.
    ///
    /// Nothing except the channels, which start out either allowed or refused
    /// depending on `open`, the server's `acl-pubsub-default`. That one setting
    /// is the whole reason this takes an argument: it is read at the moment a
    /// selector is made and never again, so a selector made before it changed
    /// keeps what it was made with.
    fn new(open: bool) -> Selector {
        Selector {
            allowed: [0; WORDS],
            all_commands: false,
            future: false,
            all_keys: false,
            all_channels: open,
            rules: Vec::new(),
            firstargs: Vec::new(),
            deniedfirst: Vec::new(),
            patterns: Vec::new(),
            channels: Vec::new(),
        }
    }

    /// Whether the command at `i` is allowed outright.
    fn bit(&self, i: usize) -> bool {
        i < table::count() && self.allowed[i / 64] & (1 << (i % 64)) != 0
    }

    /// Allow or refuse the command at `i`, and forget any first arguments it had.
    ///
    /// Taking anything away clears `all_commands`, because that flag is only
    /// there to let the gate skip the bitmap and it can no longer do that.
    fn set(&mut self, i: usize, allow: bool) {
        if i >= table::count() {
            return;
        }
        if allow {
            self.allowed[i / 64] |= 1 << (i % 64);
        } else {
            self.allowed[i / 64] &= !(1 << (i % 64));
            self.all_commands = false;
        }
        self.firstargs.retain(|(at, _)| usize::from(*at) != i);
        self.deniedfirst.retain(|(at, _)| usize::from(*at) != i);
    }

    /// Record a command rule, dropping whatever it overrides.
    ///
    /// The rule that goes on the end is the one that wins, so anything it makes
    /// irrelevant has to come off first or the list would grow forever and
    /// describe the user wrongly. Two things are irrelevant: the same rule
    /// written before, and, when this rule names a whole command, any earlier
    /// rule about one of its subcommands. `+get` after `-get|foo` leaves `+get`
    /// alone, because the subcommand rule cannot survive its parent being
    /// decided again.
    ///
    /// Note that a rule is matched on the name and not on the sign, so `+get`
    /// removes an earlier `-get` rather than sitting after it.
    fn note(&mut self, rule: &[u8], allow: bool) {
        let mut kept: Vec<u8> = Vec::with_capacity(self.rules.len() + rule.len() + 2);
        for old in self.rules.split(|b| *b == b' ') {
            if old.is_empty() {
                continue;
            }
            // The sign is not part of the name and is not compared.
            let name = &old[1..];
            let same = name == rule;
            let child =
                name.len() > rule.len() && name.starts_with(rule) && name[rule.len()] == b'|';
            if same || child {
                continue;
            }
            if !kept.is_empty() {
                kept.push(b' ');
            }
            kept.extend_from_slice(old);
        }
        if !kept.is_empty() {
            kept.push(b' ');
        }
        kept.push(if allow { b'+' } else { b'-' });
        kept.extend_from_slice(rule);
        self.rules = kept;
    }

    /// Allow this command only when its first argument is `first`.
    fn allow_first(&mut self, i: u16, first: &[u8]) {
        let lower = first.to_ascii_lowercase();
        match self.firstargs.binary_search_by_key(&i, |(at, _)| *at) {
            Ok(at) => {
                let list = &mut self.firstargs[at].1;
                if !list.contains(&lower) {
                    list.push(lower);
                }
            }
            Err(at) => self.firstargs.insert(at, (i, vec![lower])),
        }
    }

    /// The first arguments the command at `i` is allowed with.
    fn firsts(&self, i: u16) -> &[Vec<u8>] {
        match self.firstargs.binary_search_by_key(&i, |(at, _)| *at) {
            Ok(at) => &self.firstargs[at].1,
            Err(_) => &[],
        }
    }

    /// Refuse this command when its first argument is `first`.
    ///
    /// This clears `all_commands` for the same reason [`Selector::set`] does:
    /// the flag is only there so the gate can skip the bitmap, and once one
    /// subcommand is spoken for it can no longer skip anything.
    fn deny_first(&mut self, i: u16, first: &[u8]) {
        let lower = first.to_ascii_lowercase();
        self.all_commands = false;
        match self.deniedfirst.binary_search_by_key(&i, |(at, _)| *at) {
            Ok(at) => {
                let list = &mut self.deniedfirst[at].1;
                if !list.contains(&lower) {
                    list.push(lower);
                }
            }
            Err(at) => self.deniedfirst.insert(at, (i, vec![lower])),
        }
    }

    /// The first arguments the command at `i` is refused with.
    fn denied(&self, i: u16) -> &[Vec<u8>] {
        match self.deniedfirst.binary_search_by_key(&i, |(at, _)| *at) {
            Ok(at) => &self.deniedfirst[at].1,
            Err(_) => &[],
        }
    }

    /// Everything a rule about commands could have set, back to nothing.
    fn reset_commands(&mut self, all: bool) {
        self.allowed = [if all { u64::MAX } else { 0 }; WORDS];
        self.all_commands = all;
        self.future = all;
        self.rules.clear();
        self.firstargs.clear();
        self.deniedfirst.clear();
    }

    /// The command rules written the way `ACL SETUSER` would take them back.
    ///
    /// Always starts with `+@all` or `-@all` and then repeats the rule list,
    /// which is exactly what was fed in after the last time it was cleared. That
    /// is why this needs no cleverness: the list is already the answer.
    fn describe_commands(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.rules.len() + 8);
        out.extend_from_slice(if self.future { b"+@all" } else { b"-@all" });
        if !self.rules.is_empty() {
            out.push(b' ');
            out.extend_from_slice(&self.rules);
        }
        out
    }

    /// The key patterns, space separated, or `~*`.
    fn describe_keys(&self) -> Vec<u8> {
        let mut out = Vec::new();
        if self.all_keys {
            out.extend_from_slice(b"~*");
            return out;
        }
        for pattern in &self.patterns {
            if !out.is_empty() {
                out.push(b' ');
            }
            pattern.describe(&mut out);
        }
        out
    }

    /// The channel patterns, space separated, or `&*`.
    fn describe_channels(&self) -> Vec<u8> {
        let mut out = Vec::new();
        if self.all_channels {
            out.extend_from_slice(b"&*");
            return out;
        }
        for channel in &self.channels {
            if !out.is_empty() {
                out.push(b' ');
            }
            out.push(b'&');
            out.extend_from_slice(channel);
        }
        out
    }

    /// The whole selector as `ACL LIST` writes it inside a user's line.
    ///
    /// Keys first, then channels, then commands. The `resetchannels` in front of
    /// the channel list is not decoration: without it a line read back on a
    /// server whose `acl-pubsub-default` is `allchannels` would start from every
    /// channel allowed and the `&x` rules would add nothing.
    fn describe(&self) -> Vec<u8> {
        let mut out = self.describe_keys();
        if !out.is_empty() {
            out.push(b' ');
        }
        if self.all_channels {
            out.extend_from_slice(b"&* ");
        } else {
            out.extend_from_slice(b"resetchannels ");
            for channel in &self.channels {
                out.push(b'&');
                out.extend_from_slice(channel);
                out.push(b' ');
            }
        }
        out.extend_from_slice(&self.describe_commands());
        out
    }
}

/// What one of these is allowed to say went wrong with a rule.
///
/// The sentences are Redis's, word for word, because an operator reading one has
/// almost certainly found it in Redis's documentation first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bad {
    /// The rule is not one, or is malformed.
    Syntax,
    /// A command or category nobody has heard of.
    Unknown,
    /// A key pattern after `~*`.
    AfterAllKeys,
    /// A channel pattern after `&*`.
    AfterAllChannels,
    /// `<password` for a password the user has not got.
    NoSuchPassword,
    /// A `#` hash that is not sixty four hexadecimal characters.
    Hash,
    /// `+get|set|other`, which Redis has never supported.
    NestedFirstArg,
    /// A `(` with no `)`, which is reported differently from all the rest.
    Unmatched,
}

impl Bad {
    /// The sentence after `Error in ACL SETUSER modifier '<rule>': `.
    fn text(self) -> &'static str {
        match self {
            Bad::Syntax | Bad::Unmatched => "Syntax error",
            Bad::Unknown => "Unknown command or category name in ACL",
            Bad::AfterAllKeys => {
                "Adding a pattern after the * pattern (or the 'allkeys' flag) is not valid and does not have any effect. Try 'resetkeys' to start with an empty list of patterns"
            }
            Bad::AfterAllChannels => {
                "Adding a pattern after the * pattern (or the 'allchannels' flag) is not valid and does not have any effect. Try 'resetchannels' to start with an empty list of channels"
            }
            Bad::NoSuchPassword => {
                "The password you are trying to remove from the user does not exist"
            }
            Bad::Hash => {
                "The password hash must be exactly 64 characters and contain only lowercase hexadecimal characters"
            }
            Bad::NestedFirstArg => "Allowing first-arg of a subcommand is not supported",
        }
    }
}

/// One account, and everything it is allowed to do.
#[derive(Debug, Clone)]
pub(crate) struct User {
    /// The name, which is also the key it is filed under.
    name: Vec<u8>,
    /// Whether it may authenticate at all.
    enabled: bool,
    /// Whether any password gets in, which is what `nopass` means.
    nopass: bool,
    /// Whether `DEBUG` payload sanitising is skipped for this user.
    ///
    /// Nothing reads it here yet, and it is kept and reported because it is on
    /// every `ACL LIST` line a real server writes and a config file round trip
    /// that dropped it would quietly change the user.
    skip_sanitize: bool,
    /// The SHA-256 hashes of the passwords that get in, lower case hex.
    passwords: Vec<[u8; 64]>,
    /// The root selector first and any others after it.
    selectors: Vec<Selector>,
}

impl User {
    /// A new user, off, with no password and allowed nothing.
    ///
    /// `open` is the server's `acl-pubsub-default` and goes to the root
    /// selector, which is the only thing about a new user that a server setting
    /// has any say in.
    fn new(name: &[u8], open: bool) -> User {
        User {
            name: name.to_vec(),
            enabled: false,
            nopass: false,
            skip_sanitize: false,
            passwords: Vec::new(),
            selectors: vec![Selector::new(open)],
        }
    }

    /// The user a server with no `aclfile` starts with, which can do anything.
    fn default_user() -> User {
        let mut user = User::new(DEFAULT, false);
        user.enabled = true;
        user.nopass = true;
        let root = &mut user.selectors[0];
        root.reset_commands(true);
        root.all_keys = true;
        root.all_channels = true;
        user
    }

    /// The flag words `ACL LIST` and `ACL GETUSER` print, in Redis's order.
    fn flags(&self) -> Vec<&'static str> {
        let mut out = Vec::with_capacity(3);
        out.push(if self.enabled { "on" } else { "off" });
        if self.nopass {
            out.push("nopass");
        }
        // Exactly one of the two is always set, because a user is created with
        // the sanitising one on and the rules only ever swap them.
        out.push(if self.skip_sanitize {
            "skip-sanitize-payload"
        } else {
            "sanitize-payload"
        });
        out
    }

    /// The whole user as one `ACL LIST` line, without the leading `user `.
    fn describe(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(64);
        out.extend_from_slice(&self.name);
        for flag in self.flags() {
            out.push(b' ');
            out.extend_from_slice(flag.as_bytes());
        }
        for hash in &self.passwords {
            out.extend_from_slice(b" #");
            out.extend_from_slice(hash);
        }
        for (at, selector) in self.selectors.iter().enumerate() {
            out.push(b' ');
            if at == 0 {
                out.extend_from_slice(&selector.describe());
            } else {
                out.push(b'(');
                out.extend_from_slice(&selector.describe());
                out.push(b')');
            }
        }
        out
    }

    /// Whether nothing this user could be asked to do would be refused.
    ///
    /// One selector that allows every command, every key and every channel, and
    /// no other selectors, because a second one could only ever allow more and
    /// the first already allows everything. That is the shape the default user
    /// has on a server nobody has written an ACL for.
    fn unrestricted(&self) -> bool {
        self.selectors.len() == 1
            && self.selectors[0].all_commands
            && self.selectors[0].all_keys
            && self.selectors[0].all_channels
    }

    /// Whether this password gets in.
    ///
    /// Every hash is compared whatever the first one said, so the number of
    /// passwords a user has is not readable from how long a wrong guess took.
    /// The hashing has already thrown away everything about the guess that a
    /// timing difference could leak, so this is belt and braces, and it costs
    /// one compare of thirty two bytes a password.
    fn admits(&self, password: &[u8]) -> bool {
        if !self.enabled {
            return false;
        }
        if self.nopass {
            return true;
        }
        let guess = sha256::hex(password);
        let mut hit = false;
        for hash in &self.passwords {
            hit |= same(hash, &guess);
        }
        hit
    }
}

/// Whether two byte strings are equal, in time that does not depend on where
/// they stop being equal.
fn same(a: &[u8], b: &[u8]) -> bool {
    let mut diff = u8::from(a.len() != b.len());
    for i in 0..a.len().max(b.len()) {
        diff |= a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0xff);
    }
    diff == 0
}

/// Every user on the server, and a counter saying when one last changed.
///
/// One lock over the lot, because every path that writes here is an operator
/// typing a command and every path that reads is either the same or a
/// connection authenticating. The command gate does not come through here at
/// all: it holds a clone of the user it is running as, refreshed when the
/// counter moves, which is what keeps a guarded server's hot path free of this
/// lock. See [`Session::acl_user`].
#[derive(Debug)]
pub(crate) struct Users {
    /// The users, sorted by name, which is the order `ACL LIST` reports.
    ///
    /// Redis keeps them in a radix tree and walks it in order, so the sorting is
    /// not a nicety, it is the observable order of two commands.
    table: Mutex<Vec<User>>,
    /// Bumped whenever anything in the table changes.
    generation: AtomicU64,
    /// Whether a connection has to authenticate before it can do anything.
    guarded: AtomicBool,
    /// Whether any user here could be refused any command.
    ///
    /// The gate reads this and nothing else on a server nobody has written an
    /// ACL for, which is every server that only ever set `requirepass` and every
    /// server that did not even do that. Setting a password does not make it
    /// true: the default user still has every permission, so no command can be
    /// refused once a connection is past the password.
    restricted: AtomicBool,
    /// Whether a selector starts out allowed every channel.
    ///
    /// This is `acl-pubsub-default`, and it is here rather than with the other
    /// settings because the only thing that reads it is the making of a
    /// selector. It is false, meaning `resetchannels`, on a server nobody has
    /// told otherwise, which has been the default since Redis 7.
    open_channels: AtomicBool,
}

impl Default for Users {
    fn default() -> Users {
        Users {
            table: Mutex::new(vec![User::default_user()]),
            generation: AtomicU64::new(1),
            guarded: AtomicBool::new(false),
            restricted: AtomicBool::new(false),
            open_channels: AtomicBool::new(false),
        }
    }
}

impl Users {
    /// Run `f` over the table, and say that it changed if `f` says so.
    ///
    /// The two summaries are recomputed here rather than at every call site,
    /// which is the point of routing every write through one function: there is
    /// exactly one place that can leave them disagreeing with the table.
    fn with<T>(&self, f: impl FnOnce(&mut Vec<User>) -> (bool, T)) -> T {
        let mut held = self.table.lock().unwrap_or_else(|e| e.into_inner());
        let (changed, out) = f(&mut held);
        if changed {
            let guarded = held
                .binary_search_by(|u| u.name.as_slice().cmp(DEFAULT))
                .is_ok_and(|at| !held[at].nopass || !held[at].enabled);
            self.guarded.store(guarded, Relaxed);
            self.restricted
                .store(held.iter().any(|u| !u.unrestricted()), Relaxed);
            // Released after the change and acquired before a reader looks at
            // its copy, so a session that sees a new number sees the table that
            // goes with it.
            self.generation.fetch_add(1, Release);
        }
        out
    }

    /// Whether a selector made from here on starts out allowed every channel.
    pub(crate) fn open_channels(&self) -> bool {
        self.open_channels.load(Relaxed)
    }

    /// Set `acl-pubsub-default`, which changes nothing that already exists.
    pub(crate) fn set_open_channels(&self, open: bool) {
        self.open_channels.store(open, Relaxed);
    }

    /// A copy of the user called `name`, if there is one.
    fn get(&self, name: &[u8]) -> Option<User> {
        self.with(|table| {
            let found = table
                .binary_search_by(|u| u.name.as_slice().cmp(name))
                .ok()
                .map(|at| table[at].clone());
            (false, found)
        })
    }

    /// The number the session's copy of a user is stamped with.
    fn generation(&self) -> u64 {
        self.generation.load(Acquire)
    }
}

impl Server {
    /// Whether this server asks connections for a password.
    ///
    /// True when the default user has one, which is exactly what `requirepass`
    /// means: a server whose default user is `nopass` lets an unauthenticated
    /// connection straight through as that user, and a server whose default user
    /// has a password does not.
    #[must_use]
    pub(crate) fn guarded(&self) -> bool {
        self.acl.guarded.load(Relaxed)
    }

    /// Whether the gate has to ask the ACL anything at all.
    #[must_use]
    pub(crate) fn restricted(&self) -> bool {
        self.acl.restricted.load(Relaxed)
    }

    /// Set or clear the default user's password, where an empty one clears it.
    ///
    /// This is the whole of `CONFIG SET requirepass`, and on a real server it is
    /// the whole of it too: the config option is a way of writing one rule on
    /// one user. Setting it drops any password the default user already had,
    /// which is what the reference does and is worth knowing, because it means
    /// `requirepass` and `ACL SETUSER default >pw` fight rather than add up.
    pub fn set_password(&self, password: &[u8]) {
        self.acl.with(|table| {
            let Ok(at) = table.binary_search_by(|u| u.name.as_slice().cmp(DEFAULT)) else {
                return (false, ());
            };
            let user = &mut table[at];
            user.passwords.clear();
            if password.is_empty() {
                user.nopass = true;
            } else {
                user.nopass = false;
                user.passwords.push(sha256::hex(password));
            }
            (true, ())
        });
        self.plain.set(password);
    }

    /// Hand the plain `requirepass` to `f`, which is how `CONFIG GET` writes it.
    pub(crate) fn with_password<T>(&self, f: impl FnOnce(&[u8]) -> T) -> T {
        self.plain.with(f)
    }

    /// The users, for the `ACL` command and for authentication.
    pub(crate) fn users(&self) -> &Users {
        &self.acl
    }
}

/// The plain `requirepass`, kept beside the hash because `CONFIG GET` reports it.
///
/// A real server does the same and it is worth being explicit about why, because
/// it looks like the hashing was pointless. It is not: the hash is what an ACL
/// user's password is, and the config file and `ACL LIST` and `ACL GETUSER` all
/// name it rather than this. This copy exists for one command, `CONFIG GET
/// requirepass`, which a real server answers with the password itself, and a
/// server that answered a hash there would break every tool that reads its own
/// config back.
#[derive(Debug, Default)]
pub(crate) struct Plain {
    /// Empty when there is no `requirepass`.
    secret: Mutex<Vec<u8>>,
}

impl Plain {
    /// Remember the password `CONFIG SET requirepass` was given.
    fn set(&self, password: &[u8]) {
        let mut held = self.secret.lock().unwrap_or_else(|e| e.into_inner());
        held.clear();
        held.extend_from_slice(password);
    }

    /// Hand it to `f`, borrowed rather than copied so it lives in one place.
    fn with<T>(&self, f: impl FnOnce(&[u8]) -> T) -> T {
        let held = self.secret.lock().unwrap_or_else(|e| e.into_inner());
        f(&held)
    }
}

/// Let this connection in as `user` if the password is right, and say whether
/// it did.
///
/// Answers the same thing for a user that does not exist and a user whose
/// password was wrong, on purpose: the difference between the two is a list of
/// user names. Both write the same line to the ACL log, naming the user that
/// was asked for, which is where an operator goes to find out that somebody has
/// been guessing.
///
/// `args` is the whole command, and the only thing read out of it is the name at
/// the front, because that is what the log calls the object: a failed `AUTH` and
/// a failed `HELLO ... AUTH` are otherwise the same row.
///
/// The generation is read before the copy is taken rather than after. A write
/// that lands in between then leaves the session stamped with a number that is
/// too small, so its next command takes a fresh copy for nothing, and the other
/// order would leave it stamped with a number that is too large and holding a
/// user whose rules had already changed.
pub(super) fn authenticate(
    server: &Server,
    session: &mut Session,
    user: &[u8],
    password: &[u8],
    args: Args<'_>,
    out: &Out,
) -> bool {
    let stamp = server.acl.generation();
    let Some(found) = server.acl.get(user) else {
        note_auth(server, session, out, args.get(0), user);
        return false;
    };
    if !found.admits(password) {
        note_auth(server, session, out, args.get(0), user);
        return false;
    }
    session.become_user(stamp, found);
    session.admit(true);
    true
}

// ------------------------------------------------------------------ the rules

/// Apply one rule to `user`, which is `ACLSetUser` written out.
///
/// The user level rules are here and everything else falls through to the
/// selector. That split is Redis's and it is why `ACL SETUSER u (on)` is a
/// syntax error: `on` is a fact about the account and a selector is a set of
/// permissions, so there is nowhere in a selector to put it.
fn set_user(user: &mut User, rule: &[u8], open: bool) -> std::result::Result<(), Bad> {
    if rule.is_empty() {
        return Ok(());
    }
    if word(rule, b"on") {
        user.enabled = true;
    } else if word(rule, b"off") {
        user.enabled = false;
    } else if word(rule, b"skip-sanitize-payload") {
        user.skip_sanitize = true;
    } else if word(rule, b"sanitize-payload") {
        user.skip_sanitize = false;
    } else if word(rule, b"nopass") {
        user.nopass = true;
        user.passwords.clear();
    } else if word(rule, b"resetpass") {
        user.nopass = false;
        user.passwords.clear();
    } else if rule[0] == b'>' || rule[0] == b'#' {
        let hash = hash_of(rule)?;
        if !user.passwords.contains(&hash) {
            user.passwords.push(hash);
        }
        // A user with a password is not a `nopass` user, whatever it was.
        user.nopass = false;
    } else if rule[0] == b'<' || rule[0] == b'!' {
        let hash = hash_of(rule)?;
        let before = user.passwords.len();
        user.passwords.retain(|held| *held != hash);
        if user.passwords.len() == before {
            return Err(Bad::NoSuchPassword);
        }
    } else if rule[0] == b'(' && rule[rule.len() - 1] == b')' {
        let mut selector = Selector::new(open);
        for word in split(&rule[1..rule.len() - 1]) {
            set_selector(&mut selector, &word)?;
        }
        user.selectors.push(selector);
    } else if rule[0] == b'(' {
        return Err(Bad::Unmatched);
    } else if word(rule, b"clearselectors") {
        user.selectors.truncate(1);
    } else if word(rule, b"reset") {
        let name = std::mem::take(&mut user.name);
        *user = User::new(&name, open);
    } else {
        return set_selector(&mut user.selectors[0], rule);
    }
    Ok(())
}

/// The hash a `>`, `#`, `<` or `!` rule names.
///
/// The first form of each pair is a password to hash and the second is a hash
/// already, and the only difference between them is whether the sixty four
/// characters are checked or produced.
fn hash_of(rule: &[u8]) -> std::result::Result<[u8; 64], Bad> {
    let rest = &rule[1..];
    if rule[0] == b'>' || rule[0] == b'<' {
        return Ok(sha256::hex(rest));
    }
    let ok = rest.len() == 64
        && rest
            .iter()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b));
    if !ok {
        return Err(Bad::Hash);
    }
    let mut hash = [0u8; 64];
    hash.copy_from_slice(rest);
    Ok(hash)
}

/// Apply one rule to a selector, which is `ACLSetSelector` written out.
fn set_selector(selector: &mut Selector, rule: &[u8]) -> std::result::Result<(), Bad> {
    if word(rule, b"allkeys") || rule == b"~*" {
        selector.all_keys = true;
        selector.patterns.clear();
    } else if word(rule, b"resetkeys") {
        selector.all_keys = false;
        selector.patterns.clear();
    } else if word(rule, b"allchannels") || rule == b"&*" {
        selector.all_channels = true;
        selector.channels.clear();
    } else if word(rule, b"resetchannels") {
        selector.all_channels = false;
        selector.channels.clear();
    } else if word(rule, b"allcommands") || rule == b"+@all" {
        selector.reset_commands(true);
    } else if word(rule, b"nocommands") || rule == b"-@all" {
        selector.reset_commands(false);
    } else if rule[0] == b'~' || rule[0] == b'%' {
        add_pattern(selector, rule)?;
    } else if rule[0] == b'&' {
        if selector.all_channels {
            return Err(Bad::AfterAllChannels);
        }
        let glob = &rule[1..];
        if glob.contains(&b' ') {
            return Err(Bad::Syntax);
        }
        if !selector.channels.iter().any(|held| held == glob) {
            selector.channels.push(glob.to_vec());
        }
    } else if rule[0] == b'+' && rule.get(1) != Some(&b'@') {
        add_command(selector, &rule[1..], true)?;
    } else if rule[0] == b'-' && rule.get(1) != Some(&b'@') {
        add_command(selector, &rule[1..], false)?;
    } else if rule[0] == b'+' || rule[0] == b'-' {
        let allow = rule[0] == b'+';
        add_category(selector, &rule[2..], allow)?;
        // Recorded with the sigil, because `@read` is what has to be written
        // back and a bare `read` would read as a command name.
        let mut name = Vec::with_capacity(rule.len() - 1);
        name.push(b'@');
        name.extend_from_slice(&rule[2..].to_ascii_lowercase());
        selector.note(&name, allow);
    } else {
        return Err(Bad::Syntax);
    }
    Ok(())
}

/// A `~pattern` or `%RW~pattern` rule.
fn add_pattern(selector: &mut Selector, rule: &[u8]) -> std::result::Result<(), Bad> {
    if selector.all_keys {
        return Err(Bad::AfterAllKeys);
    }
    let mut flags = 0u8;
    let mut at = 1;
    if rule[0] == b'%' {
        // The letters run until the `~`, each may appear once, and there has to
        // be at least one of them. `%~x` and `%RR~x` are both syntax errors. A
        // rule that ends before the `~` is not: a bare `%R` is a pattern with an
        // empty glob, which a real server takes and describes back as `%R~`.
        let mut ok = true;
        while at < rule.len() {
            let letter = rule[at].to_ascii_uppercase();
            if letter == b'R' && flags & READ == 0 {
                flags |= READ;
            } else if letter == b'W' && flags & WRITE == 0 {
                flags |= WRITE;
            } else if rule[at] == b'~' {
                at += 1;
                break;
            } else {
                ok = false;
                break;
            }
            at += 1;
        }
        if flags == 0 || !ok {
            return Err(Bad::Syntax);
        }
    } else {
        flags = READ | WRITE;
    }
    let glob = &rule[at..];
    if glob.contains(&b' ') {
        return Err(Bad::Syntax);
    }
    // The same pattern given twice with different letters is one pattern good
    // for both, which is why `%R~x %W~x` reads back as `~x`.
    if let Some(held) = selector.patterns.iter_mut().find(|p| p.glob == glob) {
        held.flags |= flags;
    } else {
        selector.patterns.push(Pattern {
            flags,
            glob: glob.to_vec(),
        });
    }
    Ok(())
}

/// A `+command` or `+command|first` rule.
fn add_command(selector: &mut Selector, name: &[u8], allow: bool) -> std::result::Result<(), Bad> {
    let Some(bar) = name.iter().rposition(|b| *b == b'|') else {
        let Some(spec) = table::lookup(name) else {
            return Err(Bad::Unknown);
        };
        selector.set(table::index_of(spec), allow);
        selector.note(&name.to_ascii_lowercase(), allow);
        return Ok(());
    };
    let (head, first) = (&name[..bar], &name[bar + 1..]);
    // The command has to exist even though the first argument cannot be checked,
    // so `+nosuch|get` is refused and `+get|nosuch` is not.
    let Some(spec) = table::lookup(head) else {
        return Err(Bad::Unknown);
    };
    if head.contains(&b'|') {
        return Err(Bad::NestedFirstArg);
    }
    if first.is_empty() {
        return Err(Bad::Syntax);
    }
    let at = u16::try_from(table::index_of(spec)).unwrap_or(u16::MAX);
    if allow {
        // Nothing to do when the command is already allowed outright, which is
        // what makes `+get +get|set` the same user as `+get`.
        if !selector.bit(usize::from(at)) {
            selector.allow_first(at, first);
        }
    } else {
        // Taking one subcommand away only means anything for a command that has
        // subcommands. A real server refuses `-get|nope` because it looks the
        // whole name up in the table and GET has no such row, so this refuses it
        // too, and for the same answer.
        if !super::CONTAINERS.contains(&spec.name) {
            return Err(Bad::Unknown);
        }
        selector.deny_first(at, first);
    }
    selector.note(&name.to_ascii_lowercase(), allow);
    Ok(())
}

/// A `+@category` or `-@category` rule.
fn add_category(selector: &mut Selector, name: &[u8], allow: bool) -> std::result::Result<(), Bad> {
    let Some(wanted) = category(name) else {
        return Err(Bad::Unknown);
    };
    for (at, spec) in table::COMMANDS.iter().enumerate() {
        if spec.acl.iter().any(|held| &held[1..] == wanted) {
            selector.set(at, allow);
        }
    }
    // And then the subcommands that are in the category without their container
    // being in it, one first argument at a time. `-@admin` has to reach CONFIG
    // GET and leave CONFIG HELP alone, and this is the only thing that knows the
    // two are different. See the note on `table::SUBCATS`.
    for (container, sub, cats) in table::SUBCATS {
        if !cats.iter().any(|held| &held[1..] == wanted) {
            continue;
        }
        let Some(spec) = table::lookup(container.as_bytes()) else {
            continue;
        };
        let at = u16::try_from(table::index_of(spec)).unwrap_or(u16::MAX);
        if !allow {
            selector.deny_first(at, sub.as_bytes());
        } else if !selector.bit(usize::from(at)) {
            selector.allow_first(at, sub.as_bytes());
        }
    }
    Ok(())
}

/// The category called `name`, whatever case it was written in.
fn category(name: &[u8]) -> Option<&'static str> {
    CATEGORIES
        .iter()
        .find(|held| name.eq_ignore_ascii_case(held.as_bytes()))
        .copied()
}

/// Whether `rule` is the keyword `word`, case insensitively.
fn word(rule: &[u8], keyword: &[u8]) -> bool {
    rule.eq_ignore_ascii_case(keyword)
}

/// Split a selector's inside into rules on runs of spaces.
///
/// A selector arrives as one argument, `(+get ~k:*)`, so the words inside it
/// have to be taken apart here. Redis uses its config file splitter, which
/// understands quotes; this does not, because a key pattern with a space in it
/// is refused by the rule above anyway and a quoted rule has nowhere to be
/// useful.
fn split(inside: &[u8]) -> Vec<Vec<u8>> {
    inside
        .split(|b| *b == b' ')
        .filter(|part| !part.is_empty())
        .map(<[u8]>::to_vec)
        .collect()
}

// ------------------------------------------------------------- the permissions

/// Why a command was refused, ranked the way Redis ranks them.
///
/// The rank decides which of several selectors' complaints is reported: a user
/// with two selectors that both say no is told about the most specific refusal,
/// on the grounds that the selector which got as far as looking at a key was the
/// one the operator meant to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Denied {
    /// Not allowed to run the command at all.
    Command,
    /// Allowed the command, not the key at this argument.
    Key(usize),
    /// Allowed the command, not the channel at this argument.
    Channel(usize),
}

impl Denied {
    /// Redis's numeric ranking, which is what two refusals are compared on.
    fn rank(self) -> u8 {
        match self {
            Denied::Command => 1,
            Denied::Key(_) => 2,
            Denied::Channel(_) => 4,
        }
    }

    /// The argument the refusal is about, or nought for the command itself.
    fn at(self) -> usize {
        match self {
            Denied::Command => 0,
            Denied::Key(at) | Denied::Channel(at) => at,
        }
    }
}

/// Where a command's channels are, since they are not in the key specs.
///
/// Eight commands and no more, which is Redis's own table. The flags in Redis
/// are four and only two of them are checked, so what is kept here is the one
/// bit that matters, whether the argument is a pattern being subscribed to, plus
/// whether the command is checked at all: unsubscribing is always allowed,
/// because a client that has lost permission to a channel still has to be able
/// to stop listening to it.
struct Channels {
    /// The first argument that is a channel.
    first: usize,
    /// How many there are, or `None` for all the rest.
    count: Option<usize>,
    /// Whether these are patterns, which are matched literally rather than as
    /// globs. `PSUBSCRIBE news.*` needs the ACL to hold `&news.*` exactly, not
    /// something that matches it, because otherwise `&news.sport` would let a
    /// client subscribe to `news.*` and hear everything.
    pattern: bool,
}

/// The channel arguments of `name`, if it has any that are checked.
fn channels_of(name: &str) -> Option<Channels> {
    let spec = match name {
        "subscribe" | "ssubscribe" => Channels {
            first: 1,
            count: None,
            pattern: false,
        },
        "psubscribe" => Channels {
            first: 1,
            count: None,
            pattern: true,
        },
        "publish" | "spublish" => Channels {
            first: 1,
            count: Some(1),
            pattern: false,
        },
        _ => return None,
    };
    Some(spec)
}

/// Whether `selector` may reach `key` for `need`.
fn key_ok(selector: &Selector, key: &[u8], need: Access) -> bool {
    if selector.all_keys {
        return true;
    }
    selector
        .patterns
        .iter()
        .any(|p| Access::from_bits(p.flags).covers(need) && glob_matches(&p.glob, key))
}

/// Whether `selector` may reach `channel`.
fn channel_ok(selector: &Selector, channel: &[u8], pattern: bool) -> bool {
    if selector.all_channels {
        return true;
    }
    selector.channels.iter().any(|held| {
        if pattern {
            held == channel
        } else {
            glob_matches(held, channel)
        }
    })
}

/// Whether one selector allows this whole command, and what it objected to.
fn selector_ok(
    selector: &Selector,
    spec: &'static Spec,
    args: Args<'_>,
    base: usize,
) -> std::result::Result<(), Denied> {
    if !selector.all_commands && !spec.flags.contains(&"no_auth") {
        let at = table::index_of(spec);
        let index = u16::try_from(at).unwrap_or(u16::MAX);
        let sub = (args.len() > base + 1).then(|| args.get(base + 1));
        let named = |list: &[Vec<u8>]| {
            sub.is_some_and(|word| list.iter().any(|held| word.eq_ignore_ascii_case(held)))
        };
        if selector.bit(at) {
            // Allowed outright, unless this is the one subcommand that was taken
            // back off it.
            if named(selector.denied(index)) {
                return Err(Denied::Command);
            }
        } else if !named(selector.firsts(index)) {
            return Err(Denied::Command);
        }
    }

    if !selector.all_keys && keyspec::takes_keys(spec, args, base) {
        let mut refused = None;
        keyspec::find(spec, args, base, &mut |run| {
            if refused.is_some() {
                return;
            }
            let need = run.need();
            for i in 0..run.count {
                let at = run.first + i * run.step;
                if at < args.len() && !key_ok(selector, args.get(at), need) {
                    refused = Some(Denied::Key(at));
                    return;
                }
            }
        });
        if let Some(why) = refused {
            return Err(why);
        }
    }

    if !selector.all_channels
        && let Some(where_) = channels_of(spec.name)
    {
        let first = base + where_.first;
        let stop = where_
            .count
            .map_or(args.len(), |n| (first + n).min(args.len()));
        for at in first..stop {
            if !channel_ok(selector, args.get(at), where_.pattern) {
                return Err(Denied::Channel(at));
            }
        }
    }
    Ok(())
}

/// Whether `user` may run this command, and what it objected to.
///
/// Every selector is tried and the first that says yes wins. When none does, the
/// refusal reported is the highest ranked one, and on a tie the one about the
/// argument furthest along, which is Redis's rule and is what makes a two
/// selector user complain about the key rather than the command.
pub(crate) fn permits(
    user: &User,
    spec: &'static Spec,
    args: Args<'_>,
    base: usize,
) -> std::result::Result<(), Denied> {
    // The whole of the cost on a server whose users can do anything, which is
    // every server nobody has written an ACL for.
    if let Some(root) = user.selectors.first()
        && root.all_commands
        && root.all_keys
        && root.all_channels
    {
        return Ok(());
    }
    let mut worst = Denied::Command;
    for selector in &user.selectors {
        match selector_ok(selector, spec, args, base) {
            Ok(()) => return Ok(()),
            Err(why) => {
                if why.rank() > worst.rank()
                    || (why.rank() == worst.rank() && why.at() > worst.at())
                {
                    worst = why;
                }
            }
        }
    }
    Err(worst)
}

/// The sentence a refusal is reported with.
///
/// `verbose` is the difference between the gate and `ACL DRYRUN`. The gate is
/// answering a client that has just been told no and naming the key it asked
/// about would tell it which keys exist, so the terse form says only that a key
/// was the problem. `ACL DRYRUN` is answering an operator who asked the
/// question on purpose and wants to know which key, so it names it.
pub(crate) fn refusal(
    why: Denied,
    user: &[u8],
    spec: &'static Spec,
    args: Args<'_>,
    base: usize,
    verbose: bool,
) -> String {
    let name = String::from_utf8_lossy(user);
    match why {
        Denied::Command => {
            let sub = container_name(spec, args, base);
            format!("User {name} has no permissions to run the '{sub}' command")
        }
        Denied::Key(at) if verbose => {
            let key = String::from_utf8_lossy(args.get(at));
            format!("User {name} has no permissions to access the '{key}' key")
        }
        Denied::Key(_) => "No permissions to access a key".to_string(),
        Denied::Channel(at) if verbose => {
            let channel = String::from_utf8_lossy(args.get(at));
            format!("User {name} has no permissions to access the '{channel}' channel")
        }
        Denied::Channel(_) => "No permissions to access a channel".to_string(),
    }
}

/// The command's name as a refusal spells it, which is `acl|list` and not `acl`.
fn container_name(spec: &'static Spec, args: Args<'_>, base: usize) -> String {
    if super::CONTAINERS.contains(&spec.name) && args.len() > base + 1 {
        let sub = String::from_utf8_lossy(args.get(base + 1)).to_lowercase();
        return format!("{}|{sub}", spec.name);
    }
    spec.name.to_string()
}

/// Who a connection is, and the copy of that user its commands are checked
/// against.
///
/// Boxed on the session, because it is three allocations and a connection on a
/// server with no ACL never reads any of them past the first command.
#[derive(Debug)]
pub(crate) struct Identity {
    /// The name, which is what `ACL WHOAMI` and `CLIENT INFO` report.
    name: Vec<u8>,
    /// The generation the copy below was taken at.
    stamp: u64,
    /// The copy itself.
    user: User,
}

impl Default for Identity {
    /// A connection starts as the default user, with a copy stamped nought so
    /// that the first command it sends fetches the real one.
    fn default() -> Identity {
        Identity {
            name: DEFAULT.to_vec(),
            stamp: 0,
            user: User::default_user(),
        }
    }
}

impl Session {
    /// The name this connection is authenticated as.
    pub(crate) fn acl_name(&self) -> &[u8] {
        &self.acl.name
    }

    /// Say that this connection has authenticated as `user`.
    ///
    /// Written to the row as well as here, because `CLIENT LIST` reports the
    /// user of every connection and runs on whichever thread the client asking
    /// is on, which is very often not this one.
    pub(super) fn become_user(&mut self, stamp: u64, user: User) {
        yo_alloc::allow(|| {
            self.acl.name.clear();
            self.acl.name.extend_from_slice(&user.name);
        });
        self.acl.stamp = stamp;
        self.acl.user = user;
        self.sock.set_text(|text| &mut text.user, &self.acl.name);
    }

    /// Put the connection back on the default user, which is what `RESET` does.
    pub(super) fn forget_user(&mut self) {
        *self.acl = Identity::default();
        // Empty rather than the name, which is how the row spells the default
        // user, so a connection that never authenticated costs nothing.
        self.sock.set_text(|text| &mut text.user, b"");
    }

    /// The generation the copy was taken at.
    fn acl_stamp(&self) -> u64 {
        self.acl.stamp
    }

    /// Take a fresh copy, or keep the one there is if the user has gone.
    fn acl_refresh(&mut self, now: u64, fresh: Option<User>) {
        self.acl.stamp = now;
        if let Some(user) = fresh {
            self.acl.user = user;
        }
    }

    /// The copy itself.
    fn acl_cached(&self) -> &User {
        &self.acl.user
    }
}

/// What the connection is running as, refreshed if the table has moved on.
///
/// A session keeps a copy of its user rather than a handle into the table,
/// because the alternative is taking the table's lock on every command on every
/// connection. The copy is stamped with the generation it was taken at and
/// replaced when that number moves, so a `SETUSER` that tightens a user's
/// permissions reaches every connection already authenticated as it, on that
/// connection's next command. That is what a real server does, and it is the
/// half of `SETUSER` that would be easy to get wrong: the point of tightening a
/// user is usually the connection that is already open.
///
/// A user deleted out from under a connection leaves the copy in place, which is
/// also Redis's behaviour: `ACL DELUSER` closes those connections rather than
/// leaving them running as a user that no longer exists.
fn current<'a>(server: &Server, session: &'a mut Session) -> &'a User {
    let now = server.acl.generation();
    if session.acl_stamp() != now {
        let fresh = server.acl.get(session.acl_name());
        session.acl_refresh(now, fresh);
    }
    session.acl_cached()
}

/// The gate, which is every command on a server that has an ACL worth checking.
///
/// `None` means the command may run. The refusal is written by the caller
/// rather than here, because it is one of the errors that kills an open
/// transaction and the caller is what knows about transactions.
pub(super) fn gate(
    server: &Server,
    session: &mut Session,
    spec: &'static Spec,
    args: Args<'_>,
    out: &Out,
) -> Option<String> {
    // The user is borrowed out of the session and the log wants the session
    // back, so everything read off the user is copied out here and the borrow
    // ends with the block. It costs one name and one sentence on a command that
    // has just been refused, which is not a path anything is timed on.
    let (why, said) = {
        let user = current(server, session);
        let why = permits(user, spec, args, 0).err()?;
        // The code is in the line rather than in front of it, because the line
        // goes two places: straight into the reply, and spliced into the
        // `EXECABORT` an `EXEC` gets. The same reason `NOAUTH` is a whole line.
        (why, refusal(why, &user.name, spec, args, 0, false))
    };
    yo_alloc::allow(|| {
        let object = match why {
            // The name a refusal spells, which is `acl|list` and not `acl`, and
            // is the same string the sentence above names.
            Denied::Command => container_name(spec, args, 0).into_bytes(),
            Denied::Key(at) | Denied::Channel(at) => args.get(at).to_vec(),
        };
        let name = session.acl_name().to_vec();
        server
            .acl_log()
            .note(server, session, out, why.reason(), object, name);
        Some(format!("NOPERM {said}"))
    })
}

// ----------------------------------------------------------------- the log

/// How long two refusals can be apart and still be counted as the same one.
///
/// A minute, which is Redis's `ACL_LOG_GROUPING_MAX_TIME_DELTA`. The point of it
/// is that a client stuck in a retry loop against a command it may not run fills
/// the log with one entry and a count rather than with a hundred and twenty eight
/// copies of itself, and the operator who comes to look still sees what else
/// happened.
const GROUPING_MS: u64 = 60_000;

/// How far back a new refusal looks for one it matches.
///
/// Ten, which is Redis's `toscan`. It is a bound on the work rather than on the
/// grouping: a refusal that would have matched the eleventh entry gets its own
/// row instead, and the operator sees two rows where a longer scan would have
/// shown one.
const SCAN: usize = 10;

/// Why something was refused, which is what `ACL LOG` reports as `reason`.
///
/// Redis has a fifth, `tls-cert`, for a client certificate that named a user the
/// server has not got. There is no TLS here, so there is no way to reach it and
/// no value for it, and the counter it feeds in `INFO stats` is reported as the
/// nought it would always be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Reason {
    /// A command the user may not run.
    Command,
    /// A key the user may not reach.
    Key,
    /// A channel the user may not reach.
    Channel,
    /// A password that did not get in.
    Auth,
}

impl Reason {
    /// The word `ACL LOG` prints.
    fn name(self) -> &'static str {
        match self {
            Reason::Command => "command",
            Reason::Key => "key",
            Reason::Channel => "channel",
            Reason::Auth => "auth",
        }
    }
}

impl Denied {
    /// The reason a refused command is logged under.
    fn reason(self) -> Reason {
        match self {
            Denied::Command => Reason::Command,
            Denied::Key(_) => Reason::Key,
            Denied::Channel(_) => Reason::Channel,
        }
    }
}

/// Where the refused command came from, which is what `ACL LOG` reports as
/// `context`.
///
/// Redis has a fourth, `module`, for a command a module ran on a user's behalf.
/// Nothing here runs a command from a module, so there is no way to reach it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Context {
    /// The client sent it.
    Toplevel,
    /// The client was queueing it into a transaction.
    Multi,
    /// A script called it.
    Lua,
}

impl Context {
    /// The word `ACL LOG` prints.
    fn name(self) -> &'static str {
        match self {
            Context::Toplevel => "toplevel",
            Context::Multi => "multi",
            Context::Lua => "lua",
        }
    }

    /// Where this connection's next refusal is coming from.
    ///
    /// A command being queued is `multi` and a command `EXEC` is replaying is
    /// not, which is Redis's answer and falls out of where the gate sits: the
    /// check happens as the command is queued, and the replay does not go
    /// through it again.
    fn of(session: &Session) -> Context {
        if session.scripted {
            Context::Lua
        } else if session.in_multi() {
            Context::Multi
        } else {
            Context::Toplevel
        }
    }
}

/// One refusal, which is one row of `ACL LOG`.
#[derive(Debug)]
struct Entry {
    /// How many refusals have been folded into this row.
    count: u64,
    reason: Reason,
    context: Context,
    /// What was refused: the command's name, or the key or channel it named, or
    /// the command the password was sent with.
    object: Vec<u8>,
    /// Who was refused, which for a failed `AUTH` is who they said they were.
    username: Vec<u8>,
    /// When the last refusal folded into this row happened.
    ctime: u64,
    /// The `CLIENT INFO` line of the connection that was refused.
    cinfo: String,
    /// Which refusal this was, counted over the life of the server.
    entry_id: u64,
    /// When the first refusal folded into this row happened.
    created: u64,
}

/// The refusals, newest first, and the counters that go with them.
///
/// Its own lock rather than a corner of [`Users`], because the two are touched
/// at opposite moments: the table is read on the command path and written by an
/// operator, and this is written only when something has already gone wrong. A
/// server whose users can do what they are asking for never takes this lock at
/// all.
#[derive(Debug)]
pub(crate) struct Log {
    /// Newest at the front, which is the order `ACL LOG` reports.
    entries: Mutex<VecDeque<Entry>>,
    /// How many rows have ever been opened, which is the next `entry-id`.
    next_id: AtomicU64,
    /// `acllog-max-len`, and nought means keep nothing.
    max_len: AtomicU64,
    /// The four counters `INFO stats` reports, in the order [`Reason`] declares
    /// them.
    denied: [AtomicU64; 4],
}

impl Default for Log {
    fn default() -> Log {
        Log {
            entries: Mutex::new(VecDeque::new()),
            next_id: AtomicU64::new(0),
            // A hundred and twenty eight, which is the reference's default.
            max_len: AtomicU64::new(128),
            denied: Default::default(),
        }
    }
}

impl Log {
    /// Write down a refusal.
    ///
    /// The counter moves whatever the length allows, which is Redis's order and
    /// matters: an operator who turned the log off with `acllog-max-len 0` still
    /// gets the numbers in `INFO stats` and only gives up the detail.
    fn note(
        &self,
        server: &Server,
        session: &Session,
        out: &Out,
        reason: Reason,
        object: Vec<u8>,
        username: Vec<u8>,
    ) {
        self.denied[reason as usize].fetch_add(1, Relaxed);
        let max = self.max_len.load(Relaxed);
        let mut held = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if max == 0 {
            held.clear();
            return;
        }
        let context = Context::of(session);
        let now = server.now_ms();
        // The whole `CLIENT INFO` line of the connection, which is what makes
        // the log worth reading: the row says a user was refused and this says
        // which connection, from which address, running which library. Without
        // the newline on the end, which `CLIENT INFO` puts there itself and this
        // does not, so the two fields differ by that one byte on a real server
        // too.
        let mut cinfo = super::client::report(server, session, out.proto(), out.len());
        if cinfo.ends_with('\n') {
            cinfo.pop();
        }
        let matched = held.iter().take(SCAN).position(|e| {
            e.reason == reason
                && e.context == context
                && e.object == object
                && e.username == username
                && now.abs_diff(e.ctime) <= GROUPING_MS
        });
        if let Some(at) = matched {
            // Moved to the front as well as bumped, so the log is in order of
            // when something last happened rather than of when it first did.
            let mut entry = held.remove(at).expect("the position came from the deque");
            entry.cinfo = cinfo;
            entry.ctime = now;
            entry.count += 1;
            held.push_front(entry);
            return;
        }
        held.push_front(Entry {
            count: 1,
            reason,
            context,
            object,
            username,
            ctime: now,
            cinfo,
            entry_id: self.next_id.fetch_add(1, Relaxed),
            created: now,
        });
        while held.len() as u64 > max {
            held.pop_back();
        }
    }

    /// `ACL LOG RESET`.
    ///
    /// The ids do not go back to nought, which is Redis's behaviour and is the
    /// useful one: a tool that remembers the last id it read must not be shown
    /// that id again attached to a different refusal.
    fn clear(&self) {
        self.entries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    /// The five numbers `INFO stats` reports, in the order it reports them.
    pub(crate) fn counters(&self) -> [u64; 5] {
        [
            self.denied[Reason::Auth as usize].load(Relaxed),
            self.denied[Reason::Command as usize].load(Relaxed),
            self.denied[Reason::Key as usize].load(Relaxed),
            self.denied[Reason::Channel as usize].load(Relaxed),
            // `acl_access_denied_tls_cert`, which cannot move without TLS.
            0,
        ]
    }

    /// `acllog-max-len`.
    pub(crate) fn max_len(&self) -> u64 {
        self.max_len.load(Relaxed)
    }

    /// `CONFIG SET acllog-max-len`.
    ///
    /// Nothing is trimmed here, which is the reference's behaviour: lowering the
    /// number decides how long the log is allowed to be after the next refusal,
    /// and until then `ACL LOG` still reports what is already in it.
    pub(crate) fn set_max_len(&self, n: u64) {
        self.max_len.store(n, Relaxed);
    }
}

impl Server {
    /// The refusals, for the gate and for `ACL LOG`.
    pub(crate) fn acl_log(&self) -> &Log {
        &self.acllog
    }
}

/// Write down a password that did not get in.
///
/// Its own function rather than a branch in [`authenticate`], because the two
/// things it needs that the rest of that path does not are the command the
/// password arrived with and the buffer the reply is going into.
fn note_auth(server: &Server, session: &Session, out: &Out, said: &[u8], user: &[u8]) {
    yo_alloc::allow(|| {
        server.acl_log().note(
            server,
            session,
            out,
            Reason::Auth,
            // The command as the client spelled it, which is `AUTH` or `HELLO`
            // in whatever case it was typed. That is what Redis logs, and it is
            // the only way to tell the two apart in the log afterwards.
            said.to_vec(),
            user.to_vec(),
        );
    });
}

// ------------------------------------------------------------- the command

/// The subcommands and how many arguments each takes, counting `ACL` itself.
///
/// A real server gives every one of these a row in the command table with its
/// own arity, and enforces it before the body is reached, which is why `ACL
/// WHOAMI x` is a wrong number of arguments and `ACL LOG 1 2` is an unknown
/// subcommand: the second one passes its own arity check and then falls off the
/// end of the parse. There is no subcommand table here yet, so the arities live
/// in this list and the same two sentences come out. It folds into D-114.
const ARITIES: [(&[u8], i32); 12] = [
    (b"cat", -2),
    (b"deluser", -3),
    (b"dryrun", -4),
    (b"genpass", -2),
    (b"getuser", 3),
    (b"help", 2),
    (b"list", 2),
    (b"load", 2),
    (b"log", -2),
    (b"save", 2),
    (b"setuser", -3),
    (b"users", 2),
    // `whoami` is not here because its arity is exactly two, which is the same
    // check the fallthrough below makes, and a row for it would be dead weight.
];

/// `ACL <subcommand> ...`.
pub(super) fn execute(
    server: &Server,
    session: &mut Session,
    args: Args<'_>,
    out: &mut Out,
) -> Result<()> {
    let sub = args.get(1);
    // The arity of the subcommand before anything else, which is where a real
    // server makes this decision: the row is found and checked in
    // `processCommand`, so a wrong count is refused before the body runs.
    if let Some((name, arity)) = ARITIES
        .iter()
        .find(|(name, _)| sub.eq_ignore_ascii_case(name))
    {
        let n = args.len() as i32;
        if (*arity > 0 && n != *arity) || (*arity < 0 && n < -*arity) {
            let name = std::str::from_utf8(name).unwrap_or("acl");
            return Err(args::wrong_arity_sub("acl", name));
        }
    }

    if is(sub, b"whoami") {
        if args.len() != 2 {
            return Err(args::wrong_arity_sub("acl", "whoami"));
        }
        out.bulk(session.acl_name());
    } else if (is(sub, b"load") || is(sub, b"save")) && server.aclfile().is_none() {
        // In front of both bodies and in front of everything either of them
        // would check, which is where the reference puts it: a server with no
        // ACL file says so and says nothing about what was asked of it.
        return Err(Error::new(
            Code::Invalid,
            "This Redis instance is not configured to use an ACL file. You may want to specify users via the ACL SETUSER command and then issue a CONFIG REWRITE (assuming you have a Redis configuration file set) in order to store users in the Redis configuration.",
        ));
    } else if is(sub, b"load") {
        return yo_alloc::allow(|| load(server, out));
    } else if is(sub, b"save") {
        return yo_alloc::allow(|| save(server, out));
    } else if is(sub, b"log") {
        return yo_alloc::allow(|| log(server, args, out));
    } else if is(sub, b"cat") {
        cat(args, out)?;
    } else if is(sub, b"list") {
        yo_alloc::allow(|| {
            server.users().with(|table| {
                out.array(table.len());
                for user in table.iter() {
                    let mut line = b"user ".to_vec();
                    line.extend_from_slice(&user.describe());
                    out.bulk(&line);
                }
                (false, ())
            });
        });
    } else if is(sub, b"users") {
        server.users().with(|table| {
            out.array(table.len());
            for user in table.iter() {
                out.bulk(&user.name);
            }
            (false, ())
        });
    } else if is(sub, b"getuser") {
        yo_alloc::allow(|| getuser(server, args, out));
    } else if is(sub, b"setuser") {
        return yo_alloc::allow(|| setuser(server, args, out));
    } else if is(sub, b"deluser") {
        return deluser(server, args, out);
    } else if is(sub, b"genpass") {
        genpass(args, out)?;
    } else if is(sub, b"dryrun") {
        return yo_alloc::allow(|| dryrun(server, args, out));
    } else if is(sub, b"help") {
        super::server::help(out, HELP);
    } else {
        return Err(args::unknown_subcommand(sub, "ACL"));
    }
    Ok(())
}

/// `ACL CAT` and `ACL CAT <category>`.
///
/// With no argument this is the list of categories, and with one it is the
/// commands in it, in table order. Redis walks a hash table there and so reports
/// an order that is neither sorted nor stable across versions, so matching it
/// exactly is not a thing to aim at; what a client can rely on is the set, and
/// this reports the same set for every category the two servers share.
fn cat(args: Args<'_>, out: &mut Out) -> Result<()> {
    if args.len() == 2 {
        out.array(CATEGORIES.len());
        for name in CATEGORIES {
            out.bulk(name.as_bytes());
        }
        return Ok(());
    }
    if args.len() > 3 {
        return Err(args::subcommand_syntax(args.get(1), "ACL"));
    }
    let Some(wanted) = category(args.get(2)) else {
        return Err(yo_alloc::allow(|| {
            Error::fmt(
                Code::Invalid,
                format_args!(
                    "Unknown category '{}'",
                    String::from_utf8_lossy(args.get(2))
                ),
            )
        }));
    };
    // The header goes on afterwards, because how many commands are in a
    // category is not a thing the table can be asked without walking it.
    let start = out.len();
    let mut n = 0;
    for spec in table::COMMANDS {
        if spec.acl.iter().any(|held| &held[1..] == wanted) {
            out.bulk(spec.name.as_bytes());
            n += 1;
        }
    }
    out.close_array(start, n);
    Ok(())
}

/// `ACL GETUSER <username>`.
fn getuser(server: &Server, args: Args<'_>, out: &mut Out) {
    let Some(user) = server.users().get(args.get(2)) else {
        out.nil();
        return;
    };
    // Six fields: the two about the account, the root selector's three repeated
    // at the top level for the clients that were written before selectors
    // existed, and the selectors themselves.
    out.map(6);
    out.bulk(b"flags");
    // Only the flags that belong to the account. The selector flags are named in
    // the same table and an older server did list them here, but 8.10.1 does not,
    // and a client that wants to know whether a user can reach every key reads
    // the keys field rather than counting words in this set.
    let flags = user.flags();
    let root = &user.selectors[0];
    out.set(flags.len());
    for flag in &flags {
        out.bulk(flag.as_bytes());
    }
    out.bulk(b"passwords");
    out.array(user.passwords.len());
    for hash in &user.passwords {
        out.bulk(hash);
    }
    describe_selector(root, out);
    out.bulk(b"selectors");
    out.array(user.selectors.len() - 1);
    for selector in &user.selectors[1..] {
        out.map(3);
        describe_selector(selector, out);
    }
}

/// The three fields a selector contributes to `ACL GETUSER`.
fn describe_selector(selector: &Selector, out: &mut Out) {
    out.bulk(b"commands");
    out.bulk(&selector.describe_commands());
    out.bulk(b"keys");
    out.bulk(&selector.describe_keys());
    out.bulk(b"channels");
    out.bulk(&selector.describe_channels());
}

/// `ACL SETUSER <username> [rule ...]`.
///
/// Every rule is applied to a copy and the copy replaces the user only if all of
/// them worked, so a `SETUSER` that fails halfway leaves nothing behind. That is
/// worth more than it sounds: the failure case is an operator tightening a
/// user's permissions and mistyping one rule, and a server that applied the
/// first half would have left the user with the new restrictions and none of the
/// new grants.
fn setuser(server: &Server, args: Args<'_>, out: &mut Out) -> Result<()> {
    let name = args.get(2);
    if name.contains(&b' ') || name.contains(&0) {
        return Err(Error::new(
            Code::Invalid,
            "Usernames can't contain spaces or null characters",
        ));
    }
    let rules = merge(args, 3)?;
    let open = server.users().open_channels();
    let outcome = server.users().with(|table| {
        let at = table.binary_search_by(|u| u.name.as_slice().cmp(name));
        let mut staged = match at {
            Ok(at) => table[at].clone(),
            Err(_) => User::new(name, open),
        };
        for rule in &rules {
            if let Err(why) = set_user(&mut staged, rule, open) {
                return (false, Err((rule.clone(), why)));
            }
        }
        match at {
            Ok(at) => table[at] = staged,
            Err(at) => table.insert(at, staged),
        }
        (true, Ok(()))
    });
    match outcome {
        Ok(()) => {
            out.ok();
            Ok(())
        }
        Err((rule, why)) => Err(Error::fmt(
            Code::Invalid,
            format_args!(
                "Error in ACL SETUSER modifier '{}': {}",
                String::from_utf8_lossy(&rule),
                why.text()
            ),
        )),
    }
}

/// Join the arguments from `from` on, gluing a selector back together.
///
/// A selector is one rule and a client sends it as several arguments, because
/// `(+get ~k:*)` has a space in it and the wire has no way to say that was meant
/// as one word. So a `(` that does not end in `)` swallows the arguments after
/// it until one does. An opening bracket that is never closed is the one rule
/// error reported in its own sentence rather than as a modifier error, because
/// there is no single modifier to blame.
fn merge(args: Args<'_>, from: usize) -> Result<Vec<Vec<u8>>> {
    let words = (from..args.len()).map(|i| args.get(i));
    glue(words).map_err(|at| {
        Error::fmt(
            Code::Invalid,
            format_args!(
                "Unmatched parenthesis in acl selector starting at '{}'.",
                String::from_utf8_lossy(args.get(from + at))
            ),
        )
    })
}

/// The half of [`merge`] the ACL file wants too, which is everything but the
/// sentence.
///
/// The error is which word opened the bracket that was never closed, because the
/// two callers name it differently: the command quotes the word and the file
/// gives the line it was on.
fn glue<'a>(words: impl Iterator<Item = &'a [u8]>) -> std::result::Result<Vec<Vec<u8>>, usize> {
    let mut out: Vec<Vec<u8>> = Vec::new();
    let mut open: Option<usize> = None;
    for (i, word) in words.enumerate() {
        if open.is_none() && word.first() == Some(&b'(') && word.last() != Some(&b')') {
            open = Some(i);
            out.push(word.to_vec());
            continue;
        }
        if open.is_some() {
            let held = out.last_mut().expect("an open bracket left a rule behind");
            held.push(b' ');
            held.extend_from_slice(word);
            if word.last() == Some(&b')') {
                open = None;
            }
            continue;
        }
        out.push(word.to_vec());
    }
    match open {
        Some(at) => Err(at),
        None => Ok(out),
    }
}

/// `ACL DELUSER <username> [<username> ...]`.
///
/// The default user is checked for over the whole list before anything is
/// deleted, so `ACL DELUSER alice default` deletes neither. Redis does the same
/// and the reason is the same as `SETUSER`'s: a half done change to who may
/// reach a server is worse than no change.
fn deluser(server: &Server, args: Args<'_>, out: &mut Out) -> Result<()> {
    for i in 2..args.len() {
        if args.get(i) == DEFAULT {
            return Err(Error::new(
                Code::Invalid,
                "The 'default' user cannot be removed",
            ));
        }
    }
    let gone = server.users().with(|table| {
        let mut gone = 0;
        for i in 2..args.len() {
            if let Ok(at) = table.binary_search_by(|u| u.name.as_slice().cmp(args.get(i))) {
                table.remove(at);
                gone += 1;
            }
        }
        (gone > 0, gone)
    });
    out.int(gone);
    Ok(())
}

/// `ACL GENPASS [<bits>]`.
///
/// The bytes come from the operating system rather than from the engine's own
/// generator, which is seeded and reproducible on purpose. See
/// [`yo_common::entropy`].
fn genpass(args: Args<'_>, out: &mut Out) -> Result<()> {
    if args.len() > 3 {
        return Err(args::subcommand_syntax(args.get(1), "ACL"));
    }
    let bits = if args.len() == 3 { args.int(2)? } else { 256 };
    if bits <= 0 || bits > 4096 {
        return Err(Error::new(
            Code::Invalid,
            "ACL GENPASS argument must be the number of bits for the output password, a positive number up to 4096",
        ));
    }
    // One hex character is four bits, rounded up, so `GENPASS 10` is three
    // characters and holds twelve bits rather than ten. That is the reference's
    // arithmetic and it errs towards more entropy than was asked for.
    let chars = ((bits + 3) / 4) as usize;
    let mut raw = [0u8; 4096 / 8 + 1];
    let bytes = chars.div_ceil(2);
    yo_common::entropy::fill(&mut raw[..bytes]);
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut hex = [0u8; 1024];
    for (i, slot) in hex[..chars].iter_mut().enumerate() {
        let byte = raw[i / 2];
        *slot = DIGITS[usize::from(if i % 2 == 0 { byte >> 4 } else { byte & 0xf })];
    }
    out.bulk(&hex[..chars]);
    Ok(())
}

/// `ACL DRYRUN <username> <command> [<arg> ...]`.
///
/// The same question the gate asks, asked out loud. The answer is a bulk string
/// rather than an error even when it is a refusal, because the command
/// succeeded: it was asked whether something would be allowed and it found out.
fn dryrun(server: &Server, args: Args<'_>, out: &mut Out) -> Result<()> {
    let Some(user) = server.users().get(args.get(2)) else {
        return Err(Error::fmt(
            Code::Invalid,
            format_args!("User '{}' not found", String::from_utf8_lossy(args.get(2))),
        ));
    };
    let Some(spec) = table::lookup(args.get(3)) else {
        return Err(Error::fmt(
            Code::Invalid,
            format_args!(
                "Command '{}' not found",
                String::from_utf8_lossy(args.get(3))
            ),
        ));
    };
    if !table::arity_ok(spec, args.len() - 3) {
        return Err(args::wrong_arity(spec.name));
    }
    match permits(&user, spec, args, 3) {
        Ok(()) => out.ok(),
        Err(why) => {
            let text = refusal(why, &user.name, spec, args, 3, true);
            out.bulk(text.as_bytes());
        }
    }
    Ok(())
}

/// `ACL LOG [<count> | RESET]`.
fn log(server: &Server, args: Args<'_>, out: &mut Out) -> Result<()> {
    // Two arguments or three and nothing else, and a fourth is an unknown
    // subcommand rather than a wrong count. That reads like a mistake and is
    // not: a real server checks `acl|log`'s own arity first, which is a minimum
    // of two and lets `ACL LOG 1 2` through, and then falls off the end of a
    // parse that only knows two shapes.
    if args.len() > 3 {
        return Err(args::subcommand_syntax(args.get(1), "ACL"));
    }
    if args.len() == 3 && is(args.get(2), b"reset") {
        server.acl_log().clear();
        out.ok();
        return Ok(());
    }
    // Ten by default, and a negative count is nought rather than an error, which
    // is what makes `ACL LOG -1` an empty array.
    let wanted = if args.len() == 3 {
        args.int(2)?.max(0)
    } else {
        10
    };
    let now = server.now_ms();
    let held = server
        .acl_log()
        .entries
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let count = (wanted as u64).min(held.len() as u64) as usize;
    out.array(count);
    for entry in held.iter().take(count) {
        out.map(10);
        out.bulk(b"count");
        out.int(entry.count as i64);
        out.bulk(b"reason");
        out.bulk(entry.reason.name().as_bytes());
        out.bulk(b"context");
        out.bulk(entry.context.name().as_bytes());
        out.bulk(b"object");
        out.bulk(&entry.object);
        out.bulk(b"username");
        out.bulk(&entry.username);
        out.bulk(b"age-seconds");
        // Seconds with the milliseconds after the point, which is the one field
        // here that is a double, and it is the age now rather than the age when
        // the refusal happened.
        out.double(now.saturating_sub(entry.ctime) as f64 / 1000.0);
        out.bulk(b"client-info");
        out.bulk(entry.cinfo.as_bytes());
        out.bulk(b"entry-id");
        out.int(entry.entry_id as i64);
        out.bulk(b"timestamp-created");
        out.int(entry.created as i64);
        out.bulk(b"timestamp-last-updated");
        out.int(entry.ctime as i64);
    }
    Ok(())
}

/// `ACL SAVE`, which writes the users out in the format the file is read in.
///
/// A temporary file beside the real one, then fsync, then rename, then fsync of
/// the directory. That is Redis's sequence and it is the only sequence that
/// leaves a reader with either the whole old file or the whole new one: a write
/// straight over the real file would leave a half written ACL on a server that
/// lost power, and a server that came back up refusing to let anybody in is a
/// worse outcome than one that came back up with yesterday's users.
fn save(server: &Server, out: &mut Out) -> Result<()> {
    let Some(path) = server.aclfile() else {
        return Ok(());
    };
    let mut text = Vec::with_capacity(256);
    server.users().with(|table| {
        for user in table.iter() {
            text.extend_from_slice(b"user ");
            text.extend_from_slice(&user.describe());
            text.push(b'\n');
        }
        (false, ())
    });
    if write_file(path, &text).is_err() {
        // The reason is in the server log on a real server and there is nowhere
        // to put it here, so the client gets the sentence and nothing else,
        // which is also what a real server gives it.
        return Err(Error::new(
            Code::Invalid,
            "There was an error trying to save the ACLs. Please check the server logs for more information",
        ));
    }
    out.ok();
    Ok(())
}

/// The write, the rename and the two syncs.
fn write_file(path: &Path, text: &[u8]) -> std::io::Result<()> {
    use std::io::Write as _;

    let temp = path.with_file_name(format!(
        "{}.tmp-{}-{}",
        path.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis())
    ));
    let outcome = (|| {
        let mut file = std::fs::File::create(&temp)?;
        file.write_all(text)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&temp, path)
    })();
    if outcome.is_err() {
        let _ = std::fs::remove_file(&temp);
        return outcome;
    }
    // The directory, so that the rename itself is on disk and not only the
    // bytes the rename pointed at. Best effort, because opening a directory as a
    // file is a Unix thing and Windows answers an error rather than a handle,
    // and a failure here means the file is written and the entry naming it may
    // not have reached the platter yet.
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty())
        && let Ok(handle) = std::fs::File::open(dir)
    {
        let _ = handle.sync_all();
    }
    Ok(())
}

/// `ACL LOAD`, which replaces every user with what the file says.
fn load(server: &Server, out: &mut Out) -> Result<()> {
    let Some(path) = server.aclfile() else {
        return Ok(());
    };
    match load_file(server, path) {
        Ok(()) => {
            out.ok();
            Ok(())
        }
        Err(errors) => Err(Error::fmt(Code::Invalid, format_args!("{errors}"))),
    }
}

/// Read `path` and, if every line of it is good, make it the server's users.
///
/// The whole file or none of it. The users are built in a table of their own and
/// only swapped in once the last line has parsed, which is the reference's
/// design and is the only sane one: a file with a typo halfway down would
/// otherwise leave a server holding half of the new ACL and half of the old one,
/// and nobody could say which half.
///
/// # Errors
///
/// Every complaint the file raised, joined into one sentence, which is what the
/// client gets and what a server that cannot start prints.
pub(crate) fn load_file(server: &Server, path: &Path) -> std::result::Result<(), String> {
    let name = path.display().to_string();
    let text = match std::fs::read(path) {
        Ok(text) => text,
        Err(e) => {
            return Err(format!(
                "Error loading ACLs, opening file '{name}': {}",
                because(&e)
            ));
        }
    };
    let mut errors = String::new();
    let mut staged: Vec<User> = Vec::new();
    let open = server.users().open_channels();
    for (at, raw) in text.split(|b| *b == b'\n').enumerate() {
        let line = trim(raw);
        // Blank lines and comments, which is what lets a file be commented.
        if line.is_empty() || line[0] == b'#' {
            continue;
        }
        let linenum = at + 1;
        let words: Vec<&[u8]> = line.split(|b| *b == b' ').collect();
        if words[0] != b"user" || words.len() < 2 {
            errors.push_str(&format!(
                "{name}:{linenum} should start with user keyword followed by the username. "
            ));
            continue;
        }
        // A username with a space in it could not be read back, since this is
        // where the reading happens and it splits on spaces. A tab or a null is
        // refused for the same reason a space is: the file is the only place a
        // user is written down and a name that cannot be written down is a user
        // that cannot be got at again.
        let who = words[1];
        if who.iter().any(|b| b.is_ascii_whitespace() || *b == 0) {
            errors.push_str(&format!(
                "'{name}:{linenum}: username '{}' contains invalid characters. ",
                String::from_utf8_lossy(who)
            ));
            continue;
        }
        if staged.iter().any(|u| u.name == who) {
            errors.push_str(&format!(
                "WARNING: Duplicate user '{}' found on line {linenum}. ",
                String::from_utf8_lossy(who)
            ));
            continue;
        }
        let Ok(rules) = glue(words[2..].iter().copied()) else {
            errors.push_str(&format!(
                "{name}:{linenum}: Unmatched parenthesis in selector definition."
            ));
            continue;
        };
        let mut user = User::new(who, open);
        let mut said = false;
        for rule in &rules {
            let Err(why) = set_user(&mut user, trim(rule), open) else {
                continue;
            };
            if why == Bad::Unknown {
                // A name nobody has heard of is quoted back, because a command
                // name is not a secret and an operator staring at a file wants
                // to know which word was wrong. Every other complaint is about
                // the shape of a rule that may hold a password hash.
                errors.push_str(&format!(
                    "{name}:{linenum}: Error in applying operation '{}': {}. ",
                    String::from_utf8_lossy(rule),
                    why.text()
                ));
            } else if !said {
                // Only the first of the others, because a rule that failed may
                // have been what the rules after it were written against, and
                // eight complaints about one mistake is worse than one.
                errors.push_str(&format!("{name}:{linenum}: {}. ", why.text()));
                said = true;
            }
        }
        staged.push(user);
    }
    if !errors.is_empty() {
        errors.push_str(
            "WARNING: ACL errors detected, no change to the previously active ACL rules was performed",
        );
        return Err(errors);
    }
    staged.sort_by(|a, b| a.name.cmp(&b.name));
    server.users().with(|table| {
        *table = staged;
        // A file with no default user in it still leaves the server with one,
        // because there has to be a user for a connection that has not
        // authenticated to be. The reference gets there by a different route,
        // making a fresh default and copying it over the old one, and lands on
        // the same user.
        if let Err(at) = table.binary_search_by(|u| u.name.as_slice().cmp(DEFAULT)) {
            table.insert(at, User::default_user());
        }
        (true, ())
    });
    Ok(())
}

/// What went wrong with a file, in the words the C library would have used.
///
/// Rust writes `No such file or directory (os error 2)` where C's `strerror`
/// writes `No such file or directory`, and the sentence this ends up in is one
/// a client may be matching on, so the number Rust adds comes back off.
fn because(e: &std::io::Error) -> String {
    let said = e.to_string();
    match said.find(" (os error ") {
        Some(at) => said[..at].to_string(),
        None => said,
    }
}

/// A line with the blanks taken off both ends.
///
/// The same four characters the reference trims, which is why a file written on
/// Windows loads: the carriage return at the end of every line is one of them.
fn trim(line: &[u8]) -> &[u8] {
    let blank = |b: &u8| matches!(b, b' ' | b'\t' | b'\r' | b'\n');
    let from = line.iter().position(|b| !blank(b)).unwrap_or(line.len());
    let to = line.iter().rposition(|b| !blank(b)).map_or(from, |i| i + 1);
    &line[from..to]
}

/// What `ACL HELP` says.
const HELP: &[&str] = &[
    "ACL <subcommand> [<arg> [value] [opt] ...]. Subcommands are:",
    "CAT [<category>]",
    "    List all commands that belong to <category>, or all command categories",
    "    when no category is specified.",
    "DELUSER <username> [<username> ...]",
    "    Delete a list of users.",
    "DRYRUN <username> <command> [<arg> ...]",
    "    Returns whether the user can execute the given command without executing the command.",
    "GETUSER <username>",
    "    Get the user's details.",
    "GENPASS [<bits>]",
    "    Generate a secure 256-bit user password. The optional `bits` argument can",
    "    be used to specify a different size.",
    "LIST",
    "    Show users details in config file format.",
    "LOAD",
    "    Reload users from the ACL file.",
    "LOG [<count> | RESET]",
    "    Show the ACL log entries.",
    "SAVE",
    "    Save the current config to the ACL file.",
    "SETUSER <username> <attribute> [<attribute> ...]",
    "    Create or modify a user with the specified attributes.",
    "USERS",
    "    List all the registered usernames.",
    "WHOAMI",
    "    Return the current connection username.",
    "HELP",
    "    Print this help.",
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::Limits;
    use crate::request::{Argv, Step};

    /// A user built by applying rules in order, or the first rule that failed.
    fn built(rules: &[&str]) -> std::result::Result<User, Bad> {
        let mut user = User::new(b"u", false);
        for rule in rules {
            set_user(&mut user, rule.as_bytes(), false)?;
        }
        Ok(user)
    }

    /// The user's line the way `ACL LIST` writes it.
    fn listed(rules: &[&str]) -> String {
        let user = built(rules).expect("every rule here is a good one");
        String::from_utf8(user.describe()).expect("the description is text")
    }

    /// Whether this user could run this command, said as a word so a failing
    /// assertion reads like the question it was asked.
    fn allows(rules: &[&str], words: &[&str]) -> bool {
        let user = built(rules).expect("every rule here is a good one");
        let mut buf = format!("*{}\r\n", words.len()).into_bytes();
        for word in words {
            buf.extend_from_slice(format!("${}\r\n{word}\r\n", word.len()).as_bytes());
        }
        let mut argv = Argv::new();
        let Ok(Step::Command { .. }) = argv.decode(&buf, &Limits::default()) else {
            panic!("the test wrote a command that does not decode");
        };
        let args = Args::new(&argv, &buf);
        let spec = table::lookup(args.name()).expect("a command this server has");
        permits(&user, spec, args, 0).is_ok()
    }

    #[test]
    fn a_key_permission_with_no_pattern_after_it_is_an_empty_pattern() {
        // 8.10.1 takes a bare `%R` and writes it back with the `~` it never got.
        assert_eq!(
            listed(&["%R"]),
            "u off sanitize-payload %R~ resetchannels -@all"
        );
        assert_eq!(
            listed(&["%RW"]),
            "u off sanitize-payload ~ resetchannels -@all"
        );
    }

    #[test]
    fn a_key_permission_that_is_not_r_or_w_once_each_is_a_syntax_error() {
        assert_eq!(built(&["%"]).err(), Some(Bad::Syntax));
        assert_eq!(built(&["%~k:*"]).err(), Some(Bad::Syntax));
        assert_eq!(built(&["%RR~k:*"]).err(), Some(Bad::Syntax));
        assert_eq!(built(&["%X~k:*"]).err(), Some(Bad::Syntax));
    }

    #[test]
    fn a_subcommand_can_be_taken_back_off_a_container_that_was_allowed() {
        assert_eq!(
            listed(&["+config", "-config|get"]),
            "u off sanitize-payload resetchannels -@all +config -config|get"
        );
        assert!(allows(
            &["+config", "-config|get"],
            &["config", "set", "maxmemory", "0"]
        ));
        assert!(!allows(
            &["+config", "-config|get"],
            &["config", "get", "maxmemory"]
        ));
        // And allowing the container again forgets the exception, the same way
        // allowing it forgets a first argument it was limited to.
        assert_eq!(
            listed(&["+config", "-config|get", "+config"]),
            "u off sanitize-payload resetchannels -@all +config"
        );
        assert!(allows(
            &["+config", "-config|get", "+config"],
            &["config", "get", "maxmemory"]
        ));
    }

    #[test]
    fn a_first_argument_can_only_be_taken_off_a_command_that_has_subcommands() {
        // GET has no subcommands, so a real server looks up `get|nope`, finds
        // nothing, and says so. Allowing one is a different mechanism and works.
        assert_eq!(built(&["-get|nope"]).err(), Some(Bad::Unknown));
        assert_eq!(built(&["-select|0"]).err(), Some(Bad::Unknown));
        assert!(built(&["+get|nope"]).is_ok());
    }

    #[test]
    fn a_category_a_subcommand_holds_reaches_that_subcommand_and_no_further() {
        // CONFIG is only `@slow` in the table because 8.10.1 puts `@admin` on
        // `config|get` and `config|set` rather than on `config`, so a rule about
        // `@admin` has to take CONFIG GET away and leave CONFIG HELP behind.
        let deny = ["~*", "+@all", "-@admin"];
        assert!(!allows(&deny, &["config", "get", "maxmemory"]));
        assert!(allows(&deny, &["config", "help"]));
        assert!(!allows(&deny, &["client", "kill", "id", "4"]));
        assert!(allows(&deny, &["client", "setname", "x"]));
        assert!(allows(&deny, &["acl", "whoami"]));
        assert!(!allows(&deny, &["acl", "setuser", "u"]));
        // And the other way round, from a user that starts with nothing.
        let grant = ["~*", "-@all", "+@admin"];
        assert!(allows(&grant, &["config", "get", "maxmemory"]));
        assert!(!allows(&grant, &["config", "help"]));
        assert!(!allows(&grant, &["get", "k"]));
        // None of which changes what the container is listed as being in, or
        // `COMMAND INFO config` and `ACL CAT admin` would both start lying.
        let config = table::lookup(b"config").expect("a command this server has");
        assert_eq!(config.acl, ["@slow"]);
    }

    #[test]
    fn the_flags_a_user_reports_are_the_ones_about_the_account() {
        // Not `allkeys`, which an older server did report here and 8.10.1 does
        // not. A client that wants to know reads the keys field.
        let user = built(&["on", "~*", "&*", "+@all"]).expect("good rules");
        assert_eq!(user.flags(), ["on", "sanitize-payload"]);
    }

    #[test]
    fn a_line_of_the_file_loses_the_blanks_on_both_ends() {
        assert_eq!(trim(b"  user alice  "), b"user alice");
        assert_eq!(trim(b"user alice\r"), b"user alice");
        assert_eq!(trim(b"\t\r\n "), b"");
        assert_eq!(trim(b""), b"");
        // Only the ends, because a rule is separated from the next one by a
        // single space and taking the inner ones out would join two rules.
        assert_eq!(trim(b" a  b "), b"a  b");
    }

    #[test]
    fn the_reason_a_file_would_not_open_reads_the_way_c_writes_it() {
        let missing = std::io::Error::from_raw_os_error(2);
        // What is claimed here is that the number Rust puts on the end comes
        // off, and that is true everywhere. The sentence in front of it is the
        // system's own and is not: errno 2 is `No such file or directory` out
        // of a C library and `The system cannot find the file specified.` out
        // of Windows, and a server built on Windows should say what Windows
        // says rather than repeat a sentence from another operating system.
        let said = because(&missing);
        assert!(!said.contains("(os error"), "{said}");
        assert!(missing.to_string().starts_with(&said), "{said}");
        #[cfg(unix)]
        assert_eq!(said, "No such file or directory");
        // Anything with no errno behind it is left alone, since there is no
        // number on the end of it to take off.
        let made_up = std::io::Error::other("something else");
        assert_eq!(because(&made_up), "something else");
    }

    #[test]
    fn a_new_selector_starts_where_acl_pubsub_default_says() {
        let mut open = User::new(b"u", true);
        set_user(&mut open, b"on", true).expect("a good rule");
        assert!(String::from_utf8_lossy(&open.describe()).contains("&*"));
        // And a reset goes back to the same place rather than to the built in
        // one, which is what makes the setting worth having.
        set_user(&mut open, b"reset", true).expect("a good rule");
        assert!(String::from_utf8_lossy(&open.describe()).contains("&*"));
        let shut = User::new(b"u", false);
        assert!(String::from_utf8_lossy(&shut.describe()).contains("resetchannels"));
    }
}
