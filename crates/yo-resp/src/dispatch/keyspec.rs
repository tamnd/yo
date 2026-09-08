//! Where a command's keys are, and what it does to each of them.
//!
//! # Why the old triple was not enough
//!
//! Until now the answer was three numbers on every table row: the first
//! argument that is a key, the last, and how far apart they sit. That triple is
//! Redis's own legacy answer, it is what `COMMAND INFO` has reported since 2.8,
//! and it is wrong often enough to matter. `ZUNIONSTORE dst 2 a b` has three
//! keys and the triple names one of them. `XREAD COUNT 2 STREAMS a b 0 0` has
//! two and the triple names none. `SORT k STORE d` has two and the triple names
//! one.
//!
//! Nothing minded very much while the only caller was `COMMAND GETKEYS`, where
//! a wrong answer is a cluster client routing to the wrong node and a real but
//! distant problem. The ACL is a different matter. A user given `~cache:*` and
//! nothing else must not be able to reach `secret` through `ZUNIONSTORE`, and a
//! permission check built on a key finder that misses keys is not a permission
//! check. So this is the model Redis replaced the triple with in 7.0, copied
//! rather than approximated, and the triple stays beside it because
//! `COMMAND INFO` still reports it.
//!
//! # The model
//!
//! A command has a list of key specs and each one is two questions. Where does
//! this run of keys start, which is [`Begin`], and how far does it go, which is
//! [`Find`]. `ZUNIONSTORE` has two: one key at argument one, and a counted run
//! starting at argument two. `SMOVE` has two, one for each end of the move,
//! which is how the source can be a delete and the destination an insert.
//!
//! Beside them are the flags, and those are the half the ACL reads. `RO`, `RW`,
//! `OW` and `RM` say how the value is touched, and `access`, `update`, `insert`
//! and `delete` say what a caller has to be allowed to do: a key with `access`
//! needs read permission and a key with any of the other three needs write
//! permission. That is why `SET k v` needs only write while `INCR k` needs both,
//! which is not a rule anybody would guess and is not derivable from the command
//! flags. It comes off the reference one command at a time.
//!
//! # What incomplete means
//!
//! Three specs cannot be trusted on their own and say so. `SORT`'s `BY`, `GET`
//! and `STORE` keys are [`Begin::Unknown`], because the first two are patterns
//! that name keys only after the sorted value has been read and the third can
//! appear anywhere. `XREAD` and `XREADGROUP` look for the word `STREAMS`, which
//! a stream could be called. `MIGRATE` and the two `GEORADIUS` writes look for a
//! keyword that may appear twice.
//!
//! A spec carrying `incomplete`, or one whose search is unknown, means the
//! answer this module gives is a floor rather than the whole of it. `COMMAND
//! GETKEYS` fills the rest in per command, and the ACL treats a command with one
//! as needing permission over every key, which is the safe way round: a user
//! who may reach every key is allowed and one who may not is refused.

use super::args::Args;
use super::table::Spec;
use yo_common::parse_i64;

/// One run of keys in a command's arguments.
#[derive(Debug, Clone, Copy)]
pub struct KeySpec {
    /// What the reference says about this spec, empty for most of them.
    ///
    /// Carried because `COMMAND DOCS` reports it and because every one of them
    /// explains a spec that would otherwise read as a mistake.
    pub notes: &'static str,
    /// How the value is touched and what a caller has to be allowed to do.
    pub flags: &'static [&'static str],
    /// Where the run starts.
    pub begin: Begin,
    /// How far it goes.
    pub find: Find,
}

/// Where a run of keys starts.
#[derive(Debug, Clone, Copy)]
pub enum Begin {
    /// At a fixed argument, which is nearly all of them.
    At(u32),
    /// After a keyword, searched for from an argument.
    ///
    /// A positive start searches forward from there and a negative one searches
    /// backward from that far before the end, which is how `MIGRATE` finds a
    /// `KEYS` that has to be the last option.
    After(&'static [u8], i32),
    /// Somewhere only the command itself can work out.
    Unknown,
}

/// How far a run of keys goes.
#[derive(Debug, Clone, Copy)]
pub enum Find {
    /// A fixed run, counting from the start of it.
    ///
    /// `last` is relative: nought is one key, and a negative number counts back
    /// from the last argument. `limit` is nought except where the run is one
    /// part of what is left, which is `XREAD` splitting the tail into keys and
    /// ids.
    Range {
        /// The last key, relative to the start of the run.
        last: i32,
        /// How many arguments apart consecutive keys are.
        step: u32,
        /// How many ways what is left is divided, nought for all but `XREAD`.
        limit: u32,
    },
    /// A run whose length is a number in the arguments.
    ///
    /// `count` is where that number is, measured from the start of the run, and
    /// `first` is where the keys begin from the same place.
    Counted {
        /// Where the number sits, from the start of the run.
        count: u32,
        /// Where the keys begin, from the same place.
        first: u32,
        /// How many arguments apart consecutive keys are.
        step: u32,
    },
    /// A run only the command itself can work out.
    Unknown,
}

/// What a caller has to be allowed to do with a key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Access(u8);

impl Access {
    /// The value is read, so a caller needs read permission over the key.
    pub const READ: Access = Access(1);
    /// The value is changed, added or taken away.
    pub const WRITE: Access = Access(2);
    /// Both, which is what an incomplete command is treated as needing.
    pub const BOTH: Access = Access(3);
    /// Neither, which is a key a command only asks about.
    pub const NONE: Access = Access(0);

    /// Whether everything `other` asks for is in here.
    #[must_use]
    pub const fn covers(self, other: Access) -> bool {
        self.0 & other.0 == other.0
    }

    /// Both together.
    #[must_use]
    pub const fn and(self, other: Access) -> Access {
        Access(self.0 | other.0)
    }

    /// Whether this asks for nothing at all.
    #[must_use]
    pub const fn is_none(self) -> bool {
        self.0 == 0
    }
}

impl KeySpec {
    /// What a caller needs over the keys this spec names.
    #[must_use]
    pub fn access(&self) -> Access {
        access_of(self.flags)
    }

    /// Whether this spec is a floor rather than the whole answer.
    #[must_use]
    pub fn incomplete(&self) -> bool {
        self.flags.contains(&"incomplete")
            || matches!(self.begin, Begin::Unknown)
            || matches!(self.find, Find::Unknown)
    }

    /// Whether this spec names an argument that looks like a key and is not.
    ///
    /// `SPUBLISH`'s channel is one, and it is the reason that command answers
    /// `COMMAND GETKEYS` with an error rather than the channel name.
    #[must_use]
    pub fn fake(&self) -> bool {
        self.flags.contains(&"not_key")
    }

    /// How many arguments apart consecutive keys are, which is never nought.
    fn step(&self) -> u32 {
        match self.find {
            Find::Range { step, .. } | Find::Counted { step, .. } => step.max(1),
            Find::Unknown => 1,
        }
    }
}

/// One run of keys, resolved against the arguments a client actually sent.
#[derive(Debug, Clone, Copy)]
pub struct Run {
    /// The argument index of the first key.
    pub first: usize,
    /// How many keys there are, which can be nought.
    pub count: usize,
    /// How many arguments apart consecutive keys are.
    pub step: usize,
    /// The flags every key in the run carries, which is a spec's own list.
    pub flags: &'static [&'static str],
}

impl Run {
    /// What a caller needs over every key in this run.
    #[must_use]
    pub fn need(&self) -> Access {
        access_of(self.flags)
    }
}

/// What a caller needs over a key carrying `flags`.
///
/// `access` is read and any of `update`, `insert` and `delete` is write, which is
/// Redis's own rule. A key flagged `not_key` is an argument that looks like one
/// and is not, so it asks for nothing: that is `GEORADIUS`'s member and `LPOS`'s
/// element.
#[must_use]
pub fn access_of(flags: &[&str]) -> Access {
    if flags.contains(&"not_key") {
        return Access::NONE;
    }
    let mut need = Access::NONE;
    if flags.contains(&"access") {
        need = need.and(Access::READ);
    }
    if flags.contains(&"update") || flags.contains(&"insert") || flags.contains(&"delete") {
        need = need.and(Access::WRITE);
    }
    need
}

/// Every key `spec` names in `args`, handed to `each` a run at a time.
///
/// `base` is how many arguments sit in front of the command itself, which is two
/// for `COMMAND GETKEYS <command> ...` and nought for a command that is running.
///
/// False means the arguments do not resolve: a count that is not a number, a run
/// that reaches past the end, a keyword that cannot be where it says it is. That
/// is a command that is about to fail on its own arguments, and the caller
/// decides what to do about it. Nothing was handed to `each` in that case.
///
/// The specs are tried first and a finder is what is left when they cannot
/// answer, which is the order `getKeysFromCommandWithSpecs` goes in and is not
/// the same as choosing between them. `PFMERGE dst` is the plainest case: its
/// second spec starts at argument two of a two argument command, so the specs
/// fail and the finder is the only thing that names the destination.
pub fn find(spec: &Spec, args: Args<'_>, base: usize, each: &mut dyn FnMut(Run)) -> bool {
    let keys = keys_of(spec, args, base);
    let real = keys.iter().any(|k| !k.fake());
    // Three commands decide from their options what they do to their key, and
    // for those the flags a spec carries are the wrong ones, so the specs are
    // skipped rather than tried.
    let varying = keys.iter().any(|k| k.flags.contains(&"variable_flags"));
    if real && !varying && walk(keys, args, base, each) {
        return true;
    }
    if let Some(finder) = finder_for(spec.name) {
        return finder(args, base, each);
    }
    // No finder, so either the specs failed and the command is about to fail
    // with them, or there were never any specs and there are no keys.
    !real
}

/// Whether this command ever names a key, whatever it is sent.
///
/// This is not the same question as whether it names one here. `GEORADIUS`
/// without a `STORE` names one key and `SUBSCRIBE` names none ever, and the
/// second is the one a real server refuses `COMMAND GETKEYS` for. A container is
/// asked about the word behind it, since that is where its keys live.
pub fn takes_keys(spec: &Spec, args: Args<'_>, base: usize) -> bool {
    finder_for(spec.name).is_some() || keys_of(spec, args, base).iter().any(|k| !k.fake())
}

/// The specs to read for this command, which for a container are its
/// subcommand's.
///
/// A container row carries none of its own, so it is asked about the word behind
/// it. Every other row with no specs simply has no keys.
fn keys_of(spec: &Spec, args: Args<'_>, base: usize) -> &'static [KeySpec] {
    if !spec.keys.is_empty() {
        return spec.keys;
    }
    if args.len() <= base + 1 {
        return &[];
    }
    of_sub(spec.name, args.get(base + 1))
}

/// Every key a list of specs names, with no per command finder in the way.
fn walk(keys: &[KeySpec], args: Args<'_>, base: usize, each: &mut dyn FnMut(Run)) -> bool {
    // Nothing is handed over until every spec has resolved, because a command
    // that fails halfway names no keys at all rather than the ones found so far.
    let mut runs = [None::<Run>; MOST_SPECS];
    let mut at = 0;
    for key in keys {
        // An argument that looks like a key and is not is not one here either.
        if key.fake() {
            continue;
        }
        match resolve(key, args, base) {
            Resolved::Run(run) => {
                if at == runs.len() {
                    return false;
                }
                runs[at] = Some(run);
                at += 1;
                // A spec that says it is a floor has given what it can, and what
                // is left is the finder's, so this is a failure like any other.
                if key.incomplete() {
                    return false;
                }
            }
            // A keyword that is not there is not a failure. `MIGRATE k 0 1 0` has
            // no `KEYS` in it and no keys hiding behind one, and the spec that
            // looks for it has simply found that out.
            Resolved::None => {}
            Resolved::Invalid => return false,
        }
    }
    for run in runs.iter().flatten() {
        if run.count > 0 {
            each(*run);
        }
    }
    true
}

/// The most key specs any one command has, which is `GEORADIUS` with three.
const MOST_SPECS: usize = 4;

/// What one spec came to against one set of arguments.
enum Resolved {
    /// A run of keys, which can be an empty one.
    Run(Run),
    /// The keyword this spec looks for is not there, so it names nothing.
    None,
    /// The arguments do not fit the spec, so the command is about to fail.
    Invalid,
}

/// One spec against one set of arguments.
///
/// This is `getKeysUsingKeySpecs` line for line, including the two places it
/// looks careless and is not. The forward keyword search stops one short of the
/// end, because a keyword in the last argument has no key behind it. And a run
/// that reaches past the last argument is invalid rather than cut short, because
/// the command it came from is about to be refused on its arity anyway and a
/// short answer would be a wrong one.
fn resolve(key: &KeySpec, args: Args<'_>, base: usize) -> Resolved {
    let argc = (args.len() - base) as i64;
    let at = |i: i64| args.get(base + i as usize);
    let mut first = match key.begin {
        Begin::At(index) => i64::from(index),
        Begin::After(word, from) => {
            // Forward from where it says, stopping one short of the end, or
            // backward from that far before the end, stopping at argument two.
            // Both directions are the reference's own loop: the bound is a stop
            // rather than a limit, and running off either end gives up quietly.
            let start = if from > 0 {
                i64::from(from)
            } else {
                argc + i64::from(from)
            };
            let end = if from > 0 { argc - 1 } else { 1 };
            let mut found = 0;
            let mut i = start;
            while i != end {
                if i >= argc || i < 1 {
                    break;
                }
                if at(i).eq_ignore_ascii_case(word) {
                    found = i + 1;
                    break;
                }
                i += if start <= end { 1 } else { -1 };
            }
            if found == 0 {
                return Resolved::None;
            }
            found
        }
        Begin::Unknown => return Resolved::Invalid,
    };
    let step = i64::from(key.step());
    let last = match key.find {
        Find::Range { last, limit, .. } => {
            if last >= 0 {
                first + i64::from(last)
            } else if limit == 0 {
                argc + i64::from(last)
            } else {
                // The run is one part of what is left, which only `XREAD` and
                // `XREADGROUP` are: the tail after `STREAMS` is half keys and
                // half ids, so the last key is halfway along it.
                first + ((argc - first) / i64::from(limit) + i64::from(last))
            }
        }
        Find::Counted {
            count, first: from, ..
        } => {
            let index = first + i64::from(count);
            if index >= argc || index < 0 {
                return Resolved::Invalid;
            }
            // A count is read as a number from nought upward and nothing else, so
            // a negative one or a word is the command failing rather than a
            // command with no keys.
            let Some(n) = parse_i64(at(index)).filter(|&n| n >= 0) else {
                return Resolved::Invalid;
            };
            first += i64::from(from);
            match n
                .checked_sub(1)
                .and_then(|n| n.checked_mul(step))
                .and_then(|n| first.checked_add(n))
            {
                Some(last) => last,
                None => return Resolved::Invalid,
            }
        }
        Find::Unknown => return Resolved::Invalid,
    };
    // Off either end is a syntax error rather than a shorter run, because the
    // command it came from is about to be refused anyway and a short answer
    // would be a wrong one.
    if last >= argc || last < first || first >= argc {
        return Resolved::Invalid;
    }
    Resolved::Run(Run {
        first: base + first as usize,
        count: ((last - first) / step + 1) as usize,
        step: step as usize,
        flags: key.flags,
    })
}

// ------------------------------------------------------------- the finders
//
// Eight commands keep their keys somewhere a spec cannot say, and three more
// decide what they do to a key from the options they were given. A real server
// answers all eleven with a function written for that command, and so does this:
// the specs are what `COMMAND INFO` reports and these are what everything else
// reads, which is exactly the split the reference has.

/// The flags a key that is only read carries.
const READ: &[&str] = &["RO", "access"];
/// The flags a destination written over carries.
const OVERWRITE: &[&str] = &["OW", "update"];
/// The flags a key read and written carries.
const BOTH: &[&str] = &["RW", "access", "update"];
/// The flags `MIGRATE` gives every key it moves.
const MOVED: &[&str] = &["RW", "access", "delete"];
/// The flags `PFMERGE` gives the counter it merges into.
const MERGED: &[&str] = &["RW", "access", "insert"];
/// The flags a key compared against a value and then removed carries.
const COMPARED: &[&str] = &["RW", "delete"];
/// The flags a key removed without its value being looked at carries.
const REMOVED: &[&str] = &["RM", "delete"];

/// A finder written for one command, standing in for specs that cannot say
/// where its keys are or what it does to them.
type Finder = fn(Args<'_>, usize, &mut dyn FnMut(Run)) -> bool;

/// The finder for `name`, for the eleven commands that need one.
fn finder_for(name: &str) -> Option<Finder> {
    Some(match name {
        "sort" => sort_keys,
        "sort_ro" => sort_ro_keys,
        "migrate" => migrate_keys,
        "xread" | "xreadgroup" => xread_keys,
        "georadius" | "georadiusbymember" => georadius_keys,
        "set" => set_keys,
        "bitfield" => bitfield_keys,
        "delex" => delex_keys,
        "pfmerge" => pfmerge_keys,
        _ => return None,
    })
}

/// One key at one argument.
fn one(at: usize, flags: &'static [&'static str], each: &mut dyn FnMut(Run)) {
    each(Run {
        first: at,
        count: 1,
        step: 1,
        flags,
    });
}

/// `SORT key [BY pat] [LIMIT o c] [GET pat ...] [STORE dst]`.
///
/// The `BY` and `GET` patterns name keys and are not reported, because which keys
/// they name is only known once the sorted value has been read. A real server
/// deals with that by refusing the pattern outright when the user cannot reach
/// every key, which is the ACL's problem rather than this one.
///
/// `STORE` is last wins, so a command naming two destinations names the second.
fn sort_keys(args: Args<'_>, base: usize, each: &mut dyn FnMut(Run)) -> bool {
    one(base + 1, READ, each);
    let argc = args.len() - base;
    let mut store = None;
    let mut i = 2;
    while i < argc {
        let arg = args.get(base + i);
        if arg.eq_ignore_ascii_case(b"limit") {
            i += 2;
        } else if arg.eq_ignore_ascii_case(b"get") || arg.eq_ignore_ascii_case(b"by") {
            i += 1;
        } else if arg.eq_ignore_ascii_case(b"store") && i + 1 < argc {
            store = Some(base + i + 1);
        }
        i += 1;
    }
    if let Some(at) = store {
        one(at, OVERWRITE, each);
    }
    true
}

/// `SORT_RO key [BY pat] [LIMIT o c] [GET pat ...]`, which has no destination.
fn sort_ro_keys(_args: Args<'_>, base: usize, each: &mut dyn FnMut(Run)) -> bool {
    one(base + 1, READ, each);
    true
}

/// `MIGRATE host port key|"" db timeout [COPY] [REPLACE] [AUTH pw] [AUTH2 u pw] [KEYS k ...]`.
///
/// The single key form names argument three and the `KEYS` form names everything
/// behind the keyword. Naming both is a syntax error the command itself reports,
/// so this names nothing and lets it.
fn migrate_keys(args: Args<'_>, base: usize, each: &mut dyn FnMut(Run)) -> bool {
    let argc = args.len() - base;
    let mut first = 3;
    let mut count = 1;
    if argc > 6 {
        let mut i = 6;
        while i < argc {
            let arg = args.get(base + i);
            if arg.eq_ignore_ascii_case(b"keys") {
                if args.get(base + 3).is_empty() {
                    first = i + 1;
                    count = argc - first;
                } else {
                    count = 0;
                }
                break;
            }
            if arg.eq_ignore_ascii_case(b"auth") {
                i += 1;
            } else if arg.eq_ignore_ascii_case(b"auth2") {
                i += 2;
            }
            i += 1;
        }
    }
    if count > 0 {
        each(Run {
            first: base + first,
            count,
            step: 1,
            flags: MOVED,
        });
    }
    true
}

/// `XREAD [COUNT n] [BLOCK ms] STREAMS key ... id ...` and the group form.
///
/// The options in front are walked rather than counted, because a stream, a
/// group or a consumer may itself be called `STREAMS` and the first one that is
/// an option value has to be stepped over rather than matched.
fn xread_keys(args: Args<'_>, base: usize, each: &mut dyn FnMut(Run)) -> bool {
    let argc = args.len() - base;
    let mut streams = None;
    let mut i = 1;
    while i < argc {
        let arg = args.get(base + i);
        if arg.eq_ignore_ascii_case(b"block") || arg.eq_ignore_ascii_case(b"count") {
            i += 1;
        } else if arg.eq_ignore_ascii_case(b"group") {
            i += 2;
        } else if arg.eq_ignore_ascii_case(b"noack") {
            // Nothing follows it.
        } else if arg.eq_ignore_ascii_case(b"streams") {
            streams = Some(i);
            break;
        } else {
            // Anything else is a syntax error the command will report.
            break;
        }
        i += 1;
    }
    let Some(streams) = streams else {
        return false;
    };
    let tail = argc - streams - 1;
    if tail == 0 || !tail.is_multiple_of(2) {
        return false;
    }
    each(Run {
        first: base + streams + 1,
        count: tail / 2,
        step: 1,
        flags: READ,
    });
    true
}

/// `GEORADIUS key ... [STORE dst] [STOREDIST dst]` and the member form.
///
/// Both destinations write to the same slot in a real server, so naming `STORE`
/// and `STOREDIST` names one key and it is the last of the two.
fn georadius_keys(args: Args<'_>, base: usize, each: &mut dyn FnMut(Run)) -> bool {
    one(base + 1, READ, each);
    let argc = args.len() - base;
    let mut store = None;
    let mut i = 5;
    while i < argc {
        let arg = args.get(base + i);
        if (arg.eq_ignore_ascii_case(b"store") || arg.eq_ignore_ascii_case(b"storedist"))
            && i + 1 < argc
        {
            store = Some(base + i + 1);
            i += 1;
        }
        i += 1;
    }
    if let Some(at) = store {
        one(at, OVERWRITE, each);
    }
    true
}

/// `SET key value [GET] ...`, which reads the key only when it is asked to.
///
/// Without `GET` the old value is never looked at, so a user who may write the
/// key and not read it may run it. That is the whole of what `variable_flags`
/// means and it is why this cannot be a spec.
fn set_keys(args: Args<'_>, base: usize, each: &mut dyn FnMut(Run)) -> bool {
    let gets = (base + 3..args.len()).any(|i| args.get(i).eq_ignore_ascii_case(b"get"));
    one(base + 1, if gets { BOTH } else { OVERWRITE }, each);
    true
}

/// `BITFIELD key [GET ...] [SET ...] [INCRBY ...] [OVERFLOW ...]`.
///
/// A command that only gets is a read, and anything else, including a command
/// that is about to be refused for its syntax, is a read and a write.
fn bitfield_keys(args: Args<'_>, base: usize, each: &mut dyn FnMut(Run)) -> bool {
    let argc = args.len() - base;
    let mut reads = true;
    let mut i = 2;
    while i < argc {
        let left = argc - i - 1;
        let arg = args.get(base + i);
        if arg.eq_ignore_ascii_case(b"get") && left >= 2 {
            i += 2;
        } else if (arg.eq_ignore_ascii_case(b"set") || arg.eq_ignore_ascii_case(b"incrby"))
            && left >= 3
        {
            reads = false;
            break;
        } else if arg.eq_ignore_ascii_case(b"overflow") && left >= 1 {
            i += 1;
        } else {
            reads = false;
            break;
        }
        i += 1;
    }
    one(base + 1, if reads { READ } else { BOTH }, each);
    true
}

/// `DELEX key [IFEQ v|IFNE v|IFDEQ d|IFDNE d]`.
///
/// A condition of any of the four kinds reads the key before deciding, so it is
/// `RW`. Without one the key goes whatever is in it, which is `RM`. Neither form
/// needs read permission, so the two answers differ only in the letters
/// `COMMAND GETKEYSANDFLAGS` prints, which is a thing worth getting right on its
/// own.
fn delex_keys(args: Args<'_>, base: usize, each: &mut dyn FnMut(Run)) -> bool {
    let compares = args.opt(base + 2).is_some_and(|a| {
        a.eq_ignore_ascii_case(b"ifeq")
            || a.eq_ignore_ascii_case(b"ifne")
            || a.eq_ignore_ascii_case(b"ifdeq")
            || a.eq_ignore_ascii_case(b"ifdne")
    });
    one(base + 1, if compares { COMPARED } else { REMOVED }, each);
    true
}

/// `PFMERGE dst [src ...]`, whose sources are optional.
///
/// The specs cannot say that, because a run starting at argument two of a two
/// argument command is off the end and off the end is a syntax error. So a real
/// server answers this one from a function and `PFMERGE dst` names its
/// destination rather than failing.
fn pfmerge_keys(args: Args<'_>, base: usize, each: &mut dyn FnMut(Run)) -> bool {
    one(base + 1, MERGED, each);
    let argc = args.len() - base;
    if argc > 2 {
        each(Run {
            first: base + 2,
            count: argc - 2,
            step: 1,
            flags: READ,
        });
    }
    true
}

// --------------------------------------------------------------- the shapes
//
// Every distinct spec in the table, named once and shared by every command that
// has it. Fifty eight of them cover four hundred and twenty nine commands, and
// naming them rather than writing each row out is what keeps a key spec
// readable next to the command it belongs to.
//
// The numbers are the reference's own, read off `COMMAND INFO` on 8.10.1 rather
// than out of the documentation. The commands a real server has never heard of,
// which is the module groups, have a spec built from the legacy triple their row
// already carried, with read permission for a command flagged `readonly` and
// both for one flagged `write`.

/// not_key.
pub const NOT_KEY_AT1: KeySpec = KeySpec {
    notes: "",
    flags: &["not_key"],
    begin: Begin::At(1),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// not_key.
pub const NOT_KEY_AT1_RM1_1_0: KeySpec = KeySpec {
    notes: "",
    flags: &["not_key"],
    begin: Begin::At(1),
    find: Find::Range {
        last: -1,
        step: 1,
        limit: 0,
    },
};

/// OW, insert.
pub const OW_INSERT_AT1: KeySpec = KeySpec {
    notes: "",
    flags: &["OW", "insert"],
    begin: Begin::At(1),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// OW, insert.
pub const OW_INSERT_AT1_RM1_2_0: KeySpec = KeySpec {
    notes: "",
    flags: &["OW", "insert"],
    begin: Begin::At(1),
    find: Find::Range {
        last: -1,
        step: 2,
        limit: 0,
    },
};

/// OW, insert.
pub const OW_INSERT_AT2: KeySpec = KeySpec {
    notes: "",
    flags: &["OW", "insert"],
    begin: Begin::At(2),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// OW, update.
pub const OW_UPDATE_AT1: KeySpec = KeySpec {
    notes: "",
    flags: &["OW", "update"],
    begin: Begin::At(1),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// OW, update.
pub const OW_UPDATE_AT1_COUNTED: KeySpec = KeySpec {
    notes: "",
    flags: &["OW", "update"],
    begin: Begin::At(1),
    find: Find::Counted {
        count: 0,
        first: 1,
        step: 2,
    },
};

/// OW, update.
pub const OW_UPDATE_AT1_RM1_2_0: KeySpec = KeySpec {
    notes: "",
    flags: &["OW", "update"],
    begin: Begin::At(1),
    find: Find::Range {
        last: -1,
        step: 2,
        limit: 0,
    },
};

/// OW, update.
pub const OW_UPDATE_AT2: KeySpec = KeySpec {
    notes: "",
    flags: &["OW", "update"],
    begin: Begin::At(2),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// Incomplete because duplicate STORE options use last-wins; fall back to georadiusGetKeys
pub const GEORADIUS_STORE: KeySpec = KeySpec {
    notes: "Incomplete because duplicate STORE options use last-wins; fall back to georadiusGetKeys",
    flags: &["OW", "update", "incomplete"],
    begin: Begin::After(b"STORE", 6),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// Incomplete because duplicate STOREDIST options use last-wins; fall back to georadiusGetKeys
pub const GEORADIUS_STOREDIST: KeySpec = KeySpec {
    notes: "Incomplete because duplicate STOREDIST options use last-wins; fall back to georadiusGetKeys",
    flags: &["OW", "update", "incomplete"],
    begin: Begin::After(b"STOREDIST", 6),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// Incomplete because duplicate STOREDIST options use last-wins; fall back to georadiusGetKeys
pub const BYMEMBER_STOREDIST: KeySpec = KeySpec {
    notes: "Incomplete because duplicate STOREDIST options use last-wins; fall back to georadiusGetKeys",
    flags: &["OW", "update", "incomplete"],
    begin: Begin::After(b"STOREDIST", 5),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// Incomplete because duplicate STORE options use last-wins; fall back to georadiusGetKeys
pub const BYMEMBER_STORE: KeySpec = KeySpec {
    notes: "Incomplete because duplicate STORE options use last-wins; fall back to georadiusGetKeys",
    flags: &["OW", "update", "incomplete"],
    begin: Begin::After(b"STORE", 5),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// For the optional STORE keyword. It is marked 'unknown' because the keyword can appear anywhere in the argument array
pub const SORT_STORE: KeySpec = KeySpec {
    notes: "For the optional STORE keyword. It is marked 'unknown' because the keyword can appear anywhere in the argument array",
    flags: &["OW", "update"],
    begin: Begin::Unknown,
    find: Find::Unknown,
};

/// RM, delete.
pub const RM_DELETE_AT1_RM1_1_0: KeySpec = KeySpec {
    notes: "",
    flags: &["RM", "delete"],
    begin: Begin::At(1),
    find: Find::Range {
        last: -1,
        step: 1,
        limit: 0,
    },
};

/// RO, access.
pub const RO_ACCESS_AT1: KeySpec = KeySpec {
    notes: "",
    flags: &["RO", "access"],
    begin: Begin::At(1),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// RO, access.
pub const RO_ACCESS_AT1_COUNTED: KeySpec = KeySpec {
    notes: "",
    flags: &["RO", "access"],
    begin: Begin::At(1),
    find: Find::Counted {
        count: 0,
        first: 1,
        step: 1,
    },
};

/// RO, access.
pub const RO_ACCESS_AT1_R1_1_0: KeySpec = KeySpec {
    notes: "",
    flags: &["RO", "access"],
    begin: Begin::At(1),
    find: Find::Range {
        last: 1,
        step: 1,
        limit: 0,
    },
};

/// RO, access.
pub const RO_ACCESS_AT1_RM1_1_0: KeySpec = KeySpec {
    notes: "",
    flags: &["RO", "access"],
    begin: Begin::At(1),
    find: Find::Range {
        last: -1,
        step: 1,
        limit: 0,
    },
};

/// RO, access.
pub const RO_ACCESS_AT1_RM2_1_0: KeySpec = KeySpec {
    notes: "",
    flags: &["RO", "access"],
    begin: Begin::At(1),
    find: Find::Range {
        last: -2,
        step: 1,
        limit: 0,
    },
};

/// RO, access.
pub const RO_ACCESS_AT2: KeySpec = KeySpec {
    notes: "",
    flags: &["RO", "access"],
    begin: Begin::At(2),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// We cannot tell how the keys will be used so we assume the worst, RO and ACCESS
pub const SCRIPT_KEYS_RO: KeySpec = KeySpec {
    notes: "We cannot tell how the keys will be used so we assume the worst, RO and ACCESS",
    flags: &["RO", "access"],
    begin: Begin::At(2),
    find: Find::Counted {
        count: 0,
        first: 1,
        step: 1,
    },
};

/// RO, access.
pub const RO_ACCESS_AT2_COUNTED: KeySpec = KeySpec {
    notes: "",
    flags: &["RO", "access"],
    begin: Begin::At(2),
    find: Find::Counted {
        count: 0,
        first: 1,
        step: 1,
    },
};

/// RO, access.
pub const RO_ACCESS_AT2_RM1_1_0: KeySpec = KeySpec {
    notes: "",
    flags: &["RO", "access"],
    begin: Begin::At(2),
    find: Find::Range {
        last: -1,
        step: 1,
        limit: 0,
    },
};

/// RO, access.
pub const RO_ACCESS_AT3_RM1_1_0: KeySpec = KeySpec {
    notes: "",
    flags: &["RO", "access"],
    begin: Begin::At(3),
    find: Find::Range {
        last: -1,
        step: 1,
        limit: 0,
    },
};

/// Incomplete because a stream key named STREAMS (or options before it) can shift the STREAMS keyword; fall back to xreadGetKeys
pub const XREAD_STREAMS: KeySpec = KeySpec {
    notes: "Incomplete because a stream key named STREAMS (or options before it) can shift the STREAMS keyword; fall back to xreadGetKeys",
    flags: &["RO", "access", "incomplete"],
    begin: Begin::After(b"STREAMS", 1),
    find: Find::Range {
        last: -1,
        step: 1,
        limit: 2,
    },
};

/// Incomplete because a consumer/group named STREAMS (or options before GROUP) can shift the STREAMS keyword; fall back to xreadGetKeys
pub const XREADGROUP_STREAMS: KeySpec = KeySpec {
    notes: "Incomplete because a consumer/group named STREAMS (or options before GROUP) can shift the STREAMS keyword; fall back to xreadGetKeys",
    flags: &["RO", "access", "incomplete"],
    begin: Begin::After(b"STREAMS", 4),
    find: Find::Range {
        last: -1,
        step: 1,
        limit: 2,
    },
};

/// For the optional BY/GET keyword. It is marked 'unknown' because the key names derive from the content of the key we sort
pub const SORT_BY_AND_GET: KeySpec = KeySpec {
    notes: "For the optional BY/GET keyword. It is marked 'unknown' because the key names derive from the content of the key we sort",
    flags: &["RO", "access"],
    begin: Begin::Unknown,
    find: Find::Unknown,
};

/// RO.
pub const RO_AT1: KeySpec = KeySpec {
    notes: "",
    flags: &["RO"],
    begin: Begin::At(1),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// RO.
pub const RO_AT1_RM1_1_0: KeySpec = KeySpec {
    notes: "",
    flags: &["RO"],
    begin: Begin::At(1),
    find: Find::Range {
        last: -1,
        step: 1,
        limit: 0,
    },
};

/// RO.
pub const RO_AT2: KeySpec = KeySpec {
    notes: "",
    flags: &["RO"],
    begin: Begin::At(2),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// RW because it may change the internal representation of the key, and propagate to replicas
pub const RW_ACCESS_AT1_RM1_1_0: KeySpec = KeySpec {
    notes: "RW because it may change the internal representation of the key, and propagate to replicas",
    flags: &["RW", "access"],
    begin: Begin::At(1),
    find: Find::Range {
        last: -1,
        step: 1,
        limit: 0,
    },
};

/// RW, access.
pub const RW_ACCESS_AT2: KeySpec = KeySpec {
    notes: "",
    flags: &["RW", "access"],
    begin: Begin::At(2),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// RW, access, delete.
pub const RW_ACCESS_DELETE_AT1: KeySpec = KeySpec {
    notes: "",
    flags: &["RW", "access", "delete"],
    begin: Begin::At(1),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// RW, access, delete.
pub const RW_ACCESS_DELETE_AT1_COUNTED: KeySpec = KeySpec {
    notes: "",
    flags: &["RW", "access", "delete"],
    begin: Begin::At(1),
    find: Find::Counted {
        count: 0,
        first: 1,
        step: 1,
    },
};

/// RW, access, delete.
pub const RW_ACCESS_DELETE_AT1_RM2_1_0: KeySpec = KeySpec {
    notes: "",
    flags: &["RW", "access", "delete"],
    begin: Begin::At(1),
    find: Find::Range {
        last: -2,
        step: 1,
        limit: 0,
    },
};

/// RW, access, delete.
pub const RW_ACCESS_DELETE_AT2_COUNTED: KeySpec = KeySpec {
    notes: "",
    flags: &["RW", "access", "delete"],
    begin: Begin::At(2),
    find: Find::Counted {
        count: 0,
        first: 1,
        step: 1,
    },
};

/// RW, access, delete.
pub const RW_ACCESS_DELETE_AT3: KeySpec = KeySpec {
    notes: "",
    flags: &["RW", "access", "delete"],
    begin: Begin::At(3),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// RW, access, delete, incomplete.
pub const MIGRATE_KEYS: KeySpec = KeySpec {
    notes: "",
    flags: &["RW", "access", "delete", "incomplete"],
    begin: Begin::After(b"KEYS", -2),
    find: Find::Range {
        last: -1,
        step: 1,
        limit: 0,
    },
};

/// RW, access, insert.
pub const RW_ACCESS_INSERT_AT1: KeySpec = KeySpec {
    notes: "",
    flags: &["RW", "access", "insert"],
    begin: Begin::At(1),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// RW, access, update.
pub const RW_ACCESS_UPDATE_AT1: KeySpec = KeySpec {
    notes: "",
    flags: &["RW", "access", "update"],
    begin: Begin::At(1),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// RW and UPDATE because it changes the TTL
pub const RW_ACCESS_UPDATE_AT1_TTL: KeySpec = KeySpec {
    notes: "RW and UPDATE because it changes the TTL",
    flags: &["RW", "access", "update"],
    begin: Begin::At(1),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// RW, access, update.
pub const RW_ACCESS_UPDATE_AT1_R1_1_0: KeySpec = KeySpec {
    notes: "",
    flags: &["RW", "access", "update"],
    begin: Begin::At(1),
    find: Find::Range {
        last: 1,
        step: 1,
        limit: 0,
    },
};

/// RW, access, update.
pub const RW_ACCESS_UPDATE_AT1_RM1_3_0: KeySpec = KeySpec {
    notes: "",
    flags: &["RW", "access", "update"],
    begin: Begin::At(1),
    find: Find::Range {
        last: -1,
        step: 3,
        limit: 0,
    },
};

/// RW, access, update.
pub const RW_ACCESS_UPDATE_AT2: KeySpec = KeySpec {
    notes: "",
    flags: &["RW", "access", "update"],
    begin: Begin::At(2),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// We cannot tell how the keys will be used so we assume the worst, RW and UPDATE
pub const SCRIPT_KEYS_RW: KeySpec = KeySpec {
    notes: "We cannot tell how the keys will be used so we assume the worst, RW and UPDATE",
    flags: &["RW", "access", "update"],
    begin: Begin::At(2),
    find: Find::Counted {
        count: 0,
        first: 1,
        step: 1,
    },
};

/// RW, access, update.
pub const RW_ACCESS_UPDATE_AT2_COUNTED: KeySpec = KeySpec {
    notes: "",
    flags: &["RW", "access", "update"],
    begin: Begin::At(2),
    find: Find::Counted {
        count: 0,
        first: 1,
        step: 1,
    },
};

/// This command allows both access and modification of the key
pub const BITFIELD_KEY: KeySpec = KeySpec {
    notes: "This command allows both access and modification of the key",
    flags: &["RW", "access", "update", "variable_flags"],
    begin: Begin::At(1),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// RW and ACCESS due to the optional `GET` argument
pub const SET_KEY: KeySpec = KeySpec {
    notes: "RW and ACCESS due to the optional `GET` argument",
    flags: &["RW", "access", "update", "variable_flags"],
    begin: Begin::At(1),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// RW, delete.
pub const RW_DELETE_AT1: KeySpec = KeySpec {
    notes: "",
    flags: &["RW", "delete"],
    begin: Begin::At(1),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// RW, delete.
pub const RW_DELETE_AT2: KeySpec = KeySpec {
    notes: "",
    flags: &["RW", "delete"],
    begin: Begin::At(2),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// RW, delete, variable_flags.
pub const DELEX_KEY: KeySpec = KeySpec {
    notes: "",
    flags: &["RW", "delete", "variable_flags"],
    begin: Begin::At(1),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// RW, insert.
pub const RW_INSERT_AT1: KeySpec = KeySpec {
    notes: "",
    flags: &["RW", "insert"],
    begin: Begin::At(1),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// RW, insert.
pub const RW_INSERT_AT2: KeySpec = KeySpec {
    notes: "",
    flags: &["RW", "insert"],
    begin: Begin::At(2),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// RW, update.
pub const RW_UPDATE_AT1: KeySpec = KeySpec {
    notes: "",
    flags: &["RW", "update"],
    begin: Begin::At(1),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// UPDATE instead of INSERT because of the optional trimming feature
pub const RW_UPDATE_AT1_TRIMMING: KeySpec = KeySpec {
    notes: "UPDATE instead of INSERT because of the optional trimming feature",
    flags: &["RW", "update"],
    begin: Begin::At(1),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// RW, update.
pub const RW_UPDATE_AT2: KeySpec = KeySpec {
    notes: "",
    flags: &["RW", "update"],
    begin: Begin::At(2),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

/// RW, update, delete.
pub const RW_UPDATE_DELETE_AT1: KeySpec = KeySpec {
    notes: "",
    flags: &["RW", "update", "delete"],
    begin: Begin::At(1),
    find: Find::Range {
        last: 0,
        step: 1,
        limit: 0,
    },
};

// ---------------------------------------------------------- the subcommands
//
// The container commands whose keys sit inside a subcommand. The table has one
// row a container and none a subcommand, which is D-114, so these are named
// here until it does.

/// A container, one of its subcommands, and where that subcommand's keys are.
type SubSpec = (&'static [u8], &'static [u8], &'static [KeySpec]);

/// The fifteen subcommands that take a key.
static SUBS: &[SubSpec] = &[
    (b"himport", b"set", &[OW_UPDATE_AT2]),
    // `JSON.DEBUG MEMORY key [path]` is the only subcommand of a module
    // container that takes one, and the reference has never heard of it, so this
    // row is ours rather than measured.
    (b"json.debug", b"memory", &[RO_ACCESS_AT2]),
    (b"memory", b"usage", &[RO_AT2]),
    (b"object", b"encoding", &[RO_AT2]),
    (b"object", b"freq", &[RO_AT2]),
    (b"object", b"idletime", &[RO_AT2]),
    (b"object", b"refcount", &[RO_AT2]),
    (b"xgroup", b"create", &[RW_INSERT_AT2]),
    (b"xgroup", b"createconsumer", &[RW_INSERT_AT2]),
    (b"xgroup", b"delconsumer", &[RW_DELETE_AT2]),
    (b"xgroup", b"destroy", &[RW_DELETE_AT2]),
    (b"xgroup", b"setid", &[RW_UPDATE_AT2]),
    (b"xinfo", b"consumers", &[RO_ACCESS_AT2]),
    (b"xinfo", b"groups", &[RO_ACCESS_AT2]),
    (b"xinfo", b"stream", &[RO_ACCESS_AT2]),
];

/// The key specs for `container sub`, empty for a subcommand that has none.
#[must_use]
pub fn of_sub(container: &str, sub: &[u8]) -> &'static [KeySpec] {
    SUBS.iter()
        .find(|(c, s, _)| *c == container.as_bytes() && s.eq_ignore_ascii_case(sub))
        .map_or(&[][..], |(_, _, keys)| keys)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::table;
    use crate::proto::Limits;
    use crate::request::{Argv, Step};

    /// Every key a command names, with the flags each one carries.
    ///
    /// `None` where the arguments do not resolve, which is a command about to
    /// fail. Every expected answer here was read off a real 8.10.1 first.
    fn flagged(words: &[&str]) -> Option<Vec<(String, Vec<&'static str>)>> {
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
        let mut found = Vec::new();
        let whole = find(spec, args, 0, &mut |run| {
            for i in 0..run.count {
                let key = args.get(run.first + i * run.step);
                found.push((
                    String::from_utf8_lossy(key).into_owned(),
                    run.flags.to_vec(),
                ));
            }
        });
        whole.then_some(found)
    }

    /// The names alone, for the cases where the flags are not the point.
    fn named(words: &[&str]) -> Option<Vec<String>> {
        flagged(words).map(|keys| keys.into_iter().map(|(key, _)| key).collect())
    }

    #[test]
    fn a_counted_run_names_every_key_it_counts() {
        assert_eq!(
            named(&["zunionstore", "d", "2", "a", "b"]).unwrap(),
            ["d", "a", "b"]
        );
        assert_eq!(
            named(&["lmpop", "2", "a", "b", "LEFT"]).unwrap(),
            ["a", "b"]
        );
        assert_eq!(named(&["smove", "a", "b", "m"]).unwrap(), ["a", "b"]);
    }

    #[test]
    fn a_run_that_reaches_past_the_last_argument_names_nothing() {
        // Not two keys and not one. The command is about to be refused, and a
        // short answer would send a cluster client to the wrong node.
        assert_eq!(named(&["zunionstore", "d", "3", "a"]), None);
        assert_eq!(named(&["zunionstore", "d", "-1", "a"]), None);
        assert_eq!(named(&["zunionstore", "d", "x", "a"]), None);
        // A count of nought resolves to a run ending before it starts, which is
        // the same refusal, and the caller turns it into an empty reply for the
        // commands that are allowed to name no keys.
        assert_eq!(named(&["eval", "body", "0"]), None);
    }

    #[test]
    fn the_streams_keyword_is_walked_to_rather_than_counted_from() {
        let two = ["xread", "COUNT", "2", "STREAMS", "a", "b", "0", "0"];
        assert_eq!(named(&two).unwrap(), ["a", "b"]);
        // An odd tail is a stream with no id or an id with no stream.
        assert_eq!(named(&["xread", "STREAMS", "a", "b", "0"]), None);
        // A group called STREAMS is stepped over rather than matched.
        let group = ["xreadgroup", "GROUP", "STREAMS", "c", "STREAMS", "a", ">"];
        assert_eq!(named(&group).unwrap(), ["a"]);
    }

    #[test]
    fn pfmerge_names_its_destination_when_it_was_given_no_sources() {
        // The specs cannot say this, because a source run starting at argument
        // two of a two argument command is off the end.
        assert_eq!(named(&["pfmerge", "k0"]).unwrap(), ["k0"]);
        assert_eq!(named(&["pfmerge", "d", "a", "b"]).unwrap(), ["d", "a", "b"]);
    }

    #[test]
    fn an_argument_that_only_looks_like_a_key_is_not_one() {
        let spec = table::lookup(b"spublish").expect("a command this server has");
        let mut argv = Argv::new();
        let buf = b"*3\r\n$8\r\nspublish\r\n$2\r\nch\r\n$1\r\nm\r\n";
        let Ok(Step::Command { .. }) = argv.decode(buf, &Limits::default()) else {
            panic!("the test wrote a command that does not decode");
        };
        assert!(!takes_keys(spec, Args::new(&argv, buf), 0));
    }

    #[test]
    fn a_last_wins_destination_is_the_last_one_written() {
        let two = ["sort", "k", "STORE", "a", "STORE", "b"];
        assert_eq!(named(&two).unwrap(), ["k", "b"]);
        let geo = [
            "georadius",
            "k",
            "0",
            "0",
            "1",
            "m",
            "STORE",
            "d",
            "STOREDIST",
            "e",
        ];
        assert_eq!(named(&geo).unwrap(), ["k", "e"]);
        // A pattern names keys only once the value has been read, so neither BY
        // nor GET is reported and neither is the word STORE inside one.
        assert_eq!(
            named(&["sort", "k", "BY", "STORE", "GET", "d"]).unwrap(),
            ["k"]
        );
    }

    #[test]
    fn a_keyword_the_command_does_not_carry_is_not_a_failure() {
        let single = ["migrate", "h", "1", "k", "0", "0"];
        assert_eq!(named(&single).unwrap(), ["k"]);
        let listed = ["migrate", "h", "1", "", "0", "0", "KEYS", "a", "b"];
        assert_eq!(named(&listed).unwrap(), ["a", "b"]);
        // The keyword is looked for backward from two before the end, so a
        // command that stops on the word itself has nothing behind it and the
        // empty key in the middle is what is left.
        assert_eq!(
            named(&["migrate", "h", "1", "", "0", "0", "KEYS"]).unwrap(),
            [""]
        );
    }

    #[test]
    fn what_a_command_does_to_a_key_can_depend_on_its_options() {
        let flags = |words: &[&str]| flagged(words).unwrap()[0].1.clone();
        assert_eq!(flags(&["set", "k", "v"]), ["OW", "update"]);
        assert_eq!(flags(&["set", "k", "v", "GET"]), ["RW", "access", "update"]);
        assert_eq!(
            flags(&["bitfield", "k", "GET", "u8", "0"]),
            ["RO", "access"]
        );
        assert_eq!(
            flags(&["bitfield", "k", "SET", "u8", "0", "1"]),
            ["RW", "access", "update"]
        );
        assert_eq!(flags(&["delex", "k"]), ["RM", "delete"]);
        assert_eq!(flags(&["delex", "k", "IFDEQ", "d"]), ["RW", "delete"]);
    }

    #[test]
    fn a_container_is_asked_about_the_word_behind_it() {
        assert_eq!(named(&["object", "encoding", "k"]).unwrap(), ["k"]);
        assert_eq!(named(&["xgroup", "create", "s", "g", "$"]).unwrap(), ["s"]);
        assert_eq!(named(&["object", "help"]).unwrap(), Vec::<String>::new());
    }
}
