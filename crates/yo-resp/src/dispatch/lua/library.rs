//! The libraries `FUNCTION LOAD` has taken and the functions inside them.
//!
//! A library is a chunk of Lua with a shebang on the front. It runs once, at
//! load time, and everything it does that outlives the load is a call to
//! `redis.register_function`. What is kept here is the result of that run: the
//! name the library was loaded under, the code it was loaded from, and one
//! entry per function with its description and its flags. The callbacks
//! themselves are not here, because a callback is a Lua value and this server
//! has one interpreter per thread. They live in whichever interpreters have
//! compiled the library, and [`super::fcall`] is what puts one there.
//!
//! # Two dictionaries and three case rules
//!
//! A real server keeps libraries in one dictionary and functions in another,
//! and the two do not agree on case. The library dictionary is case sensitive,
//! so `FUNCTION DELETE MYLIB` does not find `mylib`. The server wide function
//! dictionary is case insensitive, so `FCALL PING 0` runs `ping`. The
//! dictionary inside one library is case sensitive again, so a single library
//! can register both `f` and `F` and both show up in `FUNCTION LIST`. None of
//! that is a design anybody would pick twice, and all three are what a client
//! written against a real server depends on.
//!
//! # Why a vector
//!
//! Because there are never many. A server with a hundred libraries is a server
//! with a hundred libraries, and every lookup here happens once per `FCALL`
//! next to an interpreter call that costs a thousand times more. The order is
//! the order they were loaded in, which is the one part of this a real server
//! does not have: it walks a hash table, so its `FUNCTION LIST` comes back in
//! an order nobody chose. That is D-109.

use mlua::{Lua, Table};
use yo_common::{Code, Error, Result};

/// Puts the one thing the prelude needs from here on the raw table.
///
/// `redis.register_function` has to turn away a function name for the same
/// reason `FUNCTION LOAD` has to turn away a library name, and by the same
/// rule. Rather than write the rule twice, the Lua side asks this.
pub(in crate::dispatch) fn statics(lua: &Lua, raw: &Table) -> mlua::Result<()> {
    raw.raw_set(
        "named",
        lua.create_function(|_, name: mlua::LuaString| Ok(named(&name.as_bytes())))?,
    )
}

/// The five words a function can be registered with, in the order a real
/// server lists them in `FUNCTION LIST`.
pub(in crate::dispatch) const FLAGS: [&str; 5] = [
    "no-writes",
    "allow-oom",
    "allow-stale",
    "no-cluster",
    "allow-cross-slot-keys",
];

/// The bit `no-writes` sets, which is the only one of the five that changes
/// what a call is allowed to do rather than where it is allowed to run.
pub(in crate::dispatch) const NO_WRITES: u32 = 1;

/// The only engine there is, and the name `FUNCTION LIST` reports.
///
/// Upper case whatever the shebang said, because the name in the reply is the
/// engine's own and not the spelling that reached it, so `#!LUA` and `#!lua`
/// both list as `LUA`.
pub(in crate::dispatch) const ENGINE: &str = "LUA";

/// One registered function.
pub(in crate::dispatch) struct Func {
    /// What `FCALL` names it by. Letters, numbers and underscores only.
    pub name: Box<str>,
    /// What the library said about it, which is any string at all or nothing.
    pub desc: Option<Box<[u8]>>,
    /// The five flags as bits, in [`FLAGS`] order.
    pub flags: u32,
}

/// One loaded library.
pub(in crate::dispatch) struct Library {
    /// The name out of the shebang, which is also the name it is deleted by.
    pub name: Box<str>,
    /// The whole code as the client sent it, shebang included, which is what
    /// `FUNCTION LIST WITHCODE` hands back and what `FUNCTION DUMP` writes.
    pub code: Box<[u8]>,
    /// Where the part an engine compiles starts, which is the newline that ends
    /// the shebang. See [`Meta::body`] for why the newline is on the inside.
    pub at: usize,
    /// The digest of the whole code.
    ///
    /// Kept so that a thread can tell the library it compiled earlier from the
    /// one a `FUNCTION LOAD REPLACE` on some other thread has since put here.
    /// Comparing names would not do it, because a replacement keeps the name.
    pub sha: [u8; 40],
    /// Every function the load registered, in the order it registered them.
    pub funcs: Vec<Func>,
}

/// Every library the server has, in the order they were loaded.
#[derive(Default)]
pub(in crate::dispatch) struct Libraries {
    held: Vec<Library>,
}

impl Libraries {
    /// The library under a name, matched exactly.
    pub(in crate::dispatch) fn library(&self, name: &[u8]) -> Option<&Library> {
        self.held.iter().find(|l| l.name.as_bytes() == name)
    }

    /// The function under a name, matched without regard to case, and the
    /// library it came from.
    pub(in crate::dispatch) fn function(&self, name: &[u8]) -> Option<(&Library, &Func)> {
        for lib in &self.held {
            for f in &lib.funcs {
                if f.name.as_bytes().eq_ignore_ascii_case(name) {
                    return Some((lib, f));
                }
            }
        }
        None
    }

    /// Whether some library other than `except` already registered `name`.
    ///
    /// This is the check that makes `FUNCTION LOAD` refuse a library whose
    /// function names collide with one that is already there, and the exception
    /// is what lets `FUNCTION LOAD REPLACE` reload a library over itself.
    pub(in crate::dispatch) fn taken(&self, name: &str, except: &str) -> bool {
        self.held.iter().any(|lib| {
            &*lib.name != except && lib.funcs.iter().any(|f| f.name.eq_ignore_ascii_case(name))
        })
    }

    /// Put a library in, replacing one of the same name if there is one.
    pub(in crate::dispatch) fn insert(&mut self, lib: Library) {
        self.remove(lib.name.as_bytes());
        self.held.push(lib);
    }

    /// Take a library out, and say whether there was one.
    pub(in crate::dispatch) fn remove(&mut self, name: &[u8]) -> bool {
        let before = self.held.len();
        self.held.retain(|l| l.name.as_bytes() != name);
        self.held.len() != before
    }

    /// Forget every one of them, which is what `FUNCTION FLUSH` asks for.
    pub(in crate::dispatch) fn wipe(&mut self) {
        self.held.clear();
    }

    /// How many libraries and how many functions, which is `FUNCTION STATS`.
    pub(in crate::dispatch) fn counts(&self) -> (usize, usize) {
        (
            self.held.len(),
            self.held.iter().map(|l| l.funcs.len()).sum(),
        )
    }

    /// Every library, for `FUNCTION LIST` to walk.
    pub(in crate::dispatch) fn all(&self) -> &[Library] {
        &self.held
    }
}

/// What the shebang line said, and where the code after it starts.
#[derive(Debug)]
pub(in crate::dispatch) struct Meta<'a> {
    /// The engine name, which is the first word with `#!` taken off.
    pub engine: Vec<u8>,
    /// The library name, from the one `name=` part.
    pub name: Vec<u8>,
    /// The code the engine compiles, which starts at the newline and includes
    /// it. That leading newline is why every message out of a library says
    /// line two, and it is not an accident anybody here can drop: the line
    /// numbers a client reads in an error have to be the line numbers of the
    /// file it wrote, shebang and all.
    pub body: &'a [u8],
}

/// Read the shebang off a library.
///
/// The order the failures come in is the order a real server checks in, and it
/// matters: `#!\n` is not `Invalid library metadata` but `Library name was not
/// given`, because an empty engine name is only a problem once there is a name
/// to look an engine up for.
pub(in crate::dispatch) fn metadata(code: &[u8]) -> Result<Meta<'_>> {
    if !code.starts_with(b"#!") {
        return Err(Error::new(Code::Invalid, "Missing library metadata"));
    }
    let Some(nl) = code.iter().position(|&b| b == b'\n') else {
        return Err(Error::new(Code::Invalid, "Invalid library metadata"));
    };
    let Some(parts) = split(&code[..nl]) else {
        return Err(Error::new(Code::Invalid, "Invalid library metadata"));
    };
    let Some(first) = parts.first() else {
        return Err(Error::new(Code::Invalid, "Invalid library metadata"));
    };
    let engine = first[2.min(first.len())..].to_vec();
    let mut name: Option<Vec<u8>> = None;
    for part in &parts[1..] {
        if part.len() >= 5 && part[..5].eq_ignore_ascii_case(b"name=") {
            if name.is_some() {
                return Err(Error::new(
                    Code::Invalid,
                    "Invalid metadata value, name argument was given multiple times",
                ));
            }
            name = Some(part[5..].to_vec());
            continue;
        }
        return Err(Error::fmt(
            Code::Invalid,
            format_args!(
                "Invalid metadata value given: {}",
                String::from_utf8_lossy(part)
            ),
        ));
    }
    let Some(name) = name else {
        return Err(Error::new(Code::Invalid, "Library name was not given"));
    };
    Ok(Meta {
        engine,
        name,
        body: &code[nl..],
    })
}

/// Whether a name is one a library or a function may be loaded under.
pub(in crate::dispatch) fn named(name: &[u8]) -> bool {
    !name.is_empty() && name.iter().all(|&b| b.is_ascii_alphanumeric() || b == b'_')
}

/// What a name that is not one answers, for a library and for a function
/// alike.
///
/// The sentence says `Library names` either way, which is a real server's own
/// slip and not one made here: it registers a function through the same check
/// and hands back the same string.
pub(in crate::dispatch) fn bad_name() -> Error {
    Error::new(
        Code::Invalid,
        "Library names can only contain letters, numbers, or underscores(_) \
         and must be at least one character long",
    )
}

/// The shebang line cut into words, the way `sdssplitargs` cuts one.
///
/// The same rules the inline protocol uses, which is why `#!lua name="q"`
/// loads a library called `q`. Written out again rather than shared with the
/// parser in [`crate::request`], because that one writes its words into a
/// buffer it owns and reuses, and this one is called once per `FUNCTION LOAD`
/// and can afford a vector.
///
/// `None` means a quote was left open, which a real server reports as invalid
/// metadata rather than as a quoting mistake.
fn split(line: &[u8]) -> Option<Vec<Vec<u8>>> {
    let mut out: Vec<Vec<u8>> = Vec::new();
    let mut i = 0;
    loop {
        while i < line.len() && line[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= line.len() {
            return Some(out);
        }
        let mut word = Vec::new();
        let mut in_double = false;
        let mut in_single = false;
        let mut done = false;
        while !done {
            let c = line.get(i).copied();
            if in_double {
                match c {
                    Some(b'\\')
                        if i + 3 < line.len()
                            && line[i + 1] == b'x'
                            && hex(line[i + 2]).is_some()
                            && hex(line[i + 3]).is_some() =>
                    {
                        let hi = hex(line[i + 2])?;
                        let lo = hex(line[i + 3])?;
                        word.push(hi * 16 + lo);
                        i += 3;
                    }
                    Some(b'\\') if i + 1 < line.len() => {
                        i += 1;
                        word.push(match line[i] {
                            b'n' => b'\n',
                            b'r' => b'\r',
                            b't' => b'\t',
                            b'b' => 0x08,
                            b'a' => 0x07,
                            other => other,
                        });
                    }
                    Some(b'"') => {
                        if line.get(i + 1).is_some_and(|b| !b.is_ascii_whitespace()) {
                            return None;
                        }
                        done = true;
                    }
                    None => return None,
                    Some(ch) => word.push(ch),
                }
            } else if in_single {
                match c {
                    Some(b'\\') if line.get(i + 1) == Some(&b'\'') => {
                        i += 1;
                        word.push(b'\'');
                    }
                    Some(b'\'') => {
                        if line.get(i + 1).is_some_and(|b| !b.is_ascii_whitespace()) {
                            return None;
                        }
                        done = true;
                    }
                    None => return None,
                    Some(ch) => word.push(ch),
                }
            } else {
                match c {
                    None => done = true,
                    Some(ch) if ch.is_ascii_whitespace() => done = true,
                    Some(b'"') => in_double = true,
                    Some(b'\'') => in_single = true,
                    Some(ch) => word.push(ch),
                }
            }
            if i < line.len() {
                i += 1;
            }
        }
        out.push(word);
    }
}

/// One hex digit as a number.
fn hex(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shebang of a library that loads, and the pieces it comes apart into.
    #[test]
    fn the_shebang_says_the_engine_and_the_name() {
        let md = metadata(b"#!lua name=x\nreturn 1").expect("read");
        assert_eq!(md.engine, b"lua");
        assert_eq!(md.name, b"x");
        // The body starts at the newline and keeps it, which is what makes the
        // first line of the code line two.
        assert_eq!(md.body, b"\nreturn 1");
        // A quoted name is a name, because the line goes through the same
        // splitter an inline command does.
        assert_eq!(metadata(b"#!lua name=\"q\"\nx").expect("read").name, b"q");
        // The engine name is whatever followed the two characters, case and
        // all, and the lookup that uses it is the case insensitive one.
        assert_eq!(metadata(b"#!LUA name=x\nx").expect("read").engine, b"LUA");
    }

    #[test]
    fn a_shebang_that_is_not_one_says_which_way_it_is_wrong() {
        let why = |code: &[u8]| metadata(code).expect_err("refused").to_string();
        assert!(why(b"return 1").contains("Missing library metadata"));
        assert!(why(b"#!lua name=x").contains("Invalid library metadata"));
        assert!(why(b"#!lua name=\"q\nx").contains("Invalid library metadata"));
        // No name at all, whether or not there is an engine, is the same
        // sentence, and it comes before anything looks at the engine.
        assert!(why(b"#!lua\nx").contains("Library name was not given"));
        assert!(why(b"#!\n").contains("Library name was not given"));
        assert!(why(b"#!lua name=a name=b\nx").contains("name argument was given multiple times"));
        assert!(why(b"#!lua nome=a\nx").contains("Invalid metadata value given: nome=a"));
    }

    #[test]
    fn a_name_is_letters_numbers_and_underscores() {
        assert!(named(b"a"));
        assert!(named(b"A_1"));
        assert!(named(b"12"));
        assert!(!named(b""));
        assert!(!named(b"a-b"));
        assert!(!named(b"a b"));
        assert!(!named("é".as_bytes()));
    }

    /// The three case rules, which are three different rules on purpose.
    #[test]
    fn a_library_is_found_by_case_and_a_function_is_not() {
        let mut held = Libraries::default();
        held.insert(Library {
            name: "mylib".into(),
            code: b"#!lua name=mylib\n".to_vec().into_boxed_slice(),
            at: 16,
            sha: [b'0'; 40],
            funcs: vec![
                Func {
                    name: "ping".into(),
                    desc: Some(b"says pong".to_vec().into_boxed_slice()),
                    flags: NO_WRITES,
                },
                Func {
                    name: "PING".into(),
                    desc: None,
                    flags: 0,
                },
            ],
        });
        assert!(held.library(b"mylib").is_some());
        assert!(held.library(b"MYLIB").is_none());
        // The server wide lookup folds case, and finds the first one that
        // matches when a library registered two names that only differ by it.
        let (lib, f) = held.function(b"PiNg").expect("found");
        assert_eq!(&*lib.name, "mylib");
        assert_eq!(&*f.name, "ping");
        assert_eq!(held.counts(), (1, 2));
        assert!(held.taken("PING", "other"));
        assert!(!held.taken("PING", "mylib"));

        assert!(!held.remove(b"MYLIB"));
        assert!(held.remove(b"mylib"));
        assert_eq!(held.counts(), (0, 0));
    }
}
