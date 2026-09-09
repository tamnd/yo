//! The command table: what each command is called, how many arguments it
//! takes, where its keys are, and what `COMMAND` reports about it.
//!
//! Every field here was read out of a running Redis 8.8 with `COMMAND INFO`
//! rather than written from the documentation, because this is the table a
//! client library builds its own routing from. A cluster aware client asks
//! `COMMAND` where the keys are and then decides which node to send a command
//! to, so an arity or a key position that is off by one does not produce a
//! wrong error message, it produces a client that sends `MSET` to the wrong
//! shard. The summaries are ours, since those are the one field nobody parses.
//!
//! `cargo xtask check` compares this table against `commands.toml` in both
//! directions, so a command cannot be dispatched without a storage plan and
//! cannot claim `wire = "verified"` without an entry here.

use super::Args;
// Every name here is a key spec shape, spelled in capitals and used once a row,
// so a glob is what keeps four hundred and twenty nine rows from carrying an
// import list longer than they are.
use super::keyspec::*;
use yo_common::parse_i64;

/// Everything `COMMAND` has to be able to say about one command.
#[derive(Debug, Clone, Copy)]
pub struct Spec {
    /// The name, lower case, which is how `COMMAND` reports it whatever case
    /// the client used.
    pub name: &'static str,
    /// Redis's arity: a positive number is exact, a negative one is a minimum
    /// of its magnitude, and both count the command name itself.
    pub arity: i32,
    /// The command flags, in the order `COMMAND INFO` lists them.
    pub flags: &'static [&'static str],
    /// The first argument that is a key, or zero when there are none.
    pub first_key: i32,
    /// The last argument that is a key, negative counting back from the end.
    pub last_key: i32,
    /// How far apart the keys are, for the commands that take pairs.
    pub step: i32,
    /// The ACL categories, which are what `COMMAND LIST FILTERBY ACLCAT` reads.
    ///
    /// A container row carries only what the container itself is in, which is
    /// what a real server reports for it. The categories its subcommands add on
    /// top are in [`SUBCATS`].
    pub acl: &'static [&'static str],
    /// The Redis this command first appeared in.
    pub since: &'static str,
    /// The cost, in the shape `COMMAND DOCS` uses.
    pub complexity: &'static str,
    /// One line about what it does, in our words.
    pub summary: &'static str,
    /// The group in `commands.toml`, which is how the two files are compared.
    pub group: &'static str,
    /// Where this command's keys are and what it does to each of them.
    ///
    /// The triple above is Redis's legacy answer and is kept because
    /// `COMMAND INFO` still reports it. This is the one that is right, and it is
    /// the one the ACL reads. See the `keyspec` module for what the difference
    /// costs.
    pub keys: &'static [KeySpec],
}

/// The four transaction commands Redis counts as fast, which is all of them
/// except `EXEC`, whose cost is whatever it was asked to run.
const AC_TX_FAST: &[&str] = &["@fast", "@transaction"];
/// The six subscribe and unsubscribe commands, none of which Redis counts as
/// fast because all of them take a list.
const AC_PUBSUB_SLOW: &[&str] = &["@pubsub", "@slow"];
/// The two publishes, which Redis does count as fast.
const AC_PUBSUB_FAST: &[&str] = &["@pubsub", "@fast"];
/// Read only, fast, one key at argument one, which is most of the getters.
const READ_FAST: &[&str] = &["readonly", "fast"];
/// A write that allocates, fast, one key at argument one.
const WRITE_FAST_OOM: &[&str] = &["write", "denyoom", "fast"];
/// A write that allocates and is not counted as fast.
const WRITE_OOM: &[&str] = &["write", "denyoom"];
/// A write that allocates and is not for ordinary clients, which is `PFDEBUG`.
const WRITE_OOM_ADMIN: &[&str] = &["write", "denyoom", "admin"];
/// The read side categories.
const AC_READ_FAST: &[&str] = &["@read", "@string", "@fast"];
/// The bitmap read side, for the two that answer without walking the value.
const AC_BIT_READ_FAST: &[&str] = &["@read", "@bitmap", "@fast"];
/// The bitmap read side for the ones that walk it.
const AC_BIT_READ: &[&str] = &["@read", "@bitmap", "@slow"];
/// The bitmap write side. Redis counts none of these as fast, `SETBIT` included.
const AC_BIT_WRITE: &[&str] = &["@write", "@bitmap", "@slow"];

/// The sketch write side, which Redis counts as fast for `PFADD` alone.
const AC_HLL_WRITE_FAST: &[&str] = &["@write", "@hyperloglog", "@fast"];
/// The sketch write side for the ones that walk every register.
const AC_HLL_WRITE: &[&str] = &["@write", "@hyperloglog", "@slow"];
/// The sketch read side, which is `PFCOUNT` and only `PFCOUNT`.
const AC_HLL_READ: &[&str] = &["@read", "@hyperloglog", "@slow"];
/// The two that are not for clients, and are tagged so an ACL can say so.
const AC_HLL_ADMIN: &[&str] = &["@hyperloglog", "@admin", "@slow", "@dangerous"];

/// What the three replication commands are in, which is the same three every
/// other administrative command is in.
const AC_ADMIN_SLOW: &[&str] = &["@admin", "@slow", "@dangerous"];
/// The read side categories for the ones that walk the value.
const AC_READ_SLOW: &[&str] = &["@read", "@string", "@slow"];
/// The write side categories.
const AC_WRITE_FAST: &[&str] = &["@write", "@string", "@fast"];
/// The write side categories for the ones that are not counted as fast.
const AC_WRITE_SLOW: &[&str] = &["@write", "@string", "@slow"];
/// A write that frees rather than allocates, so Redis does not mark it denyoom.
const WRITE_FAST: &[&str] = &["write", "fast"];
/// The set read side, for the ones that answer without walking the members.
const AC_SET_READ_FAST: &[&str] = &["@read", "@set", "@fast"];
/// The set read side for the ones that walk the members.
const AC_SET_READ_SLOW: &[&str] = &["@read", "@set", "@slow"];
/// The set write side.
const AC_SET_WRITE_FAST: &[&str] = &["@write", "@set", "@fast"];
/// The set write side for the ones that walk whole sets to decide what to
/// write, which is the whole `*STORE` family.
const AC_SET_WRITE_SLOW: &[&str] = &["@write", "@set", "@slow"];
/// The hash read side, for the ones that answer without walking the fields.
const AC_HASH_READ_FAST: &[&str] = &["@read", "@hash", "@fast"];
/// The hash read side for the ones that walk the fields.
const AC_HASH_READ_SLOW: &[&str] = &["@read", "@hash", "@slow"];
/// The hash write side.
const AC_HASH_WRITE_FAST: &[&str] = &["@write", "@hash", "@fast"];
/// `HIMPORT`, which is a container and so has no write category of its own. The
/// write flags and the key live on its `SET` subcommand, which the table does
/// not carry any more than it carries `OBJECT ENCODING`.
const AC_HASH_SLOW: &[&str] = &["@hash", "@slow"];
/// Read only and not counted as fast, which is every list read that walks.
const READ_SLOW: &[&str] = &["readonly"];
/// A write that is not counted as fast and does not allocate, which on the list
/// side is `LREM` and `LTRIM` and nothing else.
const WRITE_SLOW: &[&str] = &["write"];
/// The list read side, for the two that answer without walking the elements.
const AC_LIST_READ_FAST: &[&str] = &["@read", "@list", "@fast"];
/// The list read side for the ones that walk.
const AC_LIST_READ_SLOW: &[&str] = &["@read", "@list", "@slow"];
/// The list write side, which is the pushes and the pops. Redis counts a push
/// as fast even though it can split a chunk, because the split is amortised.
const AC_LIST_WRITE_FAST: &[&str] = &["@write", "@list", "@fast"];
/// The list write side for the ones whose cost is the length of the list.
const AC_LIST_WRITE_SLOW: &[&str] = &["@write", "@list", "@slow"];
/// The five that can wait, which carry a category of their own so that an ACL
/// can say "this user may not park a connection" without naming five commands.
const AC_LIST_WRITE_BLOCKING: &[&str] = &["@write", "@list", "@slow", "@blocking"];
/// The sorted set read side, for the ones that answer without walking members.
const AC_ZSET_READ_FAST: &[&str] = &["@read", "@sortedset", "@fast"];
/// The sorted set read side for the ones that walk members.
const AC_ZSET_READ_SLOW: &[&str] = &["@read", "@sortedset", "@slow"];
/// The sorted set write side.
const AC_ZSET_WRITE_FAST: &[&str] = &["@write", "@sortedset", "@fast"];
/// The sorted set write side for the ones whose cost is the size of the window
/// they touch, which is the removals and `ZRANGESTORE`.
const AC_ZSET_WRITE_SLOW: &[&str] = &["@write", "@sortedset", "@slow"];
/// The two sorted set pops that can wait, which Redis counts as fast because
/// each of them takes one member.
const AC_ZSET_BLOCKING_FAST: &[&str] = &["@write", "@sortedset", "@fast", "@blocking"];
/// And `BZMPOP`, whose cost is the number of keys named and the count popped.
const AC_ZSET_BLOCKING_SLOW: &[&str] = &["@write", "@sortedset", "@slow", "@blocking"];
/// The array read side, for the ones whose cost is the number of indices named
/// and not the size of the array.
/// The geo read side. Redis counts none of these as fast, not even GEODIST,
/// which is two probes and some arithmetic.
const AC_GEO_READ: &[&str] = &["@read", "@geo", "@slow"];
/// The geo write side, which is GEOADD and the four forms that can store.
const AC_GEO_WRITE: &[&str] = &["@write", "@geo", "@slow"];
/// No ACL categories at all, which is what the vector set module registers.
///
/// It is the only group here with an empty list and it is not an omission. The
/// module never calls the categories in, so a real server has no `vectorset`
/// category to name, `ACL CAT vectorset` is an unknown category there, and
/// `COMMAND LIST FILTERBY ACLCAT read` does not answer `VSIM`. Inventing a
/// category would make a rule written against this server mean something it
/// does not mean against a real one, which is the one thing a compatible ACL
/// must not do, and it would do it in the direction that grants access rather
/// than the direction that refuses it.
const AC_NONE: &[&str] = &[];
/// A vector set read, with the `module` flag every vector set command carries
/// for the same reason the JSON and Bloom ones do.
const VECTOR_READ: &[&str] = &["readonly", "module"];
/// A vector set read the module also marks fast, which is the ones that answer
/// about one element without searching.
const VECTOR_READ_FAST: &[&str] = &["readonly", "module", "fast"];
/// `VADD`, which is the one command here that can grow the set.
const VECTOR_WRITE_OOM: &[&str] = &["write", "denyoom", "module"];
/// `VREM`, which only ever frees, so the module does not mark it denyoom.
const VECTOR_WRITE: &[&str] = &["write", "module"];
/// `VSETATTR`, which the module marks fast and, though it takes a string off
/// the wire, does not mark denyoom.
const VECTOR_WRITE_FAST: &[&str] = &["write", "module", "fast"];
/// The one category nearly every search command is in. The module registers
/// `@search` on its own for most of them rather than pairing it with `@read` or
/// `@write` the way RedisJSON does, which is the module's own answer to
/// `COMMAND INFO` and is copied rather than tidied up.
const AC_SEARCH: &[&str] = &["@search"];
/// `FT.DROPINDEX` and the two spellings of it, which the module puts in the
/// dangerous category because dropping an index with `DD` deletes the documents
/// it followed.
const AC_SEARCH_DROP: &[&str] = &["@write", "@slow", "@dangerous", "@search"];
/// `FT._DROPIFX`, which is the same command with a shorter list. The module
/// leaves the slow and dangerous categories off this one and there is no
/// reading of it that makes it less dangerous than the others.
const AC_SEARCH_WRITE: &[&str] = &["@write", "@search"];
/// `FT._LIST`, which the module puts in `@admin` because listing every index is
/// a question about the server rather than about anything in it.
const AC_SEARCH_LIST: &[&str] = &["@admin", "@slow", "@search"];
/// `FT.CONFIG`, which is a question about the server rather than about an index
/// and is the only search command the module leaves the `@slow` category off
/// while still calling it admin.
const AC_SEARCH_ADMIN: &[&str] = &["@admin", "@search"];
/// `_FT.DEBUG`, which the module calls dangerous as well as admin because what
/// it answers is the module's own bookkeeping and nothing about it is promised
/// to stay the same shape between versions.
const AC_SEARCH_DEBUG: &[&str] = &["@admin", "@slow", "@dangerous", "@search"];
/// `FT.TAGVALS`, the one search read the module bothers to put in `@read` as
/// well, and the only one it calls dangerous without also calling it a write.
/// Both are fair: the whole of a tag index goes into one reply and there is no
/// way to ask for less of it.
const AC_SEARCH_TAGS: &[&str] = &["@read", "@slow", "@dangerous", "@search"];
/// The two suggestion reads. These are the search commands that work on a real
/// key, so the module pairs `@search` with the category the key access deserves
/// rather than leaving it on its own.
const AC_SEARCH_READ: &[&str] = &["@read", "@search"];
/// A search read, with the `module` flag every search command carries for the
/// same reason the JSON and vector set ones do.
const SEARCH_READ: &[&str] = &["readonly", "module"];
/// A search write that takes a schema or an alias off the wire.
const SEARCH_WRITE_OOM: &[&str] = &["write", "denyoom", "module"];
/// A search write that only ever frees, which is dropping an index or an alias.
const SEARCH_WRITE: &[&str] = &["write", "module"];
/// The JSON read side. Two categories and no speed one, which is RedisJSON's
/// own answer to `COMMAND INFO` and not an omission: the module registers
/// `@read @json` and leaves it there.
const AC_JSON_READ: &[&str] = &["@read", "@json"];
/// The JSON write side, the same way.
const AC_JSON_WRITE: &[&str] = &["@write", "@json"];
/// A JSON read, with the `module` flag every RedisJSON command carries. It is
/// there because the command came from a module on a real server, and a client
/// that reads the flags off `COMMAND INFO` should see the same list from both.
const JSON_READ: &[&str] = &["readonly", "module"];
/// A JSON write that does not grow the document.
const JSON_WRITE: &[&str] = &["write", "module"];
/// A JSON write that does, which is the four that take a value off the wire.
const JSON_WRITE_OOM: &[&str] = &["write", "denyoom", "module"];
/// A JSON read whose key is not where the arity says it is, which is
/// `JSON.DEBUG` and its subcommand.
const JSON_READ_MOVABLE: &[&str] = &["readonly", "module", "movablekeys"];
/// The Bloom filter read side. RedisBloom marks all of these `@fast` on top of
/// the two categories, including the ones that walk the whole filter, which is
/// the module's own answer to `COMMAND INFO` and is copied rather than judged.
const AC_BLOOM_READ: &[&str] = &["@read", "@bloom"];
/// The two reads the module also puts in `@fast` as a category of its own,
/// which is `BF.INFO` and `BF.CARD`. Neither reads the bits at all.
const AC_BLOOM_READ_FAST: &[&str] = &["@read", "@fast", "@bloom"];
/// The Bloom filter write side.
const AC_BLOOM_WRITE: &[&str] = &["@write", "@bloom"];
/// `BF.RESERVE`, which is the one write that does no hashing.
const AC_BLOOM_WRITE_FAST: &[&str] = &["@write", "@fast", "@bloom"];
/// A Bloom read, with the `module` flag every RedisBloom command carries for
/// the same reason the JSON ones do.
const BLOOM_READ: &[&str] = &["readonly", "module", "fast"];
/// A Bloom write. All of them can grow the filter, `BF.LOADCHUNK` included, so
/// all of them deny out of memory.
const BLOOM_WRITE: &[&str] = &["write", "denyoom", "module"];
/// The cuckoo filter read side, which is the same three flags under a category
/// of its own. `CF.COMPACT` is in here too, because the module has it down as a
/// read even though it moves fingerprints between filters.
const AC_CUCKOO_READ: &[&str] = &["@read", "@cuckoo"];
/// The one read the module also calls fast, which is `CF.INFO`.
const AC_CUCKOO_READ_FAST: &[&str] = &["@read", "@fast", "@cuckoo"];
/// The cuckoo filter write side.
const AC_CUCKOO_WRITE: &[&str] = &["@write", "@cuckoo"];
/// `CF.RESERVE`, which is the one write that does no hashing.
const AC_CUCKOO_WRITE_FAST: &[&str] = &["@write", "@fast", "@cuckoo"];
/// A cuckoo read, with the `module` flag the whole family carries.
const CUCKOO_READ: &[&str] = &["readonly", "module", "fast"];
/// A cuckoo write, all of which can grow the chain.
const CUCKOO_WRITE: &[&str] = &["write", "denyoom", "module"];
/// `CF.DEL`, the one write that only ever frees a slot and so does not deny out
/// of memory.
const CUCKOO_DELETE: &[&str] = &["write", "module", "fast"];
/// The count min sketch read side. The module does not call either of these
/// fast in the flags even though it puts `CMS.INFO` in the fast category, which
/// is a disagreement in RedisBloom's own table and is copied as it stands.
const AC_CMS_READ: &[&str] = &["@read", "@cms"];
/// `CMS.INFO`, which is the one read in the fast category.
const AC_CMS_READ_FAST: &[&str] = &["@read", "@fast", "@cms"];
/// The count min sketch write side.
const AC_CMS_WRITE: &[&str] = &["@write", "@cms"];
/// The two constructors, which allocate and then do nothing.
const AC_CMS_WRITE_FAST: &[&str] = &["@write", "@fast", "@cms"];
/// A count min sketch read, which carries `module` and not `fast`.
const CMS_READ: &[&str] = &["readonly", "module"];
/// A count min sketch write. The table never grows after it is made, so the two
/// that can allocate are the constructors, and all four deny out of memory
/// because the module marks all four.
const CMS_WRITE: &[&str] = &["write", "denyoom", "module"];
/// The top k read side, which the module does not call fast except for the one
/// that reads four numbers off the header.
const AC_TOPK_READ: &[&str] = &["@read", "@topk"];
/// `TOPK.INFO`.
const AC_TOPK_READ_FAST: &[&str] = &["@read", "@fast", "@topk"];
/// The top k write side, which is the two that count things.
const AC_TOPK_WRITE: &[&str] = &["@write", "@topk"];
/// `TOPK.RESERVE`, the one write that only allocates.
const AC_TOPK_WRITE_FAST: &[&str] = &["@write", "@fast", "@topk"];
/// A top k read, which carries `module` and not `fast`.
const TOPK_READ: &[&str] = &["readonly", "module"];
/// A top k write. The table never grows after it is made, so the only one that
/// can allocate is the constructor, and all three deny out of memory because the
/// module marks all three.
const TOPK_WRITE: &[&str] = &["write", "denyoom", "module"];
/// The t digest read side for the ones that answer off the header or off one
/// sweep of the centroids, which the module calls fast and which is all of them
/// bar the trimmed mean.
const AC_TDIGEST_READ_FAST: &[&str] = &["@read", "@fast", "@tdigest"];
/// `TDIGEST.TRIMMED_MEAN`, the one read the module does not call fast.
const AC_TDIGEST_READ: &[&str] = &["@read", "@tdigest"];
/// The t digest write side for the two that only shape a digest.
const AC_TDIGEST_WRITE_FAST: &[&str] = &["@write", "@fast", "@tdigest"];
/// The two that move weight around.
const AC_TDIGEST_WRITE: &[&str] = &["@write", "@tdigest"];
/// A t digest read, which carries `module` and not `fast`.
const TDIGEST_READ: &[&str] = &["readonly", "module"];
/// A t digest write. All four deny out of memory because all four can end up
/// asking for a set of centroids.
const TDIGEST_WRITE: &[&str] = &["write", "denyoom", "module"];
/// `TDIGEST.MERGE`, whose keys are behind a count and so cannot be found by the
/// first, last and step the rest of the table uses.
const TDIGEST_MERGE: &[&str] = &["write", "denyoom", "module", "movablekeys"];
/// The time series read side for the two that answer off the header or off the
/// last sample, which the module calls fast.
const AC_TS_READ_FAST: &[&str] = &["@read", "@fast", "@timeseries"];
/// The time series write side, which is everything that puts a sample in or
/// changes what a series does with one.
const AC_TS_WRITE: &[&str] = &["@write", "@timeseries"];
/// The time series read side for the two that walk a span, which the module
/// does not call fast.
const AC_TS_READ: &[&str] = &["@read", "@timeseries"];
/// `TS.CREATE`, the one write the module calls fast, because making an empty
/// series is an allocation and nothing else.
const AC_TS_WRITE_FAST: &[&str] = &["@write", "@fast", "@timeseries"];
/// A time series read, which carries `module` and not `fast` whatever the ACL
/// category says. That disagreement is RedisTimeSeries's own and is copied as it
/// stands, the same way the count min sketch one is.
const TS_READ: &[&str] = &["readonly", "module"];
/// The two joined reads, which carry their keys behind a count and so cannot be
/// described by the first, last and step triple.
const TS_READ_MOVABLE: &[&str] = &["readonly", "module", "movablekeys"];
/// A time series write, all of which can ask for another chunk.
const TS_WRITE: &[&str] = &["write", "denyoom", "module"];
/// `TS.DEL`, the one write that only ever frees samples and so does not deny out
/// of memory.
const TS_DELETE: &[&str] = &["write", "module"];
/// `TS.CREATERULE`, which is the one time series write that says `fast` in the
/// flags rather than only in the ACL categories, and the one that never asks for
/// room of its own.
const TS_RULE: &[&str] = &["write", "module", "fast"];
/// The graph read side, for the ones that answer without walking the plane.
const AC_GRAPH_READ_FAST: &[&str] = &["@read", "@graph", "@fast"];
/// The graph read side for the ones that walk it.
const AC_GRAPH_READ_SLOW: &[&str] = &["@read", "@graph", "@slow"];
/// The graph write side, all of which are a probe and a run.
const AC_GRAPH_WRITE_FAST: &[&str] = &["@write", "@graph", "@fast"];
const AC_ARRAY_READ_FAST: &[&str] = &["@read", "@array", "@fast"];
/// The array read side for `ARGETRANGE`, which answers once per position in the
/// range and so costs the range rather than the population.
const AC_ARRAY_READ_SLOW: &[&str] = &["@read", "@array", "@slow"];
/// The array write side.
const AC_ARRAY_WRITE_FAST: &[&str] = &["@write", "@array", "@fast"];
/// The array write side for `ARDELRANGE`, the one array command Redis does not
/// mark fast.
const AC_ARRAY_WRITE_SLOW: &[&str] = &["@write", "@array", "@slow"];
/// The stream read side, for the ones that answer without walking entries.
const AC_STREAM_READ_FAST: &[&str] = &["@read", "@stream", "@fast"];
/// The stream read side for the ranges, whose cost is what they return.
const AC_STREAM_READ_SLOW: &[&str] = &["@read", "@stream", "@slow"];
/// The stream write side, which is everything that appends, deletes or moves an
/// entry between pending lists.
const AC_STREAM_WRITE_FAST: &[&str] = &["@write", "@stream", "@fast"];
/// The stream write side for `XTRIM`, whose cost is what it removes.
const AC_STREAM_WRITE_SLOW: &[&str] = &["@write", "@stream", "@slow"];
/// `XREAD`, which waits and does not write.
const AC_STREAM_BLOCKING_READ: &[&str] = &["@read", "@stream", "@slow", "@blocking"];
/// `XREADGROUP`, which waits and does write, since handing an entry to a
/// consumer puts it on that consumer's pending list.
const AC_STREAM_BLOCKING_WRITE: &[&str] = &["@write", "@stream", "@slow", "@blocking"];
/// `XGROUP` and `XINFO`, whose keys are on the subcommand and whose categories
/// are therefore only the container's.
const AC_STREAM_CONTAINER: &[&str] = &["@slow"];
/// The two stream reads, whose keys come after `STREAMS` and are half of what
/// follows it, so nothing positional can find them.
const READ_BLOCKING_MOVABLE: &[&str] = &["readonly", "blocking", "movablekeys"];
/// The same for `XREADGROUP`, which is a write.
const WRITE_BLOCKING_MOVABLE: &[&str] = &["write", "blocking", "movablekeys"];
/// Read only and not counted as fast, for a command whose keys are counted
/// rather than positioned, so a client has to read the key specs to route it.
const READ_MOVABLE: &[&str] = &["readonly", "movablekeys"];
/// The same for a write, which is the three store forms.
const WRITE_MOVABLE: &[&str] = &["write", "denyoom", "movablekeys"];
/// `MIGRATE`, which is the one movable key write that is not `denyoom`.
///
/// It only ever frees here, since the local key goes away and nothing arrives,
/// so a server with no room left can still migrate its way out of trouble. That
/// is the same reasoning that leaves the flag off `DEL`.
const MIGRATE_FLAGS: &[&str] = &["write", "movablekeys"];
/// The connection commands' categories.
const AC_CONN: &[&str] = &["@fast", "@connection"];
/// The keyspace read side, which is `EXISTS` and `TYPE`.
const AC_KEY_READ: &[&str] = &["@keyspace", "@read", "@fast"];
/// The keyspace reads that walk something, which is `SCAN` and `RANDOMKEY`.
const AC_KEY_READ_SLOW: &[&str] = &["@keyspace", "@read", "@slow"];
/// And `KEYS`, which is the same walk without a bound on it and is the one read
/// in this group Redis calls dangerous.
const AC_KEY_READ_ALL: &[&str] = &["@keyspace", "@read", "@slow", "@dangerous"];
/// The keyspace writes that are allowed to cost what the value costs. `DEL`
/// frees on the spot and `COPY` clones a body, and `RENAME` is in here with
/// them even though it moves thirteen bytes, because Redis says slow for it and
/// this list is Redis's list rather than ours.
const AC_KEY_WRITE_SLOW: &[&str] = &["@keyspace", "@write", "@slow"];
/// `UNLINK`, which Redis does count as fast because it does not, and the
/// expiry writers, which move a deadline and never touch a value.
const AC_KEY_WRITE_FAST: &[&str] = &["@keyspace", "@write", "@fast"];
/// The two that empty a database, which are in the dangerous category.
const AC_KEY_FLUSH: &[&str] = &["@keyspace", "@write", "@slow", "@dangerous"];
/// `SWAPDB`, which is fast and dangerous at the same time. It is two pointer
/// writes and it changes what every connected client is looking at, so Redis
/// puts it in `@fast` and in `@dangerous` and both are right.
const AC_SWAPDB: &[&str] = &["@keyspace", "@write", "@fast", "@dangerous"];
/// `RESTORE`, which is dangerous for a reason worth saying out loud: it is the
/// one command that takes bytes from a client and turns them into a value
/// without any command ever having built it. `DUMP` is only `@read`, because
/// reading a value out is no more than reading it.
const AC_RESTORE: &[&str] = &["@keyspace", "@write", "@slow", "@dangerous"];
/// `WAIT` and `WAITAOF`, which are the two commands that block on something
/// that is not a key. They are not in `@keyspace` at all, because they name no
/// key and read nothing, and they carry `@blocking` for the same reason the
/// five list commands do.
const AC_WAIT: &[&str] = &["@slow", "@blocking", "@connection"];
/// `SORT`, which names three type categories because it takes any of the three
/// and a write because of `STORE`. Redis leaves `@keyspace` off both of these
/// even though the command lives in that group, and this list is Redis's.
const AC_SORT_WRITE: &[&str] = &[
    "@write",
    "@set",
    "@sortedset",
    "@list",
    "@slow",
    "@dangerous",
];
/// `SORT_RO`, which is the same list with the write turned into a read.
const AC_SORT_READ: &[&str] = &[
    "@read",
    "@set",
    "@sortedset",
    "@list",
    "@slow",
    "@dangerous",
];

/// The categories a container's subcommands hold on top of the container's own.
///
/// A real server has a row a subcommand, so `config|get` is `@admin @dangerous
/// @slow` while `config` itself is only `@slow`, and `-@admin` on a user takes
/// CONFIG GET away without touching CONFIG HELP. This table has one row a
/// container that has none, so it needs this on the side: a rule about a
/// category walks the commands for the ones it names outright and then walks
/// this for the subcommands, allowing or refusing them one first argument at a
/// time. It goes away with D-114, when the subcommands get rows of their own and
/// carry their own categories like everything else.
///
/// Every subcommand also holds every category its container holds, which was
/// read off 8.10.1 rather than assumed, and is why a rule about the container's
/// own categories can still set one bit and be done.
///
/// The list is sorted by container and then by subcommand, and a subcommand with
/// nothing to add is left out, which is why COMMAND has no rows at all.
pub static SUBCATS: &[(&str, &str, &[&str])] = &[
    ("acl", "deluser", AC_SUB_ADMIN),
    ("acl", "dryrun", AC_SUB_ADMIN),
    ("acl", "getuser", AC_SUB_ADMIN),
    ("acl", "list", AC_SUB_ADMIN),
    ("acl", "load", AC_SUB_ADMIN),
    ("acl", "log", AC_SUB_ADMIN),
    ("acl", "save", AC_SUB_ADMIN),
    ("acl", "setuser", AC_SUB_ADMIN),
    ("acl", "users", AC_SUB_ADMIN),
    ("client", "caching", AC_SUB_CONNECTION),
    ("client", "getname", AC_SUB_CONNECTION),
    ("client", "getredir", AC_SUB_CONNECTION),
    ("client", "help", AC_SUB_CONNECTION),
    ("client", "id", AC_SUB_CONNECTION),
    ("client", "info", AC_SUB_CONNECTION),
    ("client", "kill", AC_SUB_ADMIN_CONNECTION),
    ("client", "list", AC_SUB_ADMIN_CONNECTION),
    ("client", "no-evict", AC_SUB_ADMIN_CONNECTION),
    ("client", "no-touch", AC_SUB_CONNECTION),
    ("client", "pause", AC_SUB_ADMIN_CONNECTION),
    ("client", "reply", AC_SUB_CONNECTION),
    ("client", "setinfo", AC_SUB_CONNECTION),
    ("client", "setname", AC_SUB_CONNECTION),
    ("client", "tracking", AC_SUB_CONNECTION),
    ("client", "trackinginfo", AC_SUB_CONNECTION),
    ("client", "unblock", AC_SUB_ADMIN_CONNECTION),
    ("client", "unpause", AC_SUB_ADMIN_CONNECTION),
    ("config", "get", AC_SUB_ADMIN),
    ("config", "resetstat", AC_SUB_ADMIN),
    ("config", "rewrite", AC_SUB_ADMIN),
    ("config", "set", AC_SUB_ADMIN),
    ("function", "delete", AC_SUB_SCRIPTING_WRITE),
    ("function", "dump", AC_SUB_SCRIPTING),
    ("function", "flush", AC_SUB_SCRIPTING_WRITE),
    ("function", "help", AC_SUB_SCRIPTING),
    ("function", "kill", AC_SUB_SCRIPTING),
    ("function", "list", AC_SUB_SCRIPTING),
    ("function", "load", AC_SUB_SCRIPTING_WRITE),
    ("function", "restore", AC_SUB_SCRIPTING_WRITE),
    ("function", "stats", AC_SUB_SCRIPTING),
    ("memory", "usage", AC_SUB_READ),
    ("object", "encoding", AC_SUB_KEYSPACE_READ),
    ("object", "freq", AC_SUB_KEYSPACE_READ),
    ("object", "help", AC_SUB_KEYSPACE),
    ("object", "idletime", AC_SUB_KEYSPACE_READ),
    ("object", "refcount", AC_SUB_KEYSPACE_READ),
    ("pubsub", "channels", AC_SUB_PUBSUB),
    ("pubsub", "numpat", AC_SUB_PUBSUB),
    ("pubsub", "numsub", AC_SUB_PUBSUB),
    ("pubsub", "shardchannels", AC_SUB_PUBSUB),
    ("pubsub", "shardnumsub", AC_SUB_PUBSUB),
    ("script", "debug", AC_SUB_SCRIPTING),
    ("script", "exists", AC_SUB_SCRIPTING),
    ("script", "flush", AC_SUB_SCRIPTING),
    ("script", "help", AC_SUB_SCRIPTING),
    ("script", "kill", AC_SUB_SCRIPTING),
    ("script", "load", AC_SUB_SCRIPTING),
    ("xgroup", "create", AC_SUB_STREAM_WRITE),
    ("xgroup", "createconsumer", AC_SUB_STREAM_WRITE),
    ("xgroup", "delconsumer", AC_SUB_STREAM_WRITE),
    ("xgroup", "destroy", AC_SUB_STREAM_WRITE),
    ("xgroup", "help", AC_SUB_STREAM),
    ("xgroup", "setid", AC_SUB_STREAM_WRITE),
    ("xinfo", "consumers", AC_SUB_STREAM_READ),
    ("xinfo", "groups", AC_SUB_STREAM_READ),
    ("xinfo", "help", AC_SUB_STREAM),
    ("xinfo", "stream", AC_SUB_STREAM_READ),
];

/// The subcommands of ACL and CONFIG that only an administrator should have.
const AC_SUB_ADMIN: &[&str] = &["@admin", "@dangerous"];
/// The subcommands of CLIENT that anybody may use.
const AC_SUB_CONNECTION: &[&str] = &["@connection"];
/// The subcommands of CLIENT that reach another connection.
const AC_SUB_ADMIN_CONNECTION: &[&str] = &["@admin", "@connection", "@dangerous"];
/// The subcommands of FUNCTION and SCRIPT that only read.
const AC_SUB_SCRIPTING: &[&str] = &["@scripting"];
/// The subcommands of FUNCTION that change what is loaded.
const AC_SUB_SCRIPTING_WRITE: &[&str] = &["@scripting", "@write"];
/// `MEMORY USAGE`, which is the one subcommand of MEMORY that reads a key and
/// so the one that is in a category its container is not.
const AC_SUB_READ: &[&str] = &["@read"];
/// `OBJECT HELP`, which names the keyspace without reading one.
const AC_SUB_KEYSPACE: &[&str] = &["@keyspace"];
/// The subcommands of OBJECT that read a key.
const AC_SUB_KEYSPACE_READ: &[&str] = &["@keyspace", "@read"];
/// The subcommands of PUBSUB, all but HELP.
const AC_SUB_PUBSUB: &[&str] = &["@pubsub"];
/// `XGROUP HELP` and `XINFO HELP`.
const AC_SUB_STREAM: &[&str] = &["@stream"];
/// The subcommands of XGROUP that change a group.
const AC_SUB_STREAM_WRITE: &[&str] = &["@stream", "@write"];
/// The subcommands of XINFO that read a stream.
const AC_SUB_STREAM_READ: &[&str] = &["@read", "@stream"];

/// Every command this server answers, in the order the groups ship.
pub static COMMANDS: &[Spec] = &[
    // ------------------------------------------------------------- strings
    Spec {
        name: "set",
        arity: -3,
        flags: WRITE_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[SET_KEY],
        acl: AC_WRITE_SLOW,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Set a key to a string value, whatever it held before.",
        group: "string",
    },
    Spec {
        name: "get",
        arity: 2,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_READ_FAST,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "The string value of a key.",
        group: "string",
    },
    Spec {
        name: "getset",
        arity: 3,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_WRITE_FAST,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Set a key and hand back what it held.",
        group: "string",
    },
    Spec {
        name: "getdel",
        arity: 2,
        flags: &["write", "fast"],
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_DELETE_AT1],
        acl: AC_WRITE_FAST,
        since: "6.2.0",
        complexity: "O(1)",
        summary: "Read a key and delete it in the same step.",
        group: "string",
    },
    Spec {
        name: "getex",
        arity: -2,
        flags: &["write", "fast"],
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1_TTL],
        acl: AC_WRITE_FAST,
        since: "6.2.0",
        complexity: "O(1)",
        summary: "Read a key and change its deadline in the same step.",
        group: "string",
    },
    Spec {
        name: "setnx",
        arity: 3,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[OW_INSERT_AT1],
        acl: AC_WRITE_FAST,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Set a key only if it is not there.",
        group: "string",
    },
    Spec {
        name: "setex",
        arity: 4,
        flags: WRITE_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[OW_UPDATE_AT1],
        acl: AC_WRITE_SLOW,
        since: "2.0.0",
        complexity: "O(1)",
        summary: "Set a key and give it a deadline in seconds.",
        group: "string",
    },
    Spec {
        name: "psetex",
        arity: 4,
        flags: WRITE_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[OW_UPDATE_AT1],
        acl: AC_WRITE_SLOW,
        since: "2.6.0",
        complexity: "O(1)",
        summary: "Set a key and give it a deadline in milliseconds.",
        group: "string",
    },
    Spec {
        name: "mset",
        arity: -3,
        flags: WRITE_OOM,
        first_key: 1,
        last_key: -1,
        step: 2,
        keys: &[OW_UPDATE_AT1_RM1_2_0],
        acl: AC_WRITE_SLOW,
        since: "1.0.1",
        complexity: "O(N) with N the number of keys",
        summary: "Set several keys, all of them or none.",
        group: "string",
    },
    Spec {
        name: "msetnx",
        arity: -3,
        flags: WRITE_OOM,
        first_key: 1,
        last_key: -1,
        step: 2,
        keys: &[OW_INSERT_AT1_RM1_2_0],
        acl: AC_WRITE_SLOW,
        since: "1.0.1",
        complexity: "O(N) with N the number of keys",
        summary: "Set several keys only if none of them are there.",
        group: "string",
    },
    Spec {
        name: "mget",
        arity: -2,
        flags: READ_FAST,
        first_key: 1,
        last_key: -1,
        step: 1,
        keys: &[RO_ACCESS_AT1_RM1_1_0],
        acl: AC_READ_FAST,
        since: "1.0.0",
        complexity: "O(N) with N the number of keys",
        summary: "The values of several keys, in the order asked for.",
        group: "string",
    },
    Spec {
        name: "append",
        arity: 3,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_INSERT_AT1],
        acl: AC_WRITE_FAST,
        since: "2.0.0",
        complexity: "O(M) with M the length of the value being appended",
        summary: "Add to the end of a string, creating it if it is not there.",
        group: "string",
    },
    Spec {
        name: "strlen",
        arity: 2,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_AT1],
        acl: AC_READ_FAST,
        since: "2.2.0",
        complexity: "O(1)",
        summary: "How long a string value is, without reading it.",
        group: "string",
    },
    Spec {
        name: "setrange",
        arity: 4,
        flags: WRITE_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_AT1],
        acl: AC_WRITE_SLOW,
        since: "2.2.0",
        complexity: "O(M) with M the length of the replacement",
        summary: "Overwrite part of a string at an offset, zero filling the gap.",
        group: "string",
    },
    Spec {
        name: "getrange",
        arity: 4,
        flags: &["readonly"],
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_READ_SLOW,
        since: "2.4.0",
        complexity: "O(N) with N the length of the answer",
        summary: "Part of a string, by an inclusive range that may count backwards.",
        group: "string",
    },
    Spec {
        name: "substr",
        arity: 4,
        flags: &["readonly"],
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_READ_SLOW,
        since: "1.0.0",
        complexity: "O(N) with N the length of the answer",
        summary: "GETRANGE under the name it had before 2.4.",
        group: "string",
    },
    Spec {
        name: "incr",
        arity: 2,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_WRITE_FAST,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Add one, starting from zero if the key is not there.",
        group: "string",
    },
    Spec {
        name: "decr",
        arity: 2,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_WRITE_FAST,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Take one away, starting from zero if the key is not there.",
        group: "string",
    },
    Spec {
        name: "incrby",
        arity: 3,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_WRITE_FAST,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Add a number, starting from zero if the key is not there.",
        group: "string",
    },
    Spec {
        name: "decrby",
        arity: 3,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_WRITE_FAST,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Take a number away, starting from zero if the key is not there.",
        group: "string",
    },
    Spec {
        name: "incrbyfloat",
        arity: 3,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_WRITE_FAST,
        since: "2.6.0",
        complexity: "O(1)",
        summary: "Add a float, starting from zero if the key is not there.",
        group: "string",
    },
    Spec {
        name: "lcs",
        arity: -3,
        flags: &["readonly"],
        first_key: 1,
        last_key: 2,
        step: 1,
        keys: &[RO_ACCESS_AT1_R1_1_0],
        acl: AC_READ_SLOW,
        since: "7.0.0",
        complexity: "O(N*M) with N and M the lengths of the two values",
        summary: "The longest subsequence two string values have in common.",
        group: "string",
    },
    Spec {
        name: "msetex",
        arity: -4,
        flags: &["write", "denyoom", "movablekeys"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[OW_UPDATE_AT1_COUNTED],
        acl: AC_WRITE_SLOW,
        since: "8.4.0",
        complexity: "O(N) with N the number of keys",
        summary: "Set several keys with one deadline and one condition over all of them.",
        group: "string",
    },
    Spec {
        name: "delex",
        arity: -2,
        flags: &["write", "fast"],
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[DELEX_KEY],
        acl: AC_WRITE_FAST,
        since: "8.4.0",
        complexity: "O(1) by value, O(N) by digest",
        summary: "Delete a key only if it still holds what the caller thinks.",
        group: "string",
    },
    Spec {
        name: "digest",
        arity: 2,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_READ_FAST,
        since: "8.4.0",
        complexity: "O(N) with N the length of the value",
        summary: "The XXH3 of a string value, as sixteen hex characters.",
        group: "string",
    },
    Spec {
        name: "increx",
        arity: -2,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_WRITE_FAST,
        since: "8.8.0",
        complexity: "O(1)",
        summary: "Count, with a bound, a saturation policy and a deadline.",
        group: "string",
    },
    // -------------------------------------------------------------- bitmaps
    Spec {
        name: "setbit",
        arity: 4,
        flags: WRITE_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_BIT_WRITE,
        since: "2.2.0",
        complexity: "O(1)",
        summary: "Set one bit of a string, growing it to reach the offset.",
        group: "bitmap",
    },
    Spec {
        name: "getbit",
        arity: 3,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_BIT_READ_FAST,
        since: "2.2.0",
        complexity: "O(1)",
        summary: "Read one bit of a string, or nought past its end.",
        group: "bitmap",
    },
    Spec {
        name: "bitcount",
        arity: -2,
        flags: &["readonly"],
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_BIT_READ,
        since: "2.6.0",
        complexity: "O(N)",
        summary: "Count the set bits of a string, or of a range of it.",
        group: "bitmap",
    },
    Spec {
        name: "bitpos",
        arity: -3,
        flags: &["readonly"],
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_BIT_READ,
        since: "2.8.7",
        complexity: "O(N)",
        summary: "Find the first bit set to one or nought in a string.",
        group: "bitmap",
    },
    Spec {
        name: "bitop",
        arity: -4,
        flags: WRITE_OOM,
        first_key: 2,
        last_key: -1,
        step: 1,
        keys: &[OW_UPDATE_AT2, RO_ACCESS_AT3_RM1_1_0],
        acl: AC_BIT_WRITE,
        since: "2.6.0",
        complexity: "O(N) with N the length of the longest source",
        summary: "Combine strings bit by bit and store the result.",
        group: "bitmap",
    },
    Spec {
        name: "bitfield",
        arity: -2,
        flags: WRITE_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[BITFIELD_KEY],
        acl: AC_BIT_WRITE,
        since: "3.2.0",
        complexity: "O(1) per subcommand",
        summary: "Read and write packed integer fields inside a string.",
        group: "bitmap",
    },
    Spec {
        name: "bitfield_ro",
        arity: -2,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_BIT_READ_FAST,
        since: "6.0.0",
        complexity: "O(1) per subcommand",
        summary: "The read only half of BITFIELD, for a replica to answer.",
        group: "bitmap",
    },
    // --------------------------------------------------------- hyperloglogs
    Spec {
        name: "pfadd",
        arity: -2,
        flags: WRITE_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_INSERT_AT1],
        acl: AC_HLL_WRITE_FAST,
        since: "2.8.9",
        complexity: "O(1) an element",
        summary: "Add elements to a sketch, answering whether it changed.",
        group: "hyperloglog",
    },
    Spec {
        name: "pfcount",
        arity: -2,
        flags: &["readonly"],
        first_key: 1,
        last_key: -1,
        step: 1,
        keys: &[RW_ACCESS_AT1_RM1_1_0],
        acl: AC_HLL_READ,
        since: "2.8.9",
        complexity: "O(1) for one key, O(N) for N of them",
        summary: "Estimate how many distinct elements the sketches hold.",
        group: "hyperloglog",
    },
    Spec {
        name: "pfmerge",
        arity: -2,
        flags: WRITE_OOM,
        first_key: 1,
        last_key: -1,
        step: 1,
        keys: &[RW_ACCESS_INSERT_AT1, RO_ACCESS_AT2_RM1_1_0],
        acl: AC_HLL_WRITE,
        since: "2.8.9",
        complexity: "O(N) in the number of sketches",
        summary: "Merge sketches into the first one, which is a union.",
        group: "hyperloglog",
    },
    Spec {
        name: "pfdebug",
        arity: 3,
        flags: WRITE_OOM_ADMIN,
        first_key: 2,
        last_key: 2,
        step: 1,
        keys: &[RW_ACCESS_AT2],
        acl: AC_HLL_ADMIN,
        since: "2.8.9",
        complexity: "O(N)",
        summary: "Look inside a sketch, and in one case convert it.",
        group: "hyperloglog",
    },
    Spec {
        name: "pfselftest",
        arity: 1,
        flags: &["admin"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_HLL_ADMIN,
        since: "2.8.9",
        complexity: "O(1)",
        summary: "Check the sketch code, which our tests do at build time.",
        group: "hyperloglog",
    },
    // ----------------------------------------------------------------- sets
    Spec {
        name: "sadd",
        arity: -3,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_INSERT_AT1],
        acl: AC_SET_WRITE_FAST,
        since: "1.0.0",
        complexity: "O(N) with N the number of members being added",
        summary: "Add members to a set, creating it if it is not there.",
        group: "set",
    },
    Spec {
        name: "srem",
        arity: -3,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_DELETE_AT1],
        acl: AC_SET_WRITE_FAST,
        since: "1.0.0",
        complexity: "O(N) with N the number of members being removed",
        summary: "Take members out of a set, deleting the key if none are left.",
        group: "set",
    },
    Spec {
        name: "scard",
        arity: 2,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_AT1],
        acl: AC_SET_READ_FAST,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "How many members a set has.",
        group: "set",
    },
    Spec {
        name: "sismember",
        arity: 3,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_AT1],
        acl: AC_SET_READ_FAST,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Whether a member is in a set.",
        group: "set",
    },
    Spec {
        name: "smismember",
        arity: -3,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_SET_READ_FAST,
        since: "6.2.0",
        complexity: "O(N) with N the number of members being asked about",
        summary: "Whether each of several members is in a set, in the order asked.",
        group: "set",
    },
    Spec {
        name: "smembers",
        arity: 2,
        flags: &["readonly"],
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_SET_READ_SLOW,
        since: "1.0.0",
        complexity: "O(N) with N the size of the set",
        summary: "Every member of a set.",
        group: "set",
    },
    Spec {
        name: "spop",
        arity: -2,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_DELETE_AT1],
        acl: AC_SET_WRITE_FAST,
        since: "1.0.0",
        complexity: "O(1) without a count, O(N) with one",
        summary: "Take members out of a set at random and hand them back.",
        group: "set",
    },
    Spec {
        name: "srandmember",
        arity: -2,
        flags: &["readonly"],
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_SET_READ_SLOW,
        since: "1.0.0",
        complexity: "O(1) without a count, O(N) with one",
        summary: "Members of a set at random, leaving the set as it was.",
        group: "set",
    },
    Spec {
        name: "smove",
        arity: 4,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 2,
        step: 1,
        keys: &[RW_ACCESS_DELETE_AT1, RW_INSERT_AT2],
        acl: AC_SET_WRITE_FAST,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Move one member from one set to another.",
        group: "set",
    },
    Spec {
        name: "sscan",
        arity: -3,
        flags: &["readonly"],
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_SET_READ_SLOW,
        since: "2.8.0",
        complexity: "O(1) a call, O(N) for a whole iteration",
        summary: "Walk part of a set and say where to carry on from.",
        group: "set",
    },
    Spec {
        name: "sinter",
        arity: -2,
        flags: &["readonly"],
        first_key: 1,
        last_key: -1,
        step: 1,
        keys: &[RO_ACCESS_AT1_RM1_1_0],
        acl: AC_SET_READ_SLOW,
        since: "1.0.0",
        complexity: "O(N*M) worst case, N the smallest set and M the number of sets",
        summary: "The members every one of these sets has.",
        group: "set",
    },
    Spec {
        name: "sintercard",
        arity: -3,
        // The only set command whose keys are counted rather than positioned,
        // so the legacy key range cannot describe it and Redis reports zeroes
        // in these three fields too. A client that wants the keys reads the key
        // specs, which is what the count is for, and movablekeys is how it is
        // told to go and read them.
        flags: READ_MOVABLE,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[RO_ACCESS_AT1_COUNTED],
        acl: AC_SET_READ_SLOW,
        since: "7.0.0",
        complexity: "O(N*M) worst case, N the smallest set and M the number of sets",
        summary: "How many members every one of these sets has, up to a limit.",
        group: "set",
    },
    Spec {
        name: "sinterstore",
        arity: -3,
        flags: WRITE_OOM,
        first_key: 1,
        last_key: -1,
        step: 1,
        keys: &[OW_UPDATE_AT1, RO_ACCESS_AT2_RM1_1_0],
        acl: AC_SET_WRITE_SLOW,
        since: "1.0.0",
        complexity: "O(N*M) worst case, N the smallest set and M the number of sets",
        summary: "Store the members every one of these sets has.",
        group: "set",
    },
    Spec {
        name: "sunion",
        arity: -2,
        flags: &["readonly"],
        first_key: 1,
        last_key: -1,
        step: 1,
        keys: &[RO_ACCESS_AT1_RM1_1_0],
        acl: AC_SET_READ_SLOW,
        since: "1.0.0",
        complexity: "O(N) in the total number of members",
        summary: "The members any of these sets has, each once.",
        group: "set",
    },
    Spec {
        name: "sunionstore",
        arity: -3,
        flags: WRITE_OOM,
        first_key: 1,
        last_key: -1,
        step: 1,
        keys: &[OW_UPDATE_AT1, RO_ACCESS_AT2_RM1_1_0],
        acl: AC_SET_WRITE_SLOW,
        since: "1.0.0",
        complexity: "O(N) in the total number of members",
        summary: "Store the members any of these sets has.",
        group: "set",
    },
    Spec {
        name: "sdiff",
        arity: -2,
        flags: &["readonly"],
        first_key: 1,
        last_key: -1,
        step: 1,
        keys: &[RO_ACCESS_AT1_RM1_1_0],
        acl: AC_SET_READ_SLOW,
        since: "1.0.0",
        complexity: "O(N) in the total number of members",
        summary: "The members of the first set that no later set has.",
        group: "set",
    },
    Spec {
        name: "sdiffstore",
        arity: -3,
        flags: WRITE_OOM,
        first_key: 1,
        last_key: -1,
        step: 1,
        keys: &[OW_UPDATE_AT1, RO_ACCESS_AT2_RM1_1_0],
        acl: AC_SET_WRITE_SLOW,
        since: "1.0.0",
        complexity: "O(N) in the total number of members",
        summary: "Store the members of the first set that no later set has.",
        group: "set",
    },
    // The two 8.10 added, which are to SUNION and SDIFF what SINTERCARD is to
    // SINTER, and which describe their keys the same way it does and for the
    // same reason.
    Spec {
        name: "sunioncard",
        arity: -3,
        flags: READ_MOVABLE,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[RO_ACCESS_AT1_COUNTED],
        acl: AC_SET_READ_SLOW,
        since: "8.10.0",
        complexity: "O(N) in the total number of members",
        summary: "How many members any of these sets has, up to a limit.",
        group: "set",
    },
    Spec {
        name: "sdiffcard",
        arity: -3,
        flags: READ_MOVABLE,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[RO_ACCESS_AT1_COUNTED],
        acl: AC_SET_READ_SLOW,
        since: "8.10.0",
        complexity: "O(N) in the total number of members",
        summary: "How many members the first set has that no later set has, up to a limit.",
        group: "set",
    },
    // -------------------------------------------------------------- hashes
    Spec {
        name: "hset",
        arity: -4,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_AT1],
        acl: AC_HASH_WRITE_FAST,
        since: "2.0.0",
        complexity: "O(N) with N the number of pairs being written",
        summary: "Write fields into a hash, creating it if it is not there.",
        group: "hash",
    },
    Spec {
        name: "hsetnx",
        arity: 4,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_INSERT_AT1],
        acl: AC_HASH_WRITE_FAST,
        since: "2.0.0",
        complexity: "O(1)",
        summary: "Write a field only if the hash does not have it already.",
        group: "hash",
    },
    // Deprecated since 4.0 and still sent by a great deal of code, so it is
    // here rather than left out. It is HSET with an OK instead of a count.
    Spec {
        name: "hmset",
        arity: -4,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_AT1],
        acl: AC_HASH_WRITE_FAST,
        since: "2.0.0",
        complexity: "O(N) with N the number of pairs being written",
        summary: "Write fields into a hash and answer OK. Use HSET.",
        group: "hash",
    },
    Spec {
        name: "hget",
        arity: 3,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_HASH_READ_FAST,
        since: "2.0.0",
        complexity: "O(1)",
        summary: "The value of one field of a hash.",
        group: "hash",
    },
    Spec {
        name: "hmget",
        arity: -3,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_HASH_READ_FAST,
        since: "2.0.0",
        complexity: "O(N) with N the number of fields asked for",
        summary: "The values of several fields, one reply entry each.",
        group: "hash",
    },
    Spec {
        name: "hdel",
        arity: -3,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_DELETE_AT1],
        acl: AC_HASH_WRITE_FAST,
        since: "2.0.0",
        complexity: "O(N) with N the number of fields being removed",
        summary: "Take fields out of a hash, deleting the key if none are left.",
        group: "hash",
    },
    Spec {
        name: "hlen",
        arity: 2,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_AT1],
        acl: AC_HASH_READ_FAST,
        since: "2.0.0",
        complexity: "O(1)",
        summary: "How many fields a hash has.",
        group: "hash",
    },
    Spec {
        name: "hexists",
        arity: 3,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_AT1],
        acl: AC_HASH_READ_FAST,
        since: "2.0.0",
        complexity: "O(1)",
        summary: "Whether a hash has a field.",
        group: "hash",
    },
    Spec {
        name: "hstrlen",
        arity: 3,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_AT1],
        acl: AC_HASH_READ_FAST,
        since: "3.2.0",
        complexity: "O(1)",
        summary: "How many bytes a field's value is, without sending it.",
        group: "hash",
    },
    Spec {
        name: "hgetall",
        arity: 2,
        flags: &["readonly"],
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_HASH_READ_SLOW,
        since: "2.0.0",
        complexity: "O(N) in the size of the hash",
        summary: "Every field and value, as a map on RESP3.",
        group: "hash",
    },
    Spec {
        name: "hkeys",
        arity: 2,
        flags: &["readonly"],
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_HASH_READ_SLOW,
        since: "2.0.0",
        complexity: "O(N) in the size of the hash",
        summary: "Every field of a hash.",
        group: "hash",
    },
    Spec {
        name: "hvals",
        arity: 2,
        flags: &["readonly"],
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_HASH_READ_SLOW,
        since: "2.0.0",
        complexity: "O(N) in the size of the hash",
        summary: "Every value of a hash.",
        group: "hash",
    },
    Spec {
        name: "hincrby",
        arity: 4,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_HASH_WRITE_FAST,
        since: "2.0.0",
        complexity: "O(1)",
        summary: "Add an integer to a field, treating a missing one as zero.",
        group: "hash",
    },
    Spec {
        name: "hincrbyfloat",
        arity: 4,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_HASH_WRITE_FAST,
        since: "2.6.0",
        complexity: "O(1)",
        summary: "Add a float to a field, treating a missing one as zero.",
        group: "hash",
    },
    Spec {
        name: "hrandfield",
        arity: -2,
        flags: &["readonly"],
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_HASH_READ_SLOW,
        since: "6.2.0",
        complexity: "O(1) without a count, O(N) with one",
        summary: "Fields of a hash at random, leaving the hash as it was.",
        group: "hash",
    },
    Spec {
        name: "hscan",
        arity: -3,
        flags: &["readonly"],
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_HASH_READ_SLOW,
        since: "2.8.0",
        complexity: "O(1) a call, O(N) for a whole iteration",
        summary: "Walk part of a hash and say where to carry on from.",
        group: "hash",
    },
    Spec {
        name: "hexpire",
        arity: -6,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_AT1],
        acl: AC_HASH_WRITE_FAST,
        since: "7.4.0",
        complexity: "O(N) with N the number of fields named",
        summary: "Put a deadline in seconds on hash fields.",
        group: "hash",
    },
    Spec {
        name: "hpexpire",
        arity: -6,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_AT1],
        acl: AC_HASH_WRITE_FAST,
        since: "7.4.0",
        complexity: "O(N) with N the number of fields named",
        summary: "Put a deadline in milliseconds on hash fields.",
        group: "hash",
    },
    Spec {
        name: "hexpireat",
        arity: -6,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_AT1],
        acl: AC_HASH_WRITE_FAST,
        since: "7.4.0",
        complexity: "O(N) with N the number of fields named",
        summary: "Put an absolute deadline in unix seconds on hash fields.",
        group: "hash",
    },
    Spec {
        name: "hpexpireat",
        arity: -6,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_AT1],
        acl: AC_HASH_WRITE_FAST,
        since: "7.4.0",
        complexity: "O(N) with N the number of fields named",
        summary: "Put an absolute deadline in unix milliseconds on hash fields.",
        group: "hash",
    },
    Spec {
        name: "httl",
        arity: -5,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_HASH_READ_FAST,
        since: "7.4.0",
        complexity: "O(N) with N the number of fields named",
        summary: "How long hash fields have left, in seconds.",
        group: "hash",
    },
    Spec {
        name: "hpttl",
        arity: -5,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_HASH_READ_FAST,
        since: "7.4.0",
        complexity: "O(N) with N the number of fields named",
        summary: "How long hash fields have left, in milliseconds.",
        group: "hash",
    },
    Spec {
        name: "hexpiretime",
        arity: -5,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_HASH_READ_FAST,
        since: "7.4.0",
        complexity: "O(N) with N the number of fields named",
        summary: "When hash fields fall due, in unix seconds.",
        group: "hash",
    },
    Spec {
        name: "hpexpiretime",
        arity: -5,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_HASH_READ_FAST,
        since: "7.4.0",
        complexity: "O(N) with N the number of fields named",
        summary: "When hash fields fall due, in unix milliseconds.",
        group: "hash",
    },
    Spec {
        name: "hpersist",
        arity: -5,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_AT1],
        acl: AC_HASH_WRITE_FAST,
        since: "7.4.0",
        complexity: "O(N) with N the number of fields named",
        summary: "Take the deadlines off hash fields.",
        group: "hash",
    },
    Spec {
        name: "hgetdel",
        arity: -5,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_DELETE_AT1],
        acl: AC_HASH_WRITE_FAST,
        since: "8.0.0",
        complexity: "O(N) with N the number of fields named",
        summary: "Read hash fields and delete them.",
        group: "hash",
    },
    Spec {
        name: "hgetex",
        arity: -5,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1_TTL],
        acl: AC_HASH_WRITE_FAST,
        since: "8.0.0",
        complexity: "O(N) with N the number of fields named",
        summary: "Read hash fields and set their deadlines.",
        group: "hash",
    },
    Spec {
        name: "hsetex",
        arity: -6,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_AT1],
        acl: AC_HASH_WRITE_FAST,
        since: "8.0.0",
        complexity: "O(N) with N the number of fields being set",
        summary: "Set hash fields and their deadlines together.",
        group: "hash",
    },
    // A container with no flags and no keys of its own, which is what a real
    // 8.10.1 reports: the write flags and the key index live on `HIMPORT SET`
    // and this row is only the name and the categories.
    Spec {
        name: "himport",
        arity: -2,
        flags: &[],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_HASH_SLOW,
        since: "8.10.0",
        complexity: "Depends on subcommand.",
        summary: "A container for session-based hash import commands using fieldsets.",
        group: "hash",
    },
    // ---------------------------------------------------------------- lists
    Spec {
        name: "lpush",
        arity: -3,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_INSERT_AT1],
        acl: AC_LIST_WRITE_FAST,
        since: "1.0.0",
        complexity: "O(N) with N the number of elements pushed",
        summary: "Push elements onto the head of a list.",
        group: "list",
    },
    Spec {
        name: "rpush",
        arity: -3,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_INSERT_AT1],
        acl: AC_LIST_WRITE_FAST,
        since: "1.0.0",
        complexity: "O(N) with N the number of elements pushed",
        summary: "Push elements onto the tail of a list.",
        group: "list",
    },
    Spec {
        name: "lpushx",
        arity: -3,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_INSERT_AT1],
        acl: AC_LIST_WRITE_FAST,
        since: "2.2.0",
        complexity: "O(N) with N the number of elements pushed",
        summary: "Push elements onto the head of a list that already exists.",
        group: "list",
    },
    Spec {
        name: "rpushx",
        arity: -3,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_INSERT_AT1],
        acl: AC_LIST_WRITE_FAST,
        since: "2.2.0",
        complexity: "O(N) with N the number of elements pushed",
        summary: "Push elements onto the tail of a list that already exists.",
        group: "list",
    },
    Spec {
        name: "lpop",
        arity: -2,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_DELETE_AT1],
        acl: AC_LIST_WRITE_FAST,
        since: "1.0.0",
        complexity: "O(N) with N the count asked for",
        summary: "Take elements off the head of a list.",
        group: "list",
    },
    Spec {
        name: "rpop",
        arity: -2,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_DELETE_AT1],
        acl: AC_LIST_WRITE_FAST,
        since: "1.0.0",
        complexity: "O(N) with N the count asked for",
        summary: "Take elements off the tail of a list.",
        group: "list",
    },
    Spec {
        name: "llen",
        arity: 2,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_AT1],
        acl: AC_LIST_READ_FAST,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "How many elements a list holds.",
        group: "list",
    },
    Spec {
        name: "lrange",
        arity: 4,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_LIST_READ_SLOW,
        since: "1.0.0",
        complexity: "O(S+N) with S the offset of the first element and N the range",
        summary: "Read a range of a list, both ends included.",
        group: "list",
    },
    Spec {
        name: "lindex",
        arity: 3,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_LIST_READ_SLOW,
        since: "1.0.0",
        complexity: "O(N) with N the distance to the index from the nearer end",
        summary: "Read one element of a list by index.",
        group: "list",
    },
    Spec {
        name: "lset",
        arity: 4,
        flags: WRITE_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_AT1],
        acl: AC_LIST_WRITE_SLOW,
        since: "1.0.0",
        complexity: "O(N) with N the distance to the index from the nearer end",
        summary: "Replace one element of a list by index.",
        group: "list",
    },
    Spec {
        name: "linsert",
        arity: 5,
        flags: WRITE_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_INSERT_AT1],
        acl: AC_LIST_WRITE_SLOW,
        since: "2.2.0",
        complexity: "O(N) with N the distance to the pivot from the head",
        summary: "Insert an element before or after another one.",
        group: "list",
    },
    Spec {
        name: "lrem",
        arity: 4,
        flags: WRITE_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_DELETE_AT1],
        acl: AC_LIST_WRITE_SLOW,
        since: "1.0.0",
        complexity: "O(N) with N the length of the list",
        summary: "Remove elements equal to a value from a list.",
        group: "list",
    },
    Spec {
        name: "ltrim",
        arity: 4,
        flags: WRITE_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_DELETE_AT1],
        acl: AC_LIST_WRITE_SLOW,
        since: "1.0.0",
        complexity: "O(N) with N the number of elements thrown away",
        summary: "Keep a range of a list and throw the rest away.",
        group: "list",
    },
    Spec {
        name: "lpos",
        arity: -3,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_LIST_READ_SLOW,
        since: "6.0.6",
        complexity: "O(N) with N the length of the list",
        summary: "Find where a value sits in a list.",
        group: "list",
    },
    Spec {
        name: "rpoplpush",
        arity: 3,
        flags: WRITE_OOM,
        first_key: 1,
        last_key: 2,
        step: 1,
        keys: &[RW_ACCESS_DELETE_AT1, RW_INSERT_AT2],
        acl: AC_LIST_WRITE_SLOW,
        since: "1.2.0",
        complexity: "O(1)",
        summary: "Move an element from the tail of one list to the head of another.",
        group: "list",
    },
    Spec {
        name: "lmove",
        arity: 5,
        flags: WRITE_OOM,
        first_key: 1,
        last_key: 2,
        step: 1,
        keys: &[RW_ACCESS_DELETE_AT1, RW_INSERT_AT2],
        acl: AC_LIST_WRITE_SLOW,
        since: "6.2.0",
        complexity: "O(1)",
        summary: "Move an element from either end of one list to either end of another.",
        group: "list",
    },
    Spec {
        name: "lmovem",
        arity: -5,
        flags: WRITE_OOM,
        first_key: 1,
        last_key: 2,
        step: 1,
        keys: &[RW_ACCESS_DELETE_AT1, RW_INSERT_AT2],
        acl: AC_LIST_WRITE_SLOW,
        since: "8.10.0",
        complexity: "O(N) in the number of elements moved",
        summary: "Move several elements from either end of one list to either end of another.",
        group: "list",
    },
    // The keys are behind a count, so `first_key` is zero and a cluster client
    // has to ask `COMMAND GETKEYS` rather than read a position out of this row.
    // That is what `movablekeys` means and it is why the three key fields are
    // all zero rather than pointing at argument two.
    Spec {
        name: "lmpop",
        arity: -4,
        flags: &["write", "movablekeys"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[RW_ACCESS_DELETE_AT1_COUNTED],
        acl: AC_LIST_WRITE_SLOW,
        since: "7.0.0",
        complexity: "O(N+M) with N the number of keys and M the count popped",
        summary: "Pop from the first of several lists that has anything in it.",
        group: "list",
    },
    // The five that wait. `blocking` is what the dispatcher branches on to send
    // them somewhere that can park a client, so it is load bearing here rather
    // than only being reported.
    //
    // `BLPOP` and `BRPOP` take their keys up to the timeout, which is the one
    // shape in the list group where `last_key` is negative: everything from
    // argument one to the second from last.
    Spec {
        name: "blpop",
        arity: -3,
        flags: &["write", "blocking"],
        first_key: 1,
        last_key: -2,
        step: 1,
        keys: &[RW_ACCESS_DELETE_AT1_RM2_1_0],
        acl: AC_LIST_WRITE_BLOCKING,
        since: "2.0.0",
        complexity: "O(N) with N the number of keys named",
        summary: "Pop the head of the first list that has anything, waiting if none does.",
        group: "list",
    },
    Spec {
        name: "brpop",
        arity: -3,
        flags: &["write", "blocking"],
        first_key: 1,
        last_key: -2,
        step: 1,
        keys: &[RW_ACCESS_DELETE_AT1_RM2_1_0],
        acl: AC_LIST_WRITE_BLOCKING,
        since: "2.0.0",
        complexity: "O(N) with N the number of keys named",
        summary: "Pop the tail of the first list that has anything, waiting if none does.",
        group: "list",
    },
    // Redis marks the two that push somewhere `denyoom` and does not mark the
    // pops, because these are the blocking commands that can grow the keyspace.
    Spec {
        name: "blmove",
        arity: 6,
        flags: &["write", "denyoom", "blocking"],
        first_key: 1,
        last_key: 2,
        step: 1,
        keys: &[RW_ACCESS_DELETE_AT1, RW_INSERT_AT2],
        acl: AC_LIST_WRITE_BLOCKING,
        since: "6.2.0",
        complexity: "O(1)",
        summary: "Move an element between two lists, waiting for one to arrive.",
        group: "list",
    },
    Spec {
        name: "blmovem",
        arity: -6,
        flags: &["write", "denyoom", "blocking"],
        first_key: 1,
        last_key: 2,
        step: 1,
        keys: &[RW_ACCESS_DELETE_AT1, RW_INSERT_AT2],
        acl: AC_LIST_WRITE_BLOCKING,
        since: "8.10.0",
        complexity: "O(N) in the number of elements moved",
        summary: "Move several elements between two lists, waiting for them to arrive.",
        group: "list",
    },
    Spec {
        name: "brpoplpush",
        arity: 4,
        flags: &["write", "denyoom", "blocking"],
        first_key: 1,
        last_key: 2,
        step: 1,
        keys: &[RW_ACCESS_DELETE_AT1, RW_INSERT_AT2],
        acl: AC_LIST_WRITE_BLOCKING,
        since: "2.2.0",
        complexity: "O(1)",
        summary: "Move a tail element to another list's head, waiting for one to arrive.",
        group: "list",
    },
    // Keys behind a count again, so the same three zeroes `LMPOP` has.
    Spec {
        name: "blmpop",
        arity: -5,
        flags: &["write", "blocking", "movablekeys"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[RW_ACCESS_DELETE_AT2_COUNTED],
        acl: AC_LIST_WRITE_BLOCKING,
        since: "7.0.0",
        complexity: "O(N+M) with N the number of keys and M the count popped",
        summary: "Pop from the first of several lists that has anything, waiting if none does.",
        group: "list",
    },
    // ------------------------------------------------------------ sorted set
    Spec {
        name: "zadd",
        arity: -4,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_AT1],
        acl: AC_ZSET_WRITE_FAST,
        since: "1.2.0",
        complexity: "O(log(N)) for each member added",
        summary: "Add members with scores, or move the scores of members already there.",
        group: "zset",
    },
    Spec {
        name: "zincrby",
        arity: 4,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_ZSET_WRITE_FAST,
        since: "1.2.0",
        complexity: "O(log(N))",
        summary: "Add to a member's score, creating the member at zero if it is not there.",
        group: "zset",
    },
    Spec {
        name: "zcard",
        arity: 2,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_AT1],
        acl: AC_ZSET_READ_FAST,
        since: "1.2.0",
        complexity: "O(1)",
        summary: "How many members a sorted set has.",
        group: "zset",
    },
    Spec {
        name: "zscore",
        arity: 3,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_ZSET_READ_FAST,
        since: "1.2.0",
        complexity: "O(1)",
        summary: "A member's score, or nothing if it is not there.",
        group: "zset",
    },
    Spec {
        name: "zmscore",
        arity: -3,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_ZSET_READ_FAST,
        since: "6.2.0",
        complexity: "O(N) with N the number of members asked about",
        summary: "The scores of several members in one round trip.",
        group: "zset",
    },
    Spec {
        name: "zrem",
        arity: -3,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_DELETE_AT1],
        acl: AC_ZSET_WRITE_FAST,
        since: "1.2.0",
        complexity: "O(M*log(N)) with M the number of members removed",
        summary: "Remove members, deleting the key if the last one goes.",
        group: "zset",
    },
    Spec {
        name: "zrank",
        arity: -3,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_ZSET_READ_FAST,
        since: "2.0.0",
        complexity: "O(log(N))",
        summary: "Where a member sits counting up from the lowest score.",
        group: "zset",
    },
    Spec {
        name: "zrevrank",
        arity: -3,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_ZSET_READ_FAST,
        since: "2.0.0",
        complexity: "O(log(N))",
        summary: "Where a member sits counting down from the highest score.",
        group: "zset",
    },
    Spec {
        name: "zcount",
        arity: 4,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_ZSET_READ_FAST,
        since: "2.0.0",
        complexity: "O(log(N))",
        summary: "How many members have scores between two bounds.",
        group: "zset",
    },
    Spec {
        name: "zlexcount",
        arity: 4,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_ZSET_READ_FAST,
        since: "2.8.9",
        complexity: "O(log(N))",
        summary: "How many members fall between two members, by name.",
        group: "zset",
    },
    Spec {
        name: "zrange",
        arity: -4,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_ZSET_READ_SLOW,
        since: "1.2.0",
        complexity: "O(log(N)+M) with M the number of members answered",
        summary: "A window of members, by rank or by score or by name, either way round.",
        group: "zset",
    },
    Spec {
        name: "zrevrange",
        arity: -4,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_ZSET_READ_SLOW,
        since: "1.2.0",
        complexity: "O(log(N)+M) with M the number of members answered",
        summary: "A window by rank, counting down from the highest score.",
        group: "zset",
    },
    Spec {
        name: "zrangebyscore",
        arity: -4,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_ZSET_READ_SLOW,
        since: "1.0.5",
        complexity: "O(log(N)+M) with M the number of members answered",
        summary: "The members whose scores fall between two bounds.",
        group: "zset",
    },
    Spec {
        name: "zrevrangebyscore",
        arity: -4,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_ZSET_READ_SLOW,
        since: "2.2.0",
        complexity: "O(log(N)+M) with M the number of members answered",
        summary: "The same window as ZRANGEBYSCORE, highest score first and named high end first.",
        group: "zset",
    },
    Spec {
        name: "zrangebylex",
        arity: -4,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_ZSET_READ_SLOW,
        since: "2.8.9",
        complexity: "O(log(N)+M) with M the number of members answered",
        summary: "The members that fall between two names, for a set where every score is the same.",
        group: "zset",
    },
    Spec {
        name: "zrevrangebylex",
        arity: -4,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_ZSET_READ_SLOW,
        since: "2.8.9",
        complexity: "O(log(N)+M) with M the number of members answered",
        summary: "The same window as ZRANGEBYLEX, backwards and named high end first.",
        group: "zset",
    },
    Spec {
        name: "zrangestore",
        arity: -5,
        flags: WRITE_OOM,
        first_key: 1,
        last_key: 2,
        step: 1,
        keys: &[OW_UPDATE_AT1, RO_ACCESS_AT2],
        acl: AC_ZSET_WRITE_SLOW,
        since: "6.2.0",
        complexity: "O(log(N)+M) with M the number of members stored",
        summary: "Write a window of one sorted set into another key.",
        group: "zset",
    },
    Spec {
        name: "zremrangebyrank",
        arity: 4,
        flags: WRITE_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_DELETE_AT1],
        acl: AC_ZSET_WRITE_SLOW,
        since: "2.0.0",
        complexity: "O(log(N)+M) with M the number of members removed",
        summary: "Remove the members in a range of ranks.",
        group: "zset",
    },
    Spec {
        name: "zremrangebyscore",
        arity: 4,
        flags: WRITE_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_DELETE_AT1],
        acl: AC_ZSET_WRITE_SLOW,
        since: "1.2.0",
        complexity: "O(log(N)+M) with M the number of members removed",
        summary: "Remove the members whose scores fall between two bounds.",
        group: "zset",
    },
    Spec {
        name: "zremrangebylex",
        arity: 4,
        flags: WRITE_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_DELETE_AT1],
        acl: AC_ZSET_WRITE_SLOW,
        since: "2.8.9",
        complexity: "O(log(N)+M) with M the number of members removed",
        summary: "Remove the members that fall between two names.",
        group: "zset",
    },
    Spec {
        name: "zunion",
        arity: -3,
        flags: READ_MOVABLE,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[RO_ACCESS_AT1_COUNTED],
        acl: AC_ZSET_READ_SLOW,
        since: "6.2.0",
        complexity: "O(N)+O(M*log(M)) with N the total number of members and M the number in the answer",
        summary: "Every member of these sorted sets, with the scores combined.",
        group: "zset",
    },
    Spec {
        name: "zinter",
        arity: -3,
        flags: READ_MOVABLE,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[RO_ACCESS_AT1_COUNTED],
        acl: AC_ZSET_READ_SLOW,
        since: "6.2.0",
        complexity: "O(N)+O(M*log(M)) with N the total number of members and M the number in the answer",
        summary: "Only the members all of these sorted sets have, with the scores combined.",
        group: "zset",
    },
    Spec {
        name: "zdiff",
        arity: -3,
        flags: READ_MOVABLE,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[RO_ACCESS_AT1_COUNTED],
        acl: AC_ZSET_READ_SLOW,
        since: "6.2.0",
        complexity: "O(N)+O(M*log(M)) with N the total number of members and M the number in the answer",
        summary: "The members of the first that none of the rest have.",
        group: "zset",
    },
    Spec {
        name: "zunionstore",
        arity: -4,
        flags: WRITE_MOVABLE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[OW_UPDATE_AT1, RO_ACCESS_AT2_COUNTED],
        acl: AC_ZSET_WRITE_SLOW,
        since: "2.0.0",
        complexity: "O(N)+O(M*log(M)) with N the total number of members and M the number in the answer",
        summary: "Store the union in another key and say how big it is.",
        group: "zset",
    },
    Spec {
        name: "zinterstore",
        arity: -4,
        flags: WRITE_MOVABLE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[OW_UPDATE_AT1, RO_ACCESS_AT2_COUNTED],
        acl: AC_ZSET_WRITE_SLOW,
        since: "2.0.0",
        complexity: "O(N)+O(M*log(M)) with N the total number of members and M the number in the answer",
        summary: "Store the intersection in another key and say how big it is.",
        group: "zset",
    },
    Spec {
        name: "zdiffstore",
        arity: -4,
        flags: WRITE_MOVABLE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[OW_UPDATE_AT1, RO_ACCESS_AT2_COUNTED],
        acl: AC_ZSET_WRITE_SLOW,
        since: "6.2.0",
        complexity: "O(N)+O(M*log(M)) with N the total number of members and M the number in the answer",
        summary: "Store the difference in another key and say how big it is.",
        group: "zset",
    },
    Spec {
        name: "zintercard",
        arity: -3,
        flags: READ_MOVABLE,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[RO_ACCESS_AT1_COUNTED],
        acl: AC_ZSET_READ_SLOW,
        since: "7.0.0",
        complexity: "O(N*M) worst case, N the smallest input and M the number of inputs",
        summary: "How many members the intersection would have, without building it.",
        group: "zset",
    },
    Spec {
        name: "zrandmember",
        arity: -2,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_ZSET_READ_SLOW,
        since: "6.2.0",
        complexity: "O(N) with N the number of members drawn",
        summary: "Draw members at random, with or without replacement.",
        group: "zset",
    },
    Spec {
        name: "zscan",
        arity: -3,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_ZSET_READ_SLOW,
        since: "2.8.0",
        complexity: "O(1) per call, O(N) over a full walk",
        summary: "Walk the members and their scores a batch at a time.",
        group: "zset",
    },
    // The pops. Redis calls the two single key ones fast even though they cost a
    // logarithm, on the grounds that the logarithm is of a size a client chose.
    Spec {
        name: "zpopmin",
        arity: -2,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_DELETE_AT1],
        acl: AC_ZSET_WRITE_FAST,
        since: "5.0.0",
        complexity: "O(log(N)*M) with M the number of members popped",
        summary: "Take the lowest scoring members off and answer them.",
        group: "zset",
    },
    Spec {
        name: "zpopmax",
        arity: -2,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_DELETE_AT1],
        acl: AC_ZSET_WRITE_FAST,
        since: "5.0.0",
        complexity: "O(log(N)*M) with M the number of members popped",
        summary: "Take the highest scoring members off and answer them.",
        group: "zset",
    },
    // Keys behind a count, so the same three zeroes `LMPOP` has, and `write`
    // without `denyoom` because a pop cannot grow the keyspace.
    Spec {
        name: "zmpop",
        arity: -4,
        flags: &["write", "movablekeys"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[RW_ACCESS_DELETE_AT1_COUNTED],
        acl: AC_ZSET_WRITE_SLOW,
        since: "7.0.0",
        complexity: "O(K) + O(M*log(N)) with K the keys named and M the count popped",
        summary: "Pop from the first of several sorted sets that has anything in it.",
        group: "zset",
    },
    // The three that wait. `blocking` is what the dispatcher branches on, the
    // same as it is for the five list ones.
    Spec {
        name: "bzpopmin",
        arity: -3,
        flags: &["write", "blocking", "fast"],
        first_key: 1,
        last_key: -2,
        step: 1,
        keys: &[RW_ACCESS_DELETE_AT1_RM2_1_0],
        acl: AC_ZSET_BLOCKING_FAST,
        since: "5.0.0",
        complexity: "O(log(N)) with N the size of the sorted set that answers",
        summary: "Take the lowest scoring member off the first sorted set that has one, waiting if none does.",
        group: "zset",
    },
    Spec {
        name: "bzpopmax",
        arity: -3,
        flags: &["write", "blocking", "fast"],
        first_key: 1,
        last_key: -2,
        step: 1,
        keys: &[RW_ACCESS_DELETE_AT1_RM2_1_0],
        acl: AC_ZSET_BLOCKING_FAST,
        since: "5.0.0",
        complexity: "O(log(N)) with N the size of the sorted set that answers",
        summary: "Take the highest scoring member off the first sorted set that has one, waiting if none does.",
        group: "zset",
    },
    Spec {
        name: "bzmpop",
        arity: -5,
        flags: &["write", "blocking", "movablekeys"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[RW_ACCESS_DELETE_AT2_COUNTED],
        acl: AC_ZSET_BLOCKING_SLOW,
        since: "7.0.0",
        complexity: "O(K) + O(M*log(N)) with K the keys named and M the count popped",
        summary: "Pop from the first of several sorted sets that has anything, waiting if none does.",
        group: "zset",
    },
    // ----------------------------------------------------------------- geo
    Spec {
        name: "geoadd",
        arity: -5,
        flags: WRITE_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_AT1],
        acl: AC_GEO_WRITE,
        since: "3.2.0",
        complexity: "O(log(N)) per point added",
        summary: "Add places to a geo key, which is a sorted set of position hashes.",
        group: "geo",
    },
    Spec {
        name: "geopos",
        arity: -2,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_GEO_READ,
        since: "3.2.0",
        complexity: "O(1) per member asked about",
        summary: "Answer where each member is, as a longitude and a latitude.",
        group: "geo",
    },
    Spec {
        name: "geodist",
        arity: -4,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_GEO_READ,
        since: "3.2.0",
        complexity: "O(1)",
        summary: "Answer how far apart two members are, in the unit asked for.",
        group: "geo",
    },
    Spec {
        name: "geohash",
        arity: -2,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_GEO_READ,
        since: "3.2.0",
        complexity: "O(1) per member asked about",
        summary: "Answer each member's position as a standard eleven character geohash.",
        group: "geo",
    },
    Spec {
        name: "geosearch",
        arity: -7,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_GEO_READ,
        since: "6.2.0",
        complexity: "O(N+log(M)) with N the members in the boxes searched",
        summary: "Find the members inside a circle or a rectangle around a point.",
        group: "geo",
    },
    Spec {
        name: "geosearchstore",
        arity: -8,
        flags: WRITE_OOM,
        first_key: 1,
        last_key: 2,
        step: 1,
        keys: &[OW_UPDATE_AT1, RO_ACCESS_AT2],
        acl: AC_GEO_WRITE,
        since: "6.2.0",
        complexity: "O(N+log(M)) with N the members in the boxes searched",
        summary: "Run a search and write what it found into another key.",
        group: "geo",
    },
    Spec {
        name: "georadius",
        arity: -6,
        flags: WRITE_MOVABLE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1, GEORADIUS_STORE, GEORADIUS_STOREDIST],
        acl: AC_GEO_WRITE,
        since: "3.2.0",
        complexity: "O(N+log(M)) with N the members in the boxes searched",
        summary: "The older spelling of a circular search, which can also store.",
        group: "geo",
    },
    Spec {
        name: "georadius_ro",
        arity: -6,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_GEO_READ,
        since: "3.2.10",
        complexity: "O(N+log(M)) with N the members in the boxes searched",
        summary: "GEORADIUS without the store options, so a replica can serve it.",
        group: "geo",
    },
    Spec {
        name: "georadiusbymember",
        arity: -5,
        flags: WRITE_MOVABLE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1, BYMEMBER_STORE, BYMEMBER_STOREDIST],
        acl: AC_GEO_WRITE,
        since: "3.2.0",
        complexity: "O(N+log(M)) with N the members in the boxes searched",
        summary: "The same search centred on a member rather than on a point.",
        group: "geo",
    },
    Spec {
        name: "georadiusbymember_ro",
        arity: -5,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_GEO_READ,
        since: "3.2.10",
        complexity: "O(N+log(M)) with N the members in the boxes searched",
        summary: "GEORADIUSBYMEMBER without the store options.",
        group: "geo",
    },
    // --------------------------------------------------------------- graph
    Spec {
        name: "g.nadd",
        arity: -3,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_GRAPH_WRITE_FAST,
        since: "8.8.0",
        complexity: "O(N) with N the fields written",
        summary: "Write a node and its properties, creating it if it is new.",
        group: "graph",
    },
    Spec {
        name: "g.nget",
        arity: 3,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_GRAPH_READ_FAST,
        since: "8.8.0",
        complexity: "O(N) with N the fields on the node",
        summary: "Every property on a node.",
        group: "graph",
    },
    Spec {
        name: "g.ndel",
        arity: 3,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_GRAPH_WRITE_FAST,
        since: "8.8.0",
        complexity: "O(E) with E the edges on the node",
        summary: "Delete a node and every edge that touches it.",
        group: "graph",
    },
    Spec {
        name: "g.eadd",
        arity: -5,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_GRAPH_WRITE_FAST,
        since: "8.8.0",
        complexity: "O(D) with D the outgoing degree under the label",
        summary: "Write an edge and its properties, creating either end if it is new.",
        group: "graph",
    },
    Spec {
        name: "g.edel",
        arity: 5,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_GRAPH_WRITE_FAST,
        since: "8.8.0",
        complexity: "O(D) with D the outgoing degree under the label",
        summary: "Delete one edge between two nodes under a label.",
        group: "graph",
    },
    Spec {
        name: "g.out",
        arity: -4,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_GRAPH_READ_FAST,
        since: "8.8.0",
        complexity: "O(N) with N the page asked for",
        summary: "Outgoing neighbours under a label, a page at a time.",
        group: "graph",
    },
    Spec {
        name: "g.in",
        arity: -4,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_GRAPH_READ_FAST,
        since: "8.8.0",
        complexity: "O(N) with N the page asked for",
        summary: "Incoming neighbours under a label, a page at a time.",
        group: "graph",
    },
    Spec {
        name: "g.deg",
        arity: -4,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_GRAPH_READ_FAST,
        since: "8.8.0",
        complexity: "O(1)",
        summary: "How many edges a node has under a label.",
        group: "graph",
    },
    Spec {
        name: "g.neigh",
        arity: -4,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_GRAPH_READ_SLOW,
        since: "8.8.0",
        complexity: "O(V + E) over the ball the depth reaches",
        summary: "Everything reachable within a depth, each node once.",
        group: "graph",
    },
    Spec {
        name: "g.path",
        arity: -4,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_GRAPH_READ_SLOW,
        since: "8.8.0",
        complexity: "O(b^(d/2)) with b the branching factor and d the distance",
        summary: "A shortest path between two nodes, searched from both ends.",
        group: "graph",
    },
    // ---------------------------------------------------------------- json
    Spec {
        name: "json.set",
        arity: -4,
        flags: JSON_WRITE_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_JSON_WRITE,
        since: "1.0.0",
        complexity: "O(N) with N the size of the document",
        summary: "Set the value at a path, creating the document at the root.",
        group: "json",
    },
    Spec {
        name: "json.mset",
        arity: -4,
        flags: JSON_WRITE_OOM,
        first_key: 1,
        last_key: -1,
        step: 3,
        keys: &[RW_ACCESS_UPDATE_AT1_RM1_3_0],
        acl: AC_JSON_WRITE,
        since: "2.6.0",
        complexity: "O(K*N) with K the keys and N the size of each document",
        summary: "Set the value at a path in each of several documents.",
        group: "json",
    },
    Spec {
        name: "json.merge",
        arity: -4,
        flags: JSON_WRITE_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_JSON_WRITE,
        since: "2.6.0",
        complexity: "O(N) with N the size of the document",
        summary: "Apply an RFC 7386 merge patch at a path.",
        group: "json",
    },
    Spec {
        name: "json.get",
        arity: -2,
        flags: JSON_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_JSON_READ,
        since: "1.0.0",
        complexity: "O(N) with N the size of what the paths matched",
        summary: "The values one or more paths match, as JSON text.",
        group: "json",
    },
    Spec {
        name: "json.mget",
        arity: -3,
        flags: JSON_READ,
        first_key: 1,
        last_key: -2,
        step: 1,
        keys: &[RO_ACCESS_AT1_RM2_1_0],
        acl: AC_JSON_READ,
        since: "1.0.0",
        complexity: "O(K*N) with K the keys and N the size of each document",
        summary: "One path against several documents, one answer per key.",
        group: "json",
    },
    Spec {
        name: "json.del",
        arity: -2,
        flags: JSON_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_JSON_WRITE,
        since: "1.0.0",
        complexity: "O(N) with N the size of the document",
        summary: "Remove what a path matched, or the key when it is the root.",
        group: "json",
    },
    Spec {
        name: "json.forget",
        arity: -2,
        flags: JSON_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_JSON_WRITE,
        since: "1.0.0",
        complexity: "O(N) with N the size of the document",
        summary: "The same command as JSON.DEL, under its other name.",
        group: "json",
    },
    Spec {
        name: "json.type",
        arity: -2,
        flags: JSON_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_JSON_READ,
        since: "1.0.0",
        complexity: "O(N) with N the size of the document",
        summary: "The JSON type of what a path matched.",
        group: "json",
    },
    Spec {
        name: "json.toggle",
        arity: 3,
        flags: JSON_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_JSON_WRITE,
        since: "2.0.0",
        complexity: "O(N) with N the size of the document",
        summary: "Flip every boolean a path matched.",
        group: "json",
    },
    Spec {
        name: "json.clear",
        arity: -2,
        flags: JSON_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_JSON_WRITE,
        since: "2.0.0",
        complexity: "O(N) with N the size of the document",
        summary: "Empty the containers and zero the numbers a path matched.",
        group: "json",
    },
    Spec {
        name: "json.arrlen",
        arity: -2,
        flags: JSON_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_JSON_READ,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "How many elements are in the arrays a path matched.",
        group: "json",
    },
    Spec {
        name: "json.objlen",
        arity: -2,
        flags: JSON_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_JSON_READ,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "How many members are in the objects a path matched.",
        group: "json",
    },
    Spec {
        name: "json.strlen",
        arity: -2,
        flags: JSON_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_JSON_READ,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "How long the strings a path matched are, in bytes.",
        group: "json",
    },
    Spec {
        name: "json.objkeys",
        arity: -2,
        flags: JSON_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_JSON_READ,
        since: "1.0.0",
        complexity: "O(N) with N the number of members",
        summary: "The keys of the objects a path matched.",
        group: "json",
    },
    Spec {
        name: "json.arrappend",
        arity: -3,
        flags: JSON_WRITE_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_JSON_WRITE,
        since: "1.0.0",
        complexity: "O(N) with N the size of the document",
        summary: "Add values to the end of the arrays a path matched.",
        group: "json",
    },
    Spec {
        name: "json.arrinsert",
        arity: -5,
        flags: JSON_WRITE_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_JSON_WRITE,
        since: "1.0.0",
        complexity: "O(N) with N the size of the document",
        summary: "Put values into the arrays a path matched, at an index.",
        group: "json",
    },
    Spec {
        name: "json.arrtrim",
        arity: 5,
        flags: JSON_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_JSON_WRITE,
        since: "1.0.0",
        complexity: "O(N) with N the size of the document",
        summary: "Keep only a run of the arrays a path matched.",
        group: "json",
    },
    Spec {
        name: "json.arrpop",
        arity: -2,
        flags: JSON_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_JSON_WRITE,
        since: "1.0.0",
        complexity: "O(N) with N the size of the document",
        summary: "Take one element out of the arrays a path matched.",
        group: "json",
    },
    Spec {
        name: "json.arrindex",
        arity: -4,
        flags: JSON_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_JSON_READ,
        since: "1.0.0",
        complexity: "O(N) with N the number of elements",
        summary: "Where a value first sits in the arrays a path matched.",
        group: "json",
    },
    Spec {
        name: "json.numincrby",
        arity: 4,
        flags: JSON_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_JSON_WRITE,
        since: "1.0.0",
        complexity: "O(N) with N the size of the document",
        summary: "Add to every number a path matched.",
        group: "json",
    },
    Spec {
        name: "json.nummultby",
        arity: 4,
        flags: JSON_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_JSON_WRITE,
        since: "1.0.0",
        complexity: "O(N) with N the size of the document",
        summary: "Multiply every number a path matched.",
        group: "json",
    },
    Spec {
        name: "json.numpowby",
        arity: 4,
        flags: JSON_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_JSON_WRITE,
        since: "1.0.0",
        complexity: "O(N) with N the size of the document",
        summary: "Raise every number a path matched to a power.",
        group: "json",
    },
    Spec {
        name: "json.strappend",
        arity: -3,
        flags: JSON_WRITE_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_JSON_WRITE,
        since: "1.0.0",
        complexity: "O(N) with N the size of the document",
        summary: "Add to the end of every string a path matched.",
        group: "json",
    },
    Spec {
        name: "json.resp",
        arity: -2,
        flags: JSON_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_JSON_READ,
        since: "1.0.0",
        complexity: "O(N) with N the size of what the path matched",
        summary: "What a path matched, as RESP types rather than as JSON text.",
        group: "json",
    },
    Spec {
        name: "json.debug",
        arity: -2,
        flags: JSON_READ_MOVABLE,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_JSON_READ,
        since: "1.0.0",
        complexity: "O(N) with N the size of what the path matched",
        summary: "How much memory a document takes, and the help for that.",
        group: "json",
    },
    // -------------------------------------------------------------- vector
    Spec {
        name: "VADD",
        arity: -5,
        flags: VECTOR_WRITE_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_NONE,
        since: "8.0.0",
        complexity: "O(P*D) with P the partitions probed and D the dimension",
        summary: "Add a vector to a vector set under an element name.",
        group: "vector",
    },
    Spec {
        name: "VSIM",
        arity: -4,
        flags: VECTOR_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_NONE,
        since: "8.0.0",
        complexity: "O(P*D) with P the partitions probed and D the dimension",
        summary: "The elements nearest a vector or nearest another element.",
        group: "vector",
    },
    Spec {
        name: "VREM",
        arity: 3,
        flags: VECTOR_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_NONE,
        since: "8.0.0",
        complexity: "O(1)",
        summary: "Remove an element and its vector from a vector set.",
        group: "vector",
    },
    Spec {
        name: "VCARD",
        arity: 2,
        flags: VECTOR_READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_NONE,
        since: "8.0.0",
        complexity: "O(1)",
        summary: "How many elements a vector set holds.",
        group: "vector",
    },
    Spec {
        name: "VDIM",
        arity: 2,
        flags: VECTOR_READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_NONE,
        since: "8.0.0",
        complexity: "O(1)",
        summary: "How many dimensions the vectors in a vector set have.",
        group: "vector",
    },
    Spec {
        name: "VEMB",
        arity: -3,
        flags: VECTOR_READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_NONE,
        since: "8.0.0",
        complexity: "O(D) with D the dimension",
        summary: "The vector an element went in with.",
        group: "vector",
    },
    Spec {
        name: "VINFO",
        arity: 2,
        flags: VECTOR_READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_NONE,
        since: "8.0.0",
        complexity: "O(N) with N the elements, for the attribute count",
        summary: "What a vector set is and how its index is tuned.",
        group: "vector",
    },
    Spec {
        name: "VISMEMBER",
        arity: 3,
        flags: VECTOR_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_NONE,
        since: "8.0.0",
        complexity: "O(1)",
        summary: "Whether an element is in a vector set.",
        group: "vector",
    },
    Spec {
        name: "VRANDMEMBER",
        arity: -2,
        flags: VECTOR_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_NONE,
        since: "8.0.0",
        complexity: "O(1) for one, O(N) for a positive count",
        summary: "Random elements of a vector set.",
        group: "vector",
    },
    Spec {
        name: "VLINKS",
        arity: -3,
        flags: VECTOR_READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_NONE,
        since: "8.0.0",
        complexity: "O(P*D) with P the partitions probed and D the dimension",
        summary: "The elements an element is stored next to.",
        group: "vector",
    },
    Spec {
        name: "VSETATTR",
        arity: 4,
        flags: VECTOR_WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_NONE,
        since: "8.0.0",
        complexity: "O(1)",
        summary: "Set the attribute string on an element, or clear it.",
        group: "vector",
    },
    Spec {
        name: "VGETATTR",
        arity: 3,
        flags: VECTOR_READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_NONE,
        since: "8.0.0",
        complexity: "O(1)",
        summary: "The attribute string on an element.",
        group: "vector",
    },
    Spec {
        name: "VRANGE",
        arity: -4,
        flags: VECTOR_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_NONE,
        since: "8.4.0",
        complexity: "O(N log N) with N the elements the range covers",
        summary: "The elements of a vector set whose names fall in a range.",
        group: "vector",
    },
    // -------------------------------------------------------------- search
    //
    // Most of these carry no key spec, and the three zeros are the module's own
    // answer rather than a gap here. An index name is not a key: it is not in
    // the keyspace, `TYPE` has nothing to say about it, and a cluster client has
    // nothing to route on. The suggestion family and four of the five deprecated
    // document commands do name real keys and say so. `FT.MGET` is the odd one:
    // it names as many keys as a client cares to send and reports none of them,
    // which is the module's answer and is copied rather than tidied up.
    Spec {
        name: "FT.CREATE",
        arity: -5,
        flags: SEARCH_WRITE_OOM,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH,
        since: "1.0.0",
        complexity: "O(K) with K the fields declared, plus O(N) over the keyspace when the initial scan runs",
        summary: "Create an index over the keys with a prefix, with the given schema.",
        group: "search",
    },
    Spec {
        name: "FT._CREATEIFNX",
        arity: -5,
        flags: SEARCH_WRITE_OOM,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH,
        since: "1.0.0",
        complexity: "O(K) with K the fields declared, plus O(N) over the keyspace when the initial scan runs",
        summary: "Create an index, and say nothing if one of that name is already there.",
        group: "search",
    },
    Spec {
        name: "FT.ALTER",
        arity: -6,
        flags: SEARCH_WRITE_OOM,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH,
        since: "1.0.0",
        complexity: "O(N) over the keys the index follows, when the fields are backfilled",
        summary: "Add fields to an index's schema.",
        group: "search",
    },
    Spec {
        name: "FT._ALTERIFNX",
        arity: -6,
        flags: SEARCH_WRITE_OOM,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH,
        since: "1.0.0",
        complexity: "O(N) over the keys the index follows, when the fields are backfilled",
        summary: "Add fields to a schema, and say nothing about the ones already there.",
        group: "search",
    },
    Spec {
        name: "FT.DROPINDEX",
        arity: -2,
        flags: SEARCH_WRITE,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH_DROP,
        since: "2.0.0",
        complexity: "O(1), or O(N) over the documents when DD is given",
        summary: "Take an index away, and its documents with it when DD is given.",
        group: "search",
    },
    Spec {
        name: "FT._DROPINDEXIFX",
        arity: -2,
        flags: SEARCH_WRITE,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH_DROP,
        since: "2.0.0",
        complexity: "O(1), or O(N) over the documents when DD is given",
        summary: "Take an index away, and say nothing when there is none of that name.",
        group: "search",
    },
    Spec {
        name: "FT.DROP",
        arity: -1,
        flags: SEARCH_WRITE,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH_DROP,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Take an index away. Deprecated, and FT.DROPINDEX is the name to use.",
        group: "search",
    },
    Spec {
        name: "FT._DROPIFX",
        arity: -1,
        flags: SEARCH_WRITE,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH_WRITE,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Take an index away and say nothing when there is none. Deprecated.",
        group: "search",
    },
    Spec {
        name: "FT.INFO",
        arity: 2,
        flags: SEARCH_READ,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Everything the server knows about one index.",
        group: "search",
    },
    Spec {
        name: "FT._LIST",
        arity: -1,
        flags: SEARCH_READ,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH_LIST,
        since: "2.0.0",
        complexity: "O(N) with N the indexes on the server",
        summary: "Every index on the server, by name.",
        group: "search",
    },
    Spec {
        name: "FT.CONFIG",
        arity: -2,
        flags: SEARCH_READ,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH_ADMIN,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Read, write or describe the search module's settings.",
        group: "search",
    },
    Spec {
        name: "_FT.DEBUG",
        arity: -2,
        flags: SEARCH_READ,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH_DEBUG,
        since: "1.0.0",
        complexity: "O(N) with N the size of whatever is being dumped.",
        summary: "Read an index's own structures back.",
        group: "search",
    },
    Spec {
        name: "FT.ALIASADD",
        arity: 3,
        flags: SEARCH_WRITE_OOM,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Point another name at an index.",
        group: "search",
    },
    Spec {
        name: "FT._ALIASADDIFNX",
        arity: 3,
        flags: SEARCH_WRITE_OOM,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Point another name at an index, and say nothing if it is taken.",
        group: "search",
    },
    Spec {
        name: "FT.ALIASDEL",
        arity: 2,
        flags: SEARCH_WRITE,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Take an alias away.",
        group: "search",
    },
    Spec {
        name: "FT._ALIASDELIFX",
        arity: 2,
        flags: SEARCH_WRITE,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Take an alias away, and say nothing when there is none.",
        group: "search",
    },
    Spec {
        name: "FT.ALIASUPDATE",
        arity: 3,
        flags: SEARCH_WRITE_OOM,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Move an alias to another index, adding it when it was not there.",
        group: "search",
    },
    Spec {
        name: "FT.ALIASLIST",
        arity: 2,
        flags: SEARCH_READ,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH,
        since: "8.10.0",
        complexity: "O(N) with N the aliases pointing at the index",
        summary: "The aliases pointing at one index.",
        group: "search",
    },
    Spec {
        name: "FT.SEARCH",
        arity: -3,
        flags: SEARCH_READ,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH,
        since: "1.0.0",
        complexity: "O(N) with N the documents the query matches",
        summary: "The documents a query answers, with their fields.",
        group: "search",
    },
    Spec {
        name: "FT.AGGREGATE",
        arity: -3,
        flags: SEARCH_READ,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH,
        since: "1.1.0",
        complexity: "O(N) with N the documents the query matches",
        summary: "The properties a query answers, run through a pipeline.",
        group: "search",
    },
    Spec {
        name: "FT.HYBRID",
        arity: -7,
        flags: SEARCH_READ,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH,
        since: "8.4.0",
        complexity: "O(N) with N the documents either branch matches",
        summary: "A text query and a vector query over one index, folded into one ranking.",
        group: "search",
    },
    Spec {
        name: "FT.PROFILE",
        arity: -5,
        flags: SEARCH_READ,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH_READ,
        since: "2.2.0",
        complexity: "O(N) with N the documents the query matches",
        summary: "A search or an aggregation with the working shown.",
        group: "search",
    },
    Spec {
        name: "FT.CURSOR",
        arity: -2,
        flags: SEARCH_READ,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH,
        since: "1.1.0",
        complexity: "O(1)",
        summary: "The next chunk of an answer a cursor was left open on.",
        group: "search",
    },
    Spec {
        name: "FT.EXPLAIN",
        arity: -3,
        flags: SEARCH_READ,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "The tree a query parses into, as text.",
        group: "search",
    },
    Spec {
        name: "FT.EXPLAINCLI",
        arity: -3,
        flags: SEARCH_READ,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "The tree a query parses into, one line per reply element.",
        group: "search",
    },
    Spec {
        name: "FT.TAGVALS",
        arity: 3,
        flags: SEARCH_READ,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH_TAGS,
        since: "1.0.0",
        complexity: "O(N)",
        summary: "Every distinct value a tag field holds.",
        group: "search",
    },
    Spec {
        name: "FT.DICTADD",
        arity: -3,
        flags: SEARCH_WRITE_OOM,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH,
        since: "1.4.0",
        complexity: "O(1)",
        summary: "Put terms into a dictionary, making it if it is not there.",
        group: "search",
    },
    Spec {
        name: "FT.DICTDEL",
        arity: -3,
        flags: SEARCH_WRITE,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH,
        since: "1.4.0",
        complexity: "O(1)",
        summary: "Take terms back out of a dictionary.",
        group: "search",
    },
    Spec {
        name: "FT.DICTDUMP",
        arity: 2,
        flags: SEARCH_READ,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH,
        since: "1.4.0",
        complexity: "O(N)",
        summary: "Every term in a dictionary.",
        group: "search",
    },
    Spec {
        name: "FT.SYNUPDATE",
        arity: -4,
        flags: SEARCH_WRITE_OOM,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH,
        since: "1.2.0",
        complexity: "O(1)",
        summary: "Put terms in a synonym group.",
        group: "search",
    },
    Spec {
        name: "FT.SYNDUMP",
        arity: 2,
        flags: SEARCH_READ,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH,
        since: "1.2.0",
        complexity: "O(1)",
        summary: "Every term an index treats as a synonym, and the groups it is in.",
        group: "search",
    },
    Spec {
        name: "FT.SPELLCHECK",
        arity: -3,
        flags: SEARCH_READ,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH,
        since: "1.4.0",
        complexity: "O(1)",
        summary: "Suggestions for the words in a query the index does not hold.",
        group: "search",
    },
    Spec {
        name: "FT.ADD",
        arity: -1,
        flags: SEARCH_WRITE_OOM,
        first_key: 2,
        last_key: 2,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT2],
        acl: AC_SEARCH_WRITE,
        since: "1.0.0",
        complexity: "O(N) with N the tokens in the document",
        summary: "Write a hash and record what the index should think it is worth.",
        group: "search",
    },
    Spec {
        name: "FT.SAFEADD",
        arity: -1,
        flags: SEARCH_WRITE_OOM,
        first_key: 2,
        last_key: 2,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT2],
        acl: AC_SEARCH_WRITE,
        since: "1.0.0",
        complexity: "O(N) with N the tokens in the document",
        summary: "The same write, under the name a cluster client used to send.",
        group: "search",
    },
    Spec {
        name: "FT.GET",
        arity: -1,
        flags: SEARCH_READ,
        first_key: 2,
        last_key: 2,
        step: 1,
        keys: &[RO_ACCESS_AT2],
        acl: AC_SEARCH_READ,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "The hash under a key, when the index is holding it.",
        group: "search",
    },
    Spec {
        name: "FT.MGET",
        arity: -1,
        flags: SEARCH_READ,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SEARCH_READ,
        since: "1.0.0",
        complexity: "O(N) with N the keys asked about",
        summary: "The same, for as many keys as were named.",
        group: "search",
    },
    Spec {
        name: "FT.DEL",
        arity: -1,
        flags: SEARCH_WRITE,
        first_key: 2,
        last_key: 2,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT2],
        acl: AC_SEARCH_WRITE,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Delete a key, with an index name in front of it.",
        group: "search",
    },
    Spec {
        name: "FT.SUGADD",
        arity: -4,
        flags: SEARCH_WRITE_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_SEARCH_WRITE,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Put a suggestion in a dictionary, or change the one that is there.",
        group: "search",
    },
    Spec {
        name: "FT.SUGGET",
        arity: -3,
        flags: SEARCH_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_SEARCH_READ,
        since: "1.0.0",
        complexity: "O(N) with N the suggestions the prefix reaches",
        summary: "The best suggestions starting with a prefix.",
        group: "search",
    },
    Spec {
        name: "FT.SUGDEL",
        arity: 3,
        flags: SEARCH_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_SEARCH_WRITE,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Take a suggestion out of a dictionary.",
        group: "search",
    },
    Spec {
        name: "FT.SUGLEN",
        arity: 2,
        flags: SEARCH_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_SEARCH_READ,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "How many suggestions a dictionary holds.",
        group: "search",
    },
    // --------------------------------------------------------------- bloom
    Spec {
        name: "bf.reserve",
        arity: -4,
        flags: BLOOM_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_BLOOM_WRITE_FAST,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Make an empty filter with a given capacity and error rate.",
        group: "bloom",
    },
    Spec {
        name: "bf.add",
        arity: 3,
        flags: BLOOM_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_BLOOM_WRITE,
        since: "1.0.0",
        complexity: "O(K) with K the number of hash functions",
        summary: "Add an item, making the filter if the key is free.",
        group: "bloom",
    },
    Spec {
        name: "bf.madd",
        arity: -3,
        flags: BLOOM_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_BLOOM_WRITE,
        since: "1.0.0",
        complexity: "O(N * K) with N the number of items",
        summary: "Add several items, making the filter if the key is free.",
        group: "bloom",
    },
    Spec {
        name: "bf.insert",
        arity: -4,
        flags: BLOOM_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_BLOOM_WRITE,
        since: "1.0.0",
        complexity: "O(N * K) with N the number of items",
        summary: "Add several items to a filter described in the same command.",
        group: "bloom",
    },
    Spec {
        name: "bf.exists",
        arity: 3,
        flags: BLOOM_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_BLOOM_READ,
        since: "1.0.0",
        complexity: "O(K) with K the number of hash functions",
        summary: "Whether an item is probably in the filter.",
        group: "bloom",
    },
    Spec {
        name: "bf.mexists",
        arity: -3,
        flags: BLOOM_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_BLOOM_READ,
        since: "1.0.0",
        complexity: "O(N * K) with N the number of items",
        summary: "Whether each of several items is probably in the filter.",
        group: "bloom",
    },
    Spec {
        name: "bf.scandump",
        arity: 3,
        flags: BLOOM_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_BLOOM_READ,
        since: "1.0.0",
        complexity: "O(N) with N the size of the chunk",
        summary: "One chunk of the filter, to be replayed into BF.LOADCHUNK.",
        group: "bloom",
    },
    Spec {
        name: "bf.loadchunk",
        arity: 4,
        flags: BLOOM_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_BLOOM_WRITE,
        since: "1.0.0",
        complexity: "O(N) with N the size of the chunk",
        summary: "Put back a chunk that BF.SCANDUMP handed out.",
        group: "bloom",
    },
    Spec {
        name: "bf.info",
        arity: -2,
        flags: BLOOM_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_BLOOM_READ_FAST,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "The shape of the filter, or one field of it.",
        group: "bloom",
    },
    Spec {
        name: "bf.card",
        arity: 2,
        flags: BLOOM_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_BLOOM_READ_FAST,
        since: "2.4.4",
        complexity: "O(1)",
        summary: "How many items were added to the filter.",
        group: "bloom",
    },
    Spec {
        name: "bf.debug",
        arity: 2,
        flags: BLOOM_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_BLOOM_READ,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "The chain and a line for each of its links.",
        group: "bloom",
    },
    // -------------------------------------------------------------- cuckoo
    Spec {
        name: "cf.reserve",
        arity: -3,
        flags: CUCKOO_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_CUCKOO_WRITE_FAST,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Make an empty filter with a given capacity.",
        group: "cuckoo",
    },
    Spec {
        name: "cf.add",
        arity: 3,
        flags: CUCKOO_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_CUCKOO_WRITE,
        since: "1.0.0",
        complexity: "O(1) amortised, O(N) when the chain has to grow",
        summary: "Add an item, making the filter if the key is free.",
        group: "cuckoo",
    },
    Spec {
        name: "cf.addnx",
        arity: 3,
        flags: CUCKOO_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_CUCKOO_WRITE,
        since: "1.0.0",
        complexity: "O(1) amortised, O(N) when the chain has to grow",
        summary: "Add an item unless the filter already has it.",
        group: "cuckoo",
    },
    Spec {
        name: "cf.insert",
        arity: -4,
        flags: CUCKOO_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_CUCKOO_WRITE,
        since: "1.0.0",
        complexity: "O(N) with N the number of items",
        summary: "Add several items to a filter described in the same command.",
        group: "cuckoo",
    },
    Spec {
        name: "cf.insertnx",
        arity: -4,
        flags: CUCKOO_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_CUCKOO_WRITE,
        since: "1.0.0",
        complexity: "O(N) with N the number of items",
        summary: "Add several items the filter does not already have.",
        group: "cuckoo",
    },
    Spec {
        name: "cf.exists",
        arity: 3,
        flags: CUCKOO_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_CUCKOO_READ,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Whether an item is probably in the filter.",
        group: "cuckoo",
    },
    Spec {
        name: "cf.mexists",
        arity: -3,
        flags: CUCKOO_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_CUCKOO_READ,
        since: "1.0.0",
        complexity: "O(N) with N the number of items",
        summary: "Whether each of several items is probably in the filter.",
        group: "cuckoo",
    },
    Spec {
        name: "cf.count",
        arity: 3,
        flags: CUCKOO_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_CUCKOO_READ,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "How many copies of an item the filter thinks it has.",
        group: "cuckoo",
    },
    Spec {
        name: "cf.del",
        arity: 3,
        flags: CUCKOO_DELETE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_CUCKOO_WRITE,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Take one copy of an item out of the filter.",
        group: "cuckoo",
    },
    Spec {
        name: "cf.scandump",
        arity: 3,
        flags: CUCKOO_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_CUCKOO_READ,
        since: "1.0.0",
        complexity: "O(N) with N the size of the chunk",
        summary: "One chunk of the filter, to be replayed into CF.LOADCHUNK.",
        group: "cuckoo",
    },
    Spec {
        name: "cf.loadchunk",
        arity: 4,
        flags: CUCKOO_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_CUCKOO_WRITE,
        since: "1.0.0",
        complexity: "O(N) with N the size of the chunk",
        summary: "Put back a chunk that CF.SCANDUMP handed out.",
        group: "cuckoo",
    },
    Spec {
        name: "cf.info",
        arity: 2,
        flags: CUCKOO_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_CUCKOO_READ_FAST,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "The shape of the chain.",
        group: "cuckoo",
    },
    Spec {
        name: "cf.debug",
        arity: 2,
        flags: CUCKOO_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_CUCKOO_READ,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "The chain's geometry on one line.",
        group: "cuckoo",
    },
    Spec {
        name: "cf.compact",
        arity: -1,
        flags: CUCKOO_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_CUCKOO_READ,
        since: "1.0.0",
        complexity: "O(N) with N the number of items in the newer filters",
        summary: "Pull the newer filters down into the older ones.",
        group: "cuckoo",
    },
    // ----------------------------------------------------------------- cms
    Spec {
        name: "cms.initbydim",
        arity: 4,
        flags: CMS_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_CMS_WRITE_FAST,
        since: "2.0.0",
        complexity: "O(1)",
        summary: "Make an empty sketch of a given width and depth.",
        group: "cms",
    },
    Spec {
        name: "cms.initbyprob",
        arity: 4,
        flags: CMS_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_CMS_WRITE_FAST,
        since: "2.0.0",
        complexity: "O(1)",
        summary: "Make an empty sketch wide enough for a stated tolerance.",
        group: "cms",
    },
    Spec {
        name: "cms.incrby",
        arity: -4,
        flags: CMS_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_CMS_WRITE,
        since: "2.0.0",
        complexity: "O(N) with N the number of items",
        summary: "Add to the count of one or more items.",
        group: "cms",
    },
    Spec {
        name: "cms.query",
        arity: -3,
        flags: CMS_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_CMS_READ,
        since: "2.0.0",
        complexity: "O(N) with N the number of items",
        summary: "How many times the sketch has seen each item.",
        group: "cms",
    },
    Spec {
        name: "cms.merge",
        arity: -4,
        flags: CMS_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_CMS_WRITE,
        since: "2.0.0",
        complexity: "O(N * M) with N the sources and M the counters in one",
        summary: "Replace a sketch with the weighted sum of others.",
        group: "cms",
    },
    Spec {
        name: "cms.info",
        arity: 2,
        flags: CMS_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_CMS_READ_FAST,
        since: "2.0.0",
        complexity: "O(1)",
        summary: "The width, the depth and everything ever added.",
        group: "cms",
    },
    // ---------------------------------------------------------------- topk
    Spec {
        name: "topk.reserve",
        arity: -3,
        flags: TOPK_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_TOPK_WRITE_FAST,
        since: "2.0.0",
        complexity: "O(1)",
        summary: "Make an empty sketch that keeps the k commonest items.",
        group: "topk",
    },
    Spec {
        name: "topk.add",
        arity: -3,
        flags: TOPK_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_TOPK_WRITE,
        since: "2.0.0",
        complexity: "O(N * K) with N the items and K the depth",
        summary: "Count one occurrence of each item.",
        group: "topk",
    },
    Spec {
        name: "topk.incrby",
        arity: -4,
        flags: TOPK_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_TOPK_WRITE,
        since: "2.0.0",
        complexity: "O(N * K) with N the items and K the depth",
        summary: "Count a stated number of occurrences of each item.",
        group: "topk",
    },
    Spec {
        name: "topk.query",
        arity: -3,
        flags: TOPK_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_TOPK_READ,
        since: "2.0.0",
        complexity: "O(N * K) with N the items and K the kept count",
        summary: "Whether each item is one of the ones being kept.",
        group: "topk",
    },
    Spec {
        name: "topk.count",
        arity: -3,
        flags: TOPK_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_TOPK_READ,
        since: "2.0.0",
        complexity: "O(N * K) with N the items and K the depth",
        summary: "How many times the sketch thinks it has seen each item.",
        group: "topk",
    },
    Spec {
        name: "topk.list",
        arity: -2,
        flags: TOPK_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_TOPK_READ,
        since: "2.0.0",
        complexity: "O(K log K) with K the kept count",
        summary: "The kept items, heaviest first.",
        group: "topk",
    },
    Spec {
        name: "topk.info",
        arity: 2,
        flags: TOPK_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_TOPK_READ_FAST,
        since: "2.0.0",
        complexity: "O(1)",
        summary: "The four numbers the sketch was made with.",
        group: "topk",
    },
    // ------------------------------------------------------------- tdigest
    Spec {
        name: "tdigest.create",
        arity: -2,
        flags: TDIGEST_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_TDIGEST_WRITE_FAST,
        since: "2.4.0",
        complexity: "O(1)",
        summary: "Make an empty digest of a stated compression.",
        group: "tdigest",
    },
    Spec {
        name: "tdigest.reset",
        arity: 2,
        flags: TDIGEST_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_TDIGEST_WRITE_FAST,
        since: "2.4.0",
        complexity: "O(1)",
        summary: "Throw away every sample and keep the shape.",
        group: "tdigest",
    },
    Spec {
        name: "tdigest.add",
        arity: -3,
        flags: TDIGEST_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_TDIGEST_WRITE,
        since: "2.4.0",
        complexity: "O(N) with N the number of samples",
        summary: "Add samples of weight one each.",
        group: "tdigest",
    },
    Spec {
        name: "tdigest.merge",
        arity: -4,
        flags: TDIGEST_MERGE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1, RO_ACCESS_AT2_COUNTED],
        acl: AC_TDIGEST_WRITE,
        since: "2.4.0",
        complexity: "O(N) with N the number of centroids in the inputs",
        summary: "Fold digests together into one.",
        group: "tdigest",
    },
    Spec {
        name: "tdigest.min",
        arity: 2,
        flags: TDIGEST_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_TDIGEST_READ_FAST,
        since: "2.4.0",
        complexity: "O(1)",
        summary: "The smallest sample ever added.",
        group: "tdigest",
    },
    Spec {
        name: "tdigest.max",
        arity: 2,
        flags: TDIGEST_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_TDIGEST_READ_FAST,
        since: "2.4.0",
        complexity: "O(1)",
        summary: "The largest sample ever added.",
        group: "tdigest",
    },
    Spec {
        name: "tdigest.quantile",
        arity: -3,
        flags: TDIGEST_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_TDIGEST_READ_FAST,
        since: "2.4.0",
        complexity: "O(N) with N the number of centroids",
        summary: "The value each fraction of the samples falls under.",
        group: "tdigest",
    },
    Spec {
        name: "tdigest.cdf",
        arity: -3,
        flags: TDIGEST_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_TDIGEST_READ_FAST,
        since: "2.4.0",
        complexity: "O(N) with N the number of centroids",
        summary: "The fraction of the samples at or below each value.",
        group: "tdigest",
    },
    Spec {
        name: "tdigest.trimmed_mean",
        arity: 4,
        flags: TDIGEST_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_TDIGEST_READ,
        since: "2.4.0",
        complexity: "O(N) with N the number of centroids",
        summary: "The mean of what is left once both tails are cut.",
        group: "tdigest",
    },
    Spec {
        name: "tdigest.rank",
        arity: -3,
        flags: TDIGEST_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_TDIGEST_READ_FAST,
        since: "2.4.0",
        complexity: "O(N) with N the number of centroids",
        summary: "How many samples each value is above.",
        group: "tdigest",
    },
    Spec {
        name: "tdigest.revrank",
        arity: -3,
        flags: TDIGEST_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_TDIGEST_READ_FAST,
        since: "2.4.0",
        complexity: "O(N) with N the number of centroids",
        summary: "How many samples each value is below.",
        group: "tdigest",
    },
    Spec {
        name: "tdigest.byrank",
        arity: -3,
        flags: TDIGEST_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_TDIGEST_READ_FAST,
        since: "2.4.0",
        complexity: "O(N) with N the number of centroids",
        summary: "The value at each rank counting up from the smallest.",
        group: "tdigest",
    },
    Spec {
        name: "tdigest.byrevrank",
        arity: -3,
        flags: TDIGEST_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_TDIGEST_READ_FAST,
        since: "2.4.0",
        complexity: "O(N) with N the number of centroids",
        summary: "The value at each rank counting down from the largest.",
        group: "tdigest",
    },
    Spec {
        name: "tdigest.info",
        arity: 2,
        flags: TDIGEST_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_TDIGEST_READ_FAST,
        since: "2.4.0",
        complexity: "O(1)",
        summary: "The nine numbers the digest keeps about itself.",
        group: "tdigest",
    },
    // ------------------------------------------------------------------ ts
    Spec {
        name: "ts.create",
        arity: -2,
        flags: TS_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_TS_WRITE_FAST,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Make an empty series and say how it should behave.",
        group: "ts",
    },
    Spec {
        name: "ts.alter",
        arity: -2,
        flags: TS_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_TS_WRITE,
        since: "1.0.0",
        complexity: "O(N) with N the labels being set",
        summary: "Change how a series behaves, leaving what was not named alone.",
        group: "ts",
    },
    Spec {
        name: "ts.add",
        arity: -4,
        flags: TS_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_TS_WRITE,
        since: "1.0.0",
        complexity: "O(M) with M the samples in the chunk a backfill lands in",
        summary: "Put a sample in, making the series if it is not there.",
        group: "ts",
    },
    Spec {
        name: "ts.madd",
        arity: -4,
        flags: TS_WRITE,
        first_key: 1,
        last_key: -1,
        step: 3,
        keys: &[RW_ACCESS_UPDATE_AT1_RM1_3_0],
        acl: AC_TS_WRITE,
        since: "1.0.0",
        complexity: "O(N * M) with N the samples given",
        summary: "Put a sample in each of several series.",
        group: "ts",
    },
    Spec {
        name: "ts.incrby",
        arity: -3,
        flags: TS_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_TS_WRITE,
        since: "1.0.0",
        complexity: "O(M) with M the samples in the last chunk",
        summary: "Add to the newest value and store the answer.",
        group: "ts",
    },
    Spec {
        name: "ts.decrby",
        arity: -3,
        flags: TS_WRITE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_TS_WRITE,
        since: "1.0.0",
        complexity: "O(M) with M the samples in the last chunk",
        summary: "Take away from the newest value and store the answer.",
        group: "ts",
    },
    Spec {
        name: "ts.del",
        arity: 4,
        flags: TS_DELETE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_TS_WRITE,
        since: "1.6.0",
        complexity: "O(N) with N the samples in the span",
        summary: "Take out every sample between two timestamps.",
        group: "ts",
    },
    Spec {
        name: "ts.get",
        arity: -2,
        flags: TS_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_TS_READ_FAST,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "The newest sample in a series.",
        group: "ts",
    },
    Spec {
        name: "ts.info",
        arity: -2,
        flags: TS_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_TS_READ_FAST,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "The fourteen things a series says about itself.",
        group: "ts",
    },
    Spec {
        name: "ts.range",
        arity: -4,
        flags: TS_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_TS_READ,
        since: "1.0.0",
        complexity: "O(n/m+k) with n the samples, m the chunk size and k the samples in the span",
        summary: "The samples in a span, oldest first, in buckets if asked for.",
        group: "ts",
    },
    Spec {
        name: "ts.revrange",
        arity: -4,
        flags: TS_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_TS_READ,
        since: "1.4.0",
        complexity: "O(n/m+k) with n the samples, m the chunk size and k the samples in the span",
        summary: "The same span, newest first.",
        group: "ts",
    },
    Spec {
        name: "ts.nrange",
        arity: -5,
        flags: TS_READ_MOVABLE,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[RO_ACCESS_AT1_COUNTED],
        acl: AC_TS_READ,
        since: "8.10.0",
        complexity: "O(n/m+k) with n the samples, m the chunk size and k the samples in the span",
        summary: "The same span out of several series, lined up on the timestamps.",
        group: "ts",
    },
    Spec {
        name: "ts.nrevrange",
        arity: -5,
        flags: TS_READ_MOVABLE,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[RO_ACCESS_AT1_COUNTED],
        acl: AC_TS_READ,
        since: "8.10.0",
        complexity: "O(n/m+k) with n the samples, m the chunk size and k the samples in the span",
        summary: "The same rows, newest first.",
        group: "ts",
    },
    Spec {
        name: "ts.read",
        arity: -3,
        flags: TS_READ,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_TS_READ,
        since: "8.10.0",
        complexity: "O(n/m+k) with n the samples, m the chunk size and k the samples answered",
        summary: "Every sample from a timestamp to the end of the series.",
        group: "ts",
    },
    Spec {
        name: "ts.queryindex",
        arity: -2,
        flags: TS_READ,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_TS_READ,
        since: "1.0.0",
        complexity: "O(n) with n the series in the keyspace",
        summary: "The series a filter list takes, by key name.",
        group: "ts",
    },
    Spec {
        name: "ts.querylabels",
        arity: -2,
        flags: TS_READ,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_TS_READ,
        since: "8.10.0",
        complexity: "O(n) with n the series in the keyspace",
        summary: "The label names in use, or the values one of them takes.",
        group: "ts",
    },
    Spec {
        name: "ts.mget",
        arity: -3,
        flags: TS_READ,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_TS_READ,
        since: "1.0.0",
        complexity: "O(n) with n the series in the keyspace",
        summary: "The newest sample of every series a filter list takes.",
        group: "ts",
    },
    Spec {
        name: "ts.mrange",
        arity: -4,
        flags: TS_READ,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_TS_READ,
        since: "1.0.0",
        complexity: "O(n) with n the series in the keyspace",
        summary: "A span out of every series a filter list takes, oldest first.",
        group: "ts",
    },
    Spec {
        name: "ts.mrevrange",
        arity: -4,
        flags: TS_READ,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_TS_READ,
        since: "1.4.0",
        complexity: "O(n) with n the series in the keyspace",
        summary: "The same spans, newest first.",
        group: "ts",
    },
    Spec {
        name: "ts.createrule",
        arity: -5,
        flags: TS_RULE,
        first_key: 1,
        last_key: 2,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1_R1_1_0],
        acl: AC_TS_WRITE,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Fold one series into another as it is written to.",
        group: "ts",
    },
    Spec {
        name: "ts.deleterule",
        arity: 3,
        flags: TS_DELETE,
        first_key: 1,
        last_key: 2,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1_R1_1_0],
        acl: AC_TS_WRITE_FAST,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Stop folding one series into another.",
        group: "ts",
    },
    // --------------------------------------------------------------- array
    Spec {
        name: "arset",
        arity: -4,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_AT1],
        acl: AC_ARRAY_WRITE_FAST,
        since: "8.8.0",
        complexity: "O(N) with N the number of values",
        summary: "Write values into consecutive positions from an index.",
        group: "array",
    },
    Spec {
        name: "armset",
        arity: -4,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_AT1],
        acl: AC_ARRAY_WRITE_FAST,
        since: "8.8.0",
        complexity: "O(N) with N the number of pairs",
        summary: "Write index and value pairs, which need not be neighbours.",
        group: "array",
    },
    Spec {
        name: "arget",
        arity: 3,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_ARRAY_READ_FAST,
        since: "8.8.0",
        complexity: "O(1)",
        summary: "The value at one index, or a null if nothing is there.",
        group: "array",
    },
    Spec {
        name: "armget",
        arity: -3,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_ARRAY_READ_FAST,
        since: "8.8.0",
        complexity: "O(N) with N the number of indices",
        summary: "The values at the indices named, in the order named.",
        group: "array",
    },
    Spec {
        name: "argetrange",
        arity: 4,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_ARRAY_READ_SLOW,
        since: "8.8.0",
        complexity: "O(N) with N the length of the range",
        summary: "One reply per position between two indices, holes included.",
        group: "array",
    },
    Spec {
        name: "arlen",
        arity: 2,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_ARRAY_READ_FAST,
        since: "8.8.0",
        complexity: "O(1)",
        summary: "The highest populated index plus one.",
        group: "array",
    },
    Spec {
        name: "arcount",
        arity: 2,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_ARRAY_READ_FAST,
        since: "8.8.0",
        complexity: "O(1)",
        summary: "How many indices hold something.",
        group: "array",
    },
    Spec {
        name: "ardel",
        arity: -3,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_DELETE_AT1],
        acl: AC_ARRAY_WRITE_FAST,
        since: "8.8.0",
        complexity: "O(N) with N the number of indices",
        summary: "Empty the indices named and say how many held something.",
        group: "array",
    },
    Spec {
        name: "ardelrange",
        arity: -4,
        flags: WRITE_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_DELETE_AT1],
        acl: AC_ARRAY_WRITE_SLOW,
        since: "8.8.0",
        complexity: "O(N) with N the elements touched, not the span asked for",
        summary: "Empty one or more ranges of indices.",
        group: "array",
    },
    Spec {
        name: "arinsert",
        arity: -3,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_AT1],
        acl: AC_ARRAY_WRITE_FAST,
        since: "8.8.0",
        complexity: "O(N) with N the number of values",
        summary: "Append values at the insert cursor.",
        group: "array",
    },
    Spec {
        name: "arring",
        arity: -4,
        flags: WRITE_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_AT1],
        acl: AC_ARRAY_WRITE_SLOW,
        since: "8.8.0",
        complexity: "O(N) with N the values, plus the ring size when it changes",
        summary: "Append values into a ring of the given size.",
        group: "array",
    },
    Spec {
        name: "arnext",
        arity: 2,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_ARRAY_READ_FAST,
        since: "8.8.0",
        complexity: "O(1)",
        summary: "The index the next append would write to.",
        group: "array",
    },
    Spec {
        name: "arseek",
        arity: 3,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_AT1],
        acl: AC_ARRAY_WRITE_FAST,
        since: "8.8.0",
        complexity: "O(1)",
        summary: "Point the insert cursor at an index.",
        group: "array",
    },
    Spec {
        name: "arlastitems",
        arity: -3,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_ARRAY_READ_SLOW,
        since: "8.8.0",
        complexity: "O(N) with N the count asked for",
        summary: "The newest positions from the insert cursor, holes included.",
        group: "array",
    },
    Spec {
        name: "arscan",
        arity: -4,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_ARRAY_READ_SLOW,
        since: "8.8.0",
        complexity: "O(N) with N the elements found, not the span asked for",
        summary: "Index and value pairs for what a range holds, skipping holes.",
        group: "array",
    },
    Spec {
        name: "argrep",
        arity: -6,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_ARRAY_READ_SLOW,
        since: "8.8.0",
        complexity: "O(P * C) with P the positions visited and C the cost of the predicates on one element",
        summary: "The indexes in a range whose elements answer a set of textual predicates.",
        group: "array",
    },
    Spec {
        name: "arop",
        arity: -5,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_ARRAY_READ_SLOW,
        since: "8.8.0",
        complexity: "O(N) with N the elements found, not the span asked for",
        summary: "One number out of a range, added up or compared or counted.",
        group: "array",
    },
    Spec {
        name: "arinfo",
        arity: -2,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_ARRAY_READ_SLOW,
        since: "8.8.0",
        complexity: "O(1), or O(N) with N the slices when FULL is given",
        summary: "The shape of the array, and what its slices look like.",
        group: "array",
    },
    // ------------------------------------------------------------- streams
    Spec {
        name: "xadd",
        arity: -5,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_AT1_TRIMMING],
        acl: AC_STREAM_WRITE_FAST,
        since: "5.0.0",
        complexity: "O(1) for the append, plus what a trim removes.",
        summary: "Append an entry and answer with the ID it got.",
        group: "stream",
    },
    Spec {
        name: "xlen",
        arity: 2,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_AT1],
        acl: AC_STREAM_READ_FAST,
        since: "5.0.0",
        complexity: "O(1)",
        summary: "How many entries the stream holds.",
        group: "stream",
    },
    Spec {
        name: "xdel",
        arity: -3,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_DELETE_AT1],
        acl: AC_STREAM_WRITE_FAST,
        since: "5.0.0",
        complexity: "O(1) per ID.",
        summary: "Remove entries by ID and say how many were there.",
        group: "stream",
    },
    Spec {
        name: "xdelex",
        arity: -5,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_DELETE_AT1],
        acl: AC_STREAM_WRITE_FAST,
        since: "8.2.0",
        complexity: "O(1) per ID.",
        summary: "Remove entries by ID, saying what to do about the groups.",
        group: "stream",
    },
    Spec {
        name: "xackdel",
        arity: -6,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_DELETE_AT1],
        acl: AC_STREAM_WRITE_FAST,
        since: "8.2.0",
        complexity: "O(1) per ID.",
        summary: "Acknowledge entries for a group and remove them.",
        group: "stream",
    },
    Spec {
        name: "xnack",
        arity: -7,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_AT1],
        acl: AC_STREAM_WRITE_FAST,
        since: "8.8.0",
        complexity: "O(1) per ID.",
        summary: "Give entries back to the group for somebody else to claim.",
        group: "stream",
    },
    Spec {
        name: "xtrim",
        arity: -4,
        flags: WRITE_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_DELETE_AT1],
        acl: AC_STREAM_WRITE_SLOW,
        since: "5.0.0",
        complexity: "O(N) in the entries removed.",
        summary: "Cut the stream down to a length or a minimum ID.",
        group: "stream",
    },
    Spec {
        name: "xrange",
        arity: -4,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_STREAM_READ_SLOW,
        since: "5.0.0",
        complexity: "O(N) in the entries returned.",
        summary: "The entries between two IDs, oldest first.",
        group: "stream",
    },
    Spec {
        name: "xrevrange",
        arity: -4,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_STREAM_READ_SLOW,
        since: "5.0.0",
        complexity: "O(N) in the entries returned.",
        summary: "The entries between two IDs, newest first.",
        group: "stream",
    },
    Spec {
        name: "xread",
        arity: -4,
        flags: READ_BLOCKING_MOVABLE,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[XREAD_STREAMS],
        acl: AC_STREAM_BLOCKING_READ,
        since: "5.0.0",
        complexity: "O(N) in the entries returned.",
        summary: "Read from one or more streams, waiting if asked to.",
        group: "stream",
    },
    Spec {
        name: "xreadgroup",
        arity: -7,
        flags: WRITE_BLOCKING_MOVABLE,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[XREADGROUP_STREAMS],
        acl: AC_STREAM_BLOCKING_WRITE,
        since: "5.0.0",
        complexity: "O(N) in the entries returned.",
        summary: "Read as part of a consumer group, waiting if asked to.",
        group: "stream",
    },
    Spec {
        name: "xack",
        arity: -4,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_AT1],
        acl: AC_STREAM_WRITE_FAST,
        since: "5.0.0",
        complexity: "O(1) per ID.",
        summary: "Drop entries from a group's pending list.",
        group: "stream",
    },
    Spec {
        name: "xsetid",
        arity: -3,
        flags: WRITE_FAST_OOM,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_AT1],
        acl: AC_STREAM_WRITE_FAST,
        since: "5.0.0",
        complexity: "O(1)",
        summary: "Set the last ID, the entries added and the max deleted ID.",
        group: "stream",
    },
    Spec {
        name: "xgroup",
        arity: -2,
        flags: &[],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_STREAM_CONTAINER,
        since: "5.0.0",
        complexity: "O(1) for all subcommands except DESTROY, which frees the group's pending list.",
        summary: "Make, move and unmake consumer groups.",
        group: "stream",
    },
    Spec {
        name: "xinfo",
        arity: -2,
        flags: &[],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_STREAM_CONTAINER,
        since: "5.0.0",
        complexity: "O(1), or O(N) with N the entries and pending entries shown when FULL is given.",
        summary: "What a stream, its groups and its consumers look like.",
        group: "stream",
    },
    Spec {
        name: "xpending",
        arity: -3,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_STREAM_READ_SLOW,
        since: "5.0.0",
        complexity: "O(1) for the summary, O(N) in the entries returned for the list.",
        summary: "What a group has handed out and not had acknowledged.",
        group: "stream",
    },
    Spec {
        name: "xclaim",
        arity: -6,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_AT1],
        acl: AC_STREAM_WRITE_FAST,
        since: "5.0.0",
        complexity: "O(1) per ID.",
        summary: "Move named pending entries to another consumer.",
        group: "stream",
    },
    Spec {
        name: "xautoclaim",
        arity: -6,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_DELETE_AT1],
        acl: AC_STREAM_WRITE_FAST,
        since: "6.2.0",
        complexity: "O(1) per entry claimed, plus what it skips getting there.",
        summary: "Sweep a group's pending list and take what has gone idle.",
        group: "stream",
    },
    // ------------------------------------------------------------ keyspace
    Spec {
        name: "del",
        arity: -2,
        flags: &["write"],
        first_key: 1,
        last_key: -1,
        step: 1,
        keys: &[RM_DELETE_AT1_RM1_1_0],
        acl: AC_KEY_WRITE_SLOW,
        since: "1.0.0",
        complexity: "O(N) in the number of keys.",
        summary: "Delete keys and say how many were there.",
        group: "keyspace",
    },
    Spec {
        name: "unlink",
        arity: -2,
        flags: &["write", "fast"],
        first_key: 1,
        last_key: -1,
        step: 1,
        keys: &[RM_DELETE_AT1_RM1_1_0],
        acl: AC_KEY_WRITE_FAST,
        since: "4.0.0",
        complexity: "O(1) per key, since the freeing is not on this thread.",
        summary: "Delete keys and free them out of the way of the reply.",
        group: "keyspace",
    },
    Spec {
        name: "exists",
        arity: -2,
        flags: READ_FAST,
        first_key: 1,
        last_key: -1,
        step: 1,
        keys: &[RO_AT1_RM1_1_0],
        acl: AC_KEY_READ,
        since: "1.0.0",
        complexity: "O(N) in the number of keys.",
        summary: "Count how many of these keys are there, naming one twice counting twice.",
        group: "keyspace",
    },
    Spec {
        name: "type",
        arity: 2,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_AT1],
        acl: AC_KEY_READ,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "What kind of value is under a key, or none.",
        group: "keyspace",
    },
    Spec {
        name: "touch",
        arity: -2,
        flags: READ_FAST,
        first_key: 1,
        last_key: -1,
        step: 1,
        keys: &[RO_AT1_RM1_1_0],
        acl: AC_KEY_READ,
        since: "3.2.1",
        complexity: "O(N) in the number of keys.",
        summary: "Count how many of these keys are there, and move them up the eviction order.",
        group: "keyspace",
    },
    // The three that look at keys nobody named. No key positions on any of
    // them, which is what the zeroes say, and it is also why a cluster client
    // sends them to a node rather than to a slot.
    Spec {
        name: "scan",
        arity: -2,
        flags: &["readonly"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_KEY_READ_SLOW,
        since: "2.8.0",
        complexity: "O(1) a call, O(N) for a whole iteration",
        summary: "Walk part of the keyspace and say where to carry on from.",
        group: "keyspace",
    },
    Spec {
        name: "keys",
        arity: 2,
        flags: &["readonly"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_KEY_READ_ALL,
        since: "1.0.0",
        complexity: "O(N) in the number of keys.",
        summary: "Every key matching a pattern, in one reply.",
        group: "keyspace",
    },
    Spec {
        name: "randomkey",
        arity: 1,
        flags: &["readonly"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_KEY_READ_SLOW,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "One key from the database, chosen at random.",
        group: "keyspace",
    },
    // Two keys and not one, which is the 1 2 1 in the key positions. Every other
    // row in this group names a range that runs to the end of the arguments.
    Spec {
        name: "rename",
        arity: 3,
        flags: &["write"],
        first_key: 1,
        last_key: 2,
        step: 1,
        keys: &[RW_ACCESS_DELETE_AT1, OW_UPDATE_AT2],
        acl: AC_KEY_WRITE_SLOW,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Move a key to another name, over whatever was there.",
        group: "keyspace",
    },
    Spec {
        name: "renamenx",
        arity: 3,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 2,
        step: 1,
        keys: &[RW_ACCESS_DELETE_AT1, OW_INSERT_AT2],
        acl: AC_KEY_WRITE_FAST,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Move a key to another name, but only if that name is free.",
        group: "keyspace",
    },
    // `denyoom` and no `fast`, because this is the one command in the group that
    // allocates a whole second value.
    Spec {
        name: "copy",
        arity: -3,
        flags: &["write", "denyoom"],
        first_key: 1,
        last_key: 2,
        step: 1,
        keys: &[RO_ACCESS_AT1, OW_UPDATE_AT2],
        acl: AC_KEY_WRITE_SLOW,
        since: "6.2.0",
        complexity: "O(N) in the size of the value.",
        summary: "Copy a value to another key, in this database or another one.",
        group: "keyspace",
    },
    // `COPY` with the source deleted, and the only command in the group whose
    // second argument is a database rather than a key. The key spec is one key
    // at argument one and the database index is not a key, which is why this
    // does not look like `COPY` above it.
    Spec {
        name: "move",
        arity: 3,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_ACCESS_UPDATE_AT1],
        acl: AC_KEY_WRITE_FAST,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Move a key to another database, if it is not already there.",
        group: "keyspace",
    },
    // The two that block on replication rather than on a key, so they name no
    // key at all and the three zeroes below are not a placeholder.
    Spec {
        name: "wait",
        arity: 3,
        flags: &["blocking"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_WAIT,
        since: "3.0.0",
        complexity: "O(1)",
        summary: "Wait for this connection's writes to reach a number of replicas.",
        group: "keyspace",
    },
    Spec {
        name: "waitaof",
        arity: 4,
        flags: &["blocking"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_WAIT,
        since: "7.2.0",
        complexity: "O(1)",
        summary: "Wait for this connection's writes to reach the append only files.",
        group: "keyspace",
    },
    // The two that speak the file format. A payload is a value standing on its
    // own outside the process, so these are the only two commands in the group
    // that move a value rather than a name.
    Spec {
        name: "dump",
        arity: 2,
        flags: READ_SLOW,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_KEY_READ_SLOW,
        since: "2.6.0",
        complexity: "O(1) to find the key, then O(N) in the size of the value.",
        summary: "Serialize a value into a payload another server can load.",
        group: "keyspace",
    },
    Spec {
        name: "restore",
        arity: -4,
        flags: &["write", "denyoom"],
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[OW_UPDATE_AT1],
        acl: AC_RESTORE,
        since: "2.6.0",
        complexity: "O(1) to find the key, then O(N) in the size of the payload.",
        summary: "Create a key from a payload produced by DUMP.",
        group: "keyspace",
    },
    // And the third one, which is the other two with a socket in between. Its
    // keys are movable for the same reason `SORT`'s are, though for a plainer
    // reason: the `KEYS` option moves them from argument three to everything
    // after the word, so where they are depends on what was written.
    Spec {
        name: "migrate",
        arity: -6,
        flags: MIGRATE_FLAGS,
        first_key: 3,
        last_key: 3,
        step: 1,
        keys: &[RW_ACCESS_DELETE_AT3, MIGRATE_KEYS],
        acl: AC_RESTORE,
        since: "2.6.0",
        complexity: "A DUMP and a DEL here, a RESTORE there, and the bytes in between.",
        summary: "Move a key to another server.",
        group: "keyspace",
    },
    // The two whose keys cannot be read off the command. `SORT k BY w_* GET d_*`
    // touches every key those two patterns name and a client cannot know which
    // ones without the data, so both carry `movablekeys` and Redis's own key
    // specs give the same answer: the first key, and the STORE destination if
    // there is one.
    Spec {
        name: "sort",
        arity: -2,
        flags: WRITE_MOVABLE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1, SORT_BY_AND_GET, SORT_STORE],
        acl: AC_SORT_WRITE,
        since: "1.0.0",
        complexity: "O(N+M*log(M)) with N elements and M returned.",
        summary: "Sort a list, set or sorted set, optionally into another key.",
        group: "keyspace",
    },
    Spec {
        name: "sort_ro",
        arity: -2,
        flags: READ_MOVABLE,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1, SORT_BY_AND_GET],
        acl: AC_SORT_READ,
        since: "7.0.0",
        complexity: "O(N+M*log(M)) with N elements and M returned.",
        summary: "Sort a list, set or sorted set, without the STORE option.",
        group: "keyspace",
    },
    // The four writers take an optional NX, XX, GT or LT, which is the -3 in
    // the arity, and they take the same one whichever unit they are in.
    Spec {
        name: "expire",
        arity: -3,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_AT1],
        acl: AC_KEY_WRITE_FAST,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Put a deadline on a key, counted in seconds from now.",
        group: "keyspace",
    },
    Spec {
        name: "pexpire",
        arity: -3,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_AT1],
        acl: AC_KEY_WRITE_FAST,
        since: "2.6.0",
        complexity: "O(1)",
        summary: "Put a deadline on a key, counted in milliseconds from now.",
        group: "keyspace",
    },
    Spec {
        name: "expireat",
        arity: -3,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_AT1],
        acl: AC_KEY_WRITE_FAST,
        since: "1.2.0",
        complexity: "O(1)",
        summary: "Put a deadline on a key, as a unix time in seconds.",
        group: "keyspace",
    },
    Spec {
        name: "pexpireat",
        arity: -3,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_AT1],
        acl: AC_KEY_WRITE_FAST,
        since: "2.6.0",
        complexity: "O(1)",
        summary: "Put a deadline on a key, as a unix time in milliseconds.",
        group: "keyspace",
    },
    Spec {
        name: "persist",
        arity: 2,
        flags: WRITE_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RW_UPDATE_AT1],
        acl: AC_KEY_WRITE_FAST,
        since: "2.2.0",
        complexity: "O(1)",
        summary: "Take a key's deadline off, so it stops being temporary.",
        group: "keyspace",
    },
    Spec {
        name: "ttl",
        arity: 2,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_KEY_READ,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "How many seconds a key has left, -1 with no deadline, -2 if gone.",
        group: "keyspace",
    },
    Spec {
        name: "pttl",
        arity: 2,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_KEY_READ,
        since: "2.6.0",
        complexity: "O(1)",
        summary: "How many milliseconds a key has left, -1 with no deadline, -2 if gone.",
        group: "keyspace",
    },
    Spec {
        name: "expiretime",
        arity: 2,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_KEY_READ,
        since: "7.0.0",
        complexity: "O(1)",
        summary: "When a key falls due, as a unix time in seconds.",
        group: "keyspace",
    },
    Spec {
        name: "pexpiretime",
        arity: 2,
        flags: READ_FAST,
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[RO_ACCESS_AT1],
        acl: AC_KEY_READ,
        since: "7.0.0",
        complexity: "O(1)",
        summary: "When a key falls due, as a unix time in milliseconds.",
        group: "keyspace",
    },
    // A container command, so no keys and no flags of its own: the key is the
    // subcommand's and a real server reports it on `object|encoding` rather
    // than here. `@slow` is the whole ACL, checked against 8.10.1.
    Spec {
        name: "object",
        arity: -2,
        flags: &[],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: &["@slow"],
        since: "2.2.3",
        complexity: "O(1)",
        summary: "Look at the machinery under a key rather than at its value.",
        group: "keyspace",
    },
    // ----------------------------------------------------------- scripting
    // The four spellings of running a script differ in two things and nothing
    // else: whether the client sent the body or its digest, and whether the
    // script may write. So the four rows are the same row four times, and the
    // `_RO` pair says `readonly` first because that is the order a real 8.10.1
    // lists them in and `COMMAND INFO` is compared byte for byte.
    //
    // `movablekeys` is on all four because the keys are not at a fixed offset:
    // the client says how many there are. `no_mandatory_keys` is on all four
    // because it may say none. `script_runner` is what tells a client that the
    // command's real cost is whatever the script does.
    Spec {
        name: "eval",
        arity: -3,
        flags: &[
            "noscript",
            "stale",
            "skip_monitor",
            "no_mandatory_keys",
            "movablekeys",
            "script_runner",
        ],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[SCRIPT_KEYS_RW],
        acl: &["@slow", "@scripting"],
        since: "2.6.0",
        complexity: "Whatever the script does.",
        summary: "Run a Lua script sent with the command.",
        group: "scripting",
    },
    Spec {
        name: "evalsha",
        arity: -3,
        flags: &[
            "noscript",
            "stale",
            "skip_monitor",
            "no_mandatory_keys",
            "movablekeys",
            "script_runner",
        ],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[RW_ACCESS_UPDATE_AT2_COUNTED],
        acl: &["@slow", "@scripting"],
        since: "2.6.0",
        complexity: "Whatever the script does.",
        summary: "Run a Lua script the cache already holds.",
        group: "scripting",
    },
    Spec {
        name: "eval_ro",
        arity: -3,
        flags: &[
            "readonly",
            "noscript",
            "stale",
            "skip_monitor",
            "no_mandatory_keys",
            "movablekeys",
            "script_runner",
        ],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[SCRIPT_KEYS_RO],
        acl: &["@slow", "@scripting"],
        since: "7.0.0",
        complexity: "Whatever the script does.",
        summary: "Run a Lua script that is not allowed to write.",
        group: "scripting",
    },
    Spec {
        name: "evalsha_ro",
        arity: -3,
        flags: &[
            "readonly",
            "noscript",
            "stale",
            "skip_monitor",
            "no_mandatory_keys",
            "movablekeys",
            "script_runner",
        ],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[RO_ACCESS_AT2_COUNTED],
        acl: &["@slow", "@scripting"],
        since: "7.0.0",
        complexity: "Whatever the script does.",
        summary: "Run a cached Lua script that is not allowed to write.",
        group: "scripting",
    },
    Spec {
        name: "fcall",
        arity: -3,
        flags: &[
            "noscript",
            "stale",
            "skip_monitor",
            "no_mandatory_keys",
            "movablekeys",
            "script_runner",
        ],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[SCRIPT_KEYS_RW],
        acl: &["@slow", "@scripting"],
        since: "7.0.0",
        complexity: "Whatever the function does.",
        summary: "Run a function out of a loaded library.",
        group: "scripting",
    },
    Spec {
        name: "fcall_ro",
        arity: -3,
        flags: &[
            "readonly",
            "noscript",
            "stale",
            "skip_monitor",
            "no_mandatory_keys",
            "movablekeys",
            "script_runner",
        ],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[SCRIPT_KEYS_RO],
        acl: &["@slow", "@scripting"],
        since: "7.0.0",
        complexity: "Whatever the function does.",
        summary: "Run a function that was registered no-writes.",
        group: "scripting",
    },
    // Both containers have no flags and no keys of their own, which is what a
    // real 8.10.1 reports: the flags live on the subcommands.
    Spec {
        name: "script",
        arity: -2,
        flags: &[],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: &["@slow"],
        since: "2.6.0",
        complexity: "O(1) for the subcommands that are here.",
        summary: "The cache EVALSHA runs scripts out of.",
        group: "scripting",
    },
    Spec {
        name: "function",
        arity: -2,
        flags: &[],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: &["@slow"],
        since: "7.0.0",
        complexity: "O(1) for the subcommands that are here.",
        summary: "The libraries FCALL runs functions out of.",
        group: "scripting",
    },
    // ---------------------------------------------------------- connection
    Spec {
        name: "ping",
        arity: -1,
        flags: &["fast"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_CONN,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Ask whether the server is answering.",
        group: "connection",
    },
    Spec {
        name: "echo",
        arity: 2,
        flags: &["loading", "stale", "fast"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_CONN,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Send a string back unchanged.",
        group: "connection",
    },
    // `no_auth` is the flag that lets a command through before the connection
    // has a password accepted, and four commands carry it: this one, `HELLO`,
    // which carries the option that authenticates, `RESET`, which is how a
    // client says it is starting over, and `QUIT`. Everything else on the server
    // is refused until one of them has done its work.
    Spec {
        name: "auth",
        arity: -2,
        flags: &[
            "noscript",
            "loading",
            "stale",
            "fast",
            "no_auth",
            "allow_busy",
        ],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_CONN,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Hand over the password this connection is asked for.",
        group: "connection",
    },
    Spec {
        name: "hello",
        arity: -1,
        flags: &[
            "noscript",
            "loading",
            "stale",
            "fast",
            "no_auth",
            "allow_busy",
        ],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_CONN,
        since: "6.0.0",
        complexity: "O(1)",
        summary: "Agree on a protocol version and describe the server.",
        group: "connection",
    },
    Spec {
        name: "select",
        arity: 2,
        flags: &["loading", "stale", "fast"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_CONN,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Choose which database this connection works in.",
        group: "connection",
    },
    Spec {
        name: "reset",
        arity: 1,
        flags: &[
            "noscript",
            "loading",
            "stale",
            "fast",
            "no_auth",
            "allow_busy",
        ],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_CONN,
        since: "6.2.0",
        complexity: "O(1)",
        summary: "Put the connection back the way it was opened.",
        group: "connection",
    },
    Spec {
        name: "quit",
        arity: -1,
        flags: &[
            "noscript",
            "loading",
            "stale",
            "fast",
            "no_auth",
            "allow_busy",
        ],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_CONN,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Close the connection after the replies already queued.",
        group: "connection",
    },
    // -------------------------------------------------------- transactions
    // None of the five is a write and none of them names a key the table can
    // describe, `WATCH` included: a watched key is read, and the key spec Redis
    // publishes for it says `RO`. What makes them their own group is that they
    // are the only commands the funnel looks at before it decides whether to run
    // anything at all.
    Spec {
        name: "multi",
        arity: 1,
        flags: &["noscript", "loading", "stale", "fast", "allow_busy"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_TX_FAST,
        since: "2.0.0",
        complexity: "O(1)",
        summary: "Start holding commands instead of running them.",
        group: "transactions",
    },
    Spec {
        name: "exec",
        arity: 1,
        flags: &["noscript", "loading", "stale", "skip_slowlog"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: &["@slow", "@transaction"],
        since: "1.2.0",
        complexity: "Whatever the queued commands cost.",
        summary: "Run everything held since MULTI.",
        group: "transactions",
    },
    Spec {
        name: "discard",
        arity: 1,
        flags: &["noscript", "loading", "stale", "fast", "allow_busy"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_TX_FAST,
        since: "2.0.0",
        complexity: "O(N) in the number of commands held.",
        summary: "Throw away everything held since MULTI.",
        group: "transactions",
    },
    Spec {
        name: "watch",
        arity: -2,
        flags: &["noscript", "loading", "stale", "fast", "allow_busy"],
        first_key: 1,
        last_key: -1,
        step: 1,
        keys: &[RO_AT1_RM1_1_0],
        acl: AC_TX_FAST,
        since: "2.2.0",
        complexity: "O(1) a key.",
        summary: "Fail the next EXEC if any of these keys changes.",
        group: "transactions",
    },
    Spec {
        name: "unwatch",
        arity: 1,
        flags: &["noscript", "loading", "stale", "fast", "allow_busy"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_TX_FAST,
        since: "2.2.0",
        complexity: "O(N) in the number of keys watched.",
        summary: "Stop watching everything this connection was watching.",
        group: "transactions",
    },
    // -------------------------------------------------------------- pubsub
    // The three shard commands carry a key at argument one and are not about a
    // key at all. Redis marks that spec `not_key`, which is its way of saying
    // that the argument is there to be hashed to a cluster slot and nothing
    // else, so that a shard channel and the keys it belongs with land on the
    // same node. The triple is what `COMMAND GETKEYS` reads, so it is kept.
    Spec {
        name: "subscribe",
        arity: -2,
        flags: &["denyoom", "pubsub", "noscript", "loading", "stale"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_PUBSUB_SLOW,
        since: "2.0.0",
        complexity: "O(N) in the number of channels named.",
        summary: "Listen on these channels.",
        group: "pubsub",
    },
    Spec {
        name: "unsubscribe",
        arity: -1,
        flags: &["pubsub", "noscript", "loading", "stale"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_PUBSUB_SLOW,
        since: "2.0.0",
        complexity: "O(N) in the number of channels named, or held if none are.",
        summary: "Stop listening on these channels, or on all of them.",
        group: "pubsub",
    },
    Spec {
        name: "psubscribe",
        arity: -2,
        flags: &["denyoom", "pubsub", "noscript", "loading", "stale"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_PUBSUB_SLOW,
        since: "2.0.0",
        complexity: "O(N) in the number of patterns named.",
        summary: "Listen on every channel matching these patterns.",
        group: "pubsub",
    },
    Spec {
        name: "punsubscribe",
        arity: -1,
        flags: &["pubsub", "noscript", "loading", "stale"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_PUBSUB_SLOW,
        since: "2.0.0",
        complexity: "O(N) in the number of patterns named, or held if none are.",
        summary: "Stop listening on these patterns, or on all of them.",
        group: "pubsub",
    },
    Spec {
        name: "ssubscribe",
        arity: -2,
        flags: &["denyoom", "pubsub", "noscript", "loading", "stale"],
        first_key: 1,
        last_key: -1,
        step: 1,
        keys: &[NOT_KEY_AT1_RM1_1_0],
        acl: AC_PUBSUB_SLOW,
        since: "7.0.0",
        complexity: "O(N) in the number of shard channels named.",
        summary: "Listen on these shard channels.",
        group: "pubsub",
    },
    Spec {
        name: "sunsubscribe",
        arity: -1,
        flags: &["pubsub", "noscript", "loading", "stale"],
        first_key: 1,
        last_key: -1,
        step: 1,
        keys: &[NOT_KEY_AT1_RM1_1_0],
        acl: AC_PUBSUB_SLOW,
        since: "7.0.0",
        complexity: "O(N) in the number of shard channels named, or held if none are.",
        summary: "Stop listening on these shard channels, or on all of them.",
        group: "pubsub",
    },
    Spec {
        name: "publish",
        arity: 3,
        flags: &["pubsub", "loading", "stale", "fast"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_PUBSUB_FAST,
        since: "2.0.0",
        complexity: "O(N+M) with N the subscribers and M the patterns.",
        summary: "Send a message to everybody listening on a channel.",
        group: "pubsub",
    },
    Spec {
        name: "spublish",
        arity: 3,
        flags: &["pubsub", "loading", "stale", "fast"],
        first_key: 1,
        last_key: 1,
        step: 1,
        keys: &[NOT_KEY_AT1],
        acl: AC_PUBSUB_FAST,
        since: "7.0.0",
        complexity: "O(N) in the shard channel's subscribers.",
        summary: "Send a message to everybody listening on a shard channel.",
        group: "pubsub",
    },
    Spec {
        name: "pubsub",
        arity: -2,
        flags: &[],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: &["@slow"],
        since: "2.8.0",
        complexity: "O(N) in the number of channels or patterns on the server.",
        summary: "What the server's subscriptions look like from outside.",
        group: "pubsub",
    },
    // -------------------------------------------------------------- server
    // COMMAND is in the connection ACL category and in the server group, which
    // is not a contradiction: the category is about what a connection is
    // allowed to do and the group is about what the command is about. The group
    // is the one reported by COMMAND DOCS, so it is the one that has to match.
    Spec {
        name: "command",
        arity: -1,
        flags: &["loading", "stale"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: &["@slow", "@connection"],
        since: "2.8.13",
        complexity: "O(N) with N the number of commands",
        summary: "What this server can do, in the shape client libraries read.",
        group: "server",
    },
    Spec {
        name: "client",
        arity: -2,
        flags: &[],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: &["@slow"],
        since: "2.4.0",
        complexity: "Depends on the subcommand.",
        summary: "Ask about or change the connection the command arrived on.",
        group: "connection",
    },
    Spec {
        name: "acl",
        arity: -2,
        flags: &[],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: &["@slow"],
        since: "6.0.0",
        complexity: "Depends on subcommand.",
        summary: "A container for Access List Control commands.",
        group: "server",
    },
    Spec {
        name: "config",
        arity: -2,
        flags: &[],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: &["@slow"],
        since: "2.0.0",
        complexity: "Depends on the subcommand.",
        summary: "Read and change the settings a running server exposes.",
        group: "server",
    },
    // Exactly two, which is what a real 8.10.1 reports for the container even
    // though every one of its subcommands carries its own arity underneath. All
    // seven of them take two words, so nothing legal is refused by it, and the
    // one thing that reads differently is the name inside the arity error for a
    // subcommand with an argument after it. That is D-46.
    Spec {
        name: "backup",
        arity: 2,
        flags: &[],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: &["@slow"],
        since: "8.10.0",
        complexity: "Depends on subcommand.",
        summary: "A container for backup management commands.",
        group: "server",
    },
    Spec {
        name: "info",
        arity: -1,
        flags: &["loading", "stale"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: &["@slow", "@dangerous"],
        since: "1.0.0",
        complexity: "O(1)",
        summary: "The server's own numbers, in sections.",
        group: "server",
    },
    Spec {
        name: "debug",
        arity: -2,
        flags: &["admin", "noscript", "loading", "stale"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: &["@admin", "@slow", "@dangerous"],
        since: "1.0.0",
        complexity: "Depends on subcommand.",
        summary: "A container for debugging commands.",
        group: "server",
    },
    Spec {
        name: "dbsize",
        arity: 1,
        flags: READ_FAST,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_KEY_READ,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "How many keys are in the database this connection is on.",
        group: "server",
    },
    Spec {
        name: "flushall",
        arity: -1,
        flags: &["write"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_KEY_FLUSH,
        since: "1.0.0",
        complexity: "O(N) in the number of keys in every database.",
        summary: "Empty every database.",
        group: "server",
    },
    Spec {
        name: "flushdb",
        arity: -1,
        flags: &["write"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_KEY_FLUSH,
        since: "1.0.0",
        complexity: "O(N) in the number of keys in this database.",
        summary: "Empty the database this connection is on.",
        group: "server",
    },
    // In the server group and not the keyspace one, which is Redis's answer and
    // is the right one: it names no key, it takes two database indexes, and what
    // it changes is what every connected client is looking at.
    Spec {
        name: "swapdb",
        arity: 3,
        flags: WRITE_FAST,
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_SWAPDB,
        since: "4.0.0",
        complexity: "O(N) in the number of clients watching or blocked on either.",
        summary: "Swap two databases, so every client on one sees the other.",
        group: "server",
    },
    // No ACL category but `@fast`, which is Redis's answer and reads like an
    // omission. It is not: the categories are about what a command can reach and
    // this one reaches nothing.
    Spec {
        name: "time",
        arity: 1,
        flags: &["loading", "stale", "fast"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: &["@fast"],
        since: "2.6.0",
        complexity: "O(1)",
        summary: "The server's clock, as seconds and microseconds.",
        group: "server",
    },
    // The four about writing the dataset out and the one about who this server
    // is. The first three are `admin` and so stay off a monitor's feed, and the
    // last two are not and so are reported, which was measured rather than read
    // off the flags.
    //
    // `no_multi` on `SAVE` alone. A transaction has been promised that nothing
    // runs in between, and a save that blocks the server for as long as the
    // dataset takes to write is the one command here that would break that
    // promise rather than defer it.
    Spec {
        name: "save",
        arity: 1,
        flags: &["admin", "noscript", "no_async_loading", "no_multi"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: &["@admin", "@slow", "@dangerous"],
        since: "1.0.0",
        complexity: "O(N) in the number of keys",
        summary: "Write the whole dataset out as an RDB file, and wait for it.",
        group: "server",
    },
    Spec {
        name: "bgsave",
        arity: -1,
        flags: &["admin", "noscript", "no_async_loading"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: &["@admin", "@slow", "@dangerous"],
        since: "1.0.0",
        complexity: "O(N) in the number of keys",
        summary: "Write the whole dataset out as an RDB file.",
        group: "server",
    },
    Spec {
        name: "bgrewriteaof",
        arity: 1,
        flags: &["admin", "noscript", "no_async_loading"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: &["@admin", "@slow", "@dangerous"],
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Rewrite the append only file, which this server has not got.",
        group: "server",
    },
    Spec {
        name: "lastsave",
        arity: 1,
        flags: &["loading", "stale", "fast"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: &["@admin", "@fast", "@dangerous"],
        since: "1.0.0",
        complexity: "O(1)",
        summary: "When the dataset was last written out, in seconds.",
        group: "server",
    },
    // No flags at all on the container, because the container is a name and
    // every property that matters belongs to one of the six subcommands under
    // it. `USAGE` is the only one that reads a key and the only one that is
    // marked `readonly`, and the key spec on this row is what says where the key
    // is when it does.
    Spec {
        name: "memory",
        arity: -2,
        flags: &[],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: &["@slow"],
        since: "4.0.0",
        complexity: "O(1)",
        summary: "What the server is holding, and what one key of it costs.",
        group: "server",
    },
    // The three a replica sends and nothing else ever does. `REPLCONF` is the
    // only one that stays a command: by the time a replica sends the other two
    // the connection has stopped being a client, so what they answer is the head
    // of a stream rather than a reply, which is why neither is in a transaction.
    Spec {
        name: "replconf",
        arity: -1,
        flags: &["admin", "noscript", "loading", "stale", "allow_busy"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_ADMIN_SLOW,
        since: "3.0.0",
        complexity: "O(1)",
        summary: "What a replica tells a master about itself.",
        group: "server",
    },
    Spec {
        name: "psync",
        arity: -3,
        flags: &["admin", "noscript", "no_async_loading", "no_multi"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_ADMIN_SLOW,
        since: "2.8.0",
        complexity: "O(N) for a full resync, where N is the dataset.",
        summary: "Become a replica of this server, from here or from scratch.",
        group: "server",
    },
    Spec {
        name: "sync",
        arity: 1,
        flags: &["admin", "noscript", "no_async_loading", "no_multi"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_ADMIN_SLOW,
        since: "1.0.0",
        complexity: "O(N), where N is the dataset.",
        summary: "The same, from before there was anything to negotiate.",
        group: "server",
    },
    // The same command under the name it had and the name it has. Both rows are
    // real rows rather than one being an alias, because `COMMAND INFO` answers
    // for both names and a client that asks about the old one is asking a fair
    // question.
    Spec {
        name: "replicaof",
        arity: 3,
        flags: &["admin", "noscript", "stale", "no_async_loading"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_ADMIN_SLOW,
        since: "5.0.0",
        complexity: "O(1)",
        summary: "Follow another server, or stop following one.",
        group: "server",
    },
    Spec {
        name: "slaveof",
        arity: 3,
        flags: &["admin", "noscript", "stale", "no_async_loading"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_ADMIN_SLOW,
        since: "1.0.0",
        complexity: "O(1)",
        summary: "What REPLICAOF was called before it was called that.",
        group: "server",
    },
    // Arity is a minimum of one, because every word is optional and the command
    // with no words at all is a failover to whichever replica catches up first.
    Spec {
        name: "failover",
        arity: -1,
        flags: &["admin", "noscript", "stale"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: AC_ADMIN_SLOW,
        since: "6.2.0",
        complexity: "O(1)",
        summary: "Hand the master's job to a replica, on purpose.",
        group: "server",
    },
    Spec {
        name: "role",
        arity: 1,
        flags: &["noscript", "loading", "stale", "fast"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: &["@admin", "@fast", "@dangerous"],
        since: "2.8.12",
        complexity: "O(1)",
        summary: "Whether this server is a master or a replica, and of what.",
        group: "server",
    },
    // Marked `admin` for the usual reason and one more: the flag is what keeps a
    // command out of the feed, and a `MONITOR` that reported itself to the
    // monitor it had just made would be reporting the audience to itself.
    Spec {
        name: "monitor",
        arity: 1,
        flags: &["admin", "noscript", "loading", "stale"],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: &["@admin", "@slow", "@dangerous"],
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Watch every command the server runs, as it runs them.",
        group: "server",
    },
    Spec {
        name: "shutdown",
        arity: -1,
        flags: &[
            "admin",
            "noscript",
            "loading",
            "stale",
            "no_multi",
            "allow_busy",
        ],
        first_key: 0,
        last_key: 0,
        step: 0,
        keys: &[],
        acl: &["@admin", "@slow", "@dangerous"],
        since: "1.0.0",
        complexity: "O(1)",
        summary: "Stop the server, without answering.",
        group: "server",
    },
];

/// The shortest and the longest command name.
///
/// Both are facts about [`COMMANDS`], pinned by a test, and both are checked
/// before anything is read, so a name that could not be a command is rejected on
/// its length alone.
const MIN_LEN: usize = 3;
const MAX_LEN: usize = 20;

/// How many slots the index has, which is a power of two and a bit over three
/// times the number of commands.
///
/// Four kibibytes of `u16`, sixty four cache lines, and loose enough that a probe
/// for a name that is not a command stops at an empty slot almost immediately.
/// Tight enough that the whole thing stays resident next to the table it
/// indexes.
///
/// This was 512 for a long time, which was a bit over twice the number of
/// commands, and it stopped being enough at 282 of them. Then it was 1024, and
/// that stopped being enough at 337. The note on [`MIX`] has the whole story
/// both times, and the short version is the same one twice: at about half full
/// there is no multiplier left that keeps every command within two slots of
/// home, and at about a sixth full the multiplier that is already there keeps
/// every one of them within a single slot without being touched. Two kibibytes
/// is what it cost this time.
const SLOTS: usize = 2048;

/// A slot nothing was put in.
///
/// `u16::MAX` and not zero, because zero is `set` and `set` is the command this
/// most wants to be able to find.
const FREE: u16 = u16::MAX;

/// The multiplier, found by searching for one that spreads these 358 names well.
///
/// Not a magic constant in the bad sense: it is checked. Every command is looked
/// up by its own name in a test, and another test holds the worst probe length
/// at what it is now, so a command added later that made this multiplier bad
/// would fail rather than quietly cost every lookup an extra slot.
///
/// It has been searched for fifteen times, and each time because the test went red
/// rather than because somebody went looking. The first was against the 191 names
/// in the table then, the ten graph commands pushed its worst probe to three
/// slots, and the second search was run over all 201. The fifteen stream commands
/// pushed that one to four slots and fifty one extra probes, so the third was run
/// over all 216, and the three 8.x pending list commands cost that one two more
/// probes than the test allows. The fourth was over 219 and the seven bitmap
/// commands took it to three slots, and the fifth was over all 226. The five
/// HyperLogLog commands kept its worst probe at two and took it from forty nine
/// extra slots to fifty five, and the search over the 231 names found nothing
/// better, so that one stood. The ten geo commands took it to sixty, and the
/// sixth search, over eight million multipliers and all 241 names, found one at
/// fifty six. The twelve vector set commands took that one to four slots and
/// seventy extra probes, so the seventh search was run over all 254 names and
/// found one at two slots and seventy seven.
///
/// The eight JSON commands took that one to five slots, which is the worst any
/// of them has been, and the eighth search was run over four hundred million
/// multipliers and all 262 names. It found this one at two slots and fifty four,
/// which is the best the table has ever been and a third fewer extra probes than
/// the multiplier it replaced managed with eight fewer commands. Thirty one of
/// the names collide on the key itself and no multiplier can separate them, so
/// seventeen extra probes is the floor everything here is measured against.
/// `json.set` and `json.get` are one of those pairs, since every name in the
/// group starts `js` and the only thing left to tell them apart is the length
/// and the last byte.
///
/// The nine JSON array commands took that one to four slots, and this time the
/// search over the 271 names found nothing at two whatever it was given. That
/// was not the multiplier's fault. `json.arrlen`, `json.objlen` and
/// `json.strlen` all key to the same four bytes, and three names in one slot run
/// costs the third of them two probes before any other name has moved, so two
/// slots was the whole budget spent in one place. The fix was the key rather
/// than the multiplier, which is what [`key_of`] now folds the middle byte in
/// for, and the ninth search was run over the 271 names with the new key across
/// six shards. It found one at two slots and fifty seven, which is a shade over
/// a fifth of a probe a command, the same as the multiplier it replaced managed
/// over nine fewer names.
///
/// The number family and `json.strappend` took that one to three slots, and the
/// tenth search over the 275 names found one at two slots and sixty seven.
/// Three of the six shards converged on sixty seven from different seeds without
/// any of them bettering it, which is the sign that the key rather than the
/// multiplier is what is left: fourteen of the names collide on the key itself
/// and no multiplier can separate them, so fourteen extra probes is the floor
/// and that was within a quarter of it per name. The two new pairs were
/// `json.arrappend` with `json.strappend` and `json.numincrby` with
/// `json.nummultby`, and both are the same shape as the pairs already there,
/// which is a group whose names agree everywhere the key looks.
///
/// The last four JSON commands took it to three slots again, and the eleventh
/// search over the 279 names found this one at two slots and sixty two, which is
/// better than the table has ever been while carrying four more names. Only one
/// of the four collides on the key, `json.mset` with `json.mget`, so the floor
/// moved by one and the multiplier found five more probes than the floor moved.
/// Nine shards were run from different seeds and the spread was sixty two to a
/// hundred and three, which is worth knowing: one shard is not a search.
///
/// `SUNIONCARD` and `SDIFFCARD` took it to 281 names and sixty three probes, one
/// more than before, and the twelfth search is the first one that did not
/// replace it. Eight shards over 960 million multipliers did not find a single
/// one that kept the worst probe at two slots at all, let alone at two slots and
/// sixty two, and the best of them was three slots and eighty one. So that one
/// stayed and the bound went up by one, which is the opposite of what the first
/// eleven searches concluded and was the honest reading of the same procedure.
///
/// `LMOVEM` took it to 282 names and three slots, and that is where the search
/// stopped being the answer. Twelve searches had found a better multiplier
/// eleven times and the twelfth had found that there was none, which is not a
/// result about `LMOVEM`, it is a result about a 512 slot table holding 282
/// names. Fifty five percent full is where linear probing starts to cost real
/// runs, and no multiplier gets around that because the runs are the load
/// factor and not the hash.
///
/// So the other half of the remedy this note has always named was taken and the
/// table doubled. At 1024 slots the multiplier that was already here goes to two
/// slots and forty two extra probes without being touched, which on its own
/// would have been enough. A search over the doubled table across four shards
/// and eighty million multipliers then found this one at **one** slot and twenty
/// eight, so no command is more than a single slot from where it wants to be,
/// which the table has never managed at any size. Fourteen names collide on the
/// key itself and no multiplier can separate them, so fourteen is the floor and
/// this is twice it, against a floor the 512 slot table never came within four
/// times of.
///
/// The cost is a kibibyte, and the thing it buys beyond today is room. The
/// `FT.*` and `TS.*` families are still to be written and both are large, and at
/// 27 percent full there is somewhere for them to go.
///
/// The `BF.*` family is the first of those to arrive and it took the table to
/// 296 names, where the doubled table's multiplier went to two slots and thirty
/// six extra probes. That is well inside what a lookup is allowed to cost, so
/// the search was run to see whether the single slot result had been luck at 285
/// names or was a property of the table at this load, and eight shards over a
/// hundred and sixty million multipliers found this one at one slot and thirty
/// three. Eleven more names, five more probes, and the worst is still a single
/// slot. None of the eleven collides on the key, so the floor moved by one for
/// an unrelated reason and stands at fifteen, which this is a shade over twice.
///
/// The `CF.*` family took the table to 310 names and thirty five extra probes,
/// two more than the bound allowed, with the worst still a single slot. The
/// thirteenth search was run over that and it is the second one that did not
/// replace the multiplier. Ten shards over one and a half billion multipliers
/// found nothing better than thirty six at one slot, which is worse than the one
/// already here, and another two billion with the single slot rule relaxed found
/// one at two slots and thirty one. Four fewer probes spread over three hundred
/// and ten lookups is not worth giving up the property that no command is ever
/// more than one slot from home, so this one stayed and the bound went up by two.
/// None of the fourteen new names collides on the key, so the floor is still
/// fifteen and the table is at a shade over twice it while carrying fourteen more
/// commands than when that was first true.
///
/// The `CMS.*` family took it to 316 names and thirty seven extra probes, with
/// the worst still one slot. No search was run this time. The one before it
/// covered three and a half billion multipliers against a table only six names
/// smaller and found nothing better that keeps every command within a slot, and
/// six names is not enough of a change to expect a different answer, so the
/// bound went up by two again. Only one of the six new names collides on the
/// key, which is `CMS.QUERY` against `CMS.MERGE`, so the floor is sixteen and
/// the table is still a shade over twice it.
///
/// The `TOPK.*` family took it to 323 names and forty two extra probes, with the
/// worst still one slot. A short search of four hundred thousand multipliers ran
/// against the new table and the best it turned up was two slots and forty eight,
/// worse on both counts, which is what the two big searches before it already
/// said, so this multiplier stayed and the bound went up by five. None of the
/// seven new names collides on the key, so the floor is still sixteen.
///
/// The `TDIGEST.*` family took it to 337 names and broke the bound properly: the
/// worst probe went to three slots, which is the first time since the table was
/// doubled that a command was further from home than a lookup is allowed to be.
/// Fourteen names is a lot to add to a family of sketch commands that all start
/// with the same two bytes, and the key is built out of the first two bytes, so
/// the whole family lands in a handful of key values before the multiply ever
/// sees them.
///
/// So the fifteenth search ran, and it said the same thing the tenth one did at
/// 282 names. Three and a half million multipliers against the 1024 slot table
/// found nothing better than two slots and fifty two extra probes, against the
/// fifty two this one already spends at three slots. That is the shape of a
/// table that is too full rather than a multiplier that is bad, and at 337
/// names in 1024 slots it is a third full, which is where the 512 slot table
/// was when it ran out as well. Doubling the table to 2048 and touching nothing
/// else takes this same multiplier to **one** slot and thirty four, so the
/// answer was a bigger table again and not a new constant.
///
/// The search then ran over the doubled table anyway, because that is what
/// happened last time and it found something worth having. Four and a half
/// million multipliers turned up this one at one slot and twenty two, twelve
/// fewer probes than the old multiplier spends in the same table, against a
/// floor of sixteen from the names that collide on the key itself. Twelve
/// probes over three hundred and thirty seven lookups is not much, but it is
/// free, it moves both numbers the right way, and it is exactly the trade the
/// doubling from 512 made, so it was taken. The old multiplier was
/// `0x3e8668c9760e09c9` and it served for thirteen searches.
///
/// The room this buys is the same room as last time and it is worth writing down
/// again: `FT.*` and `TS.*` are still to come and both are large, and at a sixth
/// full there is somewhere for them to go.
///
/// `TS.*` then arrived and the last three of it, the two joined reads and
/// `TS.READ`, broke the bound in a way no multiplier could fix. `TS.NRANGE` made
/// `ts.create`, `ts.incrby`, `ts.mrange` and `ts.nrange` four names sharing one
/// key: all nine bytes long, all starting `ts`, and all with the same fold of
/// the last byte against the middle one. Four names in a slot run costs the
/// fourth of them three probes wherever the run starts, so the worst probe went
/// to three and doubling the table again would not have moved it, because the
/// cost is in the key and not in how much room the key has to land in.
///
/// So the sixteenth search was a search for a key rather than for a multiplier.
/// Folding the second to last byte in as well separates all four of them, and it
/// separates enough else besides to take the floor from twenty colliding names
/// down to twelve, which is the fewest of the handful of folds tried. It is the
/// cheapest byte to add, too, because it sits next to the last byte that is
/// already being read. Then the multiplier search ran over the new key, three
/// hundred and twenty million of them across eight shards, and the best keeps
/// every command within one slot at eighteen extra probes over the whole table,
/// against a floor of twelve. That is the best either number has ever been here
/// while carrying the most names it has ever carried. The old multiplier was
/// `0x2f0cc21a638ae49d` and it served for one search.
///
/// `FT.CONFIG` then took the table to 392 names and put the worst probe back to
/// three slots on its own. It does not collide on the key with anything, so the
/// floor is still twelve and this was a slot run and not a key problem, which
/// meant a seventeenth multiplier search rather than another key. Eight hundred
/// million multipliers over two independent seeds both bottomed out at one slot
/// and twenty one extra probes and neither ever went below it, so twenty one
/// looks like where this key and this table actually sit with 392 names in
/// them. Three more probes than the last search spent over one more name is the
/// ordinary cost of a name, and every command is still within one slot, which
/// is the number a lookup feels. The old multiplier was `0x55251c10f29d4c29`
/// and it served for one search as well.
///
/// The five deprecated document commands took the table to 399 names and the
/// worst probe to two slots, which is inside the bound and outside what this
/// table has held itself to since it was doubled, so an eighteenth search ran.
/// One of the five raises the floor as well: `FT.SAFEADD` agrees with
/// `FT.PROFILE` on all four key bytes, which makes thirteen colliding pairs
/// instead of twelve and thirteen the fewest extra probes any multiplier can
/// spend. Four billion multipliers over two independent seeds both reached one
/// slot, one of them at twenty two extra probes and the other at twenty one,
/// which is the same twenty one the last search settled on while carrying one
/// more collision than it did. The old multiplier was `0xda8bd262ac598c57` and
/// it served for one search as well.
///
/// The five transaction commands took the table to 411 names and the total to
/// twenty five, which is over what this table had been holding and is the first
/// time the number moved without the worst probe moving with it. A nineteenth
/// search ran anyway, four billion multipliers, and the best of them spends
/// twenty four. One probe over four hundred and eleven commands is not worth
/// changing a constant that is already at one slot for every name, so this is
/// the first search that ended by keeping the multiplier it started with. The
/// floor is still thirteen and the number a lookup feels is still one.
///
/// The nine pub/sub commands took the table to 420 names and the worst probe to
/// two slots, so the multiplier the last search declined to replace had to be
/// replaced after all. A twentieth search ran, four billion multipliers over ten
/// threads, and the best of them is back to one slot for every name at twenty
/// four extra probes, which is the same twenty four the nineteenth search found
/// and did not take. The floor is still thirteen, because none of the nine
/// agrees with anything already in the table on all four key bytes. The old
/// multiplier was `0x91de5d5e21661fbd` and it served for two searches.
///
/// The five persistence commands took the table to 427 names and the worst probe
/// to two slots again. A twenty first search ran, four billion multipliers over
/// ten threads on each of two independent seeds, and the two seeds reached one
/// slot for every name at twenty eight extra probes and at twenty nine. Four
/// probes more than the last search over seven more names is the ordinary cost
/// of a name and none of the seven collides with anything already there, so the
/// floor is still thirteen and the number a lookup feels is still one. The old
/// multiplier was `0x71ee9b00ab8a5fd7` and it served for one search.
const MIX: u64 = 0xc7f9_d8be_b27e_8381;

/// The four bytes the index is computed from: the length, the first two bytes,
/// and the last byte with the second to last and the middle folded into it, all
/// lower cased.
///
/// `None` for a name no command could be spelled as, which is decided on the
/// length before a byte is read.
///
/// Four bytes and not the whole name because the whole name has to be compared
/// at the end anyway, so the hash only has to be good enough to get to the right
/// slot, and reading less of the name is a shorter dependency chain in front of
/// the multiply. Names that agree on all four collide whatever the multiplier is
/// and probe once more, and the probe is the same compare the lookup was always
/// going to do. Over the 420 commands there are thirteen such pairs and no group
/// larger than a pair, so thirteen extra probes is the floor.
///
/// The middle byte is the part that was added last and it is worth saying why,
/// because for a long time the key was the length and the first two bytes and
/// the last and nothing else. That was fine while the groups that agreed on a
/// prefix were small: `setnx` with `setex`, `g.nadd` with `g.eadd`, `getset`
/// with `getbit`, `setbit` with `select`. The JSON group broke it, because every
/// name in it starts `js` and so every name in it was keyed on nothing but its
/// length and its last byte, and `json.arrlen`, `json.objlen` and `json.strlen`
/// agree on both. Three names in one slot run costs the third of them two probes
/// on its own, which leaves a multiplier no room anywhere else, and the number
/// families still to come are the same shape again. Folding in the middle byte
/// separates all three, and it separates `json.set` from `json.get` as well.
/// It costs one more load off a cache line the first two bytes already pulled
/// in, and the xor is on the same dependency chain as the shifts rather than in
/// front of them.
///
/// The second to last byte went in for the same reason a family later. `TS.*`
/// is the JSON shape again and worse: every name starts `ts`, so a nine byte
/// name is keyed on nothing but its length and the fold of its last byte against
/// its middle one, and `ts.create`, `ts.incrby`, `ts.mrange` and `ts.nrange` all
/// land on the same fold. Four in a run is three probes for the last of them
/// whatever the multiplier does, so the key had to carry more. The byte before
/// the last one is the cheapest one left, being on the cache line the last byte
/// already pulled in, and it separates all four of those and takes the floor
/// from twenty down to twelve besides.
///
/// `| 0x20` lower cases a letter and does not have to be told which bytes are
/// letters. It maps the two cases of a name to the same number, which is all
/// this needs, and every command name is letters. It has to be applied to each
/// of the three folded bytes separately, before the xor rather than after,
/// because `.` and `n` differ in the bit `| 0x20` sets and an xor of the raw
/// bytes would keep that difference alive.
///
/// On a three or four byte name the middle byte and the second to last byte are
/// the same byte and cancel each other out, which leaves the fold as the last
/// byte alone. That is not a loss, because on a name that short every byte the
/// fold could carry is already in the key somewhere else.
const fn key_of(name: &[u8]) -> Option<u32> {
    if name.len() < MIN_LEN || name.len() > MAX_LEN {
        return None;
    }
    let last = name.len() - 1;
    let mid = name.len() / 2;
    Some(
        name.len() as u32
            | ((name[0] | 0x20) as u32) << 8
            | ((name[1] | 0x20) as u32) << 16
            | (((name[last] | 0x20) ^ (name[last - 1] | 0x20) ^ (name[mid] | 0x20)) as u32) << 24,
    )
}

/// Where a key wants to sit.
///
/// The shift leaves the top eleven bits of the product, which are the ones the
/// multiply mixed the most, and the mask is what makes that a slot number. Eleven
/// because the table has 2048 slots, so both numbers have to move together if
/// [`SLOTS`] ever does. It was ten while the table was half this size.
const fn slot_of(key: u32) -> usize {
    ((key as u64).wrapping_mul(MIX) >> 53) as usize & (SLOTS - 1)
}

/// The index, built at compile time by inserting every command in table order.
///
/// Table order is rough order of how often a command is sent, and inserting in
/// that order means the hotter of two commands that want the same slot gets it
/// and the colder one probes, which is the right way round.
const INDEX: [u16; SLOTS] = index();

const fn index() -> [u16; SLOTS] {
    let mut out = [FREE; SLOTS];
    let mut i = 0;
    while i < COMMANDS.len() {
        let key = match key_of(COMMANDS[i].name.as_bytes()) {
            Some(key) => key,
            None => panic!("a command name is outside MIN_LEN..=MAX_LEN"),
        };
        let mut at = slot_of(key);
        while out[at] != FREE {
            at = (at + 1) & (SLOTS - 1);
        }
        out[at] = i as u16;
        i += 1;
    }
    out
}

/// The command called `name`, whatever case the client spelled it in.
///
/// This used to walk the whole table comparing lengths, and the cost of that was
/// not what it looked like. The table is written in rough order of how often a
/// command is sent, so `set` and `get` were the first two entries and cost one
/// compare, but `exists` is the hundred and forty ninth and `del` the hundred and
/// forty seventh, and every one of those compares was paid twice per command,
/// once to work out the key hash and once to dispatch.
///
/// Measured, that walk was 104 nanoseconds a command, which is more than a whole
/// `GET` costs end to end. `EXISTS` on a missing key ran at three and a half
/// times `GET` and almost none of the difference was the command: short
/// circuiting the lookup alone took it from 8.7 microseconds a batch of sixty
/// four to 2.0, and left it faster than `GET`, which it should be, because it
/// does less.
///
/// So this is one multiply and one load into two kibibytes, and then the same name
/// compare it always ended with. What it costs the hot commands is a multiply
/// they did not use to pay and a load that hits, and what it saves the rest is
/// the whole walk.
#[must_use]
pub fn lookup(name: &[u8]) -> Option<&'static Spec> {
    at(lookup_index(name))
}

/// The same, answering with a position in the table rather than a reference.
///
/// This is where the lookup actually ends, because the index is what the slots
/// hold. It is here as its own function because a position fits in a `u16` and a
/// reference does not fit anywhere a framed command can carry it cheaply, so the
/// engine resolves a command's name once when it frames it and hands the number
/// on to both the key hash and the dispatcher.
///
/// `u16::MAX` is the answer for a name that is not a command, which is not a
/// special case anybody has to write down: the table is 254 entries, so [`at`]
/// hands back `None` for it the same way it would for any other number past the
/// end.
#[must_use]
pub fn lookup_index(name: &[u8]) -> u16 {
    let Some(key) = key_of(name) else {
        return FREE;
    };
    let mut at = slot_of(key);
    loop {
        let i = INDEX[at];
        if i == FREE {
            return FREE;
        }
        if COMMANDS[i as usize]
            .name
            .as_bytes()
            .eq_ignore_ascii_case(name)
        {
            return i;
        }
        at = (at + 1) & (SLOTS - 1);
    }
}

/// The command at `i`, or `None` if there is none there.
///
/// The other half of [`lookup_index`], and the only thing that should ever be
/// handed one of its answers.
#[must_use]
pub fn at(i: u16) -> Option<&'static Spec> {
    COMMANDS.get(i as usize)
}

/// How many commands there are.
///
/// The length of a counter array that has a row per command, which is the only
/// thing that wants this number.
#[must_use]
pub const fn count() -> usize {
    COMMANDS.len()
}

/// Where in [`COMMANDS`] this spec is.
///
/// Every `&'static Spec` a caller can hold came out of [`lookup`] and therefore
/// points into that array, so its position is the distance from the front
/// measured in whole `Spec`s. That is arithmetic on two addresses and not a
/// search, which is the point: a per command counter has to be reachable from
/// the spec the dispatcher is already holding without walking the table a second
/// time.
///
/// A spec from somewhere else would answer nonsense, which is why this takes a
/// `&'static Spec` rather than a `&Spec`: the only `'static` ones are in the
/// table.
#[must_use]
pub fn index_of(spec: &'static Spec) -> usize {
    let front = COMMANDS.as_ptr().addr();
    let here = std::ptr::from_ref(spec).addr();
    (here - front) / size_of::<Spec>()
}

/// The name of the command at `at`, which is [`index_of`] the other way round.
///
/// # Panics
///
/// If `at` is past the end of the table, which only a caller that made the index
/// up rather than getting it from [`index_of`] can manage.
#[must_use]
pub fn name_at(at: usize) -> &'static str {
    COMMANDS[at].name
}

/// Whether `n` arguments, counting the name, satisfy this command's arity.
#[must_use]
pub fn arity_ok(spec: &Spec, n: usize) -> bool {
    let n = n as i32;
    if spec.arity >= 0 {
        n == spec.arity
    } else {
        n >= -spec.arity
    }
}

/// Where a command's key arguments are: the first one, how many there are, and
/// how far apart they sit.
///
/// Three numbers is enough for every command in this table, the ones Redis marks
/// `movablekeys` included, because those keep their keys behind a count and a
/// count still gives a first, a many and a step. What three numbers cannot
/// describe is a command whose keys are found by scanning for a keyword, and
/// there are none of those here.
#[derive(Debug, Clone, Copy)]
pub(crate) struct KeySpan {
    /// The argument index of the first key.
    pub first: usize,
    /// How many keys there are.
    pub count: usize,
    /// How many arguments apart consecutive keys are.
    pub step: usize,
}

/// Why a command has no key span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NoKeys {
    /// It names no keys at all, whatever it is sent.
    Never,
    /// It keeps its keys behind a count, and the count is not a usable number.
    BadCount,
}

/// Work out where `spec`'s keys are in `args`.
///
/// `base` is how many arguments sit in front of the command itself, which is two
/// for `COMMAND GETKEYS <command> ...` and zero for a command that is running.
/// The positions in the answer are absolute, so they go straight to
/// [`Args::get`].
///
/// A few commands keep their keys somewhere the first, last and step triple
/// cannot describe, behind a count of how many there are. That is why a real
/// server marks them `movablekeys` and why a cluster aware client has to ask
/// `COMMAND GETKEYS` about them at all. `MSETEX` counts pairs and the rest count
/// single keys, so what differs between them is the step and where the count
/// sits: the script family has the body in front of it and the others have
/// nothing.
///
/// The script family is also the only one where none is a real answer. A script
/// with no keys is an ordinary thing to write and `EVAL body 0` answers an empty
/// list rather than complaining, where `MSETEX 0` is a command that would do
/// nothing and is refused. So a count that makes no sense at all is an empty
/// span for the script family and a [`NoKeys::BadCount`] for the others, which
/// is a real server reading the script family through a key spec that finds no
/// keys and refusing the rest.
pub(crate) fn key_span(spec: &Spec, args: Args<'_>, base: usize) -> Result<KeySpan, NoKeys> {
    if let Some((rel, step, least, lenient)) = match spec.name {
        "msetex" => Some((1, 2, 1, false)),
        "ts.nrange" | "ts.nrevrange" => Some((1, 1, 1, false)),
        "eval" | "eval_ro" | "evalsha" | "evalsha_ro" | "fcall" | "fcall_ro" => {
            Some((2, 1, 0, true))
        }
        _ => None,
    } {
        let at = base + rel;
        let found = parse_i64(args.get(at))
            .filter(|&n| n >= least)
            .and_then(|n| usize::try_from(n).ok())
            .filter(|&n| at + 1 + step * n <= args.len());
        let count = match found {
            Some(n) => n,
            None if lenient => 0,
            None => return Err(NoKeys::BadCount),
        };
        return Ok(KeySpan {
            first: at + 1,
            count,
            step,
        });
    }
    if spec.first_key == 0 {
        return Err(NoKeys::Never);
    }
    let argc = args.len() - base;
    let last = if spec.last_key < 0 {
        (argc as i64) + i64::from(spec.last_key)
    } else {
        i64::from(spec.last_key)
    };
    let step = i64::from(spec.step).max(1);
    let first = i64::from(spec.first_key);
    let count = if last < first {
        0
    } else {
        ((last - first) / step + 1) as usize
    };
    Ok(KeySpan {
        first: base + first as usize,
        count,
        step: step as usize,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The name here is the name a client reads back, so it is spelled the way
    /// the server that registered it spelled it.
    ///
    /// That is lower case for everything the server itself registers and upper
    /// case for the two groups that come out of a module, so `COMMAND INFO vadd`
    /// answers `VADD` and `COMMAND INFO ft.create` answers `FT.CREATE`, and the
    /// arity errors quote them the same way. Nothing else in the table cares,
    /// because a lookup compares without regard to case and the index key folds
    /// the case out before it hashes.
    #[test]
    fn every_name_is_spelled_the_way_it_was_registered_and_appears_once() {
        let mut seen = std::collections::BTreeSet::new();
        for c in COMMANDS {
            let want = if c.group == "vector" || c.group == "search" {
                c.name.to_uppercase()
            } else {
                c.name.to_lowercase()
            };
            assert_eq!(c.name, want, "{} is spelled wrong for its group", c.name);
            assert!(seen.insert(c.name), "{} is in the table twice", c.name);
        }
    }

    /// Every command's index is where the table actually holds it.
    ///
    /// Checked against the position a search finds, over the whole table rather
    /// than a sample, because the arithmetic is the thing being tested and an
    /// off by one in it would put every counter on the wrong command.
    #[test]
    fn a_spec_knows_where_it_is_in_the_table() {
        assert_eq!(count(), COMMANDS.len());
        for (want, spec) in COMMANDS.iter().enumerate() {
            assert_eq!(index_of(spec), want, "{} is at the wrong index", spec.name);
        }
        assert_eq!(
            index_of(lookup(b"get").unwrap()),
            index_of(lookup(b"GET").unwrap())
        );
    }

    #[test]
    fn lookup_ignores_case_and_does_not_match_a_prefix() {
        assert_eq!(lookup(b"GET").unwrap().name, "get");
        assert_eq!(lookup(b"gEt").unwrap().name, "get");
        assert!(lookup(b"ge").is_none());
        assert!(lookup(b"gets").is_none());
    }

    /// Every command is findable under its own name, in either case.
    ///
    /// The index is built at compile time from the table it sits beside, so what
    /// a test can still catch is a command that the build put somewhere the
    /// lookup does not walk past, which is what a probe that stopped early would
    /// look like.
    #[test]
    fn every_command_is_findable_by_its_own_name() {
        for spec in COMMANDS {
            let found = lookup(spec.name.as_bytes()).expect(spec.name);
            assert_eq!(
                index_of(found),
                index_of(spec),
                "{} found the wrong spec",
                spec.name
            );
            assert_eq!(
                lookup(spec.name.to_ascii_uppercase().as_bytes()).map(index_of),
                Some(index_of(spec)),
                "{} is not found in upper case",
                spec.name,
            );
        }
    }

    /// A name that cannot be a command is answered before anything is compared.
    #[test]
    fn a_name_that_cannot_be_a_command_is_rejected_on_its_shape() {
        assert!(lookup(b"").is_none());
        assert!(key_of(b"").is_none());
        assert!(key_of(&[b'g'; 256]).is_none());
        assert!(lookup(&[b'g'; 256]).is_none());
        assert!(lookup(b"9et").is_none());
    }

    /// The two cases of a name give the same key and different names do not.
    #[test]
    fn a_key_folds_the_case_and_nothing_else() {
        assert_eq!(key_of(b"get"), key_of(b"GET"));
        assert_eq!(key_of(b"get"), key_of(b"gEt"));
        assert_ne!(key_of(b"get"), key_of(b"set"), "other first byte");
        assert_ne!(key_of(b"get"), key_of(b"gxt"), "other second byte");
        assert_ne!(key_of(b"get"), key_of(b"gex"), "other last byte");
        assert_ne!(key_of(b"get"), key_of(b"gett"), "other length");
        assert_ne!(key_of(b"abcde"), key_of(b"abxde"), "other middle byte");
        assert_eq!(key_of(b"abcde"), key_of(b"ABCDE"), "middle byte folds too");
    }

    /// The index is still worth having, which is a thing that can rot.
    ///
    /// The multiplier was searched for against the 191 commands that were in the
    /// table when it was written, and twenty one times since. Adding commands cannot
    /// make a lookup wrong, because a probe walks to an empty slot and every
    /// candidate has its name compared, but it can make one slow, and a slow
    /// lookup is exactly the thing this replaced. So the worst probe is written
    /// down here: if a command added later pushes it up, somebody searches for a
    /// new multiplier or a bigger table rather than finding out from a benchmark
    /// six months later. Both of those have now happened, and the note on
    /// [`MIX`] says which one worked when.
    ///
    /// The bound is two slots because that is what a lookup is allowed to cost,
    /// and the table is better than its bound: the multiplier in it keeps every
    /// command within one slot. The total is held at exactly what it measures so
    /// that a command which quietly spends the headroom shows up here.
    #[test]
    fn no_command_is_more_than_two_slots_from_where_it_wants_to_be() {
        let mut worst = 0;
        let mut total = 0;
        for spec in COMMANDS {
            let key = key_of(spec.name.as_bytes()).expect(spec.name);
            let home = slot_of(key);
            let mut at = home;
            let mut steps = 0;
            while INDEX[at] as usize != index_of(spec) {
                at = (at + 1) & (SLOTS - 1);
                steps += 1;
                assert!(steps < SLOTS, "{} is not in the index at all", spec.name);
            }
            worst = worst.max(steps);
            total += steps;
        }
        assert!(worst <= 2, "worst probe is {worst} slots");
        assert_eq!(
            worst, 1,
            "the multiplier stopped keeping every command close"
        );
        assert!(
            total <= 28,
            "{total} extra slots walked over the whole table"
        );
    }

    /// The table has room to probe in, which is what stops the loop.
    #[test]
    fn the_index_is_not_full() {
        assert!(
            COMMANDS.len() < SLOTS,
            "the probe would never find an empty"
        );
        assert!(
            COMMANDS.len() < FREE as usize,
            "an index would collide with FREE"
        );
        let free = INDEX.iter().filter(|&&i| i == FREE).count();
        assert_eq!(free, SLOTS - COMMANDS.len());
    }

    #[test]
    fn arity_counts_the_command_name() {
        let get = lookup(b"get").unwrap();
        assert!(!arity_ok(get, 1));
        assert!(arity_ok(get, 2));
        assert!(!arity_ok(get, 3));

        // A negative arity is a minimum, which is how SET takes its options.
        let set = lookup(b"set").unwrap();
        assert!(!arity_ok(set, 2));
        assert!(arity_ok(set, 3));
        assert!(arity_ok(set, 9));
    }

    /// A key spec that is wrong sends a cluster client to the wrong node, so
    /// the pair commands are worth stating twice.
    #[test]
    fn the_pair_commands_step_two_keys_at_a_time() {
        for name in [b"mset".as_slice(), b"msetnx"] {
            let c = lookup(name).unwrap();
            assert_eq!((c.first_key, c.last_key, c.step), (1, -1, 2));
        }
        let mget = lookup(b"mget").unwrap();
        assert_eq!((mget.first_key, mget.last_key, mget.step), (1, -1, 1));
        // MSETEX counts its keys in an argument, so there is no static spec
        // for them and a client has to ask with COMMAND GETKEYS.
        let msetex = lookup(b"msetex").unwrap();
        assert_eq!((msetex.first_key, msetex.last_key, msetex.step), (0, 0, 0));
        assert!(msetex.flags.contains(&"movablekeys"));
    }
}
