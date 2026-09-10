//! Cluster mode: the 16384 slots, who owns them, and what a client is told
//! when it asks the wrong node.
//!
//! A Redis cluster has no coordinator and no lookup service. Every key belongs
//! to one of 16384 slots, worked out from the key name alone, and every node
//! knows which node owns which slot. A client that guesses wrong is not proxied,
//! it is told where to go, and it remembers. So the whole of routing is a
//! function from a key to a number and a table from that number to a node, and
//! both of them are here.
//!
//! # The slot
//!
//! `CRC16(key) mod 16384`, with one rule on top: if the key contains a `{`
//! followed later by a `}` with at least one byte between them, only the bytes
//! between them are hashed. That is the hash tag, and it is the only way a
//! client has of making two keys land together, which is the only way a command
//! that names two keys can be run at all. `{user1000}.following` and
//! `{user1000}.followers` are one slot and one node.
//!
//! # The table
//!
//! 16384 entries, each naming the node that owns that slot or nobody. A slot
//! nobody owns is a hole, and a cluster with a hole in it is down: rather than
//! guess, every node refuses every key in the hole and, with
//! `cluster-require-full-coverage` left on, refuses every key at all. That is
//! severe on purpose. A cluster that answers for the slots it happens to have is
//! a cluster that silently loses the rest.
//!
//! Two more tables sit beside it for the slots that are moving. A slot this node
//! owns and is sending away is migrating, and a key in it that is already gone
//! is answered with `ASK`, which is a redirection for one command rather than
//! for the slot. A slot this node does not own and is receiving is importing, and
//! a command lands on it only if the connection said `ASKING` first. The pair is
//! what makes a slot move without a window where a key is on neither node.
//!
//! # What is here and what is not
//!
//! The slots, the table, the whole `CLUSTER` container, the redirections and the
//! configuration file that carries all of it across a restart. The bus, which is
//! the binary protocol nodes talk to each other over and the thing that actually
//! fills the table, is next door in [`bus`]. Nothing in this file dials anybody:
//! it reads the table and edits it, and a change an operator makes here is
//! announced by asking the bus to send a packet once the lock is gone.
//!
//! What is not here is the manual failover, so `CLUSTER FAILOVER` on a replica
//! is still refused and an operator who wants to move a master on purpose moves
//! the slots instead. A replica whose master dies promotes itself, which the
//! bus runs the election for.

use core::fmt::Write as _;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicU64};

use yo_common::lock::Lock;
use yo_common::{Code, Error, Result};

use crate::reply::Out;

use super::args::{self, Args};
use super::{Server, Session};

mod asm;
mod bus;
mod import;
use super::keyspec;
use super::table::Spec;
pub(super) use asm::Migration;

/// How many slots a cluster has, which is Redis's number and is not a setting.
///
/// It is 16384 rather than a round number because the bus gossips a bitmap of
/// every slot in every heartbeat, and 16384 bits is two kilobytes, which is
/// what the authors were willing to spend on a packet sent every second.
pub const SLOTS: usize = 16384;

/// How long a cluster has to be fully covered before it says it is up.
///
/// Redis waits before turning `cluster_state` from fail to ok, so a node that
/// has just been given its slots does not announce itself ready in the same
/// instant and then take it back when the next heartbeat disagrees. The number
/// is Redis's `CLUSTER_WRITABLE_DELAY`.
const WRITABLE_DELAY_MS: u64 = 2000;

/// How long a node that has been down waits before saying it is up again.
///
/// The two delays are different because they are about different things. The
/// first is about a node that has only just started and has never been anything
/// but ready. The second is about a node that was in the minority and has just
/// seen the coverage come back, which is the case where being hasty is how a
/// healed partition serves a key that has already been elected away somewhere
/// else. Redis works it out as the node timeout clamped between 500ms and 5s,
/// and with the default fifteen second timeout that is five seconds every time.
const REJOIN_DELAY_MS: u64 = 5000;

/// What the bus port is, which is the client port plus this.
const BUS_OFFSET: u16 = 10000;

/// How long a node id is, in hex characters.
const ID_LEN: usize = 40;

// ----------------------------------------------------------------- the slot

/// The CRC16 table Redis hashes keys with, which is CCITT with the XMODEM
/// parameters: polynomial 0x1021, no reflection, no initial value and no final
/// exclusive or.
///
/// Written out rather than computed at startup because it is what every key on
/// the command path goes through, and a table in the binary is a table that is
/// already in cache by the time the first command arrives.
#[rustfmt::skip]
const CRC16: [u16; 256] = [
    0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50a5, 0x60c6, 0x70e7,
    0x8108, 0x9129, 0xa14a, 0xb16b, 0xc18c, 0xd1ad, 0xe1ce, 0xf1ef,
    0x1231, 0x0210, 0x3273, 0x2252, 0x52b5, 0x4294, 0x72f7, 0x62d6,
    0x9339, 0x8318, 0xb37b, 0xa35a, 0xd3bd, 0xc39c, 0xf3ff, 0xe3de,
    0x2462, 0x3443, 0x0420, 0x1401, 0x64e6, 0x74c7, 0x44a4, 0x5485,
    0xa56a, 0xb54b, 0x8528, 0x9509, 0xe5ee, 0xf5cf, 0xc5ac, 0xd58d,
    0x3653, 0x2672, 0x1611, 0x0630, 0x76d7, 0x66f6, 0x5695, 0x46b4,
    0xb75b, 0xa77a, 0x9719, 0x8738, 0xf7df, 0xe7fe, 0xd79d, 0xc7bc,
    0x48c4, 0x58e5, 0x6886, 0x78a7, 0x0840, 0x1861, 0x2802, 0x3823,
    0xc9cc, 0xd9ed, 0xe98e, 0xf9af, 0x8948, 0x9969, 0xa90a, 0xb92b,
    0x5af5, 0x4ad4, 0x7ab7, 0x6a96, 0x1a71, 0x0a50, 0x3a33, 0x2a12,
    0xdbfd, 0xcbdc, 0xfbbf, 0xeb9e, 0x9b79, 0x8b58, 0xbb3b, 0xab1a,
    0x6ca6, 0x7c87, 0x4ce4, 0x5cc5, 0x2c22, 0x3c03, 0x0c60, 0x1c41,
    0xedae, 0xfd8f, 0xcdec, 0xddcd, 0xad2a, 0xbd0b, 0x8d68, 0x9d49,
    0x7e97, 0x6eb6, 0x5ed5, 0x4ef4, 0x3e13, 0x2e32, 0x1e51, 0x0e70,
    0xff9f, 0xefbe, 0xdfdd, 0xcffc, 0xbf1b, 0xaf3a, 0x9f59, 0x8f78,
    0x9188, 0x81a9, 0xb1ca, 0xa1eb, 0xd10c, 0xc12d, 0xf14e, 0xe16f,
    0x1080, 0x00a1, 0x30c2, 0x20e3, 0x5004, 0x4025, 0x7046, 0x6067,
    0x83b9, 0x9398, 0xa3fb, 0xb3da, 0xc33d, 0xd31c, 0xe37f, 0xf35e,
    0x02b1, 0x1290, 0x22f3, 0x32d2, 0x4235, 0x5214, 0x6277, 0x7256,
    0xb5ea, 0xa5cb, 0x95a8, 0x8589, 0xf56e, 0xe54f, 0xd52c, 0xc50d,
    0x34e2, 0x24c3, 0x14a0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405,
    0xa7db, 0xb7fa, 0x8799, 0x97b8, 0xe75f, 0xf77e, 0xc71d, 0xd73c,
    0x26d3, 0x36f2, 0x0691, 0x16b0, 0x6657, 0x7676, 0x4615, 0x5634,
    0xd94c, 0xc96d, 0xf90e, 0xe92f, 0x99c8, 0x89e9, 0xb98a, 0xa9ab,
    0x5844, 0x4865, 0x7806, 0x6827, 0x18c0, 0x08e1, 0x3882, 0x28a3,
    0xcb7d, 0xdb5c, 0xeb3f, 0xfb1e, 0x8bf9, 0x9bd8, 0xabbb, 0xbb9a,
    0x4a75, 0x5a54, 0x6a37, 0x7a16, 0x0af1, 0x1ad0, 0x2ab3, 0x3a92,
    0xfd2e, 0xed0f, 0xdd6c, 0xcd4d, 0xbdaa, 0xad8b, 0x9de8, 0x8dc9,
    0x7c26, 0x6c07, 0x5c64, 0x4c45, 0x3ca2, 0x2c83, 0x1ce0, 0x0cc1,
    0xef1f, 0xff3e, 0xcf5d, 0xdf7c, 0xaf9b, 0xbfba, 0x8fd9, 0x9ff8,
    0x6e17, 0x7e36, 0x4e55, 0x5e74, 0x2e93, 0x3eb2, 0x0ed1, 0x1ef0,
];

/// The CRC16 of `data`, which is only ever used to pick a slot.
#[must_use]
fn crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0;
    for &byte in data {
        let at = ((crc >> 8) ^ u16::from(byte)) & 0xff;
        crc = (crc << 8) ^ CRC16[at as usize];
    }
    crc
}

/// Which slot `key` belongs to.
///
/// The hash tag rule is Redis's `keyHashSlot` and its edge cases are worth being
/// exact about, because a client library implements the same rule and the two
/// have to agree on every key or the client routes to a node that will not serve
/// it. A `{` with no `}` after it hashes the whole key. A `{}` with nothing
/// between hashes the whole key. Only the first `}` after the first `{` counts,
/// so `{a}{b}` hashes `a`. And a key with no braces at all hashes whole, which
/// is nearly every key there has ever been.
#[must_use]
pub fn key_slot(key: &[u8]) -> u16 {
    let tagged = match key.iter().position(|&b| b == b'{') {
        Some(open) => match key[open + 1..].iter().position(|&b| b == b'}') {
            // The empty tag is not a tag, which is the `if (e == s+1)` arm.
            Some(0) | None => key,
            Some(len) => &key[open + 1..open + 1 + len],
        },
        None => key,
    };
    crc16(tagged) % SLOTS as u16
}

// ---------------------------------------------------------------- the table

/// The reference's own node flag bits.
///
/// Kept as its bits rather than as a set of booleans because they go out on the
/// bus in exactly this shape and come back in it, so anything else would be two
/// translations for no gain.
pub(crate) const FLAG_MASTER: u16 = 1;
pub(crate) const FLAG_SLAVE: u16 = 2;
pub(crate) const FLAG_PFAIL: u16 = 4;
pub(crate) const FLAG_FAIL: u16 = 8;
pub(crate) const FLAG_MYSELF: u16 = 16;
pub(crate) const FLAG_HANDSHAKE: u16 = 32;
pub(crate) const FLAG_NOADDR: u16 = 64;
pub(crate) const FLAG_MEET: u16 = 128;
pub(crate) const FLAG_MIGRATE_TO: u16 = 256;
pub(crate) const FLAG_NOFAILOVER: u16 = 512;
pub(crate) const FLAG_EXTENSIONS: u16 = 1024;

/// One node in the cluster, which is this one and everybody the bus has found.
#[derive(Clone)]
struct Node {
    /// The forty hex characters that name it, which never change while it lives.
    id: String,
    /// Where a client reaches it. Empty for this node, which is what a real
    /// server reports for itself until something tells it its own address.
    host: String,
    /// The port a client reaches it on.
    port: u16,
    /// The port the bus reaches it on, which is nearly always the client port
    /// plus ten thousand but is carried separately because a node behind a port
    /// map announces something else.
    bus: u16,
    /// Which shard it belongs to, which is a master and its replicas.
    shard: String,
    /// The config epoch it claims its slots under.
    epoch: u64,
    /// What it is and how it is doing, in the reference's bits.
    flags: u16,
    /// The master it follows, as an index into the same table.
    master: Option<u16>,
    /// When a ping went out to it with no pong back yet, or nought for none
    /// outstanding, which is the field the timeout is measured from.
    ping_sent: u64,
    /// When the last pong came back.
    pong_recv: u64,
    /// When anything at all last arrived from it. A node under a heavy pub/sub
    /// load can be slow with its pong and perfectly alive, so traffic of any
    /// kind counts as proof it is there.
    data_recv: u64,
    /// When this node decided it had failed.
    fail_time: u64,
    /// The replication offset it last put in a packet.
    offset: u64,
    /// Whether there is an outbound link to it at the moment.
    linked: bool,
    /// When this node last voted in an election about replacing it, which only
    /// means anything on a master and is what stops two of its replicas being
    /// promoted one after the other.
    voted_time: u64,
    /// Who has said it is unreachable and when they last said so, which is the
    /// weak quorum the FAIL flag needs.
    reports: Vec<(String, u64)>,
}

impl Node {
    /// A node with nothing known about it but where it is.
    fn new(id: String, host: String, port: u16, bus: u16, flags: u16, now: u64) -> Node {
        Node {
            id,
            host,
            port,
            bus,
            shard: String::from_utf8_lossy(&new_id()).into_owned(),
            epoch: 0,
            flags,
            master: None,
            ping_sent: 0,
            // Both stamps start at zero and not at now, which is the reference's
            // own rule and is what makes the `myself` line read `0 0`. Nothing
            // has been heard from a node that was made a moment ago, and saying
            // otherwise would also hold off the first ping for a second.
            pong_recv: 0,
            // This one does start at now, because it stands in for the creation
            // time as well and the handshake timeout is measured against it.
            data_recv: now,
            fail_time: 0,
            offset: 0,
            linked: false,
            voted_time: 0,
            reports: Vec::new(),
        }
    }

    /// Whether it is a master, which is the reference's rule that anything not
    /// flagged a replica counts as one.
    fn is_master(&self) -> bool {
        self.flags & FLAG_SLAVE == 0
    }

    /// Whether it is down as far as anybody can tell.
    fn down(&self) -> bool {
        self.flags & (FLAG_PFAIL | FLAG_FAIL) != 0
    }

    /// The comma separated flag list a `CLUSTER NODES` line carries, in the
    /// reference's own order, with `noflags` for a node with none of them.
    fn flag_names(&self, into: &mut String) {
        const NAMES: [(u16, &str); 8] = [
            (FLAG_MYSELF, "myself"),
            (FLAG_MASTER, "master"),
            (FLAG_SLAVE, "slave"),
            (FLAG_PFAIL, "fail?"),
            (FLAG_FAIL, "fail"),
            (FLAG_HANDSHAKE, "handshake"),
            (FLAG_NOADDR, "noaddr"),
            (FLAG_NOFAILOVER, "nofailover"),
        ];
        let mut first = true;
        for (bit, name) in NAMES {
            if self.flags & bit == 0 {
                continue;
            }
            if !first {
                into.push(',');
            }
            into.push_str(name);
            first = false;
        }
        if first {
            into.push_str("noflags");
        }
    }

    /// The `ip:port@bus` field of a `CLUSTER NODES` line.
    fn address(&self, into: &mut String) {
        let _ = write!(into, "{}:{}@{}", self.host, self.port, self.bus);
    }

    /// The same field with the aux fields on the end of it, which is what goes in
    /// the config file and not what goes on the wire. The reference keeps them
    /// off `CLUSTER NODES` on purpose, because a client that learned to parse
    /// them would break the next time somebody added one.
    fn address_on_disk(&self, into: &mut String) {
        self.address(into);
        let _ = write!(into, ",,tls-port=0,shard-id={}", self.shard);
    }
}

/// Who owns what, and what is moving.
///
/// One lock over all three tables rather than one each, because every reader
/// wants a consistent answer across them: a slot is owned, or migrating, or
/// importing, and a reader that saw two of those from different instants would
/// route a client to a node that had already handed the slot on.
struct Map {
    /// Every node this one knows about, with this one first.
    nodes: Vec<Node>,
    /// Which node owns each slot, as an index into `nodes`, or `None`.
    owner: Vec<Option<u16>>,
    /// The slots this node owns and is sending away, and to whom.
    migrating: Vec<Option<u16>>,
    /// The slots this node is receiving, and from whom.
    importing: Vec<Option<u16>>,
}

impl Map {
    /// An empty table with room for every slot.
    fn new(me: Node) -> Map {
        Map {
            nodes: vec![me],
            owner: vec![None; SLOTS],
            migrating: vec![None; SLOTS],
            importing: vec![None; SLOTS],
        }
    }

    /// The index of the node with this id, or `None` for one nobody knows.
    fn find(&self, id: &[u8]) -> Option<u16> {
        self.nodes
            .iter()
            .position(|n| n.id.as_bytes() == id)
            .map(|at| at as u16)
    }

    /// Drop a node and everything that pointed at it.
    ///
    /// Every other table here holds an index into `nodes`, so dropping one out
    /// of the middle means walking them and shifting anything above it down.
    /// That is fine because a node leaves a cluster about as often as an
    /// operator types, and holding ids instead of indices everywhere would cost
    /// a string compare on the routing path, which is the one path that has to
    /// be fast.
    fn forget(&mut self, at: u16) {
        self.nodes.remove(usize::from(at));
        let shift = |slot: &mut Option<u16>| match *slot {
            Some(node) if node == at => *slot = None,
            Some(node) if node > at => *slot = Some(node - 1),
            _ => {}
        };
        for slot in 0..SLOTS {
            shift(&mut self.owner[slot]);
            shift(&mut self.migrating[slot]);
            shift(&mut self.importing[slot]);
        }
        for node in &mut self.nodes {
            shift(&mut node.master);
        }
    }

    /// How many masters are serving at least one slot and have not failed, which
    /// is the electorate a FAIL vote is counted against.
    fn voters(&self) -> usize {
        let mut seen = vec![false; self.nodes.len()];
        for owner in self.owner.iter().flatten() {
            seen[*owner as usize] = true;
        }
        seen.iter().filter(|s| **s).count()
    }

    /// Whether this node owns `slot`, which is index nought owning it.
    fn mine(&self, slot: u16) -> bool {
        self.owner[slot as usize] == Some(0)
    }

    /// How many slots have an owner.
    fn assigned(&self) -> usize {
        self.owner.iter().filter(|o| o.is_some()).count()
    }

    /// How many nodes are serving at least one slot, which is Redis's
    /// `cluster_size` and is not the same as how many nodes there are.
    fn size(&self) -> usize {
        let mut seen = vec![false; self.nodes.len()];
        for owner in self.owner.iter().flatten() {
            seen[*owner as usize] = true;
        }
        seen.iter().filter(|s| **s).count()
    }

    /// The runs of consecutive slots one node owns, in order.
    fn runs(&self, node: u16) -> Vec<(u16, u16)> {
        let mut runs: Vec<(u16, u16)> = Vec::new();
        for slot in 0..SLOTS as u16 {
            if self.owner[slot as usize] != Some(node) {
                continue;
            }
            match runs.last_mut() {
                Some(last) if last.1 + 1 == slot => last.1 = slot,
                _ => runs.push((slot, slot)),
            }
        }
        runs
    }
}

// -------------------------------------------------------------- the settings

/// Everything about cluster mode, all of it idle on a server that was not
/// started with it on, which is nearly every server.
pub(crate) struct Cluster {
    /// Whether `--cluster-enabled yes` was given. Immutable for the life of the
    /// process, which is Redis's rule and the only sane one: a server that could
    /// be turned into a cluster node while it was holding keys would be a server
    /// whose keys were suddenly in slots it did not own.
    on: bool,
    /// The tables, empty on a server that is not a cluster node.
    map: Lock<Map>,
    /// Redis's `currentEpoch`, which is the highest epoch anybody has claimed.
    epoch: AtomicU64,
    /// When the coverage last became complete, for the delay before this node
    /// says the cluster is up. Nought means it is not covered.
    covered_at: AtomicU64,
    /// When cluster mode was turned on, which is the other end of the shorter
    /// delay.
    booted_at: AtomicU64,
    /// Whether this node has been uncovered at any point since it started, which
    /// is what picks between the two delays.
    was_down: AtomicBool,
    /// Whether every slot has to have an owner before anything is served, which
    /// is Redis's `cluster-require-full-coverage` and is on by default.
    full_coverage: AtomicBool,
    /// Whether a read is still served while the cluster is down, which is
    /// Redis's `cluster-allow-reads-when-down` and is off by default.
    ///
    /// Off is the safe answer and it is the one a real server ships with. A node
    /// that cannot see the whole cluster does not know whether the part it
    /// cannot see has already elected somebody else, so a read it served could
    /// be a read of a key that has moved on. An operator who would rather have
    /// stale reads than no reads turns it on and knows what they bought.
    reads_when_down: AtomicBool,
    /// Where the table is written, under the server's `dir`.
    file: Lock<String>,
    /// The links, the blacklist and the secret, none of which exist until the
    /// bus is started and none of which a command reads on the hot path.
    bus: bus::Bus,
    /// The election this node is standing in or voting in, if any.
    vote: bus::Vote,
    /// The slot migration this node is in, and the ones it has been in.
    asm: asm::Asm,
}

impl Default for Cluster {
    fn default() -> Cluster {
        Cluster {
            on: false,
            map: Lock::new(Map {
                nodes: Vec::new(),
                owner: Vec::new(),
                migrating: Vec::new(),
                importing: Vec::new(),
            }),
            epoch: AtomicU64::new(0),
            covered_at: AtomicU64::new(0),
            booted_at: AtomicU64::new(0),
            was_down: AtomicBool::new(false),
            full_coverage: AtomicBool::new(true),
            reads_when_down: AtomicBool::new(false),
            file: Lock::new(String::new()),
            bus: bus::Bus::default(),
            vote: bus::Vote::default(),
            asm: asm::Asm::default(),
        }
    }
}

impl Server {
    /// Whether this server is a cluster node, which is one relaxed read of a
    /// field that is false on nearly every server there is.
    #[must_use]
    pub fn cluster_enabled(&self) -> bool {
        self.cluster.on
    }

    /// Turn cluster mode on, which only whoever built the server may do.
    ///
    /// Called before the first connection is accepted and never again, which is
    /// why it takes `&mut self`: there is no lock here because there is nobody
    /// to race with.
    pub fn enable_cluster(&mut self, file: &str, port: u16) {
        self.cluster.on = true;
        self.cluster.booted_at.store(self.now_ms(), Relaxed);
        let now = self.now_ms();
        let me = yo_alloc::allow(|| {
            Node::new(
                String::from_utf8_lossy(&new_id()).into_owned(),
                String::new(),
                port,
                port + BUS_OFFSET,
                FLAG_MYSELF | FLAG_MASTER,
                now,
            )
        });
        // Under the server's directory when the name is a bare one, which is
        // what a real server does with `cluster-config-file`: it takes the
        // directory first and then opens everything relative to it. No name at
        // all is a node that keeps nothing across a restart, which is not
        // something a real server offers and is what a test wants.
        let path = yo_alloc::allow(|| {
            if file.is_empty() {
                String::new()
            } else {
                self.dir().join(file).to_string_lossy().into_owned()
            }
        });
        yo_alloc::allow(|| {
            *self.cluster.map.lock() = Map::new(me);
            *self.cluster.file.lock() = path;
            // The secret nodes recognise each other by is made here rather than
            // when the bus starts, because a node with no bus still has to have
            // one: the whole cluster ends up on the smallest secret anybody
            // started with, and a node holding an empty string would win that
            // and leave everybody with no secret at all.
            *self.cluster.bus.secret.lock() = String::from_utf8_lossy(&new_id()).into_owned();
        });
        // Whatever was written last time this node ran, if anything was. A node
        // that comes back without its slots is a node that has silently given
        // its half of the keyspace away, so a file that is there and unreadable
        // is worth more noise than a file that is not there at all.
        if let Err(e) = self.reload_cluster() {
            eprintln!("cluster config file could not be read: {e}");
        }
        self.recount_coverage();
    }

    /// The secret one node authenticates to another node's client port with.
    ///
    /// Forty hex characters on a cluster node and empty on anything else, which
    /// is what makes `AUTH "internal connection"` impossible to get past on a
    /// server that is not in a cluster: there is no secret to guess.
    ///
    /// Handed out as a copy because it is read on `AUTH` and nowhere else, so
    /// one allocation on a command that runs once a connection is cheaper than
    /// keeping the lock held across a comparison.
    pub(crate) fn cluster_secret(&self) -> String {
        let held = self.cluster.bus.secret.lock();
        yo_alloc::allow(|| held.clone())
    }

    /// Whether every slot has to be covered before anything is served.
    pub(crate) fn cluster_full_coverage(&self) -> bool {
        self.cluster.full_coverage.load(Relaxed)
    }

    /// Whether a read is served while the cluster is down.
    pub(crate) fn cluster_reads_when_down(&self) -> bool {
        self.cluster.reads_when_down.load(Relaxed)
    }

    /// Set either of the two, which `CONFIG SET` does and which really move: a
    /// cluster with a hole in it starts answering for the slots it does have.
    pub(crate) fn set_cluster_coverage(&self, full: bool, reads_when_down: bool) {
        self.cluster.full_coverage.store(full, Relaxed);
        self.cluster.reads_when_down.store(reads_when_down, Relaxed);
        self.recount_coverage();
    }

    /// Where the table is written, which `CONFIG GET` reports.
    pub(crate) fn cluster_file(&self) -> String {
        let file = self.cluster.file.lock();
        yo_alloc::allow(|| file.clone())
    }

    /// This node's id, which is what `CLUSTER MYID` answers.
    pub(crate) fn cluster_id(&self) -> String {
        let map = self.cluster.map.lock();
        yo_alloc::allow(|| map.nodes.first().map_or_else(String::new, |n| n.id.clone()))
    }

    /// Whether the cluster is up, which is every slot covered for long enough.
    ///
    /// A hole makes it down, and so does having just been covered, for the delay
    /// a real server waits. `cluster-require-full-coverage no` turns the first
    /// half off, and then a node with any slots at all says it is up and refuses
    /// only the keys in the hole.
    pub(crate) fn cluster_up(&self) -> bool {
        let at = self.cluster.covered_at.load(Relaxed);
        if at == 0 {
            return false;
        }
        let (since, wait) = if self.cluster.was_down.load(Relaxed) {
            (at, REJOIN_DELAY_MS)
        } else {
            (self.cluster.booted_at.load(Relaxed), WRITABLE_DELAY_MS)
        };
        self.now_ms().saturating_sub(since) >= wait
    }

    /// Look at the coverage again and start or stop the clock on it.
    ///
    /// Called after anything that moves a slot. Cheap enough to do on the spot
    /// rather than in a cron, and doing it on the spot is what makes
    /// `CLUSTER ADDSLOTS` followed by `CLUSTER INFO` agree.
    fn recount_coverage(&self) {
        let covered = {
            let map = self.cluster.map.lock();
            let assigned = map.assigned();
            if self.cluster_full_coverage() {
                assigned == SLOTS
            } else {
                assigned > 0
            }
        };
        if covered {
            let _ =
                self.cluster
                    .covered_at
                    .compare_exchange(0, self.now_ms().max(1), Relaxed, Relaxed);
        } else {
            self.cluster.covered_at.store(0, Relaxed);
            self.cluster.was_down.store(true, Relaxed);
        }
    }
}

/// Forty hex characters from the same entropy the replication id comes from.
fn new_id() -> [u8; ID_LEN] {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut raw = [0u8; ID_LEN / 2];
    yo_common::entropy::fill(&mut raw);
    let mut id = [0u8; ID_LEN];
    for (i, byte) in raw.iter().enumerate() {
        id[i * 2] = HEX[usize::from(byte >> 4)];
        id[i * 2 + 1] = HEX[usize::from(byte & 15)];
    }
    id
}

// ------------------------------------------------------------ the redirects

/// Whether the command carries its own `ASKING` rather than needing one sent.
///
/// One command does, `RESTORE-ASKING`, and the reference spells that as a flag on
/// the command rather than as a name the routing gate knows, so this reads the
/// flag. It is what makes a slot migration work at all: the node being sent the
/// keys does not own the slot yet, and a client that had to send `ASKING` in
/// front of every key would double the round trips for no reason.
pub(super) fn asks(spec: &Spec) -> bool {
    spec.flags.contains(&"asking")
}

/// Where a command's keys say it should run, or `None` for run it here.
///
/// Called for every command on a cluster node and for none at all on a server
/// that is not one, which is the reason the enabled flag is read first and is a
/// plain field rather than anything atomic. A command that names no key never
/// redirects, whatever the state of the cluster, which is what lets a client
/// send `PING` and `SUBSCRIBE` to a node that is not serving anything.
///
/// The order the checks go in is `getNodeByQuery`'s, and it is not the order
/// anybody would guess, so it is worth writing down. The first key decides the
/// slot and the node, and a first key in a slot nobody owns is
/// `CLUSTERDOWN Hash slot not served` before the rest of the keys have even been
/// looked at, so a command whose keys are in two slots and whose first slot is a
/// hole is told about the hole and not about the two slots. Then the rest of the
/// keys, and one in another slot is `CROSSSLOT`. Then the health of the cluster
/// as a whole. Then the slot being handed over, which is where `ASK` and
/// `TRYAGAIN` come from. And `MOVED` last, because it is the answer left when
/// nothing else applies and the node is not this one.
pub(super) fn gate(
    server: &Server,
    db: usize,
    asking: bool,
    spec: &Spec,
    args: Args<'_>,
) -> Option<Error> {
    // Nothing to route. Cheap and first, because a container command and every
    // connection command land here too.
    if !keyspec::takes_keys(spec, args, 0) {
        return None;
    }
    let mut slot: Option<u16> = None;
    let mut crossed = false;
    let mut keys = 0usize;
    let mut present = 0usize;
    let mut missing = 0usize;
    let (owner, migrating, importing, here) = {
        let map = server.cluster.map.lock();
        // The slot of the first key, and then whether everything else agrees.
        keyspec::find(spec, args, 0, &mut |run| {
            for i in 0..run.count {
                let at = run.first + i * run.step;
                if at >= args.len() {
                    continue;
                }
                let this = key_slot(args.get(at));
                keys += 1;
                match slot {
                    None => slot = Some(this),
                    Some(first) if first != this => crossed = true,
                    Some(_) => {}
                }
            }
        });
        let at = usize::from(slot?);
        (
            map.owner[at],
            map.migrating[at].map(|to| node_at(&map, to)),
            map.importing[at].is_some(),
            map.owner[at] == Some(0),
        )
    };
    let slot = slot?;
    // The first key's slot having no owner beats everything, including a command
    // whose keys are in two slots.
    let Some(owner) = owner else {
        return Some(Error::new(
            Code::Invalid,
            "CLUSTERDOWN Hash slot not served",
        ));
    };
    if crossed {
        return Some(Error::new(
            Code::Invalid,
            "CROSSSLOT Keys in request don't hash to the same slot",
        ));
    }
    // A cluster with a hole in it somewhere else. Reads are still served when
    // the operator asked for that, writes never are, because a write to a
    // cluster that cannot see all of itself is a write that may be about to be
    // written somewhere else as well.
    if !server.cluster_up() {
        if !server.cluster.reads_when_down.load(Relaxed) {
            return Some(Error::new(Code::Invalid, "CLUSTERDOWN The cluster is down"));
        }
        if spec.flags.contains(&"write") {
            return Some(Error::new(
                Code::Invalid,
                "CLUSTERDOWN The cluster is down and only accepts read commands",
            ));
        }
    }
    // The slot is moving, so which keys are still here decides the answer. This
    // is the only part of the gate that touches the keyspace, and it only runs
    // for a slot that somebody is in the middle of handing over.
    if migrating.is_some() || importing {
        let held = &server.dbs[db];
        keyspec::find(spec, args, 0, &mut |run| {
            for i in 0..run.count {
                let at = run.first + i * run.step;
                if at >= args.len() {
                    continue;
                }
                let key = args.get(at);
                let mut stripe = held.hold(key);
                if stripe.exists(key) {
                    present += 1;
                } else {
                    missing += 1;
                }
            }
        });
    }
    if let Some(to) = migrating
        && missing > 0
    {
        // Some of them are here and some have gone, so there is no node that can
        // answer the whole command and the client is told to come back.
        if present > 0 {
            return Some(Error::new(
                Code::Invalid,
                "TRYAGAIN Multiple keys request during rehashing of slot",
            ));
        }
        return Some(redirect("ASK", slot, &to));
    }
    if importing && asking {
        if keys > 1 && missing > 0 {
            return Some(Error::new(
                Code::Invalid,
                "TRYAGAIN Multiple keys request during rehashing of slot",
            ));
        }
        return None;
    }
    if here {
        return None;
    }
    let (host, port) = {
        let map = server.cluster.map.lock();
        node_at(&map, owner)
    };
    Some(redirect("MOVED", slot, &(host, port)))
}

/// Where a node is reachable, copied out while the lock is held.
fn node_at(map: &Map, at: u16) -> (String, u16) {
    let node = &map.nodes[at as usize];
    (yo_alloc::allow(|| node.host.clone()), node.port)
}

/// A `MOVED` or an `ASK` line, which are the same shape with a different word.
///
/// A node with no address of its own is written as the loopback, because a
/// redirection has to name somewhere a client can connect to and an empty host
/// is not one. A real server does the same when it has nothing better.
fn redirect(word: &str, slot: u16, node: &(String, u16)) -> Error {
    let host = if node.0.is_empty() {
        "127.0.0.1"
    } else {
        node.0.as_str()
    };
    Error::fmt(
        Code::Invalid,
        format_args!("{word} {slot} {host}:{}", node.1),
    )
}

// ------------------------------------------------------------- the container

/// `CLUSTER <subcommand> ...`.
pub(super) fn execute(
    server: &Server,
    session: &mut Session,
    args: Args<'_>,
    out: &mut Out,
) -> Result<()> {
    let session_db = session.db;
    let sub = args.get(1);
    // Every subcommand but `HELP` and the two that are only about arguments is
    // refused outright on a server that was not started as a cluster node, which
    // is Redis's rule and covers the whole container including `HELP` itself.
    if !server.cluster_enabled() {
        return match arity_of(sub) {
            Some(n) if !arity_ok(n, args.len()) => Err(wrong_sub_arity(sub)),
            Some(_) => Err(disabled()),
            None => Err(args::unknown_subcommand(sub, "CLUSTER")),
        };
    }
    let Some(n) = arity_of(sub) else {
        return Err(args::unknown_subcommand(sub, "CLUSTER"));
    };
    if !arity_ok(n, args.len()) {
        return Err(wrong_sub_arity(sub));
    }
    match () {
        () if args::is(sub, b"myid") => out.bulk(server.cluster_id().as_bytes()),
        () if args::is(sub, b"myshardid") => {
            let map = server.cluster.map.lock();
            out.bulk(map.nodes[0].shard.as_bytes());
        }
        () if args::is(sub, b"keyslot") => out.int(i64::from(key_slot(args.get(2)))),
        () if args::is(sub, b"info") => info(server, out),
        () if args::is(sub, b"nodes") => nodes(server, out),
        () if args::is(sub, b"slots") => reply_slots(server, out),
        () if args::is(sub, b"shards") => shards(server, out),
        () if args::is(sub, b"links") => server.cluster_links(out),
        () if args::is(sub, b"slaves") || args::is(sub, b"replicas") => {
            replicas(server, args.get(2), out)?;
        }
        () if args::is(sub, b"count-failure-reports") => {
            let map = server.cluster.map.lock();
            let Some(at) = map.find(args.get(2)) else {
                return Err(unknown_node(args.get(2)));
            };
            out.int(map.nodes[usize::from(at)].reports.len() as i64);
        }
        () if args::is(sub, b"countkeysinslot") => count_keys(server, session_db, args, out)?,
        () if args::is(sub, b"getkeysinslot") => get_keys(server, session_db, args, out)?,
        () if args::is(sub, b"addslots") => {
            add_or_del(server, args, true, false)?;
            out.ok();
        }
        () if args::is(sub, b"delslots") => {
            add_or_del(server, args, false, false)?;
            out.ok();
        }
        () if args::is(sub, b"addslotsrange") => {
            add_or_del(server, args, true, true)?;
            out.ok();
        }
        () if args::is(sub, b"delslotsrange") => {
            add_or_del(server, args, false, true)?;
            out.ok();
        }
        () if args::is(sub, b"setslot") => setslot(server, session_db, args, out)?,
        () if args::is(sub, b"flushslots") => flushslots(server, out)?,
        () if args::is(sub, b"bumpepoch") => bumpepoch(server, out)?,
        () if args::is(sub, b"set-config-epoch") => set_config_epoch(server, args, out)?,
        () if args::is(sub, b"reset") => {
            if args.len() > 3 {
                return Err(sub_syntax(sub));
            }
            reset(server, args, out)?;
        }
        () if args::is(sub, b"slot-stats") => slot_stats(server, session_db, args, out)?,
        () if args::is(sub, b"migration") => migration(server, args, out)?,
        () if args::is(sub, b"syncslots") => syncslots(server, session, args, out)?,
        () if args::is(sub, b"saveconfig") => {
            save(server)?;
            out.ok();
        }
        () if args::is(sub, b"forget") => {
            forget(server, args.get(2))?;
            out.ok();
        }
        () if args::is(sub, b"replicate") => {
            replicate(server, session_db, args.get(2))?;
            out.ok();
        }
        () if args::is(sub, b"failover") => {
            if args.len() > 3 {
                return Err(sub_syntax(sub));
            }
            if args.len() == 3
                && !args::is(args.get(2), b"force")
                && !args::is(args.get(2), b"takeover")
            {
                return Err(args::syntax());
            }
            return Err(Error::new(
                Code::Invalid,
                "You should send CLUSTER FAILOVER to a replica",
            ));
        }
        () if args::is(sub, b"meet") => {
            if args.len() > 5 {
                return Err(sub_syntax(sub));
            }
            let (host, port, bus) = meet(&args)?;
            server.cluster_meet(&host, port, bus);
            out.ok();
        }
        () if args::is(sub, b"help") => help(out),
        _ => return Err(args::unknown_subcommand(sub, "CLUSTER")),
    }
    Ok(())
}

/// What every subcommand answers on a server that is not a cluster node.
pub(super) fn disabled() -> Error {
    Error::new(Code::Invalid, "This instance has cluster support disabled")
}

/// The reference's sentence for a node id nobody has heard of.
fn unknown_node(id: &[u8]) -> Error {
    Error::fmt(
        Code::Invalid,
        format_args!("Unknown node {}", String::from_utf8_lossy(id)),
    )
}

/// The same, in the spelling `CLUSTER SETSLOT` uses, which is a different
/// sentence for the same thing and is the reference's.
fn dont_know(id: &[u8]) -> Error {
    Error::fmt(
        Code::Invalid,
        format_args!("I don't know about node {}", String::from_utf8_lossy(id)),
    )
}

/// The arity of each subcommand, taken from the reference's own table, or `None`
/// for a word that is not a subcommand.
///
/// A hand written list for the same reason `CONTAINERS` is one: the command
/// table has a row for the container and none for what is behind it. It goes
/// away with D-114.
fn arity_of(sub: &[u8]) -> Option<i32> {
    const TABLE: &[(&str, i32)] = &[
        ("addslots", -3),
        ("addslotsrange", -4),
        ("bumpepoch", 2),
        ("count-failure-reports", 3),
        ("countkeysinslot", 3),
        ("delslots", -3),
        ("delslotsrange", -4),
        ("failover", -2),
        ("flushslots", 2),
        ("forget", 3),
        ("getkeysinslot", 4),
        ("help", 2),
        ("info", 2),
        ("keyslot", 3),
        ("links", 2),
        ("meet", -4),
        ("migration", -4),
        ("myid", 2),
        ("myshardid", 2),
        ("nodes", 2),
        ("replicas", 3),
        ("replicate", 3),
        ("reset", -2),
        ("saveconfig", 2),
        ("set-config-epoch", 3),
        ("setslot", -4),
        ("shards", 2),
        ("slaves", 3),
        ("slot-stats", -4),
        ("slots", 2),
        ("syncslots", -3),
    ];
    TABLE
        .iter()
        .find(|(name, _)| args::is(sub, name.as_bytes()))
        .map(|(_, arity)| *arity)
}

/// Redis's arity rule: a positive number is exact and a negative one is a floor.
fn arity_ok(arity: i32, len: usize) -> bool {
    let len = len as i32;
    if arity >= 0 {
        len == arity
    } else {
        len >= -arity
    }
}

/// What a subcommand says when its own parsing gave up.
///
/// The container's arity is a floor, so a subcommand that takes a fixed number
/// of words past that floor has to check the count itself, and when it fails the
/// reference does not say the arity was wrong, it says this. The subcommand is
/// echoed back as the caller typed it and the container is upper cased, which is
/// `addReplySubcommandSyntaxError` word for word.
fn sub_syntax(sub: &[u8]) -> Error {
    Error::fmt(
        Code::Unsupported,
        format_args!(
            "unknown subcommand or wrong number of arguments for '{}'. Try CLUSTER HELP.",
            String::from_utf8_lossy(sub)
        ),
    )
}

/// The wrong arity complaint, which names the subcommand and not the container.
fn wrong_sub_arity(sub: &[u8]) -> Error {
    Error::fmt(
        Code::Invalid,
        format_args!(
            "wrong number of arguments for 'cluster|{}' command",
            String::from_utf8_lossy(sub).to_lowercase()
        ),
    )
}

// -------------------------------------------------------------- the reports

/// `CLUSTER INFO`, which is the same field names in the same order a real server
/// prints, as one bulk string rather than as a map.
fn info(server: &Server, out: &mut Out) {
    let (assigned, size, known, my_epoch) = {
        let map = server.cluster.map.lock();
        (
            map.assigned(),
            map.size(),
            map.nodes.len(),
            map.nodes[0].epoch,
        )
    };
    let state = if server.cluster_up() { "ok" } else { "fail" };
    let text = yo_alloc::allow(|| {
        let mut s = String::with_capacity(512);
        let _ = write!(
            s,
            "cluster_state:{state}\r\ncluster_slots_assigned:{assigned}\r\n\
             cluster_slots_ok:{assigned}\r\ncluster_slots_pfail:0\r\ncluster_slots_fail:0\r\n\
             cluster_known_nodes:{known}\r\ncluster_size:{size}\r\n\
             cluster_current_epoch:{}\r\ncluster_my_epoch:{my_epoch}\r\n\
             cluster_stats_messages_sent:0\r\ncluster_stats_messages_received:0\r\n\
             total_cluster_links_buffer_limit_exceeded:0\r\n\
             cluster_slot_migration_active_tasks:0\r\n\
             cluster_slot_migration_active_trim_running:0\r\n\
             cluster_slot_migration_active_trim_current_job_keys:0\r\n\
             cluster_slot_migration_active_trim_current_job_trimmed:0\r\n\
             cluster_slot_migration_stats_active_trim_started:0\r\n\
             cluster_slot_migration_stats_active_trim_completed:0\r\n\
             cluster_slot_migration_stats_active_trim_cancelled:0\r\n",
            server.cluster.epoch.load(Relaxed),
        );
        s
    });
    out.verbatim(b"txt", text.as_bytes());
}

/// `CLUSTER NODES`, which is the same text this node writes to its config file.
fn nodes(server: &Server, out: &mut Out) {
    let text = yo_alloc::allow(|| lines(server, false));
    out.verbatim(b"txt", text.as_bytes());
}

/// One line per node, in the reference's field order.
///
/// `<id> <ip:port@bus,aux> <flags> <master> <ping-sent> <pong-recv> <epoch>
/// <link-state> <slot> ...`, with the migrating and importing slots on the end
/// of the owner's line in square brackets.
fn lines(server: &Server, on_disk: bool) -> String {
    let map = server.cluster.map.lock();
    let mut s = String::with_capacity(256);
    for at in 0..map.nodes.len() as u16 {
        describe(&map, at, on_disk, &mut s);
        s.push('\n');
    }
    s
}

/// One node's line, without the newline, which `CLUSTER REPLICAS` wants one at a
/// time and `CLUSTER NODES` wants all of.
fn describe(map: &Map, at: u16, on_disk: bool, s: &mut String) {
    let node = &map.nodes[usize::from(at)];
    s.push_str(&node.id);
    s.push(' ');
    if on_disk {
        node.address_on_disk(s);
    } else {
        node.address(s);
    }
    s.push(' ');
    node.flag_names(s);
    s.push(' ');
    match node.master.and_then(|m| map.nodes.get(usize::from(m))) {
        Some(master) => s.push_str(&master.id),
        None => s.push('-'),
    }
    // A replica reports its master's epoch rather than its own, which is what a
    // client reading the line is actually asking about.
    let epoch = match node.master.and_then(|m| map.nodes.get(usize::from(m))) {
        Some(master) => master.epoch,
        None => node.epoch,
    };
    let link = if node.linked || at == 0 {
        "connected"
    } else {
        "disconnected"
    };
    let _ = write!(s, " {} {} {epoch} {link}", node.ping_sent, node.pong_recv);
    for (from, to) in map.runs(at) {
        if from == to {
            let _ = write!(s, " {from}");
        } else {
            let _ = write!(s, " {from}-{to}");
        }
    }
    if at == 0 {
        for slot in 0..SLOTS {
            if let Some(to) = map.migrating[slot] {
                let _ = write!(s, " [{slot}->-{}]", map.nodes[to as usize].id);
            }
            if let Some(from) = map.importing[slot] {
                let _ = write!(s, " [{slot}-<-{}]", map.nodes[from as usize].id);
            }
        }
    }
}

/// `CLUSTER SLOTS`, which is one entry per run of slots one node owns.
///
/// The runs come out in slot order and not in node order, because that is the
/// order the reference walks and a client that caches this reply by position
/// will notice the difference.
fn reply_slots(server: &Server, out: &mut Out) {
    let mine = server.repl_offset();
    let map = server.cluster.map.lock();
    let at = out.len();
    let mut n = 0;
    let mut run: Option<(u16, u16)> = None;
    for slot in 0..=SLOTS as u16 {
        let owner = if slot as usize == SLOTS {
            None
        } else {
            map.owner[slot as usize]
        };
        match run {
            Some((node, _)) if owner == Some(node) => {}
            Some((node, from)) => {
                slot_run(&map, node, from, slot - 1, mine, out);
                n += 1;
                run = owner.map(|node| (node, slot));
            }
            None => run = owner.map(|node| (node, slot)),
        }
    }
    out.close_array(at, n);
}

/// One `CLUSTER SLOTS` entry: the run, its owner, then the owner's replicas.
///
/// A replica everybody agrees is gone is left out, and so is one whose
/// replication offset is still zero, which is the reference's own filter and is
/// its way of saying this replica has never caught up with anything and is not
/// somewhere to send a reader yet. This node's own offset does not come off the
/// table, since nothing gossips a node its own offset, so it is passed in.
fn slot_run(map: &Map, node: u16, from: u16, to: u16, mine: u64, out: &mut Out) {
    let replicas: Vec<&Node> = map
        .nodes
        .iter()
        .enumerate()
        .filter(|(at, n)| {
            let offset = if *at == 0 { mine } else { n.offset };
            n.master == Some(node) && n.flags & FLAG_FAIL == 0 && offset != 0
        })
        .map(|(_, n)| n)
        .collect();
    out.array(3 + replicas.len());
    out.int(i64::from(from));
    out.int(i64::from(to));
    let held = &map.nodes[node as usize];
    for held in std::iter::once(held).chain(replicas.iter().copied()) {
        out.array(4);
        out.bulk(held.host.as_bytes());
        out.int(i64::from(held.port));
        out.bulk(held.id.as_bytes());
        out.array(0);
    }
}

/// `CLUSTER SHARDS`, which is the same information grouped by shard rather than
/// by run, and is what a client library reads to find the replicas of a master.
fn shards(server: &Server, out: &mut Out) {
    let map = server.cluster.map.lock();
    let at = out.len();
    let mut n = 0;
    let mut done: Vec<&str> = Vec::new();
    for node in 0..map.nodes.len() {
        let shard = map.nodes[node].shard.as_str();
        if done.contains(&shard) {
            continue;
        }
        done.push(shard);
        let members: Vec<usize> = (0..map.nodes.len())
            .filter(|other| map.nodes[*other].shard == shard)
            .collect();
        // The slots of a shard are the slots of whichever member is holding
        // them, which is the master of it except for the moment after a
        // failover when two members still think they are.
        let runs: Vec<(u16, u16)> = members
            .iter()
            .flat_map(|member| map.runs(*member as u16))
            .collect();
        out.map(2);
        out.bulk(b"slots");
        out.array(runs.len() * 2);
        for (from, to) in runs {
            out.int(i64::from(from));
            out.int(i64::from(to));
        }
        out.bulk(b"nodes");
        out.array(members.len());
        for member in members {
            let held = &map.nodes[member];
            out.map(7);
            out.bulk(b"id");
            out.bulk(held.id.as_bytes());
            out.bulk(b"port");
            out.int(i64::from(held.port));
            out.bulk(b"ip");
            out.bulk(held.host.as_bytes());
            out.bulk(b"endpoint");
            out.bulk(held.host.as_bytes());
            out.bulk(b"role");
            out.bulk(if held.is_master() {
                b"master".as_slice()
            } else {
                b"replica".as_slice()
            });
            out.bulk(b"replication-offset");
            out.int(if member == 0 {
                server.repl_offset() as i64
            } else {
                held.offset as i64
            });
            out.bulk(b"health");
            out.bulk(match held.flags {
                f if f & FLAG_FAIL != 0 => b"fail".as_slice(),
                f if f & FLAG_PFAIL != 0 => b"loading".as_slice(),
                _ => b"online".as_slice(),
            });
        }
        n += 1;
    }
    out.close_array(at, n);
}

/// The help text, word for word from the reference.
fn help(out: &mut Out) {
    const LINES: &[&str] = &[
        "CLUSTER <subcommand> [<arg> [value] [opt] ...]. Subcommands are:",
        "COUNTKEYSINSLOT <slot>",
        "    Return the number of keys in <slot>.",
        "GETKEYSINSLOT <slot> <count>",
        "    Return key names stored by current node in a slot.",
        "INFO",
        "    Return information about the cluster.",
        "KEYSLOT <key>",
        "    Return the hash slot for <key>.",
        "MYID",
        "    Return the node id.",
        "MYSHARDID",
        "    Return the node's shard id.",
        "NODES",
        "    Return cluster configuration seen by node. Output format:",
        "    <id> <ip:port@bus-port[,hostname]> <flags> <master> <pings> <pongs> <epoch> <link> <slot> ...",
        "REPLICAS <node-id>",
        "    Return <node-id> replicas.",
        "SLOTS",
        "    Return information about slots range mappings. Each range is made of:",
        "    start, end, master and replicas IP addresses, ports and ids",
        "SLOT-STATS",
        "    Return an array of slot usage statistics for slots assigned to the current node.",
        "SHARDS",
        "    Return information about slot range mappings and the nodes associated with them.",
        "ADDSLOTS <slot> [<slot> ...]",
        "    Assign slots to current node.",
        "ADDSLOTSRANGE <start slot> <end slot> [<start slot> <end slot> ...]",
        "    Assign slots which are between <start-slot> and <end-slot> to current node.",
        "BUMPEPOCH",
        "    Advance the cluster config epoch.",
        "COUNT-FAILURE-REPORTS <node-id>",
        "    Return number of failure reports for <node-id>.",
        "DELSLOTS <slot> [<slot> ...]",
        "    Delete slots information from current node.",
        "DELSLOTSRANGE <start slot> <end slot> [<start slot> <end slot> ...]",
        "    Delete slots information which are between <start-slot> and <end-slot> from current node.",
        "FAILOVER [FORCE|TAKEOVER]",
        "    Promote current replica node to being a master.",
        "FORGET <node-id>",
        "    Remove a node from the cluster.",
        "FLUSHSLOTS",
        "    Delete current node own slots information.",
        "MEET <ip> <port> [<bus-port>]",
        "    Connect nodes into a working cluster.",
        "REPLICATE <node-id>",
        "    Configure current node as replica to <node-id>.",
        "RESET [HARD|SOFT]",
        "    Reset current node (default: soft).",
        "SET-CONFIG-EPOCH <epoch>",
        "    Set config epoch of current node.",
        "SETSLOT <slot> (IMPORTING <node-id>|MIGRATING <node-id>|STABLE|NODE <node-id>)",
        "    Set slot state.",
        "SAVECONFIG",
        "    Force saving cluster configuration on disk.",
        "LINKS",
        "    Return information about all network links between this node and its peers.",
        "    Output format is an array where each array element is a map containing attributes of a link",
        "MIGRATION IMPORT <start-slot end-slot [start-slot end-slot ...]> |",
        "          STATUS [ID <task-id> | ALL] | CANCEL [ID <task-id> | ALL]",
        "    Start, monitor and cancel slot migration.",
        "HELP",
        "    Print this help.",
    ];
    out.array(LINES.len());
    for line in LINES {
        out.simple(line.as_bytes());
    }
}

// ------------------------------------------------------------ the slot moves

/// One slot number from an argument, refused the way the reference refuses one.
///
/// One sentence covers both ways of getting it wrong, which is the reference's
/// `getSlotOrReply` and is not the same as what `COUNTKEYSINSLOT` says: a word
/// that is not a number and a number that is not a slot both come back as
/// `Invalid or out of range slot` here.
fn slot_arg(args: &Args<'_>, at: usize) -> Result<u16> {
    args.int(at)
        .ok()
        .and_then(|n| u16::try_from(n).ok())
        .filter(|s| usize::from(*s) < SLOTS)
        .ok_or_else(|| Error::new(Code::Invalid, "Invalid or out of range slot"))
}

/// `CLUSTER ADDSLOTS`, `DELSLOTS`, `ADDSLOTSRANGE` and `DELSLOTSRANGE`.
///
/// All four in one body because all four are the same walk with a different
/// stride and a different word in the complaint. Every slot is checked before
/// any of them is moved, which is the reference's behaviour and is what makes a
/// list with one bad number in it change nothing at all.
fn add_or_del(server: &Server, args: Args<'_>, add: bool, ranged: bool) -> Result<()> {
    let stride = if ranged { 2 } else { 1 };
    if ranged && !(args.len() - 2).is_multiple_of(2) {
        return Err(wrong_sub_arity(args.get(1)));
    }
    let mut wanted = Vec::new();
    let mut at = 2;
    while at < args.len() {
        let from = slot_arg(&args, at)?;
        let to = if ranged {
            slot_arg(&args, at + 1)?
        } else {
            from
        };
        if from > to {
            return Err(Error::fmt(
                Code::Invalid,
                format_args!("start slot number {from} is greater than end slot number {to}"),
            ));
        }
        for slot in from..=to {
            wanted.push(slot);
        }
        at += stride;
    }
    {
        let mut map = server.cluster.map.lock();
        let mut seen = vec![false; SLOTS];
        for slot in &wanted {
            let slot = usize::from(*slot);
            if seen[slot] {
                return Err(Error::fmt(
                    Code::Invalid,
                    format_args!("Slot {slot} specified multiple times"),
                ));
            }
            seen[slot] = true;
            let busy = map.owner[slot].is_some();
            if add && busy {
                return Err(Error::fmt(
                    Code::Invalid,
                    format_args!("Slot {slot} is already busy"),
                ));
            }
            if !add && !busy {
                return Err(Error::fmt(
                    Code::Invalid,
                    format_args!("Slot {slot} is already unassigned"),
                ));
            }
        }
        for slot in &wanted {
            let slot = usize::from(*slot);
            map.owner[slot] = if add { Some(0) } else { None };
            map.migrating[slot] = None;
            map.importing[slot] = None;
        }
    }
    server.recount_coverage();
    save(server)
}

/// The address `CLUSTER MEET` was given, checked the way the reference checks it.
///
/// The port is reported back as the caller typed it rather than as it parsed,
/// which is the reference's wording and matters for a port that parsed fine and
/// was out of range.
fn meet(args: &Args<'_>) -> Result<(String, u16, u16)> {
    let host = String::from_utf8_lossy(args.get(2));
    let typed = String::from_utf8_lossy(args.get(3));
    let port = args.int(3).map_err(|_| {
        Error::fmt(
            Code::Invalid,
            format_args!("Invalid base port specified: {typed}"),
        )
    })?;
    let bus = match args.opt(4) {
        None => port + i64::from(BUS_OFFSET),
        Some(word) => args.int(4).map_err(|_| {
            Error::fmt(
                Code::Invalid,
                format_args!(
                    "Invalid bus port specified: {}",
                    String::from_utf8_lossy(word)
                ),
            )
        })?,
    };
    if !(1..=65535).contains(&port) || !(0..=65535).contains(&bus) {
        return Err(Error::fmt(
            Code::Invalid,
            format_args!("Invalid node address specified: {host}:{typed}"),
        ));
    }
    let bus = if bus == 0 {
        port + i64::from(BUS_OFFSET)
    } else {
        bus
    };
    let host = yo_alloc::allow(|| host.into_owned());
    Ok((host, port as u16, bus as u16))
}

/// `CLUSTER REPLICAS`, which is one `CLUSTER NODES` line per replica of a node.
fn replicas(server: &Server, id: &[u8], out: &mut Out) -> Result<()> {
    let map = server.cluster.map.lock();
    let Some(at) = map.find(id) else {
        return Err(unknown_node(id));
    };
    if !map.nodes[usize::from(at)].is_master() {
        return Err(Error::new(
            Code::Invalid,
            "The specified node is not a master",
        ));
    }
    let start = out.len();
    let mut n = 0;
    for other in 0..map.nodes.len() as u16 {
        if map.nodes[usize::from(other)].master != Some(at) {
            continue;
        }
        let line = yo_alloc::allow(|| {
            let mut s = String::with_capacity(256);
            describe(&map, other, false, &mut s);
            s
        });
        out.bulk(line.as_bytes());
        n += 1;
    }
    out.close_array(start, n);
    Ok(())
}

/// `CLUSTER FORGET`, which drops a node and makes the drop stick.
///
/// An id nobody has heard of is an error unless it is one this node forgot on
/// purpose a moment ago, in which case the answer is OK, because a tool that
/// sends the same forget to every node in the cluster should not hear an error
/// from the ones that had already been told by gossip.
fn forget(server: &Server, id: &[u8]) -> Result<()> {
    let at = {
        let map = server.cluster.map.lock();
        match map.find(id) {
            Some(0) => {
                return Err(Error::new(
                    Code::Invalid,
                    "I tried hard but I can't forget myself...",
                ));
            }
            Some(at) if map.nodes[0].master == Some(at) => {
                return Err(Error::new(Code::Invalid, "Can't forget my master!"));
            }
            Some(at) => at,
            None => {
                let name = String::from_utf8_lossy(id);
                if server.cluster_blacklisted(&name) {
                    return Ok(());
                }
                return Err(unknown_node(id));
            }
        }
    };
    server.cluster_forget(at);
    Ok(())
}

/// `CLUSTER REPLICATE`, which makes this node a replica of another.
///
/// A master with slots or keys refuses, which is the reference's rule and is the
/// right one: the slots would be given away silently and the keys in them would
/// be answered for by two nodes at once.
fn replicate(server: &Server, db: usize, id: &[u8]) -> Result<()> {
    let at = {
        let map = server.cluster.map.lock();
        match map.find(id) {
            None => return Err(unknown_node(id)),
            Some(0) => return Err(Error::new(Code::Invalid, "Can't replicate myself")),
            Some(at) if !map.nodes[usize::from(at)].is_master() => {
                return Err(Error::new(
                    Code::Invalid,
                    "I can only replicate a master, not a replica.",
                ));
            }
            Some(at) => {
                if map.nodes[0].is_master()
                    && (!map.runs(0).is_empty() || !server.dbs[db].is_empty())
                {
                    return Err(Error::new(
                        Code::Invalid,
                        "To set a master the node must be empty and without assigned slots.",
                    ));
                }
                at
            }
        }
    };
    let Some(shared) = server.myself() else {
        return Err(Error::new(
            Code::Invalid,
            "CLUSTER REPLICATE is not available on an embedded server",
        ));
    };
    shared.cluster_replicate(at);
    Ok(())
}

/// `CLUSTER SLOT-STATS`, which is how many keys each slot this node owns holds.
///
/// A real server keeps more metrics than this one and only when
/// `cluster-slot-stats-enabled` is on, so the default answer is the key count on
/// its own and that is what is here. The count is one walk of the keyspace with a
/// tally per slot rather than one walk per slot, which is the same D-150 cost as
/// `COUNTKEYSINSLOT` paid once instead of sixteen thousand times.
fn slot_stats(server: &Server, db: usize, args: Args<'_>, out: &mut Out) -> Result<()> {
    let sub = args.get(1);
    let mut wanted: Vec<u16> = Vec::new();
    let mut limit = SLOTS;
    let mut ascending = false;
    let ordered = args::is(args.get(2), b"orderby");
    if args::is(args.get(2), b"slotsrange") {
        if args.len() != 5 {
            return Err(sub_syntax(sub));
        }
        let from = slot_arg(&args, 3)?;
        let to = slot_arg(&args, 4)?;
        if from > to {
            return Err(Error::fmt(
                Code::Invalid,
                format_args!("Start slot number {from} is greater than end slot number {to}"),
            ));
        }
        wanted.extend(from..=to);
    } else if ordered {
        if !args::is(args.get(3), b"key-count") {
            return Err(Error::new(
                Code::Invalid,
                "Unrecognized sort metric for ORDERBY.",
            ));
        }
        let bad_limit = || {
            Error::new(
                Code::Invalid,
                "Limit has to lie in between 1 and 16384 (maximum number of slots).",
            )
        };
        let mut at = 4;
        while at < args.len() {
            let word = args.get(at);
            if args::is(word, b"limit") && at + 1 < args.len() {
                let n = args.int(at + 1).map_err(|_| bad_limit())?;
                if !(1..=SLOTS as i64).contains(&n) {
                    return Err(bad_limit());
                }
                limit = n as usize;
                at += 2;
            } else if args::is(word, b"asc") {
                ascending = true;
                at += 1;
            } else if args::is(word, b"desc") {
                at += 1;
            } else {
                return Err(args::syntax());
            }
        }
        wanted.extend(0..SLOTS as u16);
    } else {
        return Err(sub_syntax(sub));
    }
    let mut counts = vec![0i64; SLOTS];
    server.dbs[db].keys(|key| counts[key_slot(key) as usize] += 1);
    let map = server.cluster.map.lock();
    wanted.retain(|slot| map.mine(*slot));
    drop(map);
    if ordered {
        // The tie is broken by slot number ascending either way round, which is
        // what a real server does with a table nothing has been written to yet.
        if ascending {
            wanted.sort_by_key(|slot| (counts[*slot as usize], *slot));
        } else {
            wanted.sort_by_key(|slot| (-counts[*slot as usize], *slot));
        }
        wanted.truncate(limit);
    }
    out.array(wanted.len());
    for slot in wanted {
        out.array(2);
        out.int(i64::from(slot));
        out.map(1);
        out.bulk(b"key-count");
        out.int(counts[slot as usize]);
    }
    Ok(())
}

/// `CLUSTER MIGRATION IMPORT|STATUS|CANCEL`, which is the newer way of moving a
/// slot range: the importing node drives the whole move rather than a tool
/// stepping it key by key.
///
/// `STATUS` and `CANCEL` report and cancel what is really running, which on this
/// node is a migration another node started against it. Starting one from here,
/// which is what `IMPORT` is, is the other side of the protocol and is D-149.
fn migration(server: &Server, args: Args<'_>, out: &mut Out) -> Result<()> {
    let sub = args.get(1);
    let action = args.get(2);
    if args::is(action, b"status") || args::is(action, b"cancel") {
        let by_id = args::is(args.get(3), b"id");
        if by_id && args.len() != 5 {
            return Err(wrong_sub_arity(sub));
        }
        if !by_id && !args::is(args.get(3), b"all") {
            return Err(Error::new(Code::Invalid, "unknown argument"));
        }
        if !by_id && args.len() != 4 {
            return Err(wrong_sub_arity(sub));
        }
        let id = by_id.then(|| args.get(4));
        if args::is(action, b"status") {
            match id {
                Some(id) => server.cluster.asm.report_one(id, out),
                None => server.cluster.asm.report_all(out),
            }
        } else {
            out.int(server.cluster.asm.cancel(id, server.now_ms() as i64));
            // A cancelled handoff is not one whose write pause anybody should
            // have to wait out.
            server.asm_relax();
        }
        return Ok(());
    }
    if !args::is(action, b"import") {
        return Err(Error::new(Code::Invalid, "unknown argument"));
    }
    let ranges = slot_ranges(&args, 3)?;
    let (source, host, port) = import_source(server, &ranges)?;
    // An embedded server has no bus, nobody to dial and nobody to tell, so there
    // is nothing an import could even mean on one. Checked after the slot ranges
    // rather than before, so that a caller who got the ranges wrong is told about
    // the ranges wherever the command was sent.
    let Some(shared) = server.myself() else {
        return Err(Error::new(
            Code::Invalid,
            "slot migration needs a server with a cluster bus",
        ));
    };
    let id = server.asm_begin_import(source, ranges.clone())?;
    out.bulk(id.as_bytes());
    import::start(
        &shared,
        import::Job {
            id,
            host,
            port,
            slots: ranges,
        },
    );
    Ok(())
}

/// Who a set of slot ranges would have to come from, which is the reference's
/// `validateImportSlotRanges` and the owner check behind it.
///
/// In the reference's order, because the order is what a caller reads to find
/// out which of several things it got wrong. The last of them is the one an
/// operator hits most: asking a node to import slots it already has.
fn import_source(server: &Server, ranges: &[(u16, u16)]) -> Result<(Vec<u8>, String, u16)> {
    let map = server.cluster.map.lock();
    if !map.nodes[0].is_master() {
        return Err(Error::new(
            Code::Invalid,
            "slot migration not allowed on replica.",
        ));
    }
    // The two ways of moving a slot do not mix, the same as on the giving up
    // side: a slot half way through the old one has keys on two nodes at once.
    if (0..SLOTS).any(|at| map.migrating[at].is_some() || map.importing[at].is_some()) {
        return Err(Error::new(
            Code::Invalid,
            "all slot states must be STABLE to start a slot migration task.",
        ));
    }
    if let Some((from, to)) = server.cluster.asm.overlapping_import(ranges) {
        return Err(Error::fmt(
            Code::Invalid,
            format_args!("overlapping import exists for slot range: {from}-{to}"),
        ));
    }
    let mut owner = None;
    for &(from, to) in ranges {
        for slot in from..=to {
            let Some(at) = map.owner[usize::from(slot)] else {
                return Err(Error::fmt(
                    Code::Invalid,
                    format_args!("slot has no owner: {slot}"),
                ));
            };
            if *owner.get_or_insert(at) != at {
                return Err(Error::new(
                    Code::Invalid,
                    "slots belong to different source nodes",
                ));
            }
        }
    }
    let at = owner.unwrap_or(0);
    if at == 0 {
        return Err(Error::new(
            Code::Invalid,
            "this node is already the owner of the slot range",
        ));
    }
    let node = &map.nodes[usize::from(at)];
    Ok((node.id.as_bytes().to_vec(), node.host.clone(), node.port))
}

/// The list of slot ranges the tail of an argument list spells out, which is the
/// reference's `parseSlotRangesOrReply` and the validation behind it.
///
/// Sorted and with the ranges that touch joined up, because that is what the
/// reference hands back and the two checks after it are made against the joined
/// up list rather than what the caller typed. Ranges that touch join and ranges
/// that overlap do not, which is what makes `1 2 2 3` a slot named twice while
/// `1 2 3 4` is one range of four.
fn slot_ranges(args: &Args<'_>, from: usize) -> Result<Vec<(u16, u16)>> {
    let count = args.len().saturating_sub(from);
    if count < 2 || !count.is_multiple_of(2) {
        return Err(wrong_sub_arity(args.get(1)));
    }
    if count / 2 >= SLOTS {
        return Err(Error::fmt(
            Code::Invalid,
            format_args!("invalid number of slot ranges: {}", count / 2),
        ));
    }
    let mut ranges: Vec<(u16, u16)> = Vec::with_capacity(count / 2);
    let mut at = from;
    while at < args.len() {
        ranges.push((slot_arg(args, at)?, slot_arg(args, at + 1)?));
        at += 2;
    }
    ranges.sort_unstable();
    let mut joined: Vec<(u16, u16)> = Vec::with_capacity(ranges.len());
    for range in ranges {
        match joined.last_mut() {
            Some(last) if u32::from(last.1) + 1 == u32::from(range.0) => last.1 = range.1,
            _ => joined.push(range),
        }
    }
    let mut seen = vec![false; SLOTS];
    for &(start, end) in &joined {
        if start > end {
            return Err(Error::fmt(
                Code::Invalid,
                format_args!("start slot number {start} is greater than end slot number {end}"),
            ));
        }
        for slot in start..=end {
            if core::mem::replace(&mut seen[usize::from(slot)], true) {
                return Err(Error::fmt(
                    Code::Invalid,
                    format_args!("Slot {slot} specified multiple times"),
                ));
            }
        }
    }
    Ok(joined)
}

/// `TRIMSLOTS`, which drops the keys of slot ranges this node does not serve.
///
/// Not a command an operator has any reason to type, and it is a top level
/// command rather than a `CLUSTER` subcommand because of where it is sent from:
/// a node that has just handed slots over writes one of these to its replicas
/// and its append only file, so that a replica drops the same keys its master
/// just dropped rather than going on answering for data that has moved. One
/// command for the ranges rather than a deletion per key, because a slot range
/// can hold millions of keys and the far side can work the list out for itself.
///
/// The refusal in the middle is the one that matters. A node will not empty a
/// slot it is serving, whatever it is told, so a `TRIMSLOTS` that arrives late
/// or names the wrong range cannot take live data with it. A replica does not
/// make that check, because the slots are its master's and not its own and it is
/// being told what its master already did.
pub(super) fn trimslots(server: &Server, args: Args<'_>, out: &mut Out) -> Result<()> {
    if !server.cluster_enabled() {
        return Err(disabled());
    }
    if !args::is(args.get(1), b"ranges") {
        return Err(Error::new(Code::Invalid, "missing ranges argument"));
    }
    let count = args.int(2)?;
    if count < 1 || count > SLOTS as i64 || args.len() as i64 != 3 + count * 2 {
        return Err(Error::new(Code::Invalid, "invalid number of ranges"));
    }
    let ranges = slot_ranges(&args, 3)?;
    {
        let map = server.cluster.map.lock();
        if map.nodes[0].is_master() {
            for &(from, to) in &ranges {
                for slot in from..=to {
                    if map.owner[usize::from(slot)] == Some(0) {
                        return Err(Error::fmt(
                            Code::Invalid,
                            format_args!("the slot {slot} is served by this node"),
                        ));
                    }
                }
            }
        }
    }
    server.trim_named_slots(&ranges);
    out.ok();
    Ok(())
}

/// `CLUSTER SYNCSLOTS`, which is what two nodes say to each other while a slot
/// range moves from one to the other under atomic slot migration.
///
/// Not a command an operator ever types. The node taking the slots opens an
/// ordinary client connection to the node giving them up, logs in as the cluster
/// rather than as a user, and drives the whole move over that connection and one
/// more like it. So every arm here answers another node running the same code,
/// which is why the reference does not defend the state machine against being
/// driven out of order and why it hangs up on anybody who is not a node rather
/// than just saying no. Getting to the point of being able to send this is the
/// hard part and the refusal makes it worthless.
///
/// What is in so far is the gate, the argument checking and the two arms that
/// have a real answer on a node with no migration running, which is every node
/// today. Starting a migration is D-149.
fn syncslots(server: &Server, session: &mut Session, args: Args<'_>, out: &mut Out) -> Result<()> {
    if !session.internal() {
        // The hang up is the reference's and it is worth copying. The sentence on
        // its own would be enough to be correct, and dropping the connection is
        // what makes sitting there guessing at the protocol expensive.
        session.hang_up();
        return Err(Error::new(
            Code::Invalid,
            "CLUSTER SYNCSLOTS subcommands are only allowed for internal clients",
        ));
    }
    let action = args.get(2);
    if !server.cluster.map.lock().nodes[0].is_master() {
        // A replica owns nothing, so there is nothing anybody could be asking it
        // to hand over. The one thing it does take is `CONF`, and only from its
        // own master, which is how it hears about the migration its master is in
        // the middle of. Anything else from the master is dropped in silence
        // rather than refused, because an error written into the replication
        // stream is an error nobody reads.
        if !session.serving_master() {
            session.hang_up();
            return Err(Error::new(
                Code::Invalid,
                "CLUSTER SYNCSLOTS subcommands are only allowed for master",
            ));
        }
        if !args::is(action, b"conf") {
            return Ok(());
        }
    }
    if args::is(action, b"sync") && args.len() >= 6 {
        return sync(server, session, args, out);
    }
    if args::is(action, b"rdbchannel") && args.len() == 4 {
        return rdbchannel(server, session, args, out);
    }
    if (args::is(action, b"snapshot-eof") || args::is(action, b"stream-eof")) && args.len() == 3 {
        // Both of these say a transfer has ended, and on a node with no transfer
        // running the reference logs that it was not expecting them and drops the
        // connection without writing anything back.
        session.hang_up();
        return Ok(());
    }
    if args::is(action, b"ack") && args.len() == 5 {
        // The arm that answers nothing. This connection's other direction is the
        // change stream, and anything written back on it would be read as a
        // command, so a bad state word or a number that is not one is dropped in
        // silence rather than refused.
        if let Some(offset) = yo_common::num::parse_i64(args.get(4))
            && offset >= 0
        {
            server.asm_ack(session.row().id, args.get(3), offset as u64);
        }
        return Ok(());
    }
    if args::is(action, b"fail") && args.len() == 4 {
        // The other arm that never answers. `FAIL` does nothing at all on the
        // reference either: it is there so that the far side has something to
        // send that will not come back as a syntax error.
        return Ok(());
    }
    if args::is(action, b"conf") && args.len() >= 5 {
        return conf(server, session, args, out);
    }
    Err(args::syntax())
}

/// `CLUSTER SYNCSLOTS SYNC <task-id> <start> <end> [<start> <end> ...]`, which is
/// the node taking the slots asking for them.
///
/// Every check the reference makes before it starts is made here, in its order,
/// because they are what a caller reads to find out it asked the wrong node.
///
/// What comes back on the way through is `+RDBCHANNELSYNCSLOTS`, which is the far
/// side being told to open the second connection the snapshot will come down.
/// Nothing else is written to this connection yet: it is the one the changes
/// since the snapshot would go down, and those are D-149.
fn sync(server: &Server, session: &mut Session, args: Args<'_>, out: &mut Out) -> Result<()> {
    if !args.len().is_multiple_of(2) {
        return Err(wrong_sub_arity(args.get(1)));
    }
    let ranges = slot_ranges(&args, 4)?;
    {
        let map = server.cluster.map.lock();
        // The two ways of moving a slot do not mix. A slot that is half way
        // through the old one has keys on two nodes at once and a snapshot of it
        // would be a snapshot of half the data.
        if (0..SLOTS).any(|at| map.migrating[at].is_some() || map.importing[at].is_some()) {
            return Err(Error::new(
                Code::Invalid,
                "all slot states must be STABLE to start a slot migration task.",
            ));
        }
        let mut source = None;
        for slot in ranges.iter().flat_map(|(from, to)| *from..=*to) {
            let Some(owner) = map.owner[usize::from(slot)] else {
                return Err(Error::fmt(
                    Code::Invalid,
                    format_args!("slot has no owner: {slot}"),
                ));
            };
            if *source.get_or_insert(owner) != owner {
                return Err(Error::new(
                    Code::Invalid,
                    "slots belong to different source nodes",
                ));
            }
        }
        if source != Some(0) {
            return Err(Error::new(
                Code::Invalid,
                "This node is not the owner of the slots",
            ));
        }
        // A node that is not in the table, or that is somebody's replica, has
        // nowhere to put a slot. The far side only says who it is when it sent
        // `CONF NODE-ID` first, and the reference lets a connection that did not
        // through, so this is checked only when there is something to check.
        let dest = session.node_id().to_vec();
        if !dest.is_empty()
            && !map
                .find(&dest)
                .is_some_and(|at| map.nodes[at as usize].is_master())
        {
            return Err(Error::fmt(
                Code::Invalid,
                format_args!(
                    "Destination node {} is not a master",
                    String::from_utf8_lossy(&dest)
                ),
            ));
        }
    }
    server.asm_begin_migrate(args.get(3), session.node_id(), ranges, session.row())?;
    out.simple(b"RDBCHANNELSYNCSLOTS");
    Ok(())
}

/// `CLUSTER SYNCSLOTS RDBCHANNEL <task-id>`, which is the second connection of a
/// migration arriving to be handed the snapshot.
///
/// The reference answers `+SLOTSSNAPSHOT` and then forks, and the snapshot goes
/// out of the child while the parent carries on serving. There is no fork here,
/// so the snapshot is built with every write on the server held off and then
/// written, which is the same trade a full resync makes and is why the snapshot
/// is one slot range rather than the whole keyspace.
fn rdbchannel(server: &Server, session: &mut Session, args: Args<'_>, out: &mut Out) -> Result<()> {
    let id = args.get(3);
    if id.len() != ID_LEN {
        return Err(Error::new(Code::Invalid, "Invalid task id"));
    }
    let ranges = server.asm_take_rdb_channel(id, session.row().id)?;
    out.simple(b"SLOTSSNAPSHOT");
    out.raw(&server.asm_snapshot(&ranges));
    Ok(())
}

/// `CLUSTER SYNCSLOTS CONF <option> <value> [<option> <value> ...]`, which is
/// each side telling the other something it will need in a moment.
///
/// Three of the four options are the node that is talking, a hint about how big a
/// slot is so the receiving side can size its tables once instead of growing them
/// all the way up, and a capability word that is ignored on purpose so that a
/// newer node can say something an older one has never heard of. The fourth is a
/// master handing its own replicas the state of the migration it is in.
///
/// Every complaint here is written rather than returned, because the reference
/// keeps going after an option it did not understand and still says `OK` at the
/// end, so one command can answer with both an error line and an `OK`. Returning
/// would throw the first of those away.
fn conf(server: &Server, session: &mut Session, args: Args<'_>, out: &mut Out) -> Result<()> {
    let mut at = 3;
    while at < args.len() {
        if at + 1 >= args.len() {
            super::write_error(out, &wrong_sub_arity(args.get(1)));
            return Ok(());
        }
        let name = args.get(at);
        let value = args.get(at + 1);
        if args::is(name, b"node-id") {
            // Who is on the other end of this connection, which is what the
            // `SYNC` after it is checked against and what the task it starts
            // records as the node the slots are going to.
            if value.len() != ID_LEN {
                let len = value.len();
                super::write_error(
                    out,
                    &Error::fmt(Code::Invalid, format_args!("Invalid node id length {len}")),
                );
                return Ok(());
            }
            if server.cluster.map.lock().find(value).is_none() {
                super::write_error(
                    out,
                    &Error::fmt(
                        Code::Invalid,
                        format_args!(
                            "Node {} not found in cluster",
                            String::from_utf8_lossy(value)
                        ),
                    ),
                );
                return Ok(());
            }
            session.set_node_id(value);
        } else if args::is(name, b"slot-info") {
            if !slot_info(value) {
                super::write_error(
                    out,
                    &Error::fmt(
                        Code::Invalid,
                        format_args!("Invalid slot info: {}", String::from_utf8_lossy(value)),
                    ),
                );
                return Ok(());
            }
        } else if args::is(name, b"asm-task") {
            if server.cluster.map.lock().nodes[0].is_master() {
                super::write_error(
                    out,
                    &Error::new(
                        Code::Invalid,
                        "CLUSTER SYNCSLOTS CONF ASM-TASK only allowed on replica",
                    ),
                );
                return Ok(());
            }
            // A replica that hears this is being told to follow along with a
            // migration its master is running, and there is nothing here to
            // follow along with yet, so it says so and carries on with the rest
            // of the options exactly as the reference does.
            super::write_error(
                out,
                &Error::fmt(
                    Code::Invalid,
                    format_args!(
                        "Failed to handle master task: {}",
                        String::from_utf8_lossy(value)
                    ),
                ),
            );
        } else if !args::is(name, b"capa") {
            super::write_error(
                out,
                &Error::fmt(
                    Code::Invalid,
                    format_args!("Unknown option {}", String::from_utf8_lossy(name)),
                ),
            );
        }
        at += 2;
    }
    out.ok();
    Ok(())
}

/// Whether a `slot-info` value is one, which is `slot:key-count:expire-count`
/// with all three of them numbers and the first of them a slot.
///
/// What it is for is sizing: the node about to receive a slot is being told how
/// many keys are coming so that it can make room for them in one go. Nothing is
/// sized on it here, because the tables this engine keeps a slot's keys in grow
/// without the rehash a size hint exists to avoid, so the value is only checked.
fn slot_info(value: &[u8]) -> bool {
    let mut parts = value.split(|b| *b == b':');
    let Some(slot) = parts.next().and_then(yo_common::num::parse_i64) else {
        return false;
    };
    let Some(keys) = parts.next().and_then(yo_common::num::parse_i64) else {
        return false;
    };
    let Some(expires) = parts.next().and_then(yo_common::num::parse_i64) else {
        return false;
    };
    parts.next().is_none() && (0..SLOTS as i64).contains(&slot) && keys >= 0 && expires >= 0
}

/// `CLUSTER SETSLOT <slot> IMPORTING|MIGRATING|STABLE|NODE`.
///
/// The four arms and the order of their checks are the reference's, which is
/// worth being exact about because a resharding tool drives this and reads the
/// sentences it gets back.
///
/// `NODE` is where a slot migration ends and it is the one arm that does more
/// than move a field. When the node being named is this one and this node was
/// importing the slot, the config epoch goes up and the whole cluster is told
/// straight away, because until that happens every other node is still pointing
/// clients at the node the slot came from and the higher epoch is the only thing
/// that makes them stop.
fn setslot(server: &Server, session_db: usize, args: Args<'_>, out: &mut Out) -> Result<()> {
    // A replica has no say in who owns what, so this is refused before the slot
    // number is even looked at.
    if !server.cluster.map.lock().nodes[0].is_master() {
        return Err(Error::new(
            Code::Invalid,
            "Please use SETSLOT only with masters.",
        ));
    }
    let slot = slot_arg(&args, 2)?;
    // A slot an atomic migration is moving is not one to hand over by hand. The
    // two ways of moving a slot write the same three fields from two directions,
    // and the migration would finish by claiming a slot the operator had already
    // given to somebody else. Cancelling the task is the way out and the sentence
    // says so, because a resharding tool is what reads it.
    if server.cluster.asm.in_task(slot) {
        return Err(Error::fmt(
            Code::Invalid,
            format_args!(
                "Slot {slot} is currently in an active atomic slot migration. \
                 CLUSTER SETSLOT cannot be used at this time. To perform a legacy slot migration \
                 instead, first cancel the ongoing task with CLUSTER MIGRATION CANCEL"
            ),
        ));
    }
    let action = args.get(3);
    let wrong = || {
        Error::new(
            Code::Invalid,
            "Invalid CLUSTER SETSLOT action or number of arguments. Try CLUSTER HELP",
        )
    };
    // Set when the `NODE` arm decides the rest of the cluster has to be told
    // now rather than at the next ping. Done once the lock is gone, since
    // building the packet takes the same lock.
    let mut announce = false;
    // And when this node has just given its last slot away and should follow
    // whoever took it. The same, and for the same reason.
    let mut follow: Option<u16> = None;
    {
        let mut map = server.cluster.map.lock();
        let at = usize::from(slot);
        if args::is(action, b"migrating") && args.len() == 5 {
            if !map.mine(slot) {
                return Err(Error::fmt(
                    Code::Invalid,
                    format_args!("I'm not the owner of hash slot {slot}"),
                ));
            }
            let Some(to) = map.find(args.get(4)) else {
                return Err(dont_know(args.get(4)));
            };
            map.migrating[at] = Some(to);
        } else if args::is(action, b"importing") && args.len() == 5 {
            if map.mine(slot) {
                return Err(Error::fmt(
                    Code::Invalid,
                    format_args!("I'm already the owner of hash slot {slot}"),
                ));
            }
            let Some(from) = map.find(args.get(4)) else {
                return Err(dont_know(args.get(4)));
            };
            map.importing[at] = Some(from);
        } else if args::is(action, b"stable") && args.len() == 4 {
            map.migrating[at] = None;
            map.importing[at] = None;
        } else if args::is(action, b"node") && args.len() == 5 {
            let Some(to) = map.find(args.get(4)) else {
                return Err(unknown_node(args.get(4)));
            };
            if !map.nodes[usize::from(to)].is_master() {
                return Err(Error::new(Code::Invalid, "Target node is not a master"));
            }
            let was_mine = map.owner[at] == Some(0);
            // The keys are counted twice here rather than once, which reads
            // oddly and is the reference's own shape. The first count refuses
            // handing the slot to somebody else while this node is still
            // holding keys for it, since that would leave two nodes answering
            // for the same data. The second clears the migrating mark, and it
            // is a separate question because a slot can be empty and marked
            // whether or not it was ever this node's.
            let held = keys_in_slot(server, session_db, slot);
            if was_mine && to != 0 && held != 0 {
                return Err(Error::fmt(
                    Code::Invalid,
                    format_args!(
                        "Can't assign hashslot {slot} to a different node while I still hold keys for this hash slot."
                    ),
                ));
            }
            if held == 0 {
                map.migrating[at] = None;
            }
            map.owner[at] = Some(to);
            // A master that has just handed over its last slot follows whoever
            // took it, which is what stops a resharded cluster being left with
            // a node that owns nothing and serves nobody.
            if was_mine && to != 0 && map.runs(0).is_empty() {
                follow = Some(to);
            }
            // The import is finished, so the epoch goes up and everybody is
            // told. Nothing else in this function moves the epoch, because
            // nothing else is this node claiming something that used to be
            // somebody else's.
            if to == 0 && map.importing[at].is_some() {
                bump_without_consensus(server, &mut map);
                map.importing[at] = None;
                announce = true;
            }
        } else {
            return Err(wrong());
        }
    }
    server.recount_coverage();
    save(server)?;
    // Only on a real server. An embedded one has no bus and nobody to follow,
    // and the slot still changes hands the same way.
    if let Some(to) = follow
        && let Some(shared) = server.myself()
    {
        shared.cluster_replicate(to);
    }
    if announce {
        server.cluster_broadcast_pong();
    }
    out.ok();
    Ok(())
}

/// How many keys of a slot are here, which two of `SETSLOT NODE`'s refusals turn
/// on and which `CLUSTER COUNTKEYSINSLOT` answers.
fn keys_in_slot(server: &Server, at: usize, slot: u16) -> usize {
    let mut found = 0;
    server.dbs[at].keys(|key| {
        if key_slot(key) == slot {
            found += 1;
        }
    });
    found
}

/// The reference's `clusterBumpConfigEpochWithoutConsensus`.
///
/// A node that has just taken a slot off somebody needs an epoch higher than
/// theirs, or every other node will keep the old owner: a higher epoch is the
/// whole of how the cluster decides which of two claims on a slot is the newer
/// one. Without consensus means exactly that, nobody is asked, and the comment
/// in the reference is worth repeating: two nodes can end up on the same epoch
/// this way, and the collision rule sorts that out afterwards rather than the
/// bump trying to avoid it.
///
/// It does nothing at all when this node already holds the largest epoch
/// anybody has, since there is nothing left to outrank. `false` is that case,
/// and it is what `CLUSTER BUMPEPOCH` answers `STILL` for.
fn bump_without_consensus(server: &Server, map: &mut Map) -> bool {
    let highest = map
        .nodes
        .iter()
        .map(|n| n.epoch)
        .max()
        .unwrap_or(0)
        .max(server.cluster.epoch.load(Relaxed));
    let mine = map.nodes[0].epoch;
    if mine != 0 && mine == highest {
        return false;
    }
    map.nodes[0].epoch = server.cluster.epoch.fetch_add(1, Relaxed) + 1;
    true
}

/// Claim slot ranges for this node, which is the last step of an import and the
/// reference's `ASM_EVENT_TAKEOVER`.
///
/// Nobody is asked. The epoch goes up without consensus and the claim goes out
/// on the bus, and a higher epoch is the whole of how every other node decides
/// which of two claims on a slot is the newer one. That is safe here for the
/// reason it is not safe in general: the node that used to own these slots has
/// stopped taking writes for them and has said there is nothing more coming, so
/// there is no second writer to disagree with.
///
/// The old owner finds out the same way everybody else does, from the bus, and
/// what it does about it is the other half of this, in `asm_slots_moved`.
fn take_slots(server: &Server, ranges: &[(u16, u16)]) -> Result<()> {
    {
        let mut map = server.cluster.map.lock();
        for slot in ranges.iter().flat_map(|(from, to)| *from..=*to) {
            let at = usize::from(slot);
            map.owner[at] = Some(0);
            map.importing[at] = None;
            map.migrating[at] = None;
        }
        bump_without_consensus(server, &mut map);
    }
    server.recount_coverage();
    save(server)?;
    server.cluster_broadcast_pong();
    Ok(())
}

/// `CLUSTER FLUSHSLOTS`, which drops every slot this node claims.
fn flushslots(server: &Server, out: &mut Out) -> Result<()> {
    if server.dbs.iter().any(|db| !db.is_empty()) {
        return Err(Error::new(
            Code::Invalid,
            "DB must be empty to perform CLUSTER FLUSHSLOTS.",
        ));
    }
    {
        let mut map = server.cluster.map.lock();
        for slot in 0..SLOTS {
            if map.owner[slot] == Some(0) {
                map.owner[slot] = None;
            }
            map.migrating[slot] = None;
            map.importing[slot] = None;
        }
    }
    server.recount_coverage();
    save(server)?;
    out.ok();
    Ok(())
}

/// `CLUSTER BUMPEPOCH`, which takes an epoch above everybody else's.
///
/// Answers `BUMPED` and the new epoch when it moved and `STILL` and the current
/// one when there was nothing to outrank, which is what a second call in a row
/// gets. See [`bump_without_consensus`] for why doing nothing is the right
/// answer rather than a wasted epoch.
fn bumpepoch(server: &Server, out: &mut Out) -> Result<()> {
    let (moved, epoch) = {
        let mut map = server.cluster.map.lock();
        let moved = bump_without_consensus(server, &mut map);
        (moved, map.nodes[0].epoch)
    };
    if moved {
        save(server)?;
        server.cluster_broadcast_pong();
    }
    let word = if moved { "BUMPED" } else { "STILL" };
    let text = yo_alloc::allow(|| format!("{word} {epoch}"));
    out.simple(text.as_bytes());
    Ok(())
}

/// `CLUSTER SET-CONFIG-EPOCH`, which is how a brand new cluster gives each node
/// a different epoch before anything has been agreed.
fn set_config_epoch(server: &Server, args: Args<'_>, out: &mut Out) -> Result<()> {
    let epoch = args.int(2)?;
    if epoch < 0 {
        return Err(Error::fmt(
            Code::Invalid,
            format_args!("Invalid config epoch specified: {epoch}"),
        ));
    }
    {
        let mut map = server.cluster.map.lock();
        if map.nodes[0].epoch != 0 {
            return Err(Error::new(
                Code::Invalid,
                "Node config epoch is already non-zero",
            ));
        }
        map.nodes[0].epoch = epoch as u64;
    }
    let epoch = epoch as u64;
    server.cluster.epoch.fetch_max(epoch, Relaxed);
    save(server)?;
    out.ok();
    Ok(())
}

/// `CLUSTER RESET [HARD|SOFT]`, which gives up every slot and, when hard, takes
/// a new identity as well.
fn reset(server: &Server, args: Args<'_>, out: &mut Out) -> Result<()> {
    let hard = match args.opt(2) {
        None => false,
        Some(word) if args::is(word, b"hard") => true,
        Some(word) if args::is(word, b"soft") => false,
        Some(_) => return Err(args::syntax()),
    };
    if server.dbs.iter().any(|db| !db.is_empty()) {
        return Err(Error::new(
            Code::Invalid,
            "CLUSTER RESET can't be called with master nodes containing keys",
        ));
    }
    {
        let mut map = server.cluster.map.lock();
        for slot in 0..SLOTS {
            map.owner[slot] = None;
            map.migrating[slot] = None;
            map.importing[slot] = None;
        }
        map.nodes.truncate(1);
        map.nodes[0].epoch = 0;
        // A reset makes this node a master of nothing again, which is the half
        // of it a soft reset does as well: it stops following anybody and it
        // forgets everybody, it just keeps its name.
        map.nodes[0].flags = FLAG_MYSELF | FLAG_MASTER;
        map.nodes[0].master = None;
        if hard {
            yo_alloc::allow(|| {
                map.nodes[0].id = String::from_utf8_lossy(&new_id()).into_owned();
                map.nodes[0].shard = String::from_utf8_lossy(&new_id()).into_owned();
            });
        }
    }
    if hard {
        server.cluster.epoch.store(0, Relaxed);
    }
    server.recount_coverage();
    save(server)?;
    out.ok();
    Ok(())
}

// -------------------------------------------------------------- the key sets

/// `CLUSTER COUNTKEYSINSLOT`.
fn count_keys(server: &Server, at: usize, args: Args<'_>, out: &mut Out) -> Result<()> {
    let n = args.int(2)?;
    let slot = u16::try_from(n)
        .ok()
        .filter(|s| usize::from(*s) < SLOTS)
        .ok_or_else(|| Error::new(Code::Invalid, "Invalid slot"))?;
    let mut found = 0i64;
    server.dbs[at].keys(|key| {
        if key_slot(key) == slot {
            found += 1;
        }
    });
    out.int(found);
    Ok(())
}

/// `CLUSTER GETKEYSINSLOT`.
fn get_keys(server: &Server, at: usize, args: Args<'_>, out: &mut Out) -> Result<()> {
    let slot = args.int(2)?;
    let count = args.int(3)?;
    let bad = || Error::new(Code::Invalid, "Invalid slot or number of keys");
    if !(0..SLOTS as i64).contains(&slot) || count < 0 {
        return Err(bad());
    }
    let slot = slot as u16;
    let want = count as usize;
    let start = out.len();
    let mut n = 0;
    server.dbs[at].keys(|key| {
        if n < want && key_slot(key) == slot {
            out.bulk(key);
            n += 1;
        }
    });
    out.close_array(start, n);
    Ok(())
}

// ------------------------------------------------------------- the disk copy

/// Write the table where a restart will find it.
///
/// The format is the reference's, line for line, so a node's file can be read by
/// a real server and the other way round. The write goes to a temporary name and
/// is renamed over the real one, the same way the snapshot writer does it, so a
/// reader never sees half a file.
fn save(server: &Server) -> Result<()> {
    let path = {
        let file = server.cluster.file.lock();
        if file.is_empty() {
            return Ok(());
        }
        yo_alloc::allow(|| file.clone())
    };
    let epoch = server.cluster.epoch.load(Relaxed);
    // The vote goes in the file because it has to outlive the process. A node
    // that voted, restarted and voted again in the same epoch would have voted
    // twice, and two votes from one master is how two replicas of the same
    // master both come away believing they won.
    let voted = server.cluster.vote.given();
    let text = yo_alloc::allow(|| {
        let mut s = lines(server, true);
        let _ = writeln!(s, "vars currentEpoch {epoch} lastVoteEpoch {voted}");
        s
    });
    yo_alloc::allow(|| {
        let temp = format!("{path}.tmp");
        let wrote =
            std::fs::write(&temp, text.as_bytes()).and_then(|()| std::fs::rename(&temp, &path));
        match wrote {
            Ok(()) => Ok(()),
            Err(e) => Err(Error::fmt(
                Code::Invalid,
                format_args!("cluster config file could not be written: {e}"),
            )),
        }
    })
}

impl Server {
    /// Read the table back off disk, which is what makes a restart a restart
    /// rather than a node that has forgotten which half of the keyspace is its.
    ///
    /// A file that is not there is not an error, because the first start of a
    /// brand new node has none. A file that is there and is nonsense is, because
    /// the alternative is a node that comes up owning nothing and starts
    /// answering `CLUSTERDOWN` for keys it has on disk.
    fn reload_cluster(&mut self) -> Result<()> {
        let path = self.cluster_file();
        if path.is_empty() {
            return Ok(());
        }
        let text = match yo_alloc::allow(|| std::fs::read_to_string(&path)) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => {
                return Err(Error::fmt(Code::Invalid, format_args!("{path}: {e}")));
            }
        };
        yo_alloc::allow(|| self.absorb_cluster(&text))
    }

    /// The parse, split out so that a test can hand it text without a file.
    fn absorb_cluster(&mut self, text: &str) -> Result<()> {
        let bad = |what: &str| Error::fmt(Code::Invalid, format_args!("{what} in cluster config"));
        let now = self.now_ms();
        let mut nodes: Vec<Node> = Vec::new();
        // Slot runs are kept against the owner's id and not against its position
        // in the list, because this node moves to the front of the list when its
        // own line turns up and every position recorded before that moves down.
        let mut owned: Vec<(String, u16, u16)> = Vec::new();
        // The master field names a node that may not have been read yet, so
        // both ends are kept as ids and resolved once every line is in.
        let mut follows: Vec<(String, String)> = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let mut words = line.split(' ');
            let first = words.next().unwrap_or_default();
            if first == "vars" {
                // `vars currentEpoch <n> lastVoteEpoch <n>`, read as pairs so
                // that a build which adds a third one does not break the parse.
                let rest: Vec<&str> = words.collect();
                for pair in rest.chunks(2) {
                    if pair.len() == 2 && pair[0] == "currentEpoch" {
                        let epoch = pair[1].parse::<u64>().map_err(|_| bad("bad epoch"))?;
                        self.cluster.epoch.store(epoch, Relaxed);
                    }
                    if pair.len() == 2 && pair[0] == "lastVoteEpoch" {
                        let epoch = pair[1].parse::<u64>().map_err(|_| bad("bad epoch"))?;
                        self.cluster.vote.reload(epoch);
                    }
                }
                continue;
            }
            if first.len() != ID_LEN {
                return Err(bad("bad node id"));
            }
            let address = words.next().ok_or_else(|| bad("missing address"))?;
            let named = words.next().ok_or_else(|| bad("missing flags"))?;
            let master = words.next().ok_or_else(|| bad("missing master"))?;
            let ping = words.next().ok_or_else(|| bad("missing ping time"))?;
            let pong = words.next().ok_or_else(|| bad("missing pong time"))?;
            let epoch = words
                .next()
                .and_then(|w| w.parse::<u64>().ok())
                .ok_or_else(|| bad("bad config epoch"))?;
            // The link state field, which is read past rather than read.
            let _link = words.next();
            let (host, port, bus, shard) =
                split_address(address).ok_or_else(|| bad("bad address"))?;
            for word in words {
                // The migrating and importing markers are re-read from the
                // brackets below rather than here, since they name a node that
                // may not have been read yet.
                if word.starts_with('[') {
                    continue;
                }
                let (from, to) = match word.split_once('-') {
                    Some((a, b)) => (
                        a.parse::<u16>().map_err(|_| bad("bad slot"))?,
                        b.parse::<u16>().map_err(|_| bad("bad slot"))?,
                    ),
                    None => {
                        let one = word.parse::<u16>().map_err(|_| bad("bad slot"))?;
                        (one, one)
                    }
                };
                if usize::from(from) >= SLOTS || usize::from(to) >= SLOTS || from > to {
                    return Err(bad("slot out of range"));
                }
                owned.push((first.to_owned(), from, to));
            }
            let mut flags = 0u16;
            for name in named.split(',') {
                flags |= match name {
                    "myself" => FLAG_MYSELF,
                    "master" => FLAG_MASTER,
                    "slave" => FLAG_SLAVE,
                    "fail?" => FLAG_PFAIL,
                    "fail" => FLAG_FAIL,
                    "handshake" => FLAG_HANDSHAKE,
                    "noaddr" => FLAG_NOADDR,
                    "nofailover" => FLAG_NOFAILOVER,
                    _ => 0,
                };
            }
            if master != "-" {
                if master.len() != ID_LEN {
                    return Err(bad("bad master id"));
                }
                follows.push((first.to_owned(), master.to_owned()));
            }
            let node = Node {
                id: first.to_owned(),
                host,
                port,
                bus,
                shard,
                epoch,
                flags,
                master: None,
                // The two timestamps come back as now rather than as what the
                // file says, because a file written an hour ago would otherwise
                // put every node in it straight past the failure timeout. All
                // the file is really recording is whether a ping was in flight.
                ping_sent: stamp(ping, now),
                pong_recv: stamp(pong, now),
                data_recv: now,
                fail_time: 0,
                offset: 0,
                // The link state in the file is written for a human reading it
                // and is not read back. There is no link to anybody at startup,
                // and saying otherwise would stop the bus ever dialling out.
                linked: false,
                // Not in the file, which is the reference's shape as well: the
                // epoch this node last voted in is, and that is the check that
                // has to survive a restart. This one only spaces two elections
                // out and a node that has just started has not held one.
                voted_time: 0,
                reports: Vec::new(),
            };
            if named.split(',').any(|f| f == "myself") {
                // This node goes first, and everything already read moves down.
                nodes.insert(0, node);
            } else {
                nodes.push(node);
            }
        }
        if nodes.is_empty() {
            return Ok(());
        }
        let mut map = Map::new(nodes.remove(0));
        map.nodes.append(&mut nodes);
        for (id, from, to) in owned {
            let Some(at) = map.find(id.as_bytes()) else {
                return Err(bad("slots for a node nobody knows"));
            };
            for slot in from..=to {
                map.owner[usize::from(slot)] = Some(at);
            }
        }
        for (who, whose) in follows {
            let (Some(who), Some(whose)) = (map.find(who.as_bytes()), map.find(whose.as_bytes()))
            else {
                return Err(bad("master id nobody knows"));
            };
            map.nodes[usize::from(who)].master = Some(whose);
            // A replica's config epoch is its master's business, so the file's
            // value is dropped rather than believed. Keeping it would let a
            // replica that was a master an hour ago outvote the node that took
            // its slots the moment it came back.
            map.nodes[usize::from(who)].epoch = 0;
        }
        *self.cluster.map.lock() = map;
        Ok(())
    }
}

/// One of the two timestamps out of the node table, read as a yes or a no.
///
/// The reference does the same thing for the same reason. The number in the
/// file was taken from a clock that has since moved on, and all the reader can
/// usefully learn from it is whether there was a ping in flight when the file
/// was written, so a number that is not zero comes back as the time now.
fn stamp(field: &str, now: u64) -> u64 {
    match field.parse::<u64>() {
        Ok(0) | Err(_) => 0,
        Ok(_) => now,
    }
}

/// Pull the host, the port and the shard id out of one `ip:port@bus,aux` field.
///
/// The aux fields after the first comma are a list of `name=value` pairs with an
/// optional hostname in front of them, and a build that adds one has to leave a
/// build that does not able to read the file, so everything but the shard id is
/// skipped rather than being counted.
fn split_address(field: &str) -> Option<(String, u16, u16, String)> {
    let (address, aux) = match field.split_once(',') {
        Some((address, aux)) => (address, aux),
        None => (field, ""),
    };
    let (host, ports) = address.rsplit_once(':')?;
    let (client, bus) = match ports.split_once('@') {
        Some((client, bus)) => (client, bus.parse::<u16>().ok()?),
        None => (ports, 0),
    };
    let port = client.parse::<u16>().ok()?;
    let bus = if bus == 0 { port + BUS_OFFSET } else { bus };
    let shard = aux
        .split(',')
        .find_map(|pair| pair.strip_prefix("shard-id="))
        .map_or_else(
            || String::from_utf8_lossy(&new_id()).into_owned(),
            str::to_owned,
        );
    Some((host.to_owned(), port, bus, shard))
}

// ------------------------------------------------------- the hand filled map

/// What a test uses to fill the table the bus will fill later.
///
/// A single node cannot make a redirection happen: it owns every slot or it
/// owns none, and either way there is nowhere to send anybody. So the tests for
/// `MOVED`, `ASK` and `TRYAGAIN` put a second node in the table by hand, which
/// is exactly the state the bus will produce and is the state the config file
/// parser already reads.
#[cfg(test)]
impl Server {
    /// Take every slot, and be up about it straight away.
    pub(super) fn cluster_own_everything(&self) {
        {
            let mut map = self.cluster.map.lock();
            for slot in 0..SLOTS {
                map.owner[slot] = Some(0);
            }
        }
        self.recount_coverage();
        self.cluster.was_down.store(false, Relaxed);
        self.cluster.booted_at.store(0, Relaxed);
    }

    /// Put a node in the table that nobody has met, and answer its index.
    pub(super) fn cluster_pretend_node(&self, id: &str, host: &str, port: u16) -> u16 {
        let mut map = self.cluster.map.lock();
        let mut node = Node::new(
            id.to_owned(),
            host.to_owned(),
            port,
            port + BUS_OFFSET,
            FLAG_MASTER,
            0,
        );
        node.shard = id.to_owned();
        node.linked = true;
        map.nodes.push(node);
        (map.nodes.len() - 1) as u16
    }

    /// Hand one slot to another node.
    pub(super) fn cluster_hand_over(&self, slot: u16, node: u16) {
        let mut map = self.cluster.map.lock();
        map.owner[usize::from(slot)] = Some(node);
    }

    /// Make this node a replica of another one in the table.
    ///
    /// What `CLUSTER REPLICATE` does to the table, without the half of it that
    /// starts replicating, which an embedded server has no way of doing and
    /// which none of the tests that want a replica in the table care about.
    pub(super) fn cluster_pretend_follower(&self, of: u16) {
        let mut map = self.cluster.map.lock();
        map.nodes[0].flags &= !FLAG_MASTER;
        map.nodes[0].flags |= FLAG_SLAVE;
        map.nodes[0].master = Some(of);
    }

    /// Mark one slot as on its way out to, or in from, another node.
    pub(super) fn cluster_moving(&self, slot: u16, to: Option<u16>, from: Option<u16>) {
        let mut map = self.cluster.map.lock();
        map.migrating[usize::from(slot)] = to;
        map.importing[usize::from(slot)] = from;
    }
}

// ---------------------------------------------------------------- the tests

#[cfg(test)]
mod tests {
    use super::{SLOTS, key_slot};

    /// The two ways of moving a slot write the same three fields, so a slot an
    /// atomic migration is moving is refused to the old way by name.
    ///
    /// The sentence is compared against the one a real 8.10.1 sends, because a
    /// resharding tool is what reads it and the way out of the situation is
    /// named in it.
    #[test]
    fn setslot_will_not_touch_a_slot_a_migration_is_moving() {
        use crate::proto::{Limits, Proto};
        use crate::reply::Out;
        use crate::request::{Argv, Step};

        let mut server = super::Server::new();
        server.enable_cluster("", 7000);
        server.cluster_own_everything();
        let other = server.cluster_pretend_node(
            "5b1e2ce29b1e0c86bd53ee1e5b0dd7b66c0e6e0f",
            "10.0.0.9",
            7002,
        );
        assert_eq!(other, 1);

        let said = |server: &super::Server, argv: &[&[u8]]| {
            let mut wire = Vec::new();
            wire.extend_from_slice(format!("*{}\r\n", argv.len()).as_bytes());
            for arg in argv {
                wire.extend_from_slice(format!("${}\r\n", arg.len()).as_bytes());
                wire.extend_from_slice(arg);
                wire.extend_from_slice(b"\r\n");
            }
            let mut decoded = Argv::new();
            assert!(matches!(
                decoded.decode(&wire, &Limits::default()).unwrap(),
                Step::Command { .. }
            ));
            let args = super::Args::new(&decoded, &wire);
            let mut out = Out::new(Proto::Resp2);
            match super::setslot(server, 0, args, &mut out) {
                Ok(()) => String::from_utf8_lossy(out.as_slice()).into_owned(),
                Err(e) => e.to_string(),
            }
        };

        // Nothing is moving, so the old way works.
        assert_eq!(
            said(&server, &[b"CLUSTER", b"SETSLOT", b"100", b"STABLE"]),
            "+OK\r\n"
        );

        server
            .asm_begin_import(vec![b'b'; 40], vec![(100, 200)])
            .expect("nothing else is running");
        for slot in [b"100".as_slice(), b"150", b"200"] {
            let got = said(&server, &[b"CLUSTER", b"SETSLOT", slot, b"STABLE"]);
            let want = format!(
                "Slot {} is currently in an active atomic slot migration. \
                 CLUSTER SETSLOT cannot be used at this time. To perform a legacy slot migration \
                 instead, first cancel the ongoing task with CLUSTER MIGRATION CANCEL",
                String::from_utf8_lossy(slot)
            );
            assert!(got.ends_with(&want), "{got:?}");
        }
        // Either side of the range is untouched, since only the slots actually
        // moving are the ones there are two writers for.
        assert_eq!(
            said(&server, &[b"CLUSTER", b"SETSLOT", b"99", b"STABLE"]),
            "+OK\r\n"
        );
        assert_eq!(
            said(&server, &[b"CLUSTER", b"SETSLOT", b"201", b"STABLE"]),
            "+OK\r\n"
        );
        // And the gate goes when the task does.
        server.cluster.asm.cancel(None, 1);
        assert_eq!(
            said(&server, &[b"CLUSTER", b"SETSLOT", b"150", b"STABLE"]),
            "+OK\r\n"
        );
    }

    /// The slots the reference answered for these keys, read off a running
    /// 8.10.1 with cluster mode on rather than worked out from the algorithm.
    #[test]
    fn a_key_lands_in_the_slot_a_real_server_puts_it_in() {
        assert_eq!(key_slot(b"foo"), 12182);
        assert_eq!(key_slot(b"1234"), 6025);
        assert_eq!(key_slot(b""), 0);
        assert_eq!(key_slot(b"{user1000}.following"), 3443);
    }

    /// The hash tag is the whole reason a command can name two keys at all, and
    /// its edge cases are the ones a client library gets wrong.
    #[test]
    fn the_hash_tag_rules_are_the_reference_rules() {
        // Two keys with the same tag are one slot, whatever else they say.
        assert_eq!(
            key_slot(b"{user1000}.following"),
            key_slot(b"{user1000}.followers")
        );
        // A tag with nothing in it is not a tag.
        assert_eq!(key_slot(b"{}foo"), key_slot(b"{}foo"));
        assert_ne!(key_slot(b"{}foo"), key_slot(b"foo"));
        // An opening brace with no closing one is not a tag either.
        assert_ne!(key_slot(b"{foo"), key_slot(b"foo"));
        // Only the first closing brace after the first opening one counts.
        assert_eq!(key_slot(b"{a}{b}"), key_slot(b"a"));
        // A tag can hold a brace, since the search is for the first close.
        assert_eq!(key_slot(b"foo{{bar}}zap"), key_slot(b"{bar"));
    }

    /// Every slot is reachable and none is out of range, which is the only
    /// property the command path actually leans on.
    #[test]
    fn every_slot_is_in_range() {
        let mut seen = vec![false; SLOTS];
        for i in 0..200_000u32 {
            let key = i.to_string();
            let slot = key_slot(key.as_bytes());
            assert!(usize::from(slot) < SLOTS);
            seen[usize::from(slot)] = true;
        }
        assert!(seen.iter().all(|s| *s), "200k keys reach all 16384 slots");
    }

    /// Reading the node table back has to put every run of slots on the node
    /// that owns it, including when this node's own line is not the first one.
    ///
    /// It is worth a test of its own because the failure is silent and only
    /// shows up on a cluster with three nodes in it: this node moves to the
    /// front of the list when its line is read, everything read before it moves
    /// down one, and a run recorded against a position rather than against an
    /// id ends up on the wrong node.
    #[test]
    fn a_config_file_puts_every_run_on_the_node_that_owns_it() {
        let mut server = super::Server::new();
        server.enable_cluster("", 7355);
        let text = "\
3b80b05445f38bc7214f083696a2bbf90e3f30e3 127.0.0.1:7356@17356 master - 0 0 0 connected 10923-16383
30d0651b0ec5e178e082634c44fb9adcc6e4021b 127.0.0.1:7355@17355 myself,master - 0 0 1 connected 5461-10922
19a9e69b8b66016ac43c55ccdeed0283e0148e17 127.0.0.1:7354@17354 master - 0 0 2 connected 0-5460
vars currentEpoch 2 lastVoteEpoch 0
";
        server.absorb_cluster(text).expect("the file parses");
        let map = server.cluster.map.lock();
        let at = |id: &str| map.find(id.as_bytes()).expect("the node is in the table");
        assert_eq!(at("30d0651b0ec5e178e082634c44fb9adcc6e4021b"), 0, "myself");
        for (id, from, to) in [
            ("19a9e69b8b66016ac43c55ccdeed0283e0148e17", 0, 5460),
            ("30d0651b0ec5e178e082634c44fb9adcc6e4021b", 5461, 10922),
            ("3b80b05445f38bc7214f083696a2bbf90e3f30e3", 10923, 16383),
        ] {
            let owner = Some(at(id));
            for slot in from..=to {
                assert_eq!(map.owner[slot], owner, "slot {slot} belongs to {id}");
            }
        }
    }

    /// Three fields in the node table say what was true when it was written and
    /// not what is true now, and reading them back as if they were live state
    /// is what stops a restarted node ever dialling anybody again.
    #[test]
    fn a_config_file_is_not_read_back_as_live_state() {
        let mut server = super::Server::new();
        server.enable_cluster("", 7357);
        let text = "\
9fcbb7624dedbb2fd0020dd2fbf86a5eb8cec31b 127.0.0.1:7355@17355 master - 0 1789005531306 3 connected 5461-10922
ac6dd51a69741dc5130637c594866c9b7e0cfc4e 127.0.0.1:7357@17357 myself,slave 9fcbb7624dedbb2fd0020dd2fbf86a5eb8cec31b 1789005400000 1789005532315 3 connected
";
        server.absorb_cluster(text).expect("the file parses");
        let now = server.now_ms();
        let map = server.cluster.map.lock();
        for node in &map.nodes {
            assert!(!node.linked, "nothing is linked before the bus dials out");
        }
        // A ping that was in flight comes back as one sent now, and a ping that
        // was not stays at zero, which is the difference the cron reads.
        assert_eq!(map.nodes[0].ping_sent, now, "myself had a ping in flight");
        assert_eq!(map.nodes[1].ping_sent, 0, "the master did not");
        assert_eq!(map.nodes[0].pong_recv, now);
        // And a replica's config epoch belongs to its master, not to the file.
        assert_eq!(map.nodes[0].epoch, 0, "the replica's epoch is dropped");
        assert_eq!(map.nodes[1].epoch, 3, "the master's is kept");
    }
}
