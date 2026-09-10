//! The cluster bus, which is how nodes find each other and agree on who owns
//! what.
//!
//! Every cluster node listens on a second port, the client port plus ten
//! thousand, and speaks a small binary protocol on it that has nothing to do
//! with RESP. Nodes ping each other about once a second, and every ping carries
//! a header describing the sender, a handful of gossip entries describing other
//! nodes the sender has heard from lately, and a few extensions on the end. Out
//! of that falls everything a cluster knows: who exists, who owns which slots,
//! who is a replica of whom, and who has stopped answering.
//!
//! The format here is the reference's, field for field and offset for offset,
//! because the whole point of it is that a yo node and a real Redis node can be
//! in the same cluster. The reference guarantees its own layout with static
//! asserts in `cluster_legacy.h`, which is a promise that the offsets are part
//! of the protocol and not an accident of one compiler, so copying them is
//! reading a specification rather than reading an implementation.
//!
//! # Why this has its own threads
//!
//! Every client connection in this server is driven by a poller that lives in
//! the binary crate, and this crate cannot reach it, so the bus cannot join it.
//! That sounds like a limitation and is really a preference. Bus traffic is
//! roughly one packet a second per node in a cluster that is behaving, which
//! means a thread per link costs a thread that is asleep almost all of the time
//! and buys code that reads top to bottom with no state machine in it. A
//! cluster of a hundred nodes is a hundred sleeping threads, which is nothing,
//! and a cluster larger than that is not a thing anybody runs. It is registered
//! as a divergence anyway, because a reader who knows the reference will expect
//! the bus to be in the event loop and should not have to find out by grep.
//!
//! # What is not here
//!
//! The manual failover. `MFSTART` is recognised and dropped rather than
//! answered, so a real server's replica that is asked to take over by hand
//! waits its five seconds out and gives up, and `CLUSTER FAILOVER` here is
//! still refused. The vote it would use is in, since the same election runs
//! whether the master died or is being stood down, and the flag that says the
//! master is up and in on it is honoured on the side being asked.

use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use yo_common::lock::Lock;

use super::super::Server;
use super::super::pubsub::{self, Kind};
use super::{
    FLAG_FAIL, FLAG_HANDSHAKE, FLAG_MASTER, FLAG_MEET, FLAG_MIGRATE_TO, FLAG_MYSELF, FLAG_NOADDR,
    FLAG_NOFAILOVER, FLAG_PFAIL, FLAG_SLAVE, Map, Node, SLOTS, new_id,
};
use crate::reply::Out;

// ------------------------------------------------------------- the wire format

/// The four bytes every packet starts with, which is the only thing that tells
/// a bus port from anything else somebody might connect to it.
const SIG: &[u8; 4] = b"RCmb";

/// The protocol version. A packet claiming anything else is dropped without a
/// word, because a node that cannot read a packet cannot say so either.
const PROTO_VER: u16 = 1;

/// How long a node id is, in bytes of lowercase hex.
const NAME_LEN: usize = 40;

/// How much room an address gets, which is enough for the longest IPv6 text
/// form and is fixed because the header is fixed.
const IP_LEN: usize = 46;

/// The slot bitmap, one bit per slot.
const BITMAP_LEN: usize = SLOTS / 8;

/// The header every packet carries, whatever its type. The reference calls this
/// `CLUSTERMSG_MIN_LEN` and asserts it.
const HDR_LEN: usize = 2256;

/// One gossip entry.
const GOSSIP_LEN: usize = 104;

/// The most a packet may be before it is treated as noise rather than as a
/// packet. The reference grows its receive buffer to whatever the header claims,
/// which is fine when the peer is trusted and is a way to be talked into
/// allocating a lot when it is not, so there is a ceiling here. A publish of a
/// megabyte over the bus is already an unusual thing to do.
const MAX_PACKET: usize = 64 * 1024 * 1024;

/// Where each header field starts. These are the reference's, and the reference
/// asserts them, so they are the protocol.
const O_TOTLEN: usize = 4;
const O_VER: usize = 8;
const O_PORT: usize = 10;
const O_TYPE: usize = 12;
const O_COUNT: usize = 14;
const O_CURRENT_EPOCH: usize = 16;
const O_CONFIG_EPOCH: usize = 24;
const O_OFFSET: usize = 32;
const O_SENDER: usize = 40;
const O_SLOTS: usize = 80;
const O_SLAVEOF: usize = 2128;
const O_MYIP: usize = 2168;
const O_EXTENSIONS: usize = 2214;
const O_PPORT: usize = 2246;
const O_CPORT: usize = 2248;
const O_FLAGS: usize = 2250;
const O_STATE: usize = 2252;
const O_MFLAGS: usize = 2253;

/// Where each field of a gossip entry starts, from the start of the entry.
const G_NAME: usize = 0;
const G_PING: usize = 40;
const G_PONG: usize = 44;
const G_IP: usize = 48;
const G_PORT: usize = 94;
const G_CPORT: usize = 96;
const G_FLAGS: usize = 98;

/// The message types, which are the reference's numbers and cannot move.
const T_PING: u16 = 0;
const T_PONG: u16 = 1;
const T_MEET: u16 = 2;
const T_FAIL: u16 = 3;
const T_PUBLISH: u16 = 4;
const T_AUTH_REQUEST: u16 = 5;
const T_AUTH_ACK: u16 = 6;
const T_UPDATE: u16 = 7;
const T_MFSTART: u16 = 8;
const T_MODULE: u16 = 9;
const T_PUBLISHSHARD: u16 = 10;

/// The message flag that says vote for me even though my master is up, which is
/// the only thing that makes a manual failover different from any other on the
/// side being asked.
const MF_FORCEACK: u8 = 2;

/// The one message flag that goes out on everything, which says this node
/// understands the extensions on the end of a ping and is safe to send them to.
const MF_EXT_DATA: u8 = 4;

/// The extension types.
const X_HOSTNAME: u16 = 0;
const X_HUMAN_NAME: u16 = 1;
const X_FORGOTTEN: u16 = 2;
const X_SHARDID: u16 = 3;
const X_SECRET: u16 = 4;

/// `CLUSTER_OK` and `CLUSTER_FAIL`, which is the one byte of state a packet
/// carries and is what lets a node see that the rest of the cluster has given
/// up even when it has not.
const STATE_OK: u8 = 0;
const STATE_FAIL: u8 = 1;

/// How long a node id stays unwelcome after `CLUSTER FORGET`, in milliseconds.
///
/// Forgetting a node means nothing on its own, because the next gossip packet
/// from anybody who has not forgotten it would put it straight back. So a forget
/// is a forget plus a minute of refusing to learn it again, by which time the
/// forget has been passed round the cluster in the extension that carries it.
const BLACKLIST_MS: u64 = 60_000;

/// How often the bus wakes up, which is the reference's `clusterCron`.
const CRON_MS: u64 = 100;

/// How often a node is pinged when nothing else has prompted a ping. The
/// reference derives this from the node timeout and lets it be set outright;
/// this is the derived default.
const PING_MS: u64 = 1000;

/// How long a node has to be silent before this one calls it possibly failed.
const NODE_TIMEOUT_MS: u64 = 15_000;

/// How long to wait on a connect to another node's bus port before giving up
/// and trying again on the next tick.
const CONNECT_TIMEOUT: Duration = Duration::from_millis(500);

/// The fixed part of the wait before a replica asks to be promoted.
///
/// Long enough for the FAIL message to have got round the cluster, because a
/// vote asked for before the electorate agrees the master is gone is a vote
/// nobody may grant. The reference adds a random part on top of the same size,
/// which is what keeps two replicas that noticed at the same moment from asking
/// at the same moment.
const ELECTION_DELAY_MS: u64 = 500;

/// How much later a replica asks for every other replica that holds more data
/// than it does.
///
/// This is the whole of how the cluster picks the best replica without anybody
/// comparing offsets: the one with the most data asks first, and the others are
/// still waiting when it has already won.
const RANK_DELAY_MS: u64 = 1000;

/// How long an election may take before it is written off, and how long after
/// that before another one is started.
///
/// Timeout is twice the node timeout with a floor of two seconds and retry is
/// twice the timeout, which are the reference's numbers. The gap between them is
/// what keeps a replica that cannot win from asking again and again and running
/// the epoch up on every master in the cluster.
const ELECTION_TIMEOUT_MS: u64 = if NODE_TIMEOUT_MS * 2 > 2000 {
    NODE_TIMEOUT_MS * 2
} else {
    2000
};

/// How stale a replica's data may be and still be worth promoting.
///
/// The reference works this out from `repl-ping-replica-period` and
/// `cluster-replica-validity-factor`, neither of which is in the config table
/// yet, so this is those two at their defaults: ten seconds of ping period plus
/// ten node timeouts. A replica further behind than that is one whose master was
/// unreachable long before it died, and promoting it would lose more than
/// letting the shard stay down does.
const STALE_DATA_MS: u64 = 10_000 + NODE_TIMEOUT_MS * 10;

/// The election this node is standing in or voting in.
///
/// Atomics rather than a lock because they are read from the cron and written
/// from whichever link thread a packet arrived on, and nothing here is read
/// together with anything else here except by the cron, which is the only writer
/// of everything but the count.
#[derive(Default)]
pub(super) struct Vote {
    /// The epoch this node last gave its vote away in, which is the reference's
    /// `lastVoteEpoch` and is the whole of one vote per epoch.
    given: AtomicU64,
    /// When this node may start asking, or nought when it never has.
    at: AtomicU64,
    /// How many votes have come back for the election in `epoch`.
    count: AtomicU64,
    /// The epoch this node is standing in.
    epoch: AtomicU64,
    /// How many replicas of the same master held more data than this one when
    /// the delay was worked out, which is what that delay is made of.
    rank: AtomicU64,
    /// Whether the request has gone out, so that a reply is worth counting and
    /// the delay is not recomputed underneath it.
    sent: AtomicBool,
}

impl Vote {
    /// The epoch this node last voted in, for the config file.
    pub(super) fn given(&self) -> u64 {
        self.given.load(Relaxed)
    }

    /// Put back what the config file said, which is why the file has it.
    pub(super) fn reload(&self, epoch: u64) {
        self.given.store(epoch, Relaxed);
    }
}

/// Read a big endian field out of a packet, or nought when it is off the end.
///
/// Every read here is bounds checked rather than trusted, because the packet
/// came off a socket and the length checks in front of it are checks and not
/// proofs. A truncated field reading as nought loses a packet, which is what
/// losing a packet on a gossip protocol is for.
fn be16(p: &[u8], at: usize) -> u16 {
    p.get(at..at + 2)
        .map_or(0, |b| u16::from_be_bytes([b[0], b[1]]))
}

fn be32(p: &[u8], at: usize) -> u32 {
    p.get(at..at + 4)
        .map_or(0, |b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

fn be64(p: &[u8], at: usize) -> u64 {
    p.get(at..at + 8).map_or(0, |b| {
        u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
    })
}

/// Put a big endian field into a packet being built, which is always in range
/// because the buffer was sized first.
fn put16(p: &mut [u8], at: usize, v: u16) {
    p[at..at + 2].copy_from_slice(&v.to_be_bytes());
}

fn put64(p: &mut [u8], at: usize, v: u64) {
    p[at..at + 8].copy_from_slice(&v.to_be_bytes());
}

/// A fixed width text field, which is NUL padded and may not be terminated when
/// it is exactly full.
fn text(p: &[u8], at: usize, len: usize) -> String {
    let Some(raw) = p.get(at..at + len) else {
        return String::new();
    };
    let end = raw.iter().position(|b| *b == 0).unwrap_or(len);
    String::from_utf8_lossy(&raw[..end]).into_owned()
}

/// A node id out of a packet, which has to be forty hex characters or it is not
/// one. An id that is all zeros is the reference's way of saying no node, and
/// comes back as `None` here so the caller cannot mistake it for one.
fn node_id(p: &[u8], at: usize) -> Option<String> {
    let raw = p.get(at..at + NAME_LEN)?;
    if raw.iter().all(|b| *b == 0) {
        return None;
    }
    if !raw.iter().all(u8::is_ascii_hexdigit) {
        return None;
    }
    Some(String::from_utf8_lossy(raw).into_owned())
}

/// Whether a slot is set in a bitmap, which the reference stores least
/// significant bit first inside each byte.
fn bit(bitmap: &[u8], slot: usize) -> bool {
    bitmap
        .get(slot / 8)
        .is_some_and(|b| b & (1 << (slot % 8)) != 0)
}

/// Set a slot in a bitmap being built.
fn set_bit(bitmap: &mut [u8], slot: usize) {
    bitmap[slot / 8] |= 1 << (slot % 8);
}

/// A node index drawn at random out of `n`, which is how the gossip section
/// picks who to talk about. Random and not round robin because every node
/// picking independently is what spreads news in a few rounds rather than in a
/// lap of the table.
fn pick(n: usize) -> u16 {
    let mut raw = [0u8; 8];
    yo_common::entropy::fill(&mut raw);
    (u64::from_be_bytes(raw) % n as u64) as u16
}

// ------------------------------------------------------------------ the links

/// One connection to another node's bus.
///
/// There are two of these per pair of nodes and not one, which looks wasteful
/// and is the reference's design for a good reason: a link is owned by whoever
/// opened it, so a node that decides a peer is unreachable can drop its own link
/// and reconnect without having to agree with the peer about whose turn it is.
/// The outbound one carries this node's pings and the inbound one carries the
/// peer's, and each is answered on the socket it arrived on.
pub(super) struct Wire {
    /// Whether this node accepted it rather than opened it, which is the
    /// `from` and `to` of `CLUSTER LINKS`.
    pub(super) inbound: bool,
    /// The node at the far end, empty on an inbound link until a packet says
    /// who is sending it.
    pub(super) node: Lock<String>,
    /// When it was made, which `CLUSTER LINKS` reports.
    pub(super) created: u64,
    /// The address the peer connected from or was connected to, which is how a
    /// node learns the address of a peer that did not announce one.
    peer: String,
    /// The address on this side of it, which is how a node learns its own.
    local: String,
    /// The socket, behind a real mutex rather than the spin lock the tables use,
    /// because writing to it blocks and a spin lock held across a blocking write
    /// is a spin lock held for a millisecond.
    sock: Mutex<TcpStream>,
    /// Whether it has failed, so that a writer stops trying and the cron drops
    /// it on the next tick.
    dead: AtomicBool,
    /// How many bytes have been handed to the kernel on it, which is the closest
    /// honest answer to `CLUSTER LINKS`'s send buffer questions on a link that
    /// has no send queue of its own.
    sent: AtomicU64,
}

impl Wire {
    /// Wrap an open socket.
    fn new(sock: TcpStream, inbound: bool, node: &str, now: u64) -> Option<Arc<Wire>> {
        let peer = sock.peer_addr().ok()?.ip().to_string();
        let local = sock.local_addr().ok()?.ip().to_string();
        Some(Arc::new(Wire {
            inbound,
            node: Lock::new(String::from(node)),
            created: now,
            peer,
            local,
            sock: Mutex::new(sock),
            dead: AtomicBool::new(false),
            sent: AtomicU64::new(0),
        }))
    }

    /// The id of the node at the far end, or empty for one nobody has named.
    fn named(&self) -> String {
        let held = self.node.lock();
        yo_alloc::allow(|| held.clone())
    }

    /// Point this link at a node, which happens on an inbound link the first
    /// time a packet on it says who is sending.
    fn name(&self, id: &str) {
        let mut held = self.node.lock();
        yo_alloc::allow(|| {
            held.clear();
            held.push_str(id);
        });
    }

    /// Write a packet, and mark the link dead if that did not work.
    ///
    /// A failed write is not reported anywhere, because there is nobody to
    /// report it to: the peer is either coming back, in which case the cron
    /// reconnects, or it is not, in which case the timeout is the thing that
    /// notices. That is the same as the reference, which frees the link and
    /// carries on.
    fn send(&self, packet: &[u8]) {
        if self.dead.load(Relaxed) {
            return;
        }
        let Ok(mut sock) = self.sock.lock() else {
            self.dead.store(true, Relaxed);
            return;
        };
        if sock.write_all(packet).is_err() {
            self.dead.store(true, Relaxed);
            let _ = sock.shutdown(Shutdown::Both);
            return;
        }
        self.sent.store(packet.len() as u64, Relaxed);
    }

    /// Shut it down, which wakes the reader thread sitting on it.
    fn kill(&self) {
        self.dead.store(true, Relaxed);
        if let Ok(sock) = self.sock.lock() {
            let _ = sock.shutdown(Shutdown::Both);
        }
    }
}

/// Everything the bus owns that is not in the node table.
///
/// Separate from [`super::Cluster`] because none of it exists until the bus
/// starts and none of it is looked at by a command on the hot path, so it is one
/// lock rather than four fields on a struct every server has.
#[derive(Default)]
pub(super) struct Bus {
    /// Every open link, both directions.
    links: Lock<Vec<Arc<Wire>>>,
    /// Ids that may not be learned again yet, and when they stop being unwelcome.
    blacklist: Lock<Vec<(String, u64)>>,
    /// The shared secret nodes use to recognise each other. Forty hex characters
    /// made at start, and the whole cluster converges on whichever is smallest,
    /// which is a rule that needs no coordinator to settle.
    pub(super) secret: Lock<String>,
    /// Whether the threads are up, so that starting twice is a no-op.
    on: AtomicBool,
    /// Whether the table has changed since it was last written.
    dirty: AtomicBool,
}

impl Bus {
    /// Add a link, dropping any that have died since the last look.
    fn add(&self, wire: &Arc<Wire>) {
        let mut links = self.links.lock();
        yo_alloc::allow(|| {
            links.retain(|held| !held.dead.load(Relaxed));
            links.push(Arc::clone(wire));
        });
    }

    /// The outbound link to a node, or `None` when there is not one.
    fn outbound(&self, id: &str) -> Option<Arc<Wire>> {
        let links = self.links.lock();
        links
            .iter()
            .find(|held| !held.inbound && !held.dead.load(Relaxed) && *held.node.lock() == *id)
            .map(Arc::clone)
    }

    /// Every live link, for a broadcast.
    fn all(&self) -> Vec<Arc<Wire>> {
        let links = self.links.lock();
        yo_alloc::allow(|| {
            links
                .iter()
                .filter(|held| !held.dead.load(Relaxed))
                .map(Arc::clone)
                .collect()
        })
    }

    /// Drop every link to one node, which is what forgetting it means at this
    /// level.
    fn cut(&self, id: &str) {
        let mut links = self.links.lock();
        links.retain(|held| {
            let theirs = held.node.lock();
            if *theirs != *id {
                return true;
            }
            drop(theirs);
            held.kill();
            false
        });
    }

    /// Whether an id is unwelcome, and forget the entries that have aged out
    /// while looking.
    fn blacklisted(&self, id: &str, now: u64) -> bool {
        let mut list = self.blacklist.lock();
        list.retain(|(_, until)| *until > now);
        list.iter().any(|(held, _)| held == id)
    }

    /// Make an id unwelcome for the next minute.
    fn blacklist(&self, id: &str, now: u64) {
        let mut list = self.blacklist.lock();
        yo_alloc::allow(|| {
            list.retain(|(held, until)| held != id && *until > now);
            list.push((String::from(id), now + BLACKLIST_MS));
        });
    }
}

// ------------------------------------------------------------- starting it up

impl Server {
    /// Bring the bus up, which binds the bus port and starts the three kinds of
    /// thread that run it.
    ///
    /// Called once, from whoever is about to start serving clients. A server
    /// that is not a cluster node does nothing here and pays one relaxed load
    /// for asking.
    ///
    /// # Errors
    ///
    /// When the bus port cannot be bound. That is fatal on a real server and is
    /// fatal here too, because a cluster node nobody can gossip with is a node
    /// that will be voted out of its own slots in fifteen seconds.
    pub fn start_cluster_bus(self: &Arc<Server>) -> Result<(), String> {
        if !self.cluster_enabled() || self.cluster.bus.on.swap(true, Relaxed) {
            return Ok(());
        }
        let (port, id) = {
            let map = self.cluster.map.lock();
            (
                map.nodes[0].bus,
                yo_alloc::allow(|| map.nodes[0].id.clone()),
            )
        };
        let door = TcpListener::bind(("0.0.0.0", port))
            .map_err(|e| format!("cluster bus port {port} could not be bound: {e}"))?;
        let _ = id;
        let accepting = Arc::clone(self);
        spawn("yo-bus-accept", move || accept(&accepting, &door));
        let ticking = Arc::clone(self);
        spawn("yo-bus-cron", move || cron(&ticking));
        Ok(())
    }
}

/// Start a bus thread, whose name is what a stack trace will show.
fn spawn(name: &str, body: impl FnOnce() + Send + 'static) {
    yo_alloc::allow(|| {
        let _ = std::thread::Builder::new()
            .name(String::from(name))
            .spawn(body);
    });
}

/// Take connections on the bus port for as long as the process lives.
fn accept(server: &Arc<Server>, door: &TcpListener) {
    loop {
        let Ok((sock, _)) = door.accept() else {
            std::thread::sleep(Duration::from_millis(CRON_MS));
            continue;
        };
        let _ = sock.set_nodelay(true);
        let now = server.now_ms();
        let Some(wire) = Wire::new(sock, true, "", now) else {
            continue;
        };
        server.cluster.bus.add(&wire);
        let reading = Arc::clone(server);
        spawn("yo-bus-link", move || pump(&reading, &wire));
    }
}

/// Start the clock on a node this one could not open a link to.
///
/// Failure detection measures the time since a ping went out and a node with no
/// link has had no ping to send, so without this a node whose address stops
/// answering is retried for ever and never marked as failing. The reference
/// does the same thing in the same place and for the same reason: it says it
/// sent a ping now, which is true in the sense that one will go out the moment
/// there is anything to send it down.
fn unreachable(server: &Arc<Server>, id: &str) {
    let mut map = server.cluster.map.lock();
    if let Some(at) = map.find(id.as_bytes())
        && map.nodes[usize::from(at)].ping_sent == 0
    {
        map.nodes[usize::from(at)].ping_sent = server.now_ms();
    }
}

/// Open a link to a node and start reading it.
///
/// Returns whether it worked, which the cron uses to decide whether the node is
/// linked. The connect is blocking with a short timeout, which is why this is
/// never called with the node table locked.
fn dial(server: &Arc<Server>, id: &str, host: &str, bus: u16, meet: bool) -> bool {
    let Ok(addrs) = std::net::ToSocketAddrs::to_socket_addrs(&(host, bus)) else {
        unreachable(server, id);
        return false;
    };
    let mut sock = None;
    for addr in addrs {
        if let Ok(open) = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT) {
            sock = Some(open);
            break;
        }
    }
    let Some(sock) = sock else {
        unreachable(server, id);
        return false;
    };
    let _ = sock.set_nodelay(true);
    let now = server.now_ms();
    let Some(wire) = Wire::new(sock, false, id, now) else {
        unreachable(server, id);
        return false;
    };
    server.cluster.bus.add(&wire);
    // The first packet decides what this is. A node that was met by hand has to
    // hear MEET, because that is the only packet a node will accept from
    // somebody it has never heard of; everything else is a plain ping.
    let packet = {
        let mut map = server.cluster.map.lock();
        let at = map.find(id.as_bytes());
        if let Some(at) = at {
            map.nodes[usize::from(at)].linked = true;
            // The meet is sent once. If it does not land the handshake times out
            // and the node goes away, which is what should happen to an address
            // that has nothing at it.
            map.nodes[usize::from(at)].flags &= !FLAG_MEET;
            // A ping that was outstanding before the link went is left where it
            // was, so that the clock failure detection runs on is the one that
            // started when the node went quiet rather than one that is reset
            // every time the link is opened again. That is the reference's
            // `old_ping_sent` and without it a node that is up but unreachable
            // is never given up on.
            if !meet && map.nodes[usize::from(at)].ping_sent == 0 {
                map.nodes[usize::from(at)].ping_sent = now;
            }
        }
        ping(server, &map, if meet { T_MEET } else { T_PING }, at)
    };
    wire.send(&packet);
    let reading = Arc::clone(server);
    spawn("yo-bus-link", move || pump(&reading, &wire));
    true
}

/// Read packets off one link until it stops giving any.
fn pump(server: &Arc<Server>, wire: &Arc<Wire>) {
    let sock = {
        let Ok(held) = wire.sock.lock() else {
            return;
        };
        held.try_clone()
    };
    let Ok(mut sock) = sock else {
        wire.kill();
        return;
    };
    let mut head = [0u8; 8];
    let mut buf: Vec<u8> = yo_alloc::allow(|| Vec::with_capacity(HDR_LEN * 2));
    loop {
        if wire.dead.load(Relaxed) || sock.read_exact(&mut head).is_err() {
            break;
        }
        if &head[0..4] != SIG {
            break;
        }
        let total = u32::from_be_bytes([head[4], head[5], head[6], head[7]]) as usize;
        if !(16..=MAX_PACKET).contains(&total) {
            break;
        }
        yo_alloc::allow(|| {
            buf.clear();
            buf.resize(total, 0);
        });
        buf[..8].copy_from_slice(&head);
        if sock.read_exact(&mut buf[8..]).is_err() {
            break;
        }
        if !process(server, wire, &buf) {
            break;
        }
    }
    wire.kill();
    // A link that has gone is a link this node should open again, which it will
    // do on the next tick as long as the table does not still think it is there.
    let name = wire.named();
    if !name.is_empty() && !wire.inbound {
        let mut map = server.cluster.map.lock();
        if let Some(at) = map.find(name.as_bytes()) {
            map.nodes[usize::from(at)].linked = false;
        }
    }
}

// -------------------------------------------------------------- building packets

/// The fixed header, filled in from this node's view of itself.
fn header(server: &Server, map: &Map, kind: u16) -> Vec<u8> {
    let mut p = yo_alloc::allow(|| vec![0u8; HDR_LEN]);
    p[0..4].copy_from_slice(SIG);
    put16(&mut p, O_VER, PROTO_VER);
    put16(&mut p, O_TYPE, kind);
    put16(&mut p, O_PORT, map.nodes[0].port);
    put16(&mut p, O_CPORT, map.nodes[0].bus);
    put16(&mut p, O_PPORT, 0);
    put16(&mut p, O_FLAGS, map.nodes[0].flags);
    // A replica sends its master's slots and its master's epoch, flagged as a
    // replica so nobody mistakes it for the owner. That is how a replica can
    // answer a gossip round at all without having to say it knows nothing.
    let speaking = map.nodes[0]
        .master
        .filter(|_| map.nodes[0].flags & FLAG_SLAVE != 0)
        .map_or(0, usize::from);
    put64(&mut p, O_CURRENT_EPOCH, server.cluster.epoch.load(Relaxed));
    put64(&mut p, O_CONFIG_EPOCH, map.nodes[speaking].epoch);
    put64(&mut p, O_OFFSET, server.repl_offset());
    p[O_SENDER..O_SENDER + NAME_LEN].copy_from_slice(map.nodes[0].id.as_bytes());
    for slot in 0..SLOTS {
        if map.owner[slot] == Some(speaking as u16) {
            set_bit(&mut p[O_SLOTS..O_SLOTS + BITMAP_LEN], slot);
        }
    }
    if let Some(master) = map.nodes[0].master {
        let id = map.nodes[usize::from(master)].id.as_bytes();
        p[O_SLAVEOF..O_SLAVEOF + NAME_LEN].copy_from_slice(id);
    }
    // The address is left empty on purpose. A node does not know its own address
    // until somebody connects to it, and the receiver takes it off the socket,
    // which is the reference's auto discovery and is why a cluster can be built
    // out of nodes that were never told where they are.
    let _ = O_MYIP;
    p[O_STATE] = if server.cluster_up() {
        STATE_OK
    } else {
        STATE_FAIL
    };
    p[O_MFLAGS] = MF_EXT_DATA;
    p
}

/// Finish a packet by writing the length the header promises.
fn seal(p: &mut [u8]) {
    let total = p.len() as u32;
    p[O_TOTLEN..O_TOTLEN + 4].copy_from_slice(&total.to_be_bytes());
}

/// A `PING`, `PONG` or `MEET`, with its gossip and its extensions.
///
/// `to` is the node it is going to, which is left out of the gossip because
/// telling somebody about themselves is the one thing they already know.
fn ping(server: &Server, map: &Map, kind: u16, to: Option<u16>) -> Vec<u8> {
    let mut p = header(server, map, kind);
    let now = server.now_ms();
    // A tenth of the cluster with a floor of three, which is the reference's
    // rule and is what keeps the packet a fixed size as the cluster grows while
    // still passing news round in a few rounds.
    let mut want = (map.nodes.len() / 10).max(3);
    if map.nodes.len() >= 2 {
        want = want.min(map.nodes.len() - 2);
    } else {
        want = 0;
    }
    let mut count = 0u16;
    let mut sent: Vec<u16> = yo_alloc::allow(|| Vec::with_capacity(want + 4));
    let mut tries = want * 3;
    while sent.len() < want && tries > 0 {
        tries -= 1;
        let at = pick(map.nodes.len());
        if at == 0 || Some(at) == to || sent.contains(&at) {
            continue;
        }
        let node = &map.nodes[usize::from(at)];
        // A node in handshake has a made up name, a node with no address cannot
        // be reached by whoever hears about it, and a node that is not linked
        // and owns nothing is one this node is about to forget anyway.
        if node.flags & (FLAG_HANDSHAKE | FLAG_NOADDR) != 0 {
            continue;
        }
        if !node.linked && map.runs(at).is_empty() {
            continue;
        }
        sent.push(at);
    }
    // Everybody this node thinks is down goes on the end whether or not they
    // were drawn, because a failure report that arrives late is a failover that
    // happens late.
    for at in 1..map.nodes.len() as u16 {
        if map.nodes[usize::from(at)].flags & FLAG_PFAIL != 0 && !sent.contains(&at) {
            sent.push(at);
        }
    }
    for at in sent {
        let node = &map.nodes[usize::from(at)];
        let mut entry = [0u8; GOSSIP_LEN];
        entry[G_NAME..G_NAME + NAME_LEN].copy_from_slice(node.id.as_bytes());
        entry[G_PING..G_PING + 4].copy_from_slice(&((node.ping_sent / 1000) as u32).to_be_bytes());
        entry[G_PONG..G_PONG + 4].copy_from_slice(&((node.pong_recv / 1000) as u32).to_be_bytes());
        let host = node.host.as_bytes();
        let take = host.len().min(IP_LEN - 1);
        entry[G_IP..G_IP + take].copy_from_slice(&host[..take]);
        entry[G_PORT..G_PORT + 2].copy_from_slice(&node.port.to_be_bytes());
        entry[G_CPORT..G_CPORT + 2].copy_from_slice(&node.bus.to_be_bytes());
        entry[G_FLAGS..G_FLAGS + 2].copy_from_slice(&node.flags.to_be_bytes());
        yo_alloc::allow(|| p.extend_from_slice(&entry));
        count += 1;
    }
    let _ = now;
    put16(&mut p, O_COUNT, count);
    let mut exts = 0u16;
    // The shard id and the secret go on every ping. Both are forty bytes and
    // neither is optional, which is why a bare ping is never just a header.
    push_ext(&mut p, X_SHARDID, map.nodes[0].shard.as_bytes());
    exts += 1;
    let secret = server.cluster.bus.secret.lock();
    if secret.len() == NAME_LEN {
        push_ext(&mut p, X_SECRET, secret.as_bytes());
        exts += 1;
    }
    drop(secret);
    put16(&mut p, O_EXTENSIONS, exts);
    seal(&mut p);
    p
}

/// Put one extension on the end of a packet, padded to the eight byte boundary
/// the reference insists on.
fn push_ext(p: &mut Vec<u8>, kind: u16, data: &[u8]) {
    let len = 8 + data.len().div_ceil(8) * 8;
    yo_alloc::allow(|| {
        p.extend_from_slice(&(len as u32).to_be_bytes());
        p.extend_from_slice(&kind.to_be_bytes());
        p.extend_from_slice(&[0, 0]);
        p.extend_from_slice(data);
        p.resize(p.len() + (len - 8 - data.len()), 0);
    });
}

/// A `FAIL`, which says one node is gone and is believed on sight.
fn fail_packet(server: &Server, map: &Map, about: &str) -> Vec<u8> {
    let mut p = header(server, map, T_FAIL);
    yo_alloc::allow(|| p.extend_from_slice(about.as_bytes()));
    seal(&mut p);
    p
}

/// A `PUBLISH` or `PUBLISHSHARD`, which is how a message reaches a subscriber on
/// another node.
fn publish_packet(server: &Server, map: &Map, shard: bool, channel: &[u8], body: &[u8]) -> Vec<u8> {
    let kind = if shard { T_PUBLISHSHARD } else { T_PUBLISH };
    let mut p = header(server, map, kind);
    yo_alloc::allow(|| {
        p.extend_from_slice(&(channel.len() as u32).to_be_bytes());
        p.extend_from_slice(&(body.len() as u32).to_be_bytes());
        p.extend_from_slice(channel);
        p.extend_from_slice(body);
    });
    seal(&mut p);
    p
}

/// An `UPDATE`, which is what a node sends back to somebody claiming slots it
/// knows belong to a newer configuration.
fn update_packet(server: &Server, map: &Map, about: u16) -> Vec<u8> {
    let mut p = header(server, map, T_UPDATE);
    let node = &map.nodes[usize::from(about)];
    let mut body = [0u8; 8 + NAME_LEN + BITMAP_LEN];
    body[0..8].copy_from_slice(&node.epoch.to_be_bytes());
    body[8..8 + NAME_LEN].copy_from_slice(node.id.as_bytes());
    for slot in 0..SLOTS {
        if map.owner[slot] == Some(about) {
            set_bit(&mut body[8 + NAME_LEN..], slot);
        }
    }
    yo_alloc::allow(|| p.extend_from_slice(&body));
    seal(&mut p);
    p
}

// ------------------------------------------------------------ reading packets

/// What one packet asked this node to do, gathered up while the table was
/// locked and carried out after it was not.
///
/// The whole of packet processing runs with the node table locked, because it
/// reads and writes half of it and a reader that saw it half updated would route
/// a client wrongly. Nothing that blocks may happen under that lock, so a reply
/// is built while it is held and written after it is dropped, and that is what
/// this carries.
#[derive(Default)]
struct Todo {
    /// Packets to write back on the link the packet came in on.
    reply: Vec<Vec<u8>>,
    /// Packets to write to everybody.
    shout: Vec<Vec<u8>>,
    /// A message that arrived for the local subscribers.
    deliver: Option<(bool, Vec<u8>, Vec<u8>)>,
    /// Whether the table is worth writing out again.
    save: bool,
    /// Whether the slot coverage needs counting again once the lock is gone.
    recount: bool,
    /// Slots this node was serving and is not any more.
    ///
    /// Carried out of the lock rather than acted on inside it, because what
    /// happens to them is a migration finishing and a walk of the keyspace, and
    /// neither of those is a thing to do while every command on the server is
    /// waiting on the map.
    lost: Vec<u16>,
    /// Whether giving them away left this node following the node that took
    /// them, which is a node that keeps its keys rather than dropping them.
    demoted: bool,
    /// Whether the link should be closed rather than read again.
    close: bool,
}

/// Handle one packet, and say whether the link is still worth reading.
fn process(server: &Arc<Server>, wire: &Arc<Wire>, p: &[u8]) -> bool {
    if be16(p, O_VER) != PROTO_VER {
        return true;
    }
    let kind = be16(p, O_TYPE);
    let Some(explen) = expected(p, kind) else {
        return true;
    };
    if explen != p.len() {
        return true;
    }
    let now = server.now_ms();
    let mut todo = Todo::default();
    {
        let mut map = server.cluster.map.lock();
        digest(server, wire, p, kind, now, &mut map, &mut todo);
    }
    for packet in &todo.reply {
        wire.send(packet);
    }
    if !todo.shout.is_empty() {
        for link in server.cluster.bus.all() {
            for packet in &todo.shout {
                link.send(packet);
            }
        }
    }
    if let Some((shard, channel, body)) = todo.deliver
        && server.anyone_subscribed()
    {
        let kind = if shard { Kind::Shard } else { Kind::Channel };
        pubsub::deliver(server, kind, &channel, &body);
    }
    if todo.save {
        server.cluster.bus.dirty.store(true, Relaxed);
    }
    if todo.recount {
        server.recount_coverage();
    }
    // Last, and outside the lock. A slot that has gone to somebody else is where
    // a migration ends and where the keys behind it stop being this node's, and
    // both of those read the map they would otherwise be holding.
    server.asm_slots_moved(&todo.lost, todo.demoted);
    !todo.close
}

/// How long a packet of this type should be, or `None` for one that cannot be
/// worked out and is therefore not a packet.
///
/// The reference does this before it looks at anything else and refuses on a
/// mismatch rather than reading what it can. That is worth copying exactly: a
/// length field that disagrees with the body is the shape of every parser bug
/// there has ever been, and the cheapest answer is to not have a parser that
/// runs on one.
fn expected(p: &[u8], kind: u16) -> Option<usize> {
    match kind {
        T_PING | T_PONG | T_MEET => {
            let count = usize::from(be16(p, O_COUNT));
            let mut len = HDR_LEN.checked_add(count.checked_mul(GOSSIP_LEN)?)?;
            if p.get(O_MFLAGS).is_some_and(|f| f & MF_EXT_DATA != 0) {
                let mut left = be16(p, O_EXTENSIONS);
                let mut at = len;
                while left > 0 {
                    left -= 1;
                    let extlen = be32(p, at) as usize;
                    if extlen < 8 || !extlen.is_multiple_of(8) || p.len().checked_sub(len)? < extlen
                    {
                        return None;
                    }
                    len += extlen;
                    at += extlen;
                }
            }
            Some(len)
        }
        T_FAIL => Some(HDR_LEN + NAME_LEN),
        T_PUBLISH | T_PUBLISHSHARD => {
            let channel = be32(p, HDR_LEN) as usize;
            let body = be32(p, HDR_LEN + 4) as usize;
            HDR_LEN
                .checked_add(8)?
                .checked_add(channel)?
                .checked_add(body)
        }
        T_AUTH_REQUEST | T_AUTH_ACK | T_MFSTART => Some(HDR_LEN),
        T_UPDATE => Some(HDR_LEN + 8 + NAME_LEN + BITMAP_LEN),
        // A type this node does not handle is well formed by definition, which
        // is the reference's own answer and is what lets a newer node talk to an
        // older one without either of them dropping the conversation.
        _ => Some(p.len()),
    }
}

/// Everything one packet does to the node table.
#[allow(clippy::too_many_lines)]
fn digest(
    server: &Arc<Server>,
    wire: &Arc<Wire>,
    p: &[u8],
    kind: u16,
    now: u64,
    map: &mut Map,
    todo: &mut Todo,
) {
    let flags = be16(p, O_FLAGS);
    let claimed = node_id(p, O_SENDER);
    // Who sent it. An outbound link knows without looking, unless the node on
    // the other end is still in handshake and therefore still has the made up
    // name this node gave it.
    let linked = wire.named();
    let mut sender = None;
    if !linked.is_empty()
        && let Some(at) = map.find(linked.as_bytes())
        && map.nodes[usize::from(at)].flags & FLAG_HANDSHAKE == 0
    {
        sender = Some(at);
    }
    if sender.is_none()
        && let Some(id) = claimed.as_deref()
    {
        sender = map.find(id.as_bytes());
        if sender.is_some() && linked.is_empty() {
            wire.name(id);
        }
    }
    if let Some(at) = sender {
        let node = &mut map.nodes[usize::from(at)];
        if p.get(O_MFLAGS).is_some_and(|f| f & MF_EXT_DATA != 0) {
            node.flags |= super::FLAG_EXTENSIONS;
        }
        node.data_recv = now;
    }
    let sender_epoch = be64(p, O_CONFIG_EPOCH);
    if let Some(at) = sender
        && map.nodes[usize::from(at)].flags & FLAG_HANDSHAKE == 0
    {
        let theirs = be64(p, O_CURRENT_EPOCH);
        server.cluster.epoch.fetch_max(theirs, Relaxed);
        let node = &mut map.nodes[usize::from(at)];
        if sender_epoch > node.epoch {
            node.epoch = sender_epoch;
            todo.save = true;
        }
        node.offset = be64(p, O_OFFSET);
    }

    if kind == T_PING || kind == T_MEET {
        // A node learns its own address from the socket a MEET arrived on,
        // which is the only address in the cluster that is known to be reachable
        // from somewhere else. A plain ping is enough when there is no address
        // at all yet, because having a wrong one is better than having none and
        // a MEET will correct it.
        if (kind == T_MEET || map.nodes[0].host.is_empty()) && map.nodes[0].host != wire.local {
            yo_alloc::allow(|| map.nodes[0].host.clone_from(&wire.local));
            todo.save = true;
        }
        if sender.is_none() && kind == T_MEET {
            // Somebody was told to meet this node. Nothing about them is trusted
            // yet beyond where they are, so they go in as a handshake node with
            // a name this node made up, and the ping they answer with is what
            // replaces it.
            let host = {
                let announced = text(p, O_MYIP, IP_LEN);
                if announced.is_empty() {
                    yo_alloc::allow(|| wire.peer.clone())
                } else {
                    announced
                }
            };
            let id = yo_alloc::allow(|| String::from_utf8_lossy(&new_id()).into_owned());
            let node = yo_alloc::allow(|| {
                Node::new(
                    id,
                    host,
                    be16(p, O_PORT),
                    be16(p, O_CPORT),
                    FLAG_HANDSHAKE,
                    now,
                )
            });
            yo_alloc::allow(|| map.nodes.push(node));
            todo.save = true;
            // The gossip on a MEET from a stranger is taken anyway, because the
            // type of the packet is the trust: only a node that was told to meet
            // this one sends one, and it is worth knowing who else it has met.
            gossip(server, p, now, map, todo);
        }
        todo.reply.push(ping(server, map, T_PONG, sender));
    }

    match kind {
        T_PING | T_PONG | T_MEET => {}
        T_FAIL => {
            if sender.is_some()
                && let Some(id) = node_id(p, HDR_LEN)
                && let Some(at) = map.find(id.as_bytes())
                && map.nodes[usize::from(at)].flags & (FLAG_FAIL | FLAG_MYSELF) == 0
            {
                let node = &mut map.nodes[usize::from(at)];
                node.flags |= FLAG_FAIL;
                node.flags &= !FLAG_PFAIL;
                node.fail_time = now;
                todo.save = true;
            }
            return;
        }
        T_PUBLISH | T_PUBLISHSHARD => {
            if sender.is_none() {
                todo.close = true;
                return;
            }
            let channel = be32(p, HDR_LEN) as usize;
            let body = be32(p, HDR_LEN + 4) as usize;
            let at = HDR_LEN + 8;
            todo.deliver = yo_alloc::allow(|| {
                Some((
                    kind == T_PUBLISHSHARD,
                    p[at..at + channel].to_vec(),
                    p[at + channel..at + channel + body].to_vec(),
                ))
            });
            return;
        }
        T_UPDATE => {
            if sender.is_none() {
                todo.close = true;
                return;
            }
            let epoch = be64(p, HDR_LEN);
            let Some(id) = node_id(p, HDR_LEN + 8) else {
                return;
            };
            let Some(about) = map.find(id.as_bytes()) else {
                return;
            };
            if epoch <= map.nodes[usize::from(about)].epoch {
                return;
            }
            map.nodes[usize::from(about)].epoch = epoch;
            map.nodes[usize::from(about)].flags &= !FLAG_SLAVE;
            map.nodes[usize::from(about)].flags |= FLAG_MASTER;
            map.nodes[usize::from(about)].master = None;
            claim_slots(
                server,
                map,
                about,
                epoch,
                &p[HDR_LEN + 8 + NAME_LEN..],
                todo,
            );
            todo.save = true;
            return;
        }
        T_AUTH_REQUEST => {
            // A vote is only ever given to a node this one knows, because the
            // whole question is about a master this node has an opinion on.
            if let Some(at) = sender
                && vote_if_needed(server, map, at, p, now)
            {
                let mut packet = header(server, map, T_AUTH_ACK);
                seal(&mut packet);
                yo_alloc::allow(|| todo.reply.push(packet));
                todo.save = true;
            }
            return;
        }
        T_AUTH_ACK => {
            // Only a master serving slots has a vote to give, and only a vote
            // cast in the epoch this node is standing in counts. The second
            // check is what stops a late reply to a previous election being
            // counted towards this one.
            let vote = &server.cluster.vote;
            if let Some(at) = sender
                && map.nodes[usize::from(at)].is_master()
                && !map.runs(at).is_empty()
                && be64(p, O_CURRENT_EPOCH) >= vote.epoch.load(Relaxed)
            {
                vote.count.fetch_add(1, Relaxed);
            }
            return;
        }
        // Manual failover is not in yet, so the request to pause is dropped and
        // a real server's replica asking for one simply waits it out. Modules
        // are not a thing here at all.
        T_MFSTART | T_MODULE => return,
        _ => return,
    }

    // From here down is the config half of a PING, PONG or MEET, which is where
    // a cluster actually agrees on anything.
    if !wire.inbound {
        let held = map.find(linked.as_bytes());
        if let Some(at) = held
            && map.nodes[usize::from(at)].flags & FLAG_HANDSHAKE != 0
        {
            match sender {
                // This node had already met them under their real name, so the
                // handshake node is a duplicate and goes away.
                Some(known) => {
                    let host = yo_alloc::allow(|| wire.peer.clone());
                    let node = &mut map.nodes[usize::from(known)];
                    if node.host != host {
                        node.host = host;
                        node.port = be16(p, O_PORT);
                        node.bus = be16(p, O_CPORT);
                    }
                    map.forget(at);
                    todo.save = true;
                    todo.close = true;
                    return;
                }
                // The handshake worked. The made up name is replaced with the
                // real one and the node is an ordinary node from here on.
                None => {
                    let Some(id) = claimed.clone() else {
                        return;
                    };
                    wire.name(&id);
                    let node = &mut map.nodes[usize::from(at)];
                    yo_alloc::allow(|| node.id = id);
                    node.flags &= !(FLAG_HANDSHAKE | FLAG_MEET);
                    node.flags |= flags & (FLAG_MASTER | FLAG_SLAVE);
                    node.pong_recv = now;
                    node.ping_sent = 0;
                    sender = Some(at);
                    todo.save = true;
                }
            }
        } else if let Some(at) = held
            && Some(map.nodes[usize::from(at)].id.as_str()) != claimed.as_deref()
        {
            // Somebody else is answering on the address this node had written
            // down for a peer. The address is wrong rather than the peer, so the
            // peer keeps its identity and loses its address until gossip finds
            // it again.
            let node = &mut map.nodes[usize::from(at)];
            node.flags |= FLAG_NOADDR;
            node.host.clear();
            node.port = 0;
            node.bus = 0;
            node.linked = false;
            todo.save = true;
            todo.close = true;
            return;
        }
    }

    let Some(at) = sender else {
        return;
    };
    // The no failover flag is the sender's to set and everybody else's to
    // believe, because it is the sender saying whether it wants to be promoted.
    let node = &mut map.nodes[usize::from(at)];
    node.flags &= !FLAG_NOFAILOVER;
    node.flags |= flags & FLAG_NOFAILOVER;
    if kind == T_PING && !wire.inbound {
        let host = yo_alloc::allow(|| wire.peer.clone());
        if node.host != host {
            node.host = host;
            node.port = be16(p, O_PORT);
            node.bus = be16(p, O_CPORT);
            todo.save = true;
        }
    }
    if !wire.inbound && kind == T_PONG {
        node.pong_recv = now;
        node.ping_sent = 0;
        if node.flags & FLAG_PFAIL != 0 {
            node.flags &= !FLAG_PFAIL;
            todo.save = true;
        } else if node.flags & FLAG_FAIL != 0 {
            clear_failure(map, at, now);
            todo.save = true;
        }
    }

    // Master or replica, which has to settle before the slots are looked at
    // because a replica's slot claim is its master's and means something else.
    let follows = node_id(p, O_SLAVEOF);
    match follows {
        None => {
            if map.nodes[usize::from(at)].flags & FLAG_SLAVE != 0 {
                let node = &mut map.nodes[usize::from(at)];
                node.flags &= !FLAG_SLAVE;
                node.flags |= FLAG_MASTER;
                node.master = None;
                todo.save = true;
            }
        }
        Some(id) => {
            let master = map.find(id.as_bytes());
            if map.nodes[usize::from(at)].is_master() {
                // A master that has become a replica. When its new master is in
                // the same shard this is the tail of a failover, so the slots
                // move rather than vanish, and the new master is promoted here
                // to match. When it is not, the node has been moved to another
                // shard and its slots are simply not its any more.
                let same_shard = master.is_some_and(|m| {
                    map.nodes[usize::from(m)].shard == map.nodes[usize::from(at)].shard
                });
                if same_shard && sender_epoch >= map.nodes[usize::from(at)].epoch {
                    let m = master.expect("same shard means there is one");
                    for slot in 0..SLOTS {
                        if map.owner[slot] == Some(at) {
                            map.owner[slot] = Some(m);
                        }
                    }
                    let promoted = &mut map.nodes[usize::from(m)];
                    promoted.flags &= !FLAG_SLAVE;
                    promoted.flags |= FLAG_MASTER;
                    promoted.master = None;
                    promoted.epoch = sender_epoch;
                } else if !same_shard {
                    for slot in 0..SLOTS {
                        if map.owner[slot] == Some(at) {
                            map.owner[slot] = None;
                        }
                    }
                }
                let node = &mut map.nodes[usize::from(at)];
                node.flags &= !(FLAG_MASTER | FLAG_MIGRATE_TO);
                node.flags |= FLAG_SLAVE;
                todo.save = true;
            }
            if let Some(m) = master
                && map.nodes[usize::from(at)].master != Some(m)
                && m != at
            {
                map.nodes[usize::from(at)].master = Some(m);
                let shard = yo_alloc::allow(|| map.nodes[usize::from(m)].shard.clone());
                yo_alloc::allow(|| map.nodes[usize::from(at)].shard = shard);
                todo.save = true;
            }
        }
    }

    // The slots. Only a master's claim counts, and only when it differs from
    // what this node already had, which is one memcmp in front of a walk of
    // sixteen thousand slots and is worth having.
    let speaking = if map.nodes[usize::from(at)].is_master() {
        Some(at)
    } else {
        map.nodes[usize::from(at)].master
    };
    let claim = &p[O_SLOTS..O_SLOTS + BITMAP_LEN];
    let dirty = speaking
        .is_some_and(|m| (0..SLOTS).any(|slot| bit(claim, slot) != (map.owner[slot] == Some(m))));
    if dirty && map.nodes[usize::from(at)].is_master() {
        claim_slots(server, map, at, sender_epoch, claim, todo);
    }
    if dirty {
        // The other way round: the sender is claiming slots this node knows have
        // moved on to somebody with a newer epoch, so it is told about the first
        // one of them and works the rest out from there.
        for slot in 0..SLOTS {
            if !bit(claim, slot) {
                continue;
            }
            let Some(owner) = map.owner[slot] else {
                continue;
            };
            if owner == at {
                continue;
            }
            if map.nodes[usize::from(owner)].epoch > sender_epoch {
                todo.reply.push(update_packet(server, map, owner));
                break;
            }
        }
    }
    // Two masters with the same epoch cannot both be right, so the one with the
    // smaller id gives way by taking a new one. Deterministic and needs no
    // agreement, which is the only kind of tiebreak that works in a partition.
    if map.nodes[0].is_master()
        && map.nodes[usize::from(at)].is_master()
        && sender_epoch == map.nodes[0].epoch
        && map.nodes[0].id < map.nodes[usize::from(at)].id
    {
        let next = server.cluster.epoch.fetch_add(1, Relaxed) + 1;
        map.nodes[0].epoch = next;
        todo.save = true;
    }
    gossip(server, p, now, map, todo);
    extensions(server, p, now, map, at, todo);
}

/// The gossip section, which is what everybody else looks like from where the
/// sender is standing.
fn gossip(server: &Arc<Server>, p: &[u8], now: u64, map: &mut Map, todo: &mut Todo) {
    let count = usize::from(be16(p, O_COUNT));
    let sender_master = node_id(p, O_SENDER)
        .and_then(|id| map.find(id.as_bytes()))
        .is_none_or(|at| map.nodes[usize::from(at)].is_master());
    for entry in 0..count {
        let base = HDR_LEN + entry * GOSSIP_LEN;
        let Some(id) = node_id(p, base + G_NAME) else {
            continue;
        };
        let flags = be16(p, base + G_FLAGS);
        let host = text(p, base + G_IP, IP_LEN);
        let port = be16(p, base + G_PORT);
        let bus = be16(p, base + G_CPORT);
        if let Some(at) = map.find(id.as_bytes()) {
            if at == 0 {
                continue;
            }
            // A master saying somebody is down is a vote; anybody else saying it
            // is an opinion. Both are recorded, and only the votes are counted.
            if sender_master {
                let sender = node_id(p, O_SENDER).unwrap_or_default();
                if flags & (FLAG_FAIL | FLAG_PFAIL) != 0 {
                    report(map, at, &sender, now);
                    if mark_failing(map, at, now) {
                        let about = yo_alloc::allow(|| map.nodes[usize::from(at)].id.clone());
                        todo.shout.push(fail_packet(server, map, &about));
                        todo.save = true;
                    }
                } else {
                    unreport(map, at, &sender);
                }
            }
            let node = &mut map.nodes[usize::from(at)];
            if node.flags & (FLAG_FAIL | FLAG_PFAIL) == 0
                && node.ping_sent == 0
                && node.reports.is_empty()
            {
                // The sender heard from them more recently than this node did, so
                // take their word for when. The half second ceiling is the
                // reference's guard against a peer whose clock is ahead.
                let heard = u64::from(be32(p, base + G_PONG)) * 1000;
                if heard > node.pong_recv && heard <= now + 500 {
                    node.pong_recv = heard;
                }
            } else if node.down()
                && flags & (FLAG_FAIL | FLAG_PFAIL | FLAG_NOADDR | FLAG_HANDSHAKE) == 0
                && !host.is_empty()
                && (node.host != host || node.port != port)
            {
                // Down here and up there, at an address this node does not have.
                // The address is the thing that is wrong, so it is replaced and
                // the next tick tries again.
                yo_alloc::allow(|| node.host = host);
                node.port = port;
                node.bus = bus;
                node.linked = false;
                node.flags &= !FLAG_NOADDR;
                todo.save = true;
            }
            continue;
        }
        // Somebody new. Taken only from a sender this node already knows, and
        // only when the id is not one that was just forgotten on purpose.
        //
        // It goes in under the id and the flags the gossip carried rather than
        // as a handshake, which is the difference between learning about a node
        // and meeting one. A handshake exists to find out an id that is not
        // known yet, and here it is: the sender said it. Adding it as a
        // handshake instead would mean opening a link to an address whose owner
        // might have changed, which is how a node ends up in somebody else's
        // cluster.
        if flags & FLAG_NOADDR != 0 {
            continue;
        }
        if server.cluster.bus.blacklisted(&id, now) {
            continue;
        }
        let flags = flags & !(FLAG_MYSELF | FLAG_HANDSHAKE | FLAG_MEET);
        let node = yo_alloc::allow(|| Node::new(id, host, port, bus, flags, now));
        yo_alloc::allow(|| map.nodes.push(node));
        todo.save = true;
    }
}

/// The extensions on the end of a ping.
fn extensions(server: &Arc<Server>, p: &[u8], now: u64, map: &mut Map, at: u16, todo: &mut Todo) {
    if !p.get(O_MFLAGS).is_some_and(|f| f & MF_EXT_DATA != 0) {
        return;
    }
    let mut left = be16(p, O_EXTENSIONS);
    let mut base = HDR_LEN + usize::from(be16(p, O_COUNT)) * GOSSIP_LEN;
    while left > 0 && base + 8 <= p.len() {
        left -= 1;
        let extlen = be32(p, base) as usize;
        if extlen < 8 || base + extlen > p.len() {
            return;
        }
        let kind = be16(p, base + 4);
        let data = &p[base + 8..base + extlen];
        match kind {
            X_SHARDID => {
                let id = String::from_utf8_lossy(&data[..NAME_LEN.min(data.len())]);
                if id.len() == NAME_LEN && map.nodes[usize::from(at)].shard != id {
                    yo_alloc::allow(|| map.nodes[usize::from(at)].shard = id.into_owned());
                    todo.save = true;
                }
            }
            X_SECRET => {
                // The whole cluster ends up on the smallest secret anybody
                // started with, which needs no leader to decide and settles in
                // one gossip round.
                let theirs = String::from_utf8_lossy(&data[..NAME_LEN.min(data.len())]);
                let mut mine = server.cluster.bus.secret.lock();
                if theirs.len() == NAME_LEN && **mine > *theirs {
                    yo_alloc::allow(|| *mine = theirs.into_owned());
                }
            }
            X_FORGOTTEN => {
                // Somebody has been forgotten on purpose. Forget them here too,
                // and refuse to learn them again for as long as the sender says,
                // which is what stops the rest of the cluster putting them back.
                if data.len() < NAME_LEN + 8 {
                    return;
                }
                let Some(id) = node_id(data, 0) else {
                    return;
                };
                let ttl = be64(data, NAME_LEN);
                if map.nodes[0].id == id
                    || map.nodes[0].master.map(usize::from)
                        == map.find(id.as_bytes()).map(usize::from)
                {
                    base += extlen;
                    continue;
                }
                server.cluster.bus.blacklist(&id, now + ttl);
                if let Some(gone) = map.find(id.as_bytes())
                    && gone != 0
                {
                    server.cluster.bus.cut(&id);
                    map.forget(gone);
                    todo.save = true;
                }
            }
            // The hostname and the human readable name are carried for the
            // benefit of an operator reading `CLUSTER NODES` on a real server,
            // and nothing here routes on either of them.
            X_HOSTNAME | X_HUMAN_NAME => {}
            _ => {}
        }
        base += extlen;
    }
}

/// Record that somebody thinks a node is down.
fn report(map: &mut Map, at: u16, from: &str, now: u64) {
    if from.is_empty() {
        return;
    }
    let node = &mut map.nodes[usize::from(at)];
    if let Some(held) = node.reports.iter_mut().find(|(who, _)| who == from) {
        held.1 = now;
        return;
    }
    yo_alloc::allow(|| node.reports.push((String::from(from), now)));
}

/// Take one back, which is somebody saying they can reach a node after all.
fn unreport(map: &mut Map, at: u16, from: &str) {
    map.nodes[usize::from(at)]
        .reports
        .retain(|(who, _)| who != from);
}

/// Whether enough of the cluster agrees that a node is gone, and flag it if so.
///
/// The quorum is a majority of the masters that are serving slots, which is the
/// same electorate a failover vote is counted against, and this node counts
/// itself when it is one of them. Reports older than twice the node timeout are
/// dropped first, so a node that flickered a long time ago does not help
/// condemn one that is flickering now.
fn mark_failing(map: &mut Map, at: u16, now: u64) -> bool {
    let cutoff = now.saturating_sub(NODE_TIMEOUT_MS * 2);
    map.nodes[usize::from(at)]
        .reports
        .retain(|(_, when)| *when > cutoff);
    if map.nodes[usize::from(at)].flags & (FLAG_PFAIL | FLAG_FAIL) == 0 {
        return false;
    }
    if map.nodes[usize::from(at)].flags & FLAG_FAIL != 0 {
        return false;
    }
    let needed = map.voters() / 2 + 1;
    let mut votes = map.nodes[usize::from(at)].reports.len();
    if map.nodes[0].is_master() && !map.runs(0).is_empty() {
        votes += 1;
    }
    if votes < needed {
        return false;
    }
    let node = &mut map.nodes[usize::from(at)];
    node.flags &= !FLAG_PFAIL;
    node.flags |= FLAG_FAIL;
    node.fail_time = now;
    true
}

/// Undo a failure verdict, which is only allowed for a replica that came back or
/// for a master that came back still owning nothing.
///
/// A master that comes back owning slots is not cleared, because the cluster has
/// probably promoted its replica by now and clearing it would leave two nodes
/// claiming the same slots. It stays failed until it hears about the promotion
/// and stands down on its own.
fn clear_failure(map: &mut Map, at: u16, now: u64) {
    let node = &map.nodes[usize::from(at)];
    let replica = !node.is_master();
    let empty = map.runs(at).is_empty();
    let stale = now.saturating_sub(node.fail_time) > NODE_TIMEOUT_MS * 2;
    if replica || empty || stale {
        let node = &mut map.nodes[usize::from(at)];
        node.flags &= !FLAG_FAIL;
        node.fail_time = 0;
    }
}

/// Take a master's word for which slots are its.
///
/// This is the one place a slot changes hands without an operator asking, so it
/// is careful about which claims it believes. A slot nobody owns is taken on
/// sight. A slot owned by somebody with an older epoch changes hands, because a
/// higher epoch is the cluster's way of saying this configuration is newer. A
/// slot this node is importing is left alone, because an import in progress is
/// an operator's decision and outranks gossip. And a slot this node loses while
/// still holding keys for it is remembered, because those keys now belong to
/// somebody else and keeping them would mean two nodes answering for the same
/// data.
fn claim_slots(
    server: &Arc<Server>,
    map: &mut Map,
    owner: u16,
    epoch: u64,
    claim: &[u8],
    todo: &mut Todo,
) {
    // Whose slots decide where this node belongs afterwards. A master answers
    // for its own and a replica answers for its master's, because a replica of a
    // master that has just lost everything is a replica of nothing and is how a
    // second replica of a failed master finds the one that replaced it.
    let ours = if map.nodes[0].is_master() {
        Some(0)
    } else {
        map.nodes[0].master
    };
    let mut lost = 0usize;
    for slot in 0..SLOTS {
        if !bit(claim, slot) {
            continue;
        }
        if map.owner[slot] == Some(owner) || map.importing[slot].is_some() {
            continue;
        }
        let held = map.owner[slot];
        let newer = held.is_none_or(|at| map.nodes[usize::from(at)].epoch <= epoch);
        if !newer {
            continue;
        }
        if held == ours {
            lost += 1;
        }
        if held == Some(0) {
            yo_alloc::allow(|| todo.lost.push(slot as u16));
        }
        map.owner[slot] = Some(owner);
        map.migrating[slot] = None;
        todo.save = true;
    }
    // A master that has just given away its last slot is not a master any more.
    // Following the node that took them is what a real server does and is what
    // makes a failed master come back as a replica of whoever replaced it.
    if lost > 0 && ours.is_some_and(|at| map.runs(at).is_empty()) && owner != 0 {
        todo.demoted = true;
        map.nodes[0].flags &= !FLAG_MASTER;
        map.nodes[0].flags |= FLAG_SLAVE;
        map.nodes[0].master = Some(owner);
        let shard = yo_alloc::allow(|| map.nodes[usize::from(owner)].shard.clone());
        yo_alloc::allow(|| map.nodes[0].shard = shard);
        let (host, port) = {
            let node = &map.nodes[usize::from(owner)];
            (yo_alloc::allow(|| node.host.clone()), node.port)
        };
        let following = Arc::clone(server);
        spawn("yo-bus-follow", move || {
            following.follow_master(&host, port);
        });
    }
    // The coverage count is not touched here on purpose. It reads the same map
    // this runs under, and the lock is not reentrant, so the caller does it
    // once the lock is gone.
    todo.recount = true;
}

// ------------------------------------------------------------ the failover vote

/// Answer a replica asking to be promoted, if every condition holds.
///
/// This is the reference's `clusterSendFailoverAuthIfNeeded` and the order of
/// the checks is its order, because the order is the safety. A master gets one
/// vote per epoch and gives it to the first replica that asks, and everything
/// in front of that is about making sure the question was a fair one to ask.
///
/// Nothing is sent back when a condition fails. There is no such thing as a no
/// vote on this protocol: a replica counts the yeses it got and gives up when
/// the election times out, which means a master that has crashed and a master
/// that disapproves look the same from where the replica is standing. That is
/// deliberate, since the alternative is a reply that a replica could be made to
/// wait for.
fn vote_if_needed(server: &Arc<Server>, map: &mut Map, at: u16, p: &[u8], now: u64) -> bool {
    // Only a master serving a slot has a vote, because the electorate is the
    // masters that serve slots and nothing else would make the quorum add up.
    if map.nodes[0].flags & FLAG_SLAVE != 0 || map.runs(0).is_empty() {
        return false;
    }
    // The asking node's epoch cannot be behind this node's. It cannot really be
    // ahead either, since reading the packet has already pulled this node's
    // epoch up to it, so what this catches is a request that was already stale
    // when it arrived.
    let epoch = server.cluster.epoch.load(Relaxed);
    if be64(p, O_CURRENT_EPOCH) < epoch {
        return false;
    }
    // One vote per epoch, and it has already gone.
    if server.cluster.vote.given.load(Relaxed) == epoch {
        return false;
    }
    // It has to be a replica, this node has to know whose, and that master has
    // to be one the cluster has given up on. A manual failover is the exception
    // and says so in the packet, because there the master is up and is in on it.
    let asking = &map.nodes[usize::from(at)];
    let Some(master) = asking.master.filter(|_| asking.flags & FLAG_SLAVE != 0) else {
        return false;
    };
    let forced = p.get(O_MFLAGS).is_some_and(|f| f & MF_FORCEACK != 0);
    if map.nodes[usize::from(master)].flags & FLAG_FAIL == 0 && !forced {
        return false;
    }
    // Not twice about the same master inside two node timeouts. A second replica
    // of the same master asking straight after the first is either the first one
    // having failed to win or two of them racing, and in both cases the answer
    // that keeps the shard with one master is to wait and see how the first one
    // went.
    if now.saturating_sub(map.nodes[usize::from(master)].voted_time) < NODE_TIMEOUT_MS * 2 {
        return false;
    }
    // The slots it is claiming have to be ones it would be claiming under an
    // epoch at least as new as whoever is serving them here. A replica asking to
    // take over slots that have already moved somewhere newer is a replica that
    // has been out of touch, and voting for it would undo the move.
    let claimed = be64(p, O_CONFIG_EPOCH);
    for slot in 0..SLOTS {
        if !bit(&p[O_SLOTS..O_SLOTS + BITMAP_LEN], slot) {
            continue;
        }
        if map.owner[slot].is_some_and(|held| map.nodes[usize::from(held)].epoch > claimed) {
            return false;
        }
    }
    server.cluster.vote.given.store(epoch, Relaxed);
    map.nodes[usize::from(master)].voted_time = now;
    true
}

/// How many replicas of the same master hold more data than this node does.
///
/// The reference's `clusterGetSlaveRank`, and the offsets it compares are the
/// ones the other replicas put in their own packets rather than anything asked
/// for, so a rank is always a little out of date and that is fine: it decides a
/// delay and not an outcome.
fn rank_of(map: &Map, master: u16, mine: u64) -> u64 {
    map.nodes
        .iter()
        .skip(1)
        .filter(|node| {
            node.master == Some(master)
                && node.flags & FLAG_SLAVE != 0
                && node.flags & FLAG_NOFAILOVER == 0
                && node.offset > mine
        })
        .count() as u64
}

/// What the cron has to do about the election, worked out under the lock.
enum Step {
    /// Nothing, which is the answer on nearly every tick of nearly every node.
    Idle,
    /// The delay has just been worked out, so tell the other replicas of this
    /// master how far this node has got in case it changes their minds.
    Announce,
    /// Ask every node for a vote.
    Ask(Vec<u8>),
    /// The quorum is in and the slots are this node's now.
    Won,
}

/// Work out what this tick of the election does, under the map lock.
///
/// The reference's `clusterHandleSlaveFailover`. Every branch of it returns and
/// waits for the next tick rather than carrying on, which is what makes the
/// whole thing readable: the state is in one place and each tick moves it at
/// most one step.
fn decide(server: &Arc<Server>, map: &mut Map, now: u64) -> Step {
    let vote = &server.cluster.vote;
    // A replica of a master the cluster has given up on, which is willing to be
    // promoted, and whose master was serving something worth taking over.
    let me = &map.nodes[0];
    if me.flags & (FLAG_SLAVE | FLAG_NOFAILOVER) != FLAG_SLAVE {
        return Step::Idle;
    }
    let Some(master) = me.master else {
        return Step::Idle;
    };
    if map.nodes[usize::from(master)].flags & FLAG_FAIL == 0 || map.runs(master).is_empty() {
        return Step::Idle;
    }
    // How far out of touch with the master this node was before it died. The
    // node timeout comes off because that much silence is what made it dead in
    // the first place and is not the replica's fault.
    if server.master_silence(now).saturating_sub(NODE_TIMEOUT_MS) > STALE_DATA_MS {
        return Step::Idle;
    }
    let since = now as i64 - vote.at.load(Relaxed) as i64;
    if since > (ELECTION_TIMEOUT_MS * 2) as i64 {
        // Nothing running, or the last one is long enough ago to try again. The
        // delay is a fixed part, a random part and a part per replica that holds
        // more data than this one.
        let rank = rank_of(map, master, server.repl_offset());
        vote.at.store(
            now + ELECTION_DELAY_MS
                + u64::from(pick(ELECTION_DELAY_MS as usize))
                + rank * RANK_DELAY_MS,
            Relaxed,
        );
        vote.rank.store(rank, Relaxed);
        vote.count.store(0, Relaxed);
        vote.sent.store(false, Relaxed);
        return Step::Announce;
    }
    if !vote.sent.load(Relaxed) {
        // Another replica may have said something since the delay was worked
        // out. Falling further down the order pushes the delay back, and
        // climbing it does not pull it forward, which is the reference's rule
        // and is what keeps two replicas swapping places from both asking at
        // once.
        let rank = rank_of(map, master, server.repl_offset());
        let was = vote.rank.load(Relaxed);
        if rank > was {
            vote.at.fetch_add((rank - was) * RANK_DELAY_MS, Relaxed);
            vote.rank.store(rank, Relaxed);
        }
        if now < vote.at.load(Relaxed) {
            return Step::Idle;
        }
    }
    if since > ELECTION_TIMEOUT_MS as i64 {
        // Too late to be worth counting. The retry window above is what starts
        // the next one.
        return Step::Idle;
    }
    if !vote.sent.load(Relaxed) {
        let epoch = server.cluster.epoch.fetch_add(1, Relaxed) + 1;
        vote.epoch.store(epoch, Relaxed);
        vote.sent.store(true, Relaxed);
        let mut packet = header(server, map, T_AUTH_REQUEST);
        seal(&mut packet);
        return Step::Ask(packet);
    }
    if vote.count.load(Relaxed) < (map.size() / 2 + 1) as u64 {
        return Step::Idle;
    }
    let epoch = vote.epoch.load(Relaxed);
    if map.nodes[0].epoch < epoch {
        map.nodes[0].epoch = epoch;
    }
    // Everything the old master was serving is this node's now, and saying so
    // under an epoch nobody else has used is what makes every other node believe
    // it over whatever it had written down.
    map.nodes[0].flags &= !FLAG_SLAVE;
    map.nodes[0].flags |= FLAG_MASTER;
    map.nodes[0].master = None;
    for slot in 0..SLOTS {
        if map.owner[slot] == Some(master) {
            map.owner[slot] = Some(0);
        }
    }
    Step::Won
}

/// Stand for election when the master is gone, and take over on winning.
fn failover(server: &Arc<Server>, now: u64) {
    let step = {
        let mut map = server.cluster.map.lock();
        decide(server, &mut map, now)
    };
    match step {
        Step::Idle => {}
        // The reference sends this to the other replicas of the same master and
        // this sends it to everybody, which is a packet more per node on a
        // cluster that has just lost one and is the same answer.
        Step::Announce => server.cluster_broadcast_pong(),
        Step::Ask(packet) => {
            for link in server.cluster.bus.all() {
                link.send(&packet);
            }
            server.cluster.bus.dirty.store(true, Relaxed);
        }
        Step::Won => {
            server.stop_following();
            server.recount_coverage();
            server.cluster.bus.dirty.store(true, Relaxed);
            let _ = super::save(server);
            server.cluster_broadcast_pong();
        }
    }
}

// ---------------------------------------------------------------- the cron

/// The bus's own clock, which is where everything that is not a reply happens.
fn cron(server: &Arc<Server>) {
    let mut tick = 0u64;
    loop {
        std::thread::sleep(Duration::from_millis(CRON_MS));
        tick += 1;
        let now = server.now_ms();
        // Who needs a link, who needs a ping, and who has been quiet too long.
        // All decided under the lock and all carried out after it, because a
        // connect takes half a second in the worst case and the routing path
        // reads this table on every command.
        let mut dial_list: Vec<(String, String, u16, bool)> = Vec::new();
        let mut ping_list: Vec<(String, Vec<u8>)> = Vec::new();
        let mut shout: Vec<Vec<u8>> = Vec::new();
        let mut follow: Option<(String, u16)> = None;
        let mut save = false;
        {
            let mut map = server.cluster.map.lock();
            let count = map.nodes.len();
            for at in 1..count as u16 {
                let node = &map.nodes[usize::from(at)];
                // A handshake that never completed is a node that is not there.
                if node.flags & FLAG_HANDSHAKE != 0
                    && now.saturating_sub(node.data_recv) > NODE_TIMEOUT_MS.max(1000)
                {
                    map.forget(at);
                    save = true;
                    break;
                }
                if node.flags & FLAG_NOADDR != 0 || node.host.is_empty() {
                    continue;
                }
                if !node.linked {
                    // Only a node this one was told to meet hears MEET, because
                    // MEET is the packet that says trust me, I am not in another
                    // cluster. Everything else is a plain ping, including a node
                    // in handshake that was met the other way round.
                    let meet = node.flags & FLAG_MEET != 0;
                    yo_alloc::allow(|| {
                        dial_list.push((node.id.clone(), node.host.clone(), node.bus, meet));
                    });
                    continue;
                }
                // A ping every second, or sooner for whoever this node has heard
                // from least recently out of five, which is the reference's way
                // of getting round a cluster in far fewer than N rounds.
                let quiet = now.saturating_sub(node.pong_recv);
                let due = node.ping_sent == 0 && quiet > PING_MS;
                if due || (tick.is_multiple_of(10) && oldest_of_five(&map, now) == Some(at)) {
                    let packet = ping(server, &map, T_PING, Some(at));
                    let id = yo_alloc::allow(|| map.nodes[usize::from(at)].id.clone());
                    map.nodes[usize::from(at)].ping_sent = now;
                    ping_list.push((id, packet));
                }
            }
            // Anybody who has not answered in time is possibly failed, which is
            // this node's opinion and becomes the cluster's once enough masters
            // share it.
            for at in 1..map.nodes.len() as u16 {
                let node = &map.nodes[usize::from(at)];
                if node.flags & (FLAG_HANDSHAKE | FLAG_FAIL | FLAG_PFAIL) != 0 {
                    continue;
                }
                let waiting = if node.ping_sent == 0 {
                    0
                } else {
                    now.saturating_sub(node.ping_sent)
                };
                let quiet = now.saturating_sub(node.data_recv);
                if waiting.min(quiet) > NODE_TIMEOUT_MS {
                    map.nodes[usize::from(at)].flags |= FLAG_PFAIL;
                    save = true;
                }
                if mark_failing(&mut map, at, now) {
                    let about = yo_alloc::allow(|| map.nodes[usize::from(at)].id.clone());
                    shout.push(fail_packet(server, &map, &about));
                    save = true;
                }
            }
            // A node whose table says it is a replica but which is not following
            // anybody starts following. That is how a replica comes back after a
            // restart, since the node table is on disk and the replication link
            // is not, and it is also how one that was told to replicate a node
            // it had no address for yet gets going once the address turns up.
            if map.nodes[0].flags & FLAG_SLAVE != 0
                && !server.following()
                && let Some(master) = map.nodes[0].master
                && let Some(node) = map.nodes.get(usize::from(master))
                && node.flags & FLAG_NOADDR == 0
                && !node.host.is_empty()
            {
                follow = yo_alloc::allow(|| Some((node.host.clone(), node.port)));
            }
        }
        // Called here and not on a thread of its own, because it starts the
        // replica thread and returns, and doing it here is what makes the guard
        // above true again before the next tick rather than a tick later.
        if let Some((host, port)) = follow {
            server.follow_master(&host, port);
        }
        for (id, host, bus, meet) in dial_list {
            if !dial(server, &id, &host, bus, meet) {
                continue;
            }
            let mut map = server.cluster.map.lock();
            if let Some(at) = map.find(id.as_bytes()) {
                map.nodes[usize::from(at)].linked = true;
            }
        }
        for (id, packet) in ping_list {
            if let Some(link) = server.cluster.bus.outbound(&id) {
                link.send(&packet);
            } else {
                // The link went between deciding to ping and sending it, so the
                // node is dialled again on the next tick. The ping is left
                // standing rather than taken back, because it is what failure
                // detection measures and a node that cannot be sent a ping is
                // exactly the node that should be running out of time.
                let mut map = server.cluster.map.lock();
                if let Some(at) = map.find(id.as_bytes()) {
                    map.nodes[usize::from(at)].linked = false;
                }
            }
        }
        if !shout.is_empty() {
            for link in server.cluster.bus.all() {
                for packet in &shout {
                    link.send(packet);
                }
            }
        }
        if save {
            server.cluster.bus.dirty.store(true, Relaxed);
        }
        server.recount_coverage();
        failover(server, now);
        server.asm_cron();
        server.asm_relax();
        // The file is written at most ten times a second and only when something
        // moved, which keeps a busy cluster from writing it on every packet.
        if tick.is_multiple_of(10) && server.cluster.bus.dirty.swap(false, Relaxed) {
            let _ = super::save(server);
        }
    }
}

/// Whoever this node has heard from least recently, out of five drawn at random.
///
/// Five and not all of them because the point is to bias the ping order towards
/// the nodes that need one without walking the table every tenth of a second.
fn oldest_of_five(map: &Map, now: u64) -> Option<u16> {
    if map.nodes.len() < 2 {
        return None;
    }
    let mut best: Option<(u16, u64)> = None;
    for _ in 0..5 {
        let at = pick(map.nodes.len());
        if at == 0 {
            continue;
        }
        let node = &map.nodes[usize::from(at)];
        if node.ping_sent != 0 || node.flags & (FLAG_HANDSHAKE | FLAG_NOADDR) != 0 {
            continue;
        }
        let quiet = now.saturating_sub(node.pong_recv);
        if best.is_none_or(|(_, held)| quiet > held) {
            best = Some((at, quiet));
        }
    }
    best.map(|(at, _)| at)
}

// ------------------------------------------------------- what the commands do

impl Server {
    /// Start a handshake with a node at an address, which is `CLUSTER MEET`.
    ///
    /// Nothing is known about them yet, not even their id, so they go into the
    /// table under a name this node made up and the next tick opens a link and
    /// sends them a MEET. Their answer carries their real id, which replaces the
    /// made up one. That is why meeting a node twice is harmless and why meeting
    /// one that is already known is a no-op rather than a duplicate.
    pub(super) fn cluster_meet(&self, host: &str, port: u16, bus: u16) {
        let now = self.now_ms();
        let mut map = self.cluster.map.lock();
        let known = map
            .nodes
            .iter()
            .any(|node| node.host == host && node.port == port);
        if known {
            return;
        }
        let id = yo_alloc::allow(|| String::from_utf8_lossy(&new_id()).into_owned());
        let node = yo_alloc::allow(|| {
            Node::new(
                id,
                yo_alloc::allow(|| String::from(host)),
                port,
                bus,
                FLAG_HANDSHAKE | FLAG_MEET,
                now,
            )
        });
        yo_alloc::allow(|| map.nodes.push(node));
    }

    /// Whether an id was forgotten on purpose recently, which is what makes
    /// forgetting the same node twice answer OK rather than complain.
    pub(super) fn cluster_blacklisted(&self, id: &str) -> bool {
        self.cluster.bus.blacklisted(id, self.now_ms())
    }

    /// Drop a node and tell everybody else to, which is `CLUSTER FORGET`.
    ///
    /// The telling is the part that matters. Dropping a node on its own would
    /// last until the next gossip packet from anybody who still had it, so the
    /// id is refused for a minute here and the refusal is passed round in the
    /// extension that carries it.
    pub(super) fn cluster_forget(&self, at: u16) {
        let now = self.now_ms();
        let id = {
            let mut map = self.cluster.map.lock();
            let id = yo_alloc::allow(|| map.nodes[usize::from(at)].id.clone());
            map.forget(at);
            id
        };
        self.cluster.bus.blacklist(&id, now);
        self.cluster.bus.cut(&id);
        let packet = {
            let map = self.cluster.map.lock();
            let mut p = ping(self, &map, T_PING, None);
            // The forgotten node rides on the next ping rather than in a packet
            // of its own, which is the reference's design: there is no reliable
            // delivery on the bus, so a fact that has to reach everybody is
            // repeated rather than sent once.
            let mut body = [0u8; NAME_LEN + 8];
            body[..NAME_LEN].copy_from_slice(id.as_bytes());
            body[NAME_LEN..].copy_from_slice(&BLACKLIST_MS.to_be_bytes());
            push_ext(&mut p, X_FORGOTTEN, &body);
            let exts = be16(&p, O_EXTENSIONS) + 1;
            put16(&mut p, O_EXTENSIONS, exts);
            seal(&mut p);
            p
        };
        for link in self.cluster.bus.all() {
            link.send(&packet);
        }
        self.cluster.bus.dirty.store(true, Relaxed);
    }

    /// Become a replica of another node, which is `CLUSTER REPLICATE`.
    pub(super) fn cluster_replicate(self: &Arc<Server>, at: u16) {
        let (host, port) = {
            let mut map = self.cluster.map.lock();
            // Whatever slots this node was holding are not its any more, which
            // is the reference's rule and is why the command refuses when there
            // are keys in them.
            for slot in 0..SLOTS {
                if map.owner[slot] == Some(0) {
                    map.owner[slot] = None;
                }
            }
            map.nodes[0].flags &= !(FLAG_MASTER | FLAG_MIGRATE_TO);
            map.nodes[0].flags |= FLAG_SLAVE;
            map.nodes[0].master = Some(at);
            let shard = yo_alloc::allow(|| map.nodes[usize::from(at)].shard.clone());
            yo_alloc::allow(|| map.nodes[0].shard = shard);
            let node = &map.nodes[usize::from(at)];
            (yo_alloc::allow(|| node.host.clone()), node.port)
        };
        self.recount_coverage();
        self.cluster.bus.dirty.store(true, Relaxed);
        if !host.is_empty() {
            self.follow_master(&host, port);
        }
    }

    /// Send a published message to every other node, which is what makes a
    /// subscriber on one node hear a publish on another.
    ///
    /// Every node gets a copy of an ordinary publish, because an ordinary
    /// subscription is not tied to a slot and could be anywhere. A shard publish
    /// only needs to reach the shard that owns the slot, but the reference
    /// broadcasts it too and lets the far side drop it, so this does the same.
    pub(crate) fn cluster_publish(&self, shard: bool, channel: &[u8], body: &[u8]) {
        if !self.cluster_enabled() || !self.cluster.bus.on.load(Relaxed) {
            return;
        }
        let packet = {
            let map = self.cluster.map.lock();
            publish_packet(self, &map, shard, channel, body)
        };
        for link in self.cluster.bus.all() {
            link.send(&packet);
        }
    }

    /// Send everybody a `PONG` right now rather than waiting for the cron.
    ///
    /// A `PONG` nobody asked for is how the reference tells the cluster about a
    /// configuration change it has just made to itself, and the only thing that
    /// makes it a `PONG` rather than a `PING` is that nobody is expected to
    /// answer it. What the far side actually reads is the header, which carries
    /// this node's slots and its config epoch, so one packet is the whole of the
    /// announcement.
    ///
    /// It matters after a slot import because the epoch has just gone up and the
    /// rest of the cluster is still pointing clients at the node the slot came
    /// from. Waiting the ordinary ping interval would leave every client that
    /// asked the wrong node being redirected to a node that no longer owns it.
    pub(super) fn cluster_broadcast_pong(&self) {
        if !self.cluster_enabled() || !self.cluster.bus.on.load(Relaxed) {
            return;
        }
        let packet = {
            let map = self.cluster.map.lock();
            ping(self, &map, T_PONG, None)
        };
        for link in self.cluster.bus.all() {
            link.send(&packet);
        }
    }

    /// `CLUSTER LINKS`, which is one map per open link in each direction.
    pub(super) fn cluster_links(&self, out: &mut Out) {
        let links = self.cluster.bus.all();
        let at = out.len();
        let mut n = 0;
        for link in links {
            let node = link.named();
            // The reference only lists links it has associated with a node, and
            // an inbound link stays unassociated until a packet on it says who
            // is sending, so a connection that has said nothing is not a link
            // yet as far as this report is concerned.
            if node.is_empty() {
                continue;
            }
            out.map(6);
            out.bulk(b"direction");
            out.bulk(if link.inbound {
                b"from".as_slice()
            } else {
                b"to".as_slice()
            });
            out.bulk(b"node");
            out.bulk(node.as_bytes());
            out.bulk(b"create-time");
            out.int(link.created as i64);
            out.bulk(b"events");
            out.bulk(b"r");
            out.bulk(b"send-buffer-allocated");
            out.int(link.sent.load(Relaxed) as i64);
            out.bulk(b"send-buffer-used");
            out.int(0);
            n += 1;
        }
        out.close_array(at, n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bitmap_round_trips_through_the_reference_bit_order() {
        let mut bitmap = [0u8; BITMAP_LEN];
        for slot in [0usize, 1, 7, 8, 1234, 5061, 12182, SLOTS - 1] {
            set_bit(&mut bitmap, slot);
        }
        for slot in 0..SLOTS {
            let want = matches!(slot, 0 | 1 | 7 | 8 | 1234 | 5061 | 12182) || slot == SLOTS - 1;
            assert_eq!(bit(&bitmap, slot), want, "slot {slot}");
        }
        // Least significant bit first inside each byte, which is the one detail
        // a reader is likely to get backwards and the one the reference will not
        // forgive.
        assert_eq!(bitmap[0], 0b1000_0011);
        assert_eq!(bitmap[1], 0b0000_0001);
    }

    #[test]
    fn an_extension_is_padded_to_the_eight_byte_boundary() {
        let mut p = Vec::new();
        push_ext(&mut p, X_SHARDID, &[b'a'; 40]);
        assert_eq!(p.len(), 48);
        assert_eq!(be32(&p, 0), 48);
        assert_eq!(be16(&p, 4), X_SHARDID);
        // A hostname is not a multiple of eight and has to be rounded up, which
        // is what `getAlignedPingExtSize` does.
        let mut q = Vec::new();
        push_ext(&mut q, X_HOSTNAME, b"node1.example\0");
        assert_eq!(q.len(), 8 + 16);
        assert_eq!(be32(&q, 0), 24);
    }

    #[test]
    fn a_packet_of_the_wrong_length_is_refused() {
        let mut p = vec![0u8; HDR_LEN];
        p[0..4].copy_from_slice(SIG);
        put16(&mut p, O_VER, PROTO_VER);
        put16(&mut p, O_TYPE, T_PING);
        put16(&mut p, O_COUNT, 0);
        p[O_MFLAGS] = 0;
        seal(&mut p);
        assert_eq!(expected(&p, T_PING), Some(HDR_LEN));
        // One gossip entry claimed and none present is the shape every parser
        // bug has, so it has to come out as a length that does not match.
        put16(&mut p, O_COUNT, 1);
        assert_eq!(expected(&p, T_PING), Some(HDR_LEN + GOSSIP_LEN));
        assert_ne!(expected(&p, T_PING), Some(p.len()));
    }

    /// A node table with a master that is about to fail, one replica of it and
    /// two other masters, which is the smallest cluster an election means
    /// anything on.
    ///
    /// Node 0 is this server and is the replica standing for election. Node 1 is
    /// its master and owns the first third of the slots, and nodes 2 and 3 own
    /// the rest, so the quorum is two and this node is not part of it.
    /// A wall clock reading, since every span in an election is measured
    /// against one and a node that has just started is not two node timeouts
    /// away from the epoch.
    const T0: u64 = 1_700_000_000_000;

    fn shard() -> Arc<Server> {
        let mut server = Server::new();
        server.enable_cluster("", 7000);
        let server = Arc::new(server);
        let master = server.cluster_pretend_node("1".repeat(40).as_str(), "10.0.0.1", 7001);
        let other = server.cluster_pretend_node("2".repeat(40).as_str(), "10.0.0.2", 7002);
        let third = server.cluster_pretend_node("3".repeat(40).as_str(), "10.0.0.3", 7003);
        server.cluster_pretend_follower(master);
        {
            let mut map = server.cluster.map.lock();
            for slot in 0..SLOTS {
                map.owner[slot] = Some(match slot {
                    0..=5460 => master,
                    5461..=10922 => other,
                    _ => third,
                });
            }
        }
        server.recount_coverage();
        server
    }

    /// An `AUTH_REQUEST` as a replica of node 1 would send it.
    fn asking(epoch: u64, config: u64, forced: bool) -> Vec<u8> {
        let mut p = vec![0u8; HDR_LEN];
        put64(&mut p, O_CURRENT_EPOCH, epoch);
        put64(&mut p, O_CONFIG_EPOCH, config);
        for slot in 0..=5460 {
            set_bit(&mut p[O_SLOTS..O_SLOTS + BITMAP_LEN], slot);
        }
        p[O_MFLAGS] = if forced { MF_FORCEACK } else { 0 };
        p
    }

    /// A node that cannot be dialled has to start running out of time, or the
    /// election it should be losing never happens.
    ///
    /// Failure detection measures the time since a ping went out, and a node
    /// with no link never gets one sent, so the address that stops answering is
    /// the one case where the clock has to be started by hand.
    #[test]
    fn a_node_that_cannot_be_reached_is_treated_as_pinged() {
        let server = shard();
        let id = "1".repeat(40);
        assert_eq!(server.cluster.map.lock().nodes[1].ping_sent, 0);
        unreachable(&server, &id);
        let first = server.cluster.map.lock().nodes[1].ping_sent;
        assert!(first > 0, "the clock is running");
        // And the next failed dial leaves it where it is, because the span that
        // matters is the one since the node went quiet and not the one since
        // the last attempt to reach it.
        unreachable(&server, &id);
        assert_eq!(server.cluster.map.lock().nodes[1].ping_sent, first);
        // A name nobody knows is not an error, it is a node that has been
        // forgotten between the decision to dial it and the attempt.
        unreachable(&server, &"9".repeat(40));
    }

    /// Every reason a master has for not answering a replica that wants its
    /// master's slots, in the order a real server checks them.
    ///
    /// The order is the safety rather than a detail of the implementation, so
    /// each one is arranged to be the only thing wrong.
    #[test]
    fn a_vote_is_given_once_and_only_when_every_condition_holds() {
        let server = shard();
        // Stand this node up as a master serving the last third, and put a
        // replica of node 1 in the table for it to be asked about.
        {
            let mut map = server.cluster.map.lock();
            map.nodes[0].flags &= !FLAG_SLAVE;
            map.nodes[0].flags |= FLAG_MASTER;
            map.nodes[0].master = None;
            for slot in 10923..SLOTS {
                map.owner[slot] = Some(0);
            }
            let mut replica = Node::new(
                "4".repeat(40),
                "10.0.0.4".to_owned(),
                7004,
                7004 + 10000,
                FLAG_SLAVE,
                0,
            );
            replica.master = Some(1);
            map.nodes.push(replica);
        }
        let at = 4u16;
        let ask = |server: &Arc<Server>, p: &[u8], now: u64| {
            let mut map = server.cluster.map.lock();
            vote_if_needed(server, &mut map, at, p, now)
        };
        // A replica bumps the epoch before it asks and reading the packet pulls
        // this node's up to it, so by the time the question is put the two
        // agree and neither is nought.
        server.cluster.epoch.store(1, Relaxed);

        // The master it wants to replace is up and nobody said this was a manual
        // failover, so there is nothing to vote about.
        assert!(!ask(&server, &asking(1, 0, false), T0));
        // A manual failover says so in the packet, and that is the whole
        // difference from where the voter is standing.
        assert!(ask(&server, &asking(1, 0, true), T0));
        // One vote per epoch, and it has gone.
        assert!(!ask(&server, &asking(1, 0, true), T0));

        // Give up on node 1 and move the epoch on, which is what a real replica
        // would have done before asking again.
        server.cluster.epoch.store(2, Relaxed);
        {
            let mut map = server.cluster.map.lock();
            map.nodes[1].flags |= FLAG_FAIL;
        }
        // Not twice about the same master inside two node timeouts, however
        // dead it is, because the first replica may still be winning.
        assert!(!ask(&server, &asking(2, 0, false), T0 + NODE_TIMEOUT_MS));
        let later = T0 + NODE_TIMEOUT_MS * 2 + 1;
        // A request that was stale before it arrived.
        assert!(!ask(&server, &asking(1, 0, false), later));
        // A node that is not a replica has no master to replace.
        {
            let mut map = server.cluster.map.lock();
            map.nodes[4].flags &= !FLAG_SLAVE;
        }
        assert!(!ask(&server, &asking(2, 0, false), later));
        {
            let mut map = server.cluster.map.lock();
            map.nodes[4].flags |= FLAG_SLAVE;
        }
        // Slots that have already moved somewhere newer than the epoch the
        // replica would claim them under. Voting for that would undo the move.
        {
            let mut map = server.cluster.map.lock();
            map.owner[3000] = Some(2);
            map.nodes[2].epoch = 7;
        }
        assert!(!ask(&server, &asking(2, 6, false), later));
        // The same request under an epoch that is not behind is fine.
        assert!(ask(&server, &asking(2, 7, false), later));
        assert_eq!(server.cluster.vote.given(), 2);

        // And a node with no slots of its own is not part of the electorate at
        // all, however much it knows about the shard.
        server.cluster.epoch.store(3, Relaxed);
        {
            let mut map = server.cluster.map.lock();
            for slot in 0..SLOTS {
                if map.owner[slot] == Some(0) {
                    map.owner[slot] = Some(3);
                }
            }
        }
        assert!(!ask(
            &server,
            &asking(3, 7, false),
            later + NODE_TIMEOUT_MS * 3
        ));
    }

    /// A replica of a master the cluster has given up on waits its turn, asks
    /// once, and takes the slots over when the quorum is in.
    #[test]
    fn an_election_waits_then_asks_then_wins() {
        let server = shard();
        let step = |server: &Arc<Server>, now: u64| {
            let mut map = server.cluster.map.lock();
            decide(server, &mut map, now)
        };

        // The master is up, so there is no election to hold.
        assert!(matches!(step(&server, T0), Step::Idle));
        {
            let mut map = server.cluster.map.lock();
            map.nodes[1].flags |= FLAG_FAIL;
        }
        // The first tick after that works out when this node may ask and tells
        // the other replicas how far it has got.
        assert!(matches!(step(&server, T0), Step::Announce));
        let at = server.cluster.vote.at.load(Relaxed);
        assert!(
            (T0 + 500..=T0 + 1000).contains(&at),
            "half a second plus up to half a second more, got {at}"
        );
        // Until then there is nothing to do.
        assert!(matches!(step(&server, at - 1), Step::Idle));

        // A replica that turns out to hold more data than this one pushes the
        // turn back by a second, and one that falls behind does not pull it
        // forward again.
        {
            let mut map = server.cluster.map.lock();
            let mut ahead = Node::new(
                "5".repeat(40),
                "10.0.0.5".to_owned(),
                7005,
                7005 + 10000,
                FLAG_SLAVE,
                0,
            );
            ahead.master = Some(1);
            ahead.offset = 900;
            map.nodes.push(ahead);
        }
        assert!(matches!(step(&server, at), Step::Idle));
        assert_eq!(server.cluster.vote.at.load(Relaxed), at + RANK_DELAY_MS);
        {
            let mut map = server.cluster.map.lock();
            map.nodes[4].offset = 0;
        }
        assert!(matches!(step(&server, at), Step::Idle));
        assert_eq!(server.cluster.vote.at.load(Relaxed), at + RANK_DELAY_MS);

        // Then it asks, once, under an epoch of its own.
        let now = at + RANK_DELAY_MS;
        let Step::Ask(packet) = step(&server, now) else {
            panic!("the turn has come");
        };
        assert_eq!(be64(&packet, O_CURRENT_EPOCH), 1);
        assert_eq!(server.cluster.vote.epoch.load(Relaxed), 1);
        // The slots in the request are the master's, because those are the ones
        // the answer is about.
        assert!(bit(&packet[O_SLOTS..O_SLOTS + BITMAP_LEN], 5460));
        assert!(!bit(&packet[O_SLOTS..O_SLOTS + BITMAP_LEN], 5461));
        assert!(matches!(step(&server, now), Step::Idle), "asked already");

        // One vote out of the three masters is not a quorum.
        server.cluster.vote.count.store(1, Relaxed);
        assert!(matches!(step(&server, now), Step::Idle));
        server.cluster.vote.count.store(2, Relaxed);
        assert!(matches!(step(&server, now), Step::Won));

        let map = server.cluster.map.lock();
        assert!(map.nodes[0].is_master());
        assert_eq!(map.nodes[0].master, None);
        assert_eq!(map.nodes[0].epoch, 1, "the epoch it stood under");
        assert_eq!(map.owner[0], Some(0));
        assert_eq!(map.owner[5460], Some(0));
        assert_eq!(map.owner[5461], Some(2), "somebody else's slots are theirs");
    }

    /// An election this node has no business holding is not held.
    #[test]
    fn a_replica_that_should_not_stand_does_not() {
        let refused = |now: u64, set: fn(&Arc<Server>)| {
            let server = shard();
            {
                let mut map = server.cluster.map.lock();
                map.nodes[1].flags |= FLAG_FAIL;
            }
            set(&server);
            let mut map = server.cluster.map.lock();
            matches!(decide(&server, &mut map, now), Step::Idle)
        };
        // A master does not stand for election, whatever has happened to
        // anybody else.
        assert!(refused(T0, |server| {
            let mut map = server.cluster.map.lock();
            map.nodes[0].flags &= !FLAG_SLAVE;
            map.nodes[0].flags |= FLAG_MASTER;
        }));
        // A replica that has been told not to.
        assert!(refused(T0, |server| {
            let mut map = server.cluster.map.lock();
            map.nodes[0].flags |= FLAG_NOFAILOVER;
        }));
        // A master that was serving nothing has nothing to take over.
        assert!(refused(T0, |server| {
            let mut map = server.cluster.map.lock();
            for slot in 0..=5460 {
                map.owner[slot] = Some(2);
            }
        }));
        // Data too old to be worth promoting. The link went down long enough
        // ago that this node has missed more than a failover is allowed to
        // lose, on top of the silence that made the master dead.
        let stale = T0 + NODE_TIMEOUT_MS + STALE_DATA_MS + 1;
        assert!(refused(stale, |server| {
            server.pretend_following("10.0.0.1", 7001, false);
            server.pretend_master_down_at(T0);
        }));
        // A second less and it stands, which is what makes the line above a
        // test of the bound rather than of the setup.
        assert!(!refused(stale - 1000, |server| {
            server.pretend_following("10.0.0.1", 7001, false);
            server.pretend_master_down_at(T0);
        }));
    }

    #[test]
    fn a_node_id_has_to_be_forty_hex_characters() {
        let mut p = vec![0u8; NAME_LEN * 3];
        assert_eq!(node_id(&p, 0), None, "all zeros is the reference's no node");
        p[0..NAME_LEN].copy_from_slice(&[b'a'; NAME_LEN]);
        assert_eq!(node_id(&p, 0).as_deref(), Some("a".repeat(40).as_str()));
        p[5] = b'z';
        assert_eq!(node_id(&p, 0), None);
        assert_eq!(node_id(&p, NAME_LEN * 3), None, "off the end is not an id");
    }
}
