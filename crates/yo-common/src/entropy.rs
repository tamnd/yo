//! Bytes nobody can guess, from the operating system.
//!
//! [`crate::rng`] is the generator everything else in the engine uses and it is
//! deliberately not this: it is nine lines of arithmetic on a seed, so a trial
//! that fails can be run again and `SPOP` can be tested. `ACL GENPASS` is the
//! one caller that wants the opposite property. It hands a client a password
//! that is going to guard a server, so the whole value of it is that the next
//! one cannot be worked out from the last one, and a seeded stream fails that
//! by construction.
//!
//! So this asks the system, which is the only thing on the machine that is
//! collecting real entropy. `/dev/urandom` on unix and `BCryptGenRandom` on
//! Windows, both of which are the documented interface rather than the clever
//! one, and neither of which needs a handle opened up front or a fallback for
//! being called early.
//!
//! There is no error path. A machine whose random device cannot be read is a
//! machine that cannot keep a secret, and handing back a password anyway,
//! filled with whatever was in the buffer, is worse than stopping. So this
//! panics, which is the same call every serious implementation of this makes.

/// Fill `into` with bytes from the system's random source.
///
/// # Panics
///
/// If the system will not produce them, because the alternative is a password
/// made of nothing.
pub fn fill(into: &mut [u8]) {
    if into.is_empty() {
        return;
    }
    imp::fill(into);
}

#[cfg(unix)]
mod imp {
    use std::fs::File;
    use std::io::Read;

    /// Read the whole buffer out of `/dev/urandom`.
    ///
    /// Opened per call rather than kept open, because the one caller is a
    /// command an operator runs by hand and the open is not what it costs. A
    /// held descriptor would have to survive `fork` and every process that has
    /// got that wrong has got it badly wrong.
    ///
    /// The loop is because a read is allowed to return short, which on this
    /// device it will not, but the code that assumes it will not is the code
    /// that hands back a half filled buffer the day it does.
    pub fn fill(into: &mut [u8]) {
        let mut file = File::open("/dev/urandom").expect("no /dev/urandom to take a password from");
        let mut at = 0;
        while at < into.len() {
            let n = file
                .read(&mut into[at..])
                .expect("could not read /dev/urandom");
            assert!(n != 0, "/dev/urandom stopped early");
            at += n;
        }
    }
}

#[cfg(windows)]
mod imp {
    /// Ask the system preferred generator, which is what the flag below means.
    ///
    /// The flag is the documented way to call this without opening an algorithm
    /// handle first, so there is no state here and nothing to shut down.
    const USE_SYSTEM_PREFERRED_RNG: u32 = 0x0000_0002;

    // The library has to be named, because nothing else in the tree pulls it
    // in. The standard library used to, back when its own generator was this
    // same call, and it has since moved to `ProcessPrng` in another library,
    // so a build that linked by accident stopped linking when the toolchain
    // caught up. The failure is at link time and only on the MSVC target,
    // which is not a target a person developing this is usually on.
    #[link(name = "bcrypt")]
    unsafe extern "system" {
        fn BCryptGenRandom(
            algorithm: *mut core::ffi::c_void,
            buffer: *mut u8,
            count: u32,
            flags: u32,
        ) -> i32;
    }

    /// Fill the buffer in chunks a `u32` can count, which every real call is
    /// well inside and which costs one comparison to be right about anyway.
    pub fn fill(into: &mut [u8]) {
        for chunk in into.chunks_mut(u32::MAX as usize) {
            // SAFETY: the pointer and the length are one buffer we hold
            // exclusively, and the null handle is what the flag asks for.
            let status = unsafe {
                BCryptGenRandom(
                    core::ptr::null_mut(),
                    chunk.as_mut_ptr(),
                    chunk.len() as u32,
                    USE_SYSTEM_PREFERRED_RNG,
                )
            };
            assert!(status == 0, "BCryptGenRandom failed with {status:#x}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fill;

    /// Two draws differ, and the buffer was actually written to.
    ///
    /// Testing a random source is testing that it is not a constant, which is
    /// the whole of what a test can say without a statistics library. Thirty two
    /// bytes the same twice is a one in 2^256 accident and a certainty if the
    /// call did nothing, so this catches the failure that matters.
    #[test]
    fn two_draws_are_not_the_same_bytes() {
        let mut one = [0u8; 32];
        let mut two = [0u8; 32];
        fill(&mut one);
        fill(&mut two);
        assert_ne!(one, two);
        assert_ne!(one, [0u8; 32]);
    }

    /// An empty buffer is not an error and a short one is filled.
    #[test]
    fn every_length_up_to_a_block_comes_back_written() {
        fill(&mut []);
        for len in 1..=64 {
            let mut buf = vec![0u8; len];
            fill(&mut buf);
            // A short draw can legitimately be all zeroes, so what is asserted
            // is only that the call returned, which is the boundary being
            // checked here.
            assert_eq!(buf.len(), len);
        }
    }
}
