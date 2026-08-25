//! Positioned reads and writes, and the one call that actually costs anything.
//!
//! Thin on purpose. Everything here is a wrapper over what the standard library
//! already offers, and the only reason it exists is that the standard library
//! spells positioned I/O differently on unix and on Windows, and that
//! `fdatasync` is not in the standard library at all.
//!
//! **No seeking.** Every read and write carries its own offset, so nothing in
//! this crate holds a file cursor and two shards writing to the same file
//! cannot race over one. That is what makes a `File` shareable here without a
//! lock.
//!
//! **`fdatasync` where there is one.** A log page write changes data and the
//! file length, but not the file's metadata in any way a reader depends on,
//! and `fdatasync` skips the inode write that `fsync` does not. On a device
//! where a sync is milliseconds that is a real difference, and on macOS, where
//! the honest call is `F_FULLFSYNC`, it is the difference between a durable
//! commit and a commit that is durable if the drive feels like it.

use std::fs::File;
use std::io;

/// Reads exactly `buf.len()` bytes from `offset`, or says how many there were.
///
/// A short read is not an error here. It is the ordinary answer at the end of
/// a file, and the caller knows whether a short page means damage or means the
/// file simply stops there.
///
/// # Errors
///
/// Whatever the operating system returns, except that an interrupted read is
/// retried rather than surfaced.
pub fn read_at(f: &File, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
    let mut done = 0;
    while done < buf.len() {
        let n = read_once(f, offset + done as u64, &mut buf[done..])?;
        if n == 0 {
            break;
        }
        done += n;
    }
    Ok(done)
}

/// Writes all of `buf` at `offset`.
///
/// # Errors
///
/// Whatever the operating system returns. A partial write is retried from
/// where it stopped, so a success means every byte went.
pub fn write_at(f: &File, offset: u64, buf: &[u8]) -> io::Result<()> {
    let mut done = 0;
    while done < buf.len() {
        let n = write_once(f, offset + done as u64, &buf[done..])?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "the file accepted no bytes",
            ));
        }
        done += n;
    }
    Ok(())
}

#[cfg(unix)]
fn read_once(f: &File, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;
    loop {
        match f.read_at(buf, offset) {
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            other => return other,
        }
    }
}

#[cfg(unix)]
fn write_once(f: &File, offset: u64, buf: &[u8]) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;
    loop {
        match f.write_at(buf, offset) {
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            other => return other,
        }
    }
}

#[cfg(windows)]
fn read_once(f: &File, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;
    // `seek_read` moves the file pointer as a side effect, which is exactly
    // what this module promises not to depend on. It is fine because nothing
    // here ever reads the pointer, and Windows has no positioned read that
    // leaves it alone.
    loop {
        match f.seek_read(buf, offset) {
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            other => return other,
        }
    }
}

#[cfg(windows)]
fn write_once(f: &File, offset: u64, buf: &[u8]) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;
    loop {
        match f.seek_write(buf, offset) {
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            other => return other,
        }
    }
}

/// Makes everything written so far durable.
///
/// The strongest thing the platform offers, not the fastest thing that
/// returns. On macOS that is `F_FULLFSYNC`, which asks the drive to flush its
/// own write cache; plain `fsync` there returns once the data has reached the
/// drive, which is not the same as it surviving a power cut. A durability mode
/// that lies is worse than not having one.
///
/// # Errors
///
/// Whatever the operating system returns.
pub fn sync_data(f: &File) -> io::Result<()> {
    strongest(f)
}

/// Makes the data and the metadata durable, which is what a rename or a length
/// change needs on top of its bytes.
///
/// # Errors
///
/// Whatever the operating system returns.
pub fn sync_all(f: &File) -> io::Result<()> {
    #[cfg(target_os = "macos")]
    {
        // Already the strongest thing there is on this platform, and it covers
        // the metadata too.
        strongest(f)
    }
    #[cfg(not(target_os = "macos"))]
    {
        f.sync_all()
    }
}

#[cfg(target_os = "macos")]
fn strongest(f: &File) -> io::Result<()> {
    // SAFETY: `fcntl` with `F_FULLFSYNC` takes no argument beyond the command
    // and reads nothing through a pointer. The descriptor is valid for as long
    // as `f` is borrowed.
    let rc = unsafe {
        use std::os::unix::io::AsRawFd;
        libc::fcntl(f.as_raw_fd(), libc::F_FULLFSYNC)
    };
    if rc == -1 {
        // Not every filesystem implements it. A network mount or a tmpfs
        // returns ENOTTY or EINVAL, and there the plain sync is the strongest
        // thing available rather than a shortcut.
        let e = io::Error::last_os_error();
        return match e.raw_os_error() {
            Some(libc::ENOTTY | libc::EINVAL | libc::ENOTSUP) => f.sync_data(),
            _ => Err(e),
        };
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn strongest(f: &File) -> io::Result<()> {
    // `sync_data` is `fdatasync` where there is one, which skips the inode
    // write that a log page append does not need.
    f.sync_data()
}

/// Grows the file to `len`, without writing anything.
///
/// Growing rather than allocating. The extended range reads as zeroes and
/// costs no blocks until something is written into it, which is what keeps a
/// database that has been given room to grow from occupying that room on disk
/// before it needs it.
///
/// # Errors
///
/// Whatever the operating system returns.
pub fn grow_to(f: &File, len: u64) -> io::Result<()> {
    if f.metadata()?.len() >= len {
        return Ok(());
    }
    f.set_len(len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn temp(name: &str) -> (std::path::PathBuf, File) {
        let mut p = std::env::temp_dir();
        p.push(format!("yo-io-{name}-{}.bin", std::process::id()));
        let _ = std::fs::remove_file(&p);
        let f = File::options()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&p)
            .unwrap();
        (p, f)
    }

    #[test]
    fn a_write_lands_where_it_was_addressed_and_reads_back() {
        let (p, f) = temp("roundtrip");
        write_at(&f, 4096, b"hello").unwrap();
        let mut buf = [0u8; 5];
        assert_eq!(read_at(&f, 4096, &mut buf).unwrap(), 5);
        assert_eq!(&buf, b"hello");
        std::fs::remove_file(p).unwrap();
    }

    #[test]
    fn the_gap_a_positioned_write_leaves_reads_as_zeroes() {
        let (p, f) = temp("hole");
        write_at(&f, 100, b"x").unwrap();
        let mut buf = [0xffu8; 100];
        assert_eq!(read_at(&f, 0, &mut buf).unwrap(), 100);
        assert!(buf.iter().all(|b| *b == 0));
        std::fs::remove_file(p).unwrap();
    }

    #[test]
    fn a_read_past_the_end_is_short_rather_than_an_error() {
        let (p, mut f) = temp("short");
        f.write_all(b"abc").unwrap();
        let mut buf = [0u8; 16];
        assert_eq!(read_at(&f, 0, &mut buf).unwrap(), 3);
        assert_eq!(read_at(&f, 99, &mut buf).unwrap(), 0);
        std::fs::remove_file(p).unwrap();
    }

    #[test]
    fn growing_extends_with_zeroes_and_never_shrinks() {
        let (p, f) = temp("grow");
        grow_to(&f, 8192).unwrap();
        assert_eq!(f.metadata().unwrap().len(), 8192);
        grow_to(&f, 4096).unwrap();
        assert_eq!(f.metadata().unwrap().len(), 8192, "growing does not shrink");
        std::fs::remove_file(p).unwrap();
    }

    #[test]
    fn a_sync_on_a_plain_file_succeeds() {
        let (p, f) = temp("sync");
        write_at(&f, 0, b"durable").unwrap();
        sync_data(&f).unwrap();
        sync_all(&f).unwrap();
        std::fs::remove_file(p).unwrap();
    }
}
