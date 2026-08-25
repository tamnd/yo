//! Positioned reads, and nothing else.
//!
//! The reader never writes, never maps and never seeks. A file it is pointed at
//! may be one somebody is still running a database on, and a `seek` plus `read`
//! pair on a shared descriptor is a race with anyone else holding it. Positioned
//! reads take the offset as an argument and leave no file position to race over.

use std::fs::File;
use std::io;

/// Fills `buf` from `off`, or says how far it got.
///
/// Short is not an error here. Reading past the end of the file is how the
/// reader finds out where the end is, and a caller that needs the whole buffer
/// checks the count.
///
/// # Errors
///
/// Whatever the operating system says, once interruptions have been retried.
pub fn read_at(f: &File, off: u64, buf: &mut [u8]) -> io::Result<usize> {
    let mut done = 0;
    while done < buf.len() {
        match read_once(f, off + done as u64, &mut buf[done..]) {
            Ok(0) => break,
            Ok(n) => done += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(done)
}

/// Fills `buf` from `off` and insists on all of it.
///
/// # Errors
///
/// [`io::ErrorKind::UnexpectedEof`] if the file ends first.
pub fn read_exact_at(f: &File, off: u64, buf: &mut [u8]) -> io::Result<()> {
    let n = read_at(f, off, buf)?;
    if n != buf.len() {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!(
                "wanted {} bytes at offset {off} and the file had {n}",
                buf.len()
            ),
        ));
    }
    Ok(())
}

#[cfg(unix)]
fn read_once(f: &File, off: u64, buf: &mut [u8]) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;
    f.read_at(buf, off)
}

#[cfg(windows)]
fn read_once(f: &File, off: u64, buf: &mut [u8]) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;
    f.seek_read(buf, off)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    struct Tmp(std::path::PathBuf);

    impl Tmp {
        fn new(name: &str, bytes: &[u8]) -> Tmp {
            let mut p = std::env::temp_dir();
            p.push(format!("yo-reader-io-{name}-{}.bin", std::process::id()));
            let mut f = File::create(&p).unwrap();
            f.write_all(bytes).unwrap();
            Tmp(p)
        }

        fn open(&self) -> File {
            File::open(&self.0).unwrap()
        }
    }

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn a_read_lands_where_it_was_told_to() {
        let t = Tmp::new("at", b"0123456789");
        let f = t.open();
        let mut buf = [0u8; 4];
        assert_eq!(read_at(&f, 3, &mut buf).unwrap(), 4);
        assert_eq!(&buf, b"3456");
    }

    #[test]
    fn reading_past_the_end_is_short_and_not_an_error() {
        let t = Tmp::new("short", b"abc");
        let f = t.open();
        let mut buf = [0u8; 16];
        assert_eq!(read_at(&f, 1, &mut buf).unwrap(), 2);
        assert_eq!(read_at(&f, 99, &mut buf).unwrap(), 0);
    }

    #[test]
    fn insisting_on_the_whole_buffer_fails_at_the_end() {
        let t = Tmp::new("exact", b"abc");
        let f = t.open();
        let mut buf = [0u8; 16];
        let e = read_exact_at(&f, 0, &mut buf).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::UnexpectedEof);
        // And the message says what it wanted, because this one surfaces to a
        // person looking at a file that will not open.
        assert!(e.to_string().contains("16 bytes"), "{e}");
    }
}
