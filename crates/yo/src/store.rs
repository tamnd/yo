//! How a value becomes the bytes in a record, and back (`15` section 4).
//!
//! There is no serialization here in the sense the word usually has. A `u64` is
//! eight bytes little endian, a string is its own bytes, and neither of them
//! goes near a format that would have to be parsed. Nothing allocates on the
//! way in, and the borrowed form on the way out does not allocate either.
//!
//! What is missing is the part `#[derive(Yo)]` writes: a struct's fields laid
//! out in the order its shape declares. Until the derive lands, a collection
//! holds the primitives, which is what `Map<K, V>` needs to be worth measuring.

use core::str;

use yo_common::{Code, Error, Result};
use yo_shape::Shape;

/// A type that can be written into a record.
///
/// Implemented for a borrowed form as well as an owned one, so that a lookup
/// can take `&str` where the collection holds `String`, exactly as
/// `HashMap<String, _>::get` does.
pub trait Encode: Shape {
    /// Hand the bytes of `self` to `f`.
    ///
    /// A callback rather than a returned `Vec` because a returned `Vec` is an
    /// allocation, and this is the write path. A type whose bytes are already
    /// contiguous passes them straight through, and a fixed width one builds
    /// them on the stack.
    fn encode<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R;
}

/// A type that can be read back out of a record.
///
/// [`Decode::Ref`] is the borrowed view: the form that reads out of the arena
/// without copying, which is Y29 and where G6's point read budget lives. The
/// owned form is one call away for the code that does not care.
pub trait Decode: Encode + Sized {
    /// The borrowed view of this type.
    type Ref<'a>;

    /// Read an owned value.
    ///
    /// # Errors
    ///
    /// [`Code::Corrupt`] when the bytes are not this type, which means the
    /// collection holds something the shape said it did not.
    fn decode(bytes: &[u8]) -> Result<Self>;

    /// Read a borrowed view, which copies nothing.
    ///
    /// # Errors
    ///
    /// The same as [`Decode::decode`].
    fn view(bytes: &[u8]) -> Result<Self::Ref<'_>>;
}

fn wrong_len(what: &str, want: usize, got: usize) -> Error {
    Error::fmt(
        Code::Corrupt,
        format_args!("a {what} in this collection is {got} bytes and should be {want}"),
    )
}

macro_rules! fixed {
    ($($t:ty),* $(,)?) => {
        $(
            impl Encode for $t {
                #[inline]
                fn encode<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
                    f(&self.to_le_bytes())
                }
            }

            impl Decode for $t {
                type Ref<'a> = $t;

                #[inline]
                fn decode(bytes: &[u8]) -> Result<$t> {
                    let want = size_of::<$t>();
                    let array = bytes
                        .try_into()
                        .map_err(|_| wrong_len(stringify!($t), want, bytes.len()))?;
                    Ok(<$t>::from_le_bytes(array))
                }

                #[inline]
                fn view(bytes: &[u8]) -> Result<$t> {
                    <$t as Decode>::decode(bytes)
                }
            }
        )*
    };
}

fixed!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

impl Encode for bool {
    #[inline]
    fn encode<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        f(&[u8::from(*self)])
    }
}

impl Decode for bool {
    type Ref<'a> = bool;

    #[inline]
    fn decode(bytes: &[u8]) -> Result<bool> {
        match bytes {
            [0] => Ok(false),
            [1] => Ok(true),
            [_] => Err(Error::new(
                Code::Corrupt,
                "a bool in this collection is neither 0 nor 1",
            )),
            other => Err(wrong_len("bool", 1, other.len())),
        }
    }

    #[inline]
    fn view(bytes: &[u8]) -> Result<bool> {
        <bool as Decode>::decode(bytes)
    }
}

impl Encode for str {
    #[inline]
    fn encode<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        f(self.as_bytes())
    }
}

impl Encode for String {
    #[inline]
    fn encode<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        f(self.as_bytes())
    }
}

impl Decode for String {
    type Ref<'a> = &'a str;

    fn decode(bytes: &[u8]) -> Result<String> {
        <String as Decode>::view(bytes).map(ToOwned::to_owned)
    }

    #[inline]
    fn view(bytes: &[u8]) -> Result<&str> {
        str::from_utf8(bytes).map_err(|e| {
            Error::fmt(
                Code::Corrupt,
                format_args!("a str in this collection is not UTF-8: {e}"),
            )
        })
    }
}

impl Encode for [u8] {
    #[inline]
    fn encode<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        f(self)
    }
}

impl Encode for Vec<u8> {
    #[inline]
    fn encode<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        f(self)
    }
}

impl Decode for Vec<u8> {
    type Ref<'a> = &'a [u8];

    #[inline]
    fn decode(bytes: &[u8]) -> Result<Vec<u8>> {
        Ok(bytes.to_vec())
    }

    #[inline]
    fn view(bytes: &[u8]) -> Result<&[u8]> {
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes_of(v: &(impl Encode + ?Sized)) -> Vec<u8> {
        v.encode(<[u8]>::to_vec)
    }

    #[test]
    fn fixed_widths_are_little_endian_and_their_own_size() {
        assert_eq!(bytes_of(&1u32), vec![1, 0, 0, 0]);
        assert_eq!(bytes_of(&-2i16), vec![0xfe, 0xff]);
        assert_eq!(bytes_of(&1.5f64), 1.5f64.to_le_bytes().to_vec());
        assert_eq!(bytes_of(&true), vec![1]);
        assert_eq!(u64::decode(&bytes_of(&9u64)).unwrap(), 9);
        assert_eq!(f32::decode(&bytes_of(&0.5f32)).unwrap(), 0.5);
        assert!(bool::decode(&bytes_of(&false)).unwrap().eq(&false));
    }

    #[test]
    fn text_and_bytes_pass_straight_through() {
        assert_eq!(bytes_of("hello"), b"hello".to_vec());
        assert_eq!(String::view(b"hello").unwrap(), "hello");
        assert_eq!(Vec::<u8>::view(b"\x00\xff").unwrap(), b"\x00\xff");
    }

    /// The bytes in a record are the only thing that says what a value is, so
    /// the wrong number of them is a corruption and says which type it was
    /// expecting.
    #[test]
    fn the_wrong_number_of_bytes_is_corruption() {
        let e = u64::decode(b"1234").expect_err("four bytes is not a u64");
        assert_eq!(e.code(), Code::Corrupt);
        assert_eq!(
            e.message(),
            "a u64 in this collection is 4 bytes and should be 8"
        );

        assert_eq!(
            bool::decode(&[2]).expect_err("2 is not a bool").code(),
            Code::Corrupt
        );
        assert_eq!(
            bool::decode(&[0, 0])
                .expect_err("two bytes is not a bool")
                .code(),
            Code::Corrupt
        );
        assert!(
            String::decode(&[0xff, 0xfe])
                .expect_err("that is not UTF-8")
                .message()
                .contains("not UTF-8")
        );
    }
}
