//! A minimal big-endian Java NBT cursor.
//!
//! The reader never builds a tree: values are decoded in place and unknown
//! subtrees are skipped in O(length) without allocation, so a decompressed
//! litematic buffer is scanned exactly once.

use crate::error::{Error, Result};

/// Maximum NBT nesting depth accepted by [`Reader::skip_value`].
pub(crate) const MAX_DEPTH: usize = 64;

pub(crate) const TAG_END: u8 = 0;
pub(crate) const TAG_INT: u8 = 3;
pub(crate) const TAG_LONG: u8 = 4;
pub(crate) const TAG_BYTE_ARRAY: u8 = 7;
pub(crate) const TAG_STRING: u8 = 8;
pub(crate) const TAG_LIST: u8 = 9;
pub(crate) const TAG_COMPOUND: u8 = 10;
pub(crate) const TAG_INT_ARRAY: u8 = 11;
pub(crate) const TAG_LONG_ARRAY: u8 = 12;

/// Payload size in bytes for fixed-size scalar tags, indexed by tag id.
/// Entries are `None` for tags with variable or prefixed sizes.
const SCALAR_SIZE: [Option<u64>; 13] = [
    None,    // 0 end (never a standalone value)
    Some(1), // 1 byte
    Some(2), // 2 short
    Some(4), // 3 int
    Some(8), // 4 long
    Some(4), // 5 float
    Some(8), // 6 double
    None,    // 7 byte array
    None,    // 8 string
    None,    // 9 list
    None,    // 10 compound
    None,    // 11 int array
    None,    // 12 long array
];

/// A forward-only cursor over an uncompressed NBT buffer.
pub(crate) struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    /// Creates a reader positioned at the start of `buf`.
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// The current byte offset.
    pub(crate) fn pos(&self) -> usize {
        self.pos
    }

    /// Consumes and returns the next `n` bytes as a slice borrowed from the
    /// input buffer, or fails with [`Error::UnexpectedEof`].
    #[inline]
    pub(crate) fn take(&mut self, n: u64) -> Result<&'a [u8]> {
        let end = self.pos as u64 + n;
        if end > self.buf.len() as u64 {
            return Err(Error::UnexpectedEof { offset: self.pos });
        }
        let start = self.pos;
        self.pos = end as usize;
        Ok(&self.buf[start..self.pos])
    }

    /// Reads one unsigned byte.
    #[inline]
    pub(crate) fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    /// Reads one big-endian `i32`.
    #[inline]
    pub(crate) fn i32(&mut self) -> Result<i32> {
        let b = self.take(4)?;
        Ok(i32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Reads one big-endian `i64`.
    #[inline]
    pub(crate) fn i64(&mut self) -> Result<i64> {
        let b = self.take(8)?;
        Ok(i64::from_be_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// Reads an NBT string (u16 length prefix + bytes) and decodes it as
    /// UTF-8, replacing invalid sequences. Java's modified UTF-8 differs from
    /// standard UTF-8 only in encodings that never appear in litematic files,
    /// so a lossy decode is sufficient here.
    pub(crate) fn string(&mut self) -> Result<String> {
        let b = self.take(2)?;
        let n = u16::from_be_bytes([b[0], b[1]]) as u64;
        let bytes = self.take(n)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    /// Reads an i32 element or byte count and rejects negative values.
    pub(crate) fn count(&mut self) -> Result<usize> {
        let offset = self.pos;
        let n = self.i32()?;
        if n < 0 {
            return Err(Error::NegativeLength { length: n, offset });
        }
        Ok(n as usize)
    }

    /// Reads a tag id byte and validates that it is in 0..=12.
    pub(crate) fn tag_id(&mut self) -> Result<u8> {
        let offset = self.pos;
        let id = self.u8()?;
        if id > TAG_LONG_ARRAY {
            return Err(Error::InvalidTagId { id, offset });
        }
        Ok(id)
    }

    /// Computes the depth of a nested container, enforcing [`MAX_DEPTH`].
    fn descend(&self, enclosing_depth: usize) -> Result<usize> {
        let depth = enclosing_depth + 1;
        if depth > MAX_DEPTH {
            return Err(Error::DepthLimitExceeded { offset: self.pos });
        }
        Ok(depth)
    }

    /// Skips the payload of `tag`, which is a value nested inside a container
    /// at `enclosing_depth`. Sized arrays jump directly by length * element
    /// size; compounds and lists walk their headers only.
    pub(crate) fn skip_value(&mut self, tag: u8, enclosing_depth: usize) -> Result<()> {
        if let Some(n) = SCALAR_SIZE[tag as usize] {
            return self.take(n).map(|_| ());
        }
        match tag {
            TAG_BYTE_ARRAY => {
                let n = self.count()?;
                self.take(n as u64)?;
            }
            TAG_STRING => {
                self.string()?;
            }
            TAG_LIST => {
                let elem = self.tag_id()?;
                let count = self.count()?;
                if count > 0 {
                    let depth = self.descend(enclosing_depth)?;
                    if elem == TAG_END {
                        return Err(Error::UnexpectedTag {
                            field: "list element",
                            tag: elem,
                            offset: self.pos,
                        });
                    }
                    for _ in 0..count {
                        self.skip_value(elem, depth)?;
                    }
                }
            }
            TAG_COMPOUND => {
                let depth = self.descend(enclosing_depth)?;
                loop {
                    let t = self.tag_id()?;
                    if t == TAG_END {
                        break;
                    }
                    self.string()?;
                    self.skip_value(t, depth)?;
                }
            }
            TAG_INT_ARRAY => {
                let n = self.count()?;
                self.take(n as u64 * 4)?;
            }
            TAG_LONG_ARRAY => {
                let n = self.count()?;
                self.take(n as u64 * 8)?;
            }
            // Unreachable for validated ids: every id is covered above.
            _ => {
                return Err(Error::UnexpectedTag {
                    field: "value",
                    tag,
                    offset: self.pos,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{SCALAR_SIZE, TAG_INT, TAG_LONG};

    #[test]
    fn scalar_sizes_match_tag_ids() {
        assert_eq!(SCALAR_SIZE[1], Some(1)); // byte
        assert_eq!(SCALAR_SIZE[2], Some(2)); // short
        assert_eq!(SCALAR_SIZE[TAG_INT as usize], Some(4));
        assert_eq!(SCALAR_SIZE[5], Some(4)); // float
        assert_eq!(SCALAR_SIZE[TAG_LONG as usize], Some(8));
        assert_eq!(SCALAR_SIZE[6], Some(8)); // double
    }
}
