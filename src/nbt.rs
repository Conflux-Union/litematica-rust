//! Big-endian NBT primitives shared by the litematic reader and writer.
//!
//! The crate never builds an NBT tree. Reading is a forward-only cursor that
//! decodes known values in place and skips unknown subtrees in O(length), so
//! a decompressed litematic buffer is scanned exactly once. Writing is the
//! mirror image: a forward-only encoder that pushes values straight to any
//! `std::io::Write`, which lets a `TAG_Long_Array` stream word by word instead
//! of being materialized.
//!
//! Both directions name tags through [`Tag`], a closed enum, rather than raw
//! bytes. An id read from a file is validated once, at [`Tag::from_id`], and
//! from then on every comparison and dispatch is spelled by name.

use std::io::Read as IoRead;
use std::io::Write as IoWrite;

use crate::error::{Error, Result};

/// Maximum NBT nesting depth accepted by [`Reader::skip_value`].
pub(crate) const MAX_DEPTH: usize = 64;

// Tag
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// The thirteen NBT tag ids, as a closed set.
///
/// Keeping the tag space in an enum means every consumer states intent in
/// names (`Tag::Compound`) and a tag id read from a file is validated exactly
/// once, at [`Tag::from_id`]. The representation stays `u8` on the wire, so
/// the enum converts cheaply in both directions.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tag {
    /// Closes a compound, or marks an empty list element type.
    End = 0,
    /// A signed 8-bit integer.
    Byte = 1,
    /// A signed 16-bit integer.
    Short = 2,
    /// A signed 32-bit integer.
    Int = 3,
    /// A signed 64-bit integer.
    Long = 4,
    /// An IEEE 754 single-precision float.
    Float = 5,
    /// An IEEE 754 double-precision float.
    Double = 6,
    /// A length-prefixed array of bytes.
    ByteArray = 7,
    /// A length-prefixed UTF-8 string.
    String = 8,
    /// A homogeneous sequence of values.
    List = 9,
    /// A sequence of named tags.
    Compound = 10,
    /// A length-prefixed array of 32-bit integers.
    IntArray = 11,
    /// A length-prefixed array of 64-bit integers.
    LongArray = 12,
}

impl Tag {
    /// The tag id written to or read from the wire.
    pub(crate) const fn id(self) -> u8 {
        self as u8
    }

    /// Resolves a wire id, or `None` when it is outside the tag space.
    pub(crate) const fn from_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(Self::End),
            1 => Some(Self::Byte),
            2 => Some(Self::Short),
            3 => Some(Self::Int),
            4 => Some(Self::Long),
            5 => Some(Self::Float),
            6 => Some(Self::Double),
            7 => Some(Self::ByteArray),
            8 => Some(Self::String),
            9 => Some(Self::List),
            10 => Some(Self::Compound),
            11 => Some(Self::IntArray),
            12 => Some(Self::LongArray),
            _ => None,
        }
    }

    /// Payload size of a fixed-width scalar, or `None` for tags whose payload
    /// is length-prefixed or structural.
    ///
    /// Skipping uses this to jump over scalars in one step instead of
    /// matching each tag individually.
    const fn scalar_size(self) -> Option<u64> {
        match self {
            Self::Byte => Some(1),
            Self::Short => Some(2),
            Self::Int | Self::Float => Some(4),
            Self::Long | Self::Double => Some(8),
            Self::End
            | Self::ByteArray
            | Self::String
            | Self::List
            | Self::Compound
            | Self::IntArray
            | Self::LongArray => None,
        }
    }
}

// Encode
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// A value that knows its own NBT shape.
///
/// This is the crate's polymorphic seam: a scalar writes a payload, a
/// compound opens its fields and closes itself, a list declares its element
/// type. A schema node therefore states *what it is* rather than making a
/// caller restate its tag and field order, and [`Encode::encode`] contributes
/// the shared tag-name header as a template method.
pub(crate) trait Encode {
    /// The tag this value is written as.
    const TAG: Tag;

    /// Writes this tag's payload, without a tag id or a name.
    fn body<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()>;

    /// Writes this value as the named tag `name`.
    fn encode<W: IoWrite>(&self, name: &str, out: &mut Writer<W>) -> Result<()> {
        out.tag(Self::TAG, name)?;
        self.body(out)
    }
}

impl Encode for i32 {
    const TAG: Tag = Tag::Int;

    fn body<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()> {
        out.int(*self)
    }
}

impl Encode for i64 {
    const TAG: Tag = Tag::Long;

    fn body<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()> {
        out.long(*self)
    }
}

impl Encode for str {
    const TAG: Tag = Tag::String;

    fn body<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()> {
        out.string(self)
    }
}

impl Encode for String {
    const TAG: Tag = Tag::String;

    fn body<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()> {
        out.string(self)
    }
}

// Decode
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// A value that knows its own NBT shape on the read side.
///
/// The reading mirror of [`Encode`]: a schema node implements this once and
/// is then read wherever it appears, instead of every caller restating which
/// fields to pull and in what order. `field` names the node in diagnostics,
/// since a value's identity lives in the name its parent gave it, not in its
/// payload; `enclosing` is the depth of the container holding the value, so a
/// compound can derive the depth of its own fields and hand unknown members
/// to [`Reader::skip_value`].
pub(crate) trait Decode<S: Source>: Sized {
    /// Reads this value's payload, without a tag id or a name.
    fn decode(r: &mut Reader<S>, field: &'static str, enclosing: usize) -> Result<Self>;
}

impl<S: Source> Decode<S> for i32 {
    fn decode(r: &mut Reader<S>, _field: &'static str, _enclosing: usize) -> Result<Self> {
        r.i32()
    }
}

impl<S: Source> Decode<S> for i64 {
    fn decode(r: &mut Reader<S>, _field: &'static str, _enclosing: usize) -> Result<Self> {
        r.i64()
    }
}

impl<S: Source> Decode<S> for String {
    fn decode(r: &mut Reader<S>, _field: &'static str, _enclosing: usize) -> Result<Self> {
        r.string()
    }
}

// Reader
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// A byte source that an NBT [`Reader`] consumes.
///
pub(crate) trait Source {
    /// Reads exactly `out.len()` bytes into `out`.
    fn read_exact(&mut self, out: &mut [u8]) -> Result<()>;

    /// Advances past `n` bytes without materializing them.
    fn skip(&mut self, n: u64) -> Result<()>;

    /// Number of bytes consumed so far.
    fn pos(&self) -> usize;
}

/// A [`Source`] over an in-memory buffer.
///
/// Keeps borrowed data addressable, which is what the random-access region
/// reader needs for its zero-copy `BlockStates` view.
pub(crate) struct SliceSource<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> SliceSource<'a> {
    /// Creates a source positioned at the start of `buf`.
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Borrows the next `n` bytes directly from the buffer.
    pub(crate) fn take(&mut self, n: u64) -> Result<&'a [u8]> {
        let end = self.pos as u64 + n;
        if end > self.buf.len() as u64 {
            return Err(Error::UnexpectedEof { offset: self.pos });
        }
        let start = self.pos;
        self.pos = end as usize;
        Ok(&self.buf[start..self.pos])
    }
}

impl Source for SliceSource<'_> {
    fn read_exact(&mut self, out: &mut [u8]) -> Result<()> {
        let bytes = self.take(out.len() as u64)?;
        out.copy_from_slice(bytes);
        Ok(())
    }

    fn skip(&mut self, n: u64) -> Result<()> {
        self.take(n).map(|_| ())
    }

    fn pos(&self) -> usize {
        self.pos
    }
}

/// A [`Source`] over any `std::io::Read`, with a cap on consumed bytes.
///
/// The streaming entry point wraps a gzip decoder in a buffered reader, so
/// this type adds only the byte counter used in diagnostics and the cap that
/// stops a hostile member from inflating without bound.
pub(crate) struct IoSource<R> {
    inner: R,
    pos: u64,
    limit: u64,
    scratch: Box<[u8]>,
}

impl<R: IoRead> IoSource<R> {
    /// Wraps `inner`, failing once more than `limit` bytes are consumed.
    pub(crate) fn new(inner: R, limit: u64) -> Self {
        Self {
            inner,
            pos: 0,
            limit,
            scratch: vec![0u8; 64 * 1024].into_boxed_slice(),
        }
    }

    fn check(&self, n: u64) -> Result<()> {
        if self.pos.saturating_add(n) > self.limit {
            return Err(Error::SizeLimitExceeded { limit: self.limit });
        }
        Ok(())
    }
}

/// Maps an I/O failure to the NBT-level error that describes it.
fn map_io(error: std::io::Error, offset: usize) -> Error {
    match error.kind() {
        std::io::ErrorKind::UnexpectedEof => Error::UnexpectedEof { offset },
        _ => Error::CorruptGzip(error),
    }
}

impl<R: IoRead> Source for IoSource<R> {
    fn read_exact(&mut self, out: &mut [u8]) -> Result<()> {
        let n = out.len() as u64;
        self.check(n)?;
        let offset = self.pos as usize;
        self.inner
            .read_exact(out)
            .map_err(|error| map_io(error, offset))?;
        self.pos += n;
        Ok(())
    }

    fn skip(&mut self, n: u64) -> Result<()> {
        self.check(n)?;
        let mut left = n;
        while left > 0 {
            let take = left.min(self.scratch.len() as u64) as usize;
            let offset = self.pos as usize;
            let chunk = &mut self.scratch[..take];
            self.inner
                .read_exact(chunk)
                .map_err(|error| map_io(error, offset))?;
            self.pos += take as u64;
            left -= take as u64;
        }
        Ok(())
    }

    fn pos(&self) -> usize {
        usize::try_from(self.pos).unwrap_or(usize::MAX)
    }
}

/// A forward-only cursor over an NBT byte source.
///
/// The source is generic so the same schema decoders serve both the
/// random-access reader (a [`SliceSource`]) and the streaming reader (an
/// [`IoSource`]); only payload-owning fields such as `BlockStates` branch on
/// the concrete source.
pub(crate) struct Reader<S> {
    src: S,
}

impl<S: Source> Reader<S> {
    /// Creates a reader positioned at the start of `src`.
    pub(crate) fn new(src: S) -> Self {
        Self { src }
    }

    /// The current byte offset.
    pub(crate) fn pos(&self) -> usize {
        self.src.pos()
    }

    /// Reads exactly `out.len()` bytes.
    #[inline]
    pub(crate) fn read_exact(&mut self, out: &mut [u8]) -> Result<()> {
        self.src.read_exact(out)
    }

    /// Advances past `n` bytes.
    #[inline]
    pub(crate) fn skip(&mut self, n: u64) -> Result<()> {
        self.src.skip(n)
    }

    /// Reads one unsigned byte.
    #[inline]
    pub(crate) fn u8(&mut self) -> Result<u8> {
        let mut b = [0u8; 1];
        self.read_exact(&mut b)?;
        Ok(b[0])
    }

    /// Reads one big-endian `i32`.
    #[inline]
    pub(crate) fn i32(&mut self) -> Result<i32> {
        let mut b = [0u8; 4];
        self.read_exact(&mut b)?;
        Ok(i32::from_be_bytes(b))
    }

    /// Reads one big-endian `i64`.
    #[inline]
    pub(crate) fn i64(&mut self) -> Result<i64> {
        let mut b = [0u8; 8];
        self.read_exact(&mut b)?;
        Ok(i64::from_be_bytes(b))
    }

    /// Reads an NBT string (u16 length prefix + bytes) and decodes it as
    /// UTF-8, replacing invalid sequences. Java's modified UTF-8 differs from
    /// standard UTF-8 only in encodings that never appear in litematic files,
    /// so a lossy decode is sufficient here.
    pub(crate) fn string(&mut self) -> Result<String> {
        let mut len = [0u8; 2];
        self.read_exact(&mut len)?;
        let n = u16::from_be_bytes(len) as usize;
        let mut bytes = vec![0u8; n];
        self.read_exact(&mut bytes)?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Reads an i32 element or byte count and rejects negative values.
    pub(crate) fn count(&mut self) -> Result<usize> {
        let offset = self.pos();
        let n = self.i32()?;
        if n < 0 {
            return Err(Error::NegativeLength { length: n, offset });
        }
        Ok(n as usize)
    }

    /// Reads a tag id and resolves it to a [`Tag`], rejecting ids outside the
    /// tag space.
    pub(crate) fn tag(&mut self) -> Result<Tag> {
        let offset = self.pos();
        let id = self.u8()?;
        Tag::from_id(id).ok_or(Error::InvalidTagId { id, offset })
    }

    /// Rejects `actual` unless it is `expected`, naming the offending field.
    pub(crate) fn expect(&self, field: &'static str, actual: Tag, expected: Tag) -> Result<()> {
        if actual != expected {
            return Err(Error::UnexpectedTag {
                field,
                tag: actual.id(),
                offset: self.pos(),
            });
        }
        Ok(())
    }

    /// The depth of a container nested inside `enclosing`, enforcing
    /// [`MAX_DEPTH`].
    pub(crate) fn nested(&self, enclosing: usize) -> Result<usize> {
        let depth = enclosing + 1;
        if depth > MAX_DEPTH {
            return Err(Error::DepthLimitExceeded { offset: self.pos() });
        }
        Ok(depth)
    }

    /// Skips the payload of `tag`, which is a value nested inside a container
    /// at `enclosing_depth`. Sized arrays jump directly by length * element
    /// size; compounds and lists walk their headers only.
    pub(crate) fn skip_value(&mut self, tag: Tag, enclosing_depth: usize) -> Result<()> {
        if let Some(n) = tag.scalar_size() {
            return self.skip(n);
        }
        match tag {
            Tag::ByteArray => {
                let n = self.count()?;
                self.skip(n as u64)?;
            }
            Tag::String => {
                self.string()?;
            }
            Tag::List => {
                let elem = self.tag()?;
                let count = self.count()?;
                if count > 0 {
                    let depth = self.nested(enclosing_depth)?;
                    if elem == Tag::End {
                        return Err(Error::UnexpectedTag {
                            field: "list element",
                            tag: elem.id(),
                            offset: self.pos(),
                        });
                    }
                    for _ in 0..count {
                        self.skip_value(elem, depth)?;
                    }
                }
            }
            Tag::Compound => {
                let depth = self.nested(enclosing_depth)?;
                loop {
                    let t = self.tag()?;
                    if t == Tag::End {
                        break;
                    }
                    self.string()?;
                    self.skip_value(t, depth)?;
                }
            }
            Tag::IntArray => {
                let n = self.count()?;
                self.skip(n as u64 * 4)?;
            }
            Tag::LongArray => {
                let n = self.count()?;
                self.skip(n as u64 * 8)?;
            }
            // Scalars were handled above; this arm keeps the match exhaustive.
            Tag::End | Tag::Byte | Tag::Short | Tag::Int | Tag::Long | Tag::Float | Tag::Double => {
                return Err(Error::UnexpectedTag {
                    field: "value",
                    tag: tag.id(),
                    offset: self.pos(),
                });
            }
        }
        Ok(())
    }
}

impl<'a> Reader<SliceSource<'a>> {
    /// Consumes and returns the next `n` bytes as a slice borrowed from the
    /// underlying buffer.
    #[inline]
    pub(crate) fn take(&mut self, n: u64) -> Result<&'a [u8]> {
        self.src.take(n)
    }
}

// Writer
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// A forward-only NBT encoder writing big-endian values to a sink.
///
/// Methods come in two layers. The structural layer (`root`, `tag`,
/// `compound`, `list`, `end`, `long_array`) opens and closes the document
/// shape; the scalar layer (`int`, `long`, `string`) writes one named
/// field. Callers therefore express the schema top-down instead of emitting
/// raw tag bytes themselves.
pub(crate) struct Writer<W: IoWrite> {
    out: W,
}

impl<W: IoWrite> Writer<W> {
    /// Creates a writer that serializes into `out`.
    pub(crate) fn new(out: W) -> Self {
        Self { out }
    }

    /// Writes all bytes, mapping sink failures to [`Error::WriteIo`].
    fn put(&mut self, bytes: &[u8]) -> Result<()> {
        self.out
            .write_all(bytes)
            .map_err(|source| Error::WriteIo { source })
    }

    /// Writes one tag id byte.
    #[inline]
    fn tag_id(&mut self, tag: Tag) -> Result<()> {
        self.put(&[tag.id()])
    }

    /// Writes a tag id and a length-prefixed name.
    pub(crate) fn tag(&mut self, tag: Tag, name: &str) -> Result<()> {
        self.tag_id(tag)?;
        self.put_string(name)
    }

    /// Writes the root compound header: `TAG_Compound` with an empty name.
    pub(crate) fn root(&mut self) -> Result<()> {
        self.tag(Tag::Compound, "")
    }

    /// Opens a named compound.
    pub(crate) fn compound(&mut self, name: &str) -> Result<()> {
        self.tag(Tag::Compound, name)
    }

    /// Writes a `TAG_End` marker closing the innermost compound.
    pub(crate) fn end(&mut self) -> Result<()> {
        self.tag_id(Tag::End)
    }

    /// Writes a list payload header: its element tag and element count.
    pub(crate) fn list(&mut self, element: Tag, len: i32) -> Result<()> {
        self.tag_id(element)?;
        self.put_i32(len)
    }

    /// Writes an empty named list.
    pub(crate) fn empty_list(&mut self, name: &str) -> Result<()> {
        self.tag(Tag::List, name)?;
        self.list(Tag::End, 0)
    }

    /// Writes an `i32` payload.
    pub(crate) fn int(&mut self, value: i32) -> Result<()> {
        self.put_i32(value)
    }

    /// Writes an `i64` payload.
    pub(crate) fn long(&mut self, value: i64) -> Result<()> {
        self.put_i64(value)
    }

    /// Writes a length-prefixed UTF-8 payload.
    pub(crate) fn string(&mut self, value: &str) -> Result<()> {
        self.put_string(value)
    }

    /// Writes a `TAG_Long_Array` payload whose words are pulled on demand.
    ///
    /// The length goes out first, then one word at a time, so an array far
    /// larger than memory can still be serialized. A failing word stops the
    /// stream and is returned as the call's error, leaving the sink partially
    /// written exactly as any other mid-document failure would.
    pub(crate) fn long_array<I>(&mut self, len: i32, words: I) -> Result<()>
    where
        I: IntoIterator<Item = Result<i64>>,
    {
        self.put_i32(len)?;
        for word in words {
            self.put_i64(word?)?;
        }
        Ok(())
    }

    /// Writes an `i32` payload.
    #[inline]
    fn put_i32(&mut self, value: i32) -> Result<()> {
        self.put(&value.to_be_bytes())
    }

    /// Writes an `i64` payload.
    #[inline]
    fn put_i64(&mut self, value: i64) -> Result<()> {
        self.put(&value.to_be_bytes())
    }

    /// Writes a length-prefixed UTF-8 payload.
    fn put_string(&mut self, value: &str) -> Result<()> {
        let bytes = value.as_bytes();
        let len =
            u16::try_from(bytes.len()).map_err(|_| Error::StringTooLong { len: bytes.len() })?;
        self.put(&len.to_be_bytes())?;
        self.put(bytes)
    }

    /// Flushes and returns the underlying sink.
    pub(crate) fn into_inner(mut self) -> std::io::Result<W> {
        self.out.flush()?;
        Ok(self.out)
    }
}

// Tests
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

#[cfg(test)]
mod tests {
    use super::{Encode, Reader, SliceSource, Tag, Writer, MAX_DEPTH};
    use crate::Error;

    #[test]
    fn tag_ids_round_trip_and_reject_unknown_values() {
        for id in 0..=12 {
            let tag = Tag::from_id(id).unwrap();
            assert_eq!(tag.id(), id);
        }
        assert_eq!(Tag::from_id(13), None);
        assert_eq!(Tag::from_id(u8::MAX), None);
    }

    #[test]
    fn writer_scalars_round_trip_through_reader() {
        let mut w = Writer::new(Vec::new());
        w.root().unwrap();
        (-5).encode("a", &mut w).unwrap();
        "h\u{e9}llo \u{2713}"
            .encode("\u{540d}\u{5b57}", &mut w)
            .unwrap();
        i64::MIN.encode("c", &mut w).unwrap();
        w.end().unwrap();
        let buf = w.into_inner().unwrap();

        let mut r = Reader::new(SliceSource::new(&buf));
        assert_eq!(r.tag().unwrap(), Tag::Compound);
        assert_eq!(r.string().unwrap(), "");
        assert_eq!(r.tag().unwrap(), Tag::Int);
        assert_eq!(r.string().unwrap(), "a");
        assert_eq!(r.i32().unwrap(), -5);
        assert_eq!(r.tag().unwrap(), Tag::String);
        assert_eq!(r.string().unwrap(), "\u{540d}\u{5b57}");
        assert_eq!(r.string().unwrap(), "h\u{e9}llo \u{2713}");
        assert_eq!(r.tag().unwrap(), Tag::Long);
        assert_eq!(r.string().unwrap(), "c");
        assert_eq!(r.i64().unwrap(), i64::MIN);
        assert_eq!(r.tag().unwrap(), Tag::End);
        assert_eq!(r.pos(), buf.len());
    }

    #[test]
    fn writer_long_array_round_trips_and_skips() {
        let mut w = Writer::new(Vec::new());
        w.root().unwrap();
        let words = [Ok(0x0102_0304_0506_0708), Ok(-1), Ok(0)];
        w.tag(Tag::LongArray, "data").unwrap();
        w.long_array(words.len() as i32, words).unwrap();
        w.end().unwrap();
        let buf = w.into_inner().unwrap();

        let mut r = Reader::new(SliceSource::new(&buf));
        assert_eq!(r.tag().unwrap(), Tag::Compound);
        r.string().unwrap();
        let tag = r.tag().unwrap();
        r.string().unwrap(); // skip_value skips payloads only, not names
        r.skip_value(tag, 1).unwrap();
        assert_eq!(r.tag().unwrap(), Tag::End);
        assert_eq!(r.pos(), buf.len());
    }

    #[test]
    fn long_array_propagates_element_errors() {
        let mut w = Writer::new(Vec::new());
        w.root().unwrap();
        let words = [Ok(1), Err(Error::StringTooLong { len: 1 })];
        assert!(matches!(
            w.long_array(2, words),
            Err(Error::StringTooLong { len: 1 })
        ));
    }

    #[test]
    fn writer_propagates_sink_io_errors() {
        struct Failing;
        impl std::io::Write for Failing {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("boom"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut w = Writer::new(Failing);
        assert!(matches!(w.root(), Err(Error::WriteIo { .. })));
    }

    #[test]
    fn writer_rejects_strings_over_u16_limit() {
        let mut w = Writer::new(Vec::new());
        let s = "x".repeat(70_000);
        assert!(matches!(
            s.encode("s", &mut w),
            Err(Error::StringTooLong { len: 70_000 })
        ));
    }

    #[test]
    fn reader_rejects_unknown_tag_ids() {
        let mut r = Reader::new(SliceSource::new(&[13]));
        assert!(matches!(
            r.tag(),
            Err(Error::InvalidTagId { id: 13, offset: 0 })
        ));
    }

    #[test]
    fn skip_value_enforces_the_depth_limit() {
        let nested = |depth: usize| {
            let mut bytes = Vec::new();
            for _ in 0..depth {
                bytes.extend_from_slice(&[Tag::Compound.id(), 0, 0]);
            }
            for _ in 0..=depth {
                bytes.push(Tag::End.id());
            }
            bytes
        };

        // The compound passed to `skip_value` is itself the first level, so
        // `MAX_DEPTH - 1` nested fields reach the limit exactly.
        let bytes = nested(MAX_DEPTH - 1);
        Reader::new(SliceSource::new(&bytes))
            .skip_value(Tag::Compound, 0)
            .unwrap();

        let bytes = nested(MAX_DEPTH);
        assert!(matches!(
            Reader::new(SliceSource::new(&bytes)).skip_value(Tag::Compound, 0),
            Err(Error::DepthLimitExceeded { .. })
        ));
    }
}
