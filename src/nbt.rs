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
        if id as usize >= TAGS.len() {
            return None;
        }
        TAGS[id as usize]
    }

    /// How this tag's payload is laid out on the wire.
    const fn shape(self) -> Option<Shape> {
        SHAPES[self.id() as usize]
    }
}

/// How an NBT value is laid out on the wire, as skipping sees it.
///
/// Skipping never needs the semantic kind of a value, only its byte layout: a
/// value is a fixed-width scalar, a count followed by same-width elements, a
/// list, or a compound. Collapsing the twelve tags into these forms keeps tag
/// knowledge in tables instead of in a branch of every read loop.
#[derive(Clone, Copy)]
enum Shape {
    /// A payload of a fixed byte length.
    Fixed(u64),
    /// A count followed by that many elements of `width` bytes.
    Sized {
        /// How the element count is encoded.
        prefix: Prefix,
        /// Width of one element in bytes.
        width: u64,
    },
    /// An element tag, a count, and that many elements.
    List,
    /// A named sequence of members.
    Compound,
}

/// How a sized NBT value announces its element count.
#[derive(Clone, Copy)]
enum Prefix {
    /// A `u16`, as NBT strings use.
    U16,
    /// A non-negative `i32`, as arrays use.
    I32,
}

/// Every tag id, indexed by its wire value.
const TAGS: [Option<Tag>; 13] = [
    Some(Tag::End),
    Some(Tag::Byte),
    Some(Tag::Short),
    Some(Tag::Int),
    Some(Tag::Long),
    Some(Tag::Float),
    Some(Tag::Double),
    Some(Tag::ByteArray),
    Some(Tag::String),
    Some(Tag::List),
    Some(Tag::Compound),
    Some(Tag::IntArray),
    Some(Tag::LongArray),
];

/// The skip shape of every tag id, indexed by its wire value.
const SHAPES: [Option<Shape>; 13] = [
    None,                  // End
    Some(Shape::Fixed(1)), // Byte
    Some(Shape::Fixed(2)), // Short
    Some(Shape::Fixed(4)), // Int
    Some(Shape::Fixed(8)), // Long
    Some(Shape::Fixed(4)), // Float
    Some(Shape::Fixed(8)), // Double
    Some(Shape::Sized {
        prefix: Prefix::I32,
        width: 1,
    }), // ByteArray
    Some(Shape::Sized {
        prefix: Prefix::U16,
        width: 1,
    }), // String
    Some(Shape::List),     // List
    Some(Shape::Compound), // Compound
    Some(Shape::Sized {
        prefix: Prefix::I32,
        width: 4,
    }), // IntArray
    Some(Shape::Sized {
        prefix: Prefix::I32,
        width: 8,
    }), // LongArray
];

impl Shape {
    /// Skips one value of this shape nested inside `enclosing`.
    fn skip<S: Source>(self, r: &mut Reader<S>, enclosing: usize) -> Result<()> {
        match self {
            Self::Fixed(width) => r.skip(width),
            Self::Sized { prefix, width } => {
                let count = prefix.count(r)?;
                r.skip(count * width)
            }
            Self::List => skip_list(r, enclosing),
            Self::Compound => skip_compound(r, enclosing),
        }
    }
}

impl Prefix {
    /// Reads this prefix as a non-negative element count.
    fn count<S: Source>(self, r: &mut Reader<S>) -> Result<u64> {
        match self {
            Self::U16 => Ok(u64::from(r.u16()?)),
            Self::I32 => Ok(r.count()? as u64),
        }
    }
}

/// Skips a list: its element tag, count, and every element.
fn skip_list<S: Source>(r: &mut Reader<S>, enclosing: usize) -> Result<()> {
    let elem = r.tag()?;
    let count = r.count()?;
    if count == 0 {
        return Ok(());
    }
    let depth = r.nested(enclosing)?;
    if elem == Tag::End {
        return Err(Error::UnexpectedTag {
            field: "list element",
            tag: elem.id(),
            offset: r.pos(),
        });
    }
    for _ in 0..count {
        r.skip_value(elem, depth)?;
    }
    Ok(())
}

/// Skips a compound by the shapes of its members.
fn skip_compound<S: Source>(r: &mut Reader<S>, enclosing: usize) -> Result<()> {
    let depth = r.nested(enclosing)?;
    while let Some((_, tag)) = r.next_field()? {
        r.skip_value(tag, depth)?;
    }
    Ok(())
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
    /// The tag this value is written as.
    const TAG: Tag;

    /// Reads this value's payload, without a tag id or a name.
    fn decode(r: &mut Reader<S>, field: &'static str, enclosing: usize) -> Result<Self>;
}

impl<S: Source> Decode<S> for i32 {
    const TAG: Tag = Tag::Int;

    fn decode(r: &mut Reader<S>, _field: &'static str, _enclosing: usize) -> Result<Self> {
        r.i32()
    }
}

impl<S: Source> Decode<S> for i64 {
    const TAG: Tag = Tag::Long;

    fn decode(r: &mut Reader<S>, _field: &'static str, _enclosing: usize) -> Result<Self> {
        r.i64()
    }
}

impl<S: Source> Decode<S> for String {
    const TAG: Tag = Tag::String;

    fn decode(r: &mut Reader<S>, _field: &'static str, _enclosing: usize) -> Result<Self> {
        r.string()
    }
}

// Slot
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// The position of one field in a compound accumulator.
///
/// NBT compounds are unordered, so a decoder cannot assign a field as it
/// walks them; it deposits each member it meets into its slot and asks for
/// the result once the compound ends. The slot records presence, so the rule
/// for a missing required field lives with the field rather than in a pile of
/// `Option` sentinels and `ok_or` calls after the scan.
#[derive(Debug)]
pub(crate) struct Slot<T> {
    value: Option<T>,
}

impl<T> Default for Slot<T> {
    fn default() -> Self {
        Self { value: None }
    }
}

impl<T> Slot<T> {
    /// Stores the value read for this field, replacing an earlier one.
    pub(crate) fn set(&mut self, value: T) {
        self.value = Some(value);
    }

    /// The stored value, borrowed.
    pub(crate) fn get(&self) -> Option<&T> {
        self.value.as_ref()
    }

    /// The value, or an error naming `field` when the file omitted it.
    pub(crate) fn required(self, field: &'static str, offset: usize) -> Result<T> {
        self.value.ok_or(Error::MissingField { field, offset })
    }

    /// The value, or `T::default()` when the file omitted this field.
    pub(crate) fn or_default(self) -> T
    where
        T: Default,
    {
        self.value.unwrap_or_default()
    }
}

// Fields
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// What consuming a compound member leaves for the caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FieldOutcome {
    /// The member's value was consumed or skipped.
    Value,
    /// The member marks a payload whose bytes the caller consumes, so the
    /// schema node reported it without reading further.
    Deferred,
}

/// One named member of a schema compound and the way it deposits its value.
///
/// A field is data: it carries the name it answers to, the slot filler that
/// accepts its value, and whether consuming it defers the payload to the
/// caller. A schema node states its fields as a table and lets the table find
/// the member it just met, so adding a field never means editing a dispatch
/// `match`.
pub(crate) struct FieldEntry<S, R: Source> {
    name: &'static str,
    outcome: FieldOutcome,
    apply: fn(&mut S, &mut Reader<R>, Tag, usize) -> Result<()>,
}

impl<S, R: Source> FieldEntry<S, R> {
    /// A field whose value is consumed here, or skipped when unknown.
    pub(crate) const fn new(
        name: &'static str,
        apply: fn(&mut S, &mut Reader<R>, Tag, usize) -> Result<()>,
    ) -> Self {
        Self {
            name,
            outcome: FieldOutcome::Value,
            apply,
        }
    }

    /// A field that only validates its own header and hands the payload back
    /// to the caller, such as a `BlockStates` long array.
    pub(crate) const fn deferred(
        name: &'static str,
        apply: fn(&mut S, &mut Reader<R>, Tag, usize) -> Result<()>,
    ) -> Self {
        Self {
            name,
            outcome: FieldOutcome::Deferred,
            apply,
        }
    }
}

/// A schema compound's declared members, looked up by the name just read.
///
/// The table is the whole dispatch: a member either matches an entry and is
/// applied by it, or is unknown and skipped uniformly. NBT compounds are
/// unordered, so the table is consulted once per member and carries no
/// position.
pub(crate) struct FieldTable<'a, S, R: Source> {
    entries: &'a [FieldEntry<S, R>],
}

impl<'a, S, R: Source> FieldTable<'a, S, R> {
    /// Wraps a schema node's declared fields.
    pub(crate) fn new(entries: &'a [FieldEntry<S, R>]) -> Self {
        Self { entries }
    }

    /// Applies the member `name` to `schema`, or skips it when undeclared.
    pub(crate) fn apply(
        &self,
        schema: &mut S,
        r: &mut Reader<R>,
        name: &str,
        tag: Tag,
        enclosing: usize,
    ) -> Result<FieldOutcome> {
        if let Some(entry) = self.entries.iter().find(|entry| entry.name == name) {
            (entry.apply)(schema, r, tag, enclosing)?;
            return Ok(entry.outcome);
        }
        r.skip_value(tag, enclosing)?;
        Ok(FieldOutcome::Value)
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

    /// Reads one big-endian `u16`.
    #[inline]
    pub(crate) fn u16(&mut self) -> Result<u16> {
        let mut b = [0u8; 2];
        self.read_exact(&mut b)?;
        Ok(u16::from_be_bytes(b))
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
        let n = usize::from(self.u16()?);
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

    /// Reads the header of a compound's next member, or `None` at its end.
    ///
    /// This is the whole wire protocol of a compound: a tag id, the member
    /// name, and then a value the caller decides how to treat. Centralizing it
    /// keeps every schema node from restating the end-of-compound rule.
    pub(crate) fn next_field(&mut self) -> Result<Option<(String, Tag)>> {
        let tag = self.tag()?;
        if tag == Tag::End {
            return Ok(None);
        }
        Ok(Some((self.string()?, tag)))
    }

    /// Reads the payload of `T`, which the caller found under `tag`.
    ///
    /// The declared tag check and the payload decode are one step, so a schema
    /// node names a field and its type without restating the type's tag.
    pub(crate) fn decode_field<T: Decode<S>>(
        &mut self,
        field: &'static str,
        tag: Tag,
        enclosing: usize,
    ) -> Result<T> {
        self.expect(field, tag, T::TAG)?;
        T::decode(self, field, enclosing)
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
    /// at `enclosing_depth`. The tag's declared shape decides how many bytes
    /// to advance; sized payloads jump by length * element size, while lists
    /// and compounds walk only headers.
    pub(crate) fn skip_value(&mut self, tag: Tag, enclosing_depth: usize) -> Result<()> {
        let shape = tag.shape().ok_or_else(|| Error::UnexpectedTag {
            field: "value",
            tag: tag.id(),
            offset: self.pos(),
        })?;
        shape.skip(self, enclosing_depth)
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
