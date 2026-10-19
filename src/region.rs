//! Regions, block state palettes, and the bit-unpacking block iterator.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;

use crate::bitstream::{bits_for, BitStream, BlockIter};
use crate::error::{Error, Result};
use crate::mc::{BlockPos, BlockSize, BlockState};
use crate::nbt::{
    Decode, FieldEntry, FieldOutcome, FieldTable, Reader, SliceSource, Slot, Source, Tag,
};

// Block state
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

impl<S: Source> Decode<S> for BlockState {
    const TAG: Tag = Tag::Compound;

    /// Reads one palette entry: a compound with a `Name` and an optional
    /// `Properties` compound.
    fn decode(r: &mut Reader<S>, _field: &'static str, enclosing: usize) -> Result<Self> {
        let depth = r.nested(enclosing)?;
        let offset = r.pos();
        let fields = EntryFields::fields();
        let mut entry = EntryFields::default();
        while let Some((field, tag)) = r.next_field()? {
            FieldTable::new(&fields).apply(&mut entry, r, &field, tag, depth)?;
        }
        Ok(Self {
            name: Cow::Owned(entry.name.required("palette entry Name", offset)?),
            properties: entry.properties.or_default(),
        })
    }
}

/// The members of a palette entry while they are still arriving.
///
/// `Properties` is optional, so it stays in a [`Slot`] until the compound
/// ends; `Name` is required and reports the entry's byte offset when missing.
#[derive(Default)]
struct EntryFields {
    name: Slot<String>,
    properties: Slot<BTreeMap<Cow<'static, str>, Cow<'static, str>>>,
}

impl EntryFields {
    /// The members of a palette entry and the way each one lands here.
    fn fields<R: Source>() -> [FieldEntry<Self, R>; 2] {
        [
            FieldEntry::new("Name", |entry, r, tag, depth| {
                entry
                    .name
                    .set(r.decode_field("palette entry Name", tag, depth)?);
                Ok(())
            }),
            FieldEntry::new("Properties", |entry, r, tag, depth| {
                entry.properties.set(decode_properties(r, tag, depth)?);
                Ok(())
            }),
        ]
    }
}

/// Reads a `Properties` compound as the keyed map the shared [`BlockState`]
/// orders by, so a file's member order cannot leak into state identity.
fn decode_properties<S: Source>(
    r: &mut Reader<S>,
    tag: Tag,
    enclosing: usize,
) -> Result<BTreeMap<Cow<'static, str>, Cow<'static, str>>> {
    r.expect("palette entry Properties", tag, Tag::Compound)?;
    let depth = r.nested(enclosing)?;
    let mut properties = BTreeMap::new();
    while let Some((key, tag)) = r.next_field()? {
        let value = r.decode_field("palette property value", tag, depth)?;
        properties.insert(Cow::Owned(key), Cow::Owned(value));
    }
    Ok(properties)
}

// Region header
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// The header fields of one region, accumulated in whatever order they arrive.
///
/// The random-access and streaming readers differ only in where the packed
/// payload lives; both hand the same field names to this accumulator, so the
/// region schema is stated once.
#[derive(Default)]
pub(crate) struct RegionHeader {
    position: Slot<BlockPos>,
    size: Slot<BlockPos>,
    palette: Slot<Palette>,
}

impl RegionHeader {
    /// Creates an empty header; every field defaults until one is read.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// The members of a region compound and the way each one lands here.
    /// `BlockStates` is deferred: its long array length and payload belong to
    /// the caller, which chooses between borrowing and copying them.
    fn fields<R: Source>() -> [FieldEntry<Self, R>; 4] {
        [
            FieldEntry::new("Position", |h, r, tag, depth| {
                h.position.set(r.decode_field("Position", tag, depth)?);
                Ok(())
            }),
            FieldEntry::new("Size", |h, r, tag, depth| {
                h.size.set(r.decode_field("Size", tag, depth)?);
                Ok(())
            }),
            FieldEntry::new("BlockStatePalette", |h, r, tag, depth| {
                h.palette
                    .set(r.decode_field("BlockStatePalette", tag, depth)?);
                Ok(())
            }),
            FieldEntry::deferred("BlockStates", |_, r, tag, _| {
                r.expect("BlockStates", tag, Tag::LongArray)
            }),
        ]
    }

    /// Consumes one field whose name has already been read.
    ///
    /// A deferred outcome means `BlockStates` was reached; its length prefix
    /// has not been read, so the caller decides whether to borrow or copy the
    /// payload. `enclosing` is the depth of the region's fields.
    pub(crate) fn read_field<R: Source>(
        &mut self,
        r: &mut Reader<R>,
        field: &str,
        tag: Tag,
        enclosing: usize,
    ) -> Result<FieldOutcome> {
        let fields = Self::fields();
        FieldTable::new(&fields).apply(self, r, field, tag, enclosing)
    }

    /// Returns the header, or the first required field that never arrived.
    pub(crate) fn into_parts(self, offset: usize) -> Result<RegionParts> {
        Ok(RegionParts {
            position: self.position.required("Position", offset)?,
            size: self.size.required("Size", offset)?,
            palette: self.palette.required("BlockStatePalette", offset)?,
        })
    }
}

/// The required header of a region, once all of it has arrived.
pub(crate) struct RegionParts {
    pub(crate) position: BlockPos,
    pub(crate) size: BlockPos,
    pub(crate) palette: Palette,
}

// Region
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// One region of a schematic.
///
/// The block data itself is *not* materialized: palette indices are decoded
/// from the borrowed `BlockStates` payload on demand, either one at a time in
/// storage order through [`Region::iter_blocks`] or at random coordinates
/// through [`Region::block_at`]. Both routes share one bit-stream decoder.
#[derive(Clone)]
pub struct Region<'a> {
    /// Name of the region, unique within a schematic.
    pub name: String,
    /// Position of the region in schematic coordinates.
    pub position: BlockPos,
    /// Size of the region as stored in the file. Components may be negative,
    /// which means the region extends in the negative axis direction; the
    /// box actually addressed is [`Region::dims`].
    pub size: BlockPos,
    dims: BlockSize,
    palette: Palette,
    blocks: BitStream<'a>,
    volume: u64,
}

impl fmt::Debug for Region<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Region")
            .field("name", &self.name)
            .field("position", &self.position)
            .field("size", &self.size)
            .field("palette", &self.palette.states())
            .field("block_states_len", &self.blocks.byte_len())
            .finish()
    }
}

impl<'a> Region<'a> {
    /// Reads a region compound's fields. `name` is the compound's member name
    /// in `Regions`; `region_offset` anchors errors about missing fields.
    pub(crate) fn decode(
        r: &mut Reader<SliceSource<'a>>,
        name: String,
        region_offset: usize,
        enclosing: usize,
    ) -> Result<Self> {
        let depth = r.nested(enclosing)?;
        let mut header = RegionHeader::new();
        let mut payload: Slot<(&'a [u8], usize)> = Slot::default();

        while let Some((field, tag)) = r.next_field()? {
            if let FieldOutcome::Deferred = header.read_field(r, &field, tag, depth)? {
                let offset = r.pos();
                let count = r.count()?;
                // Borrow the payload directly; never copy or expand it.
                payload.set((r.take(count as u64 * 8)?, offset));
            }
        }

        let parts = header.into_parts(region_offset)?;
        let (block_states, states_offset) = payload.required("BlockStates", region_offset)?;
        Self::from_parts(name, parts, block_states, states_offset)
    }

    /// Assembles a region from parsed pieces and validates the block data
    /// against the palette and volume. `offset` is the byte offset of the
    /// region's `BlockStates` payload and is used in error reports.
    fn from_parts(
        name: String,
        parts: RegionParts,
        block_states: &'a [u8],
        offset: usize,
    ) -> Result<Self> {
        let RegionParts {
            position,
            size,
            palette,
        } = parts;
        let geometry = region_geometry(size, &palette, block_states.len() as u64, offset)?;
        Ok(Self {
            name,
            position,
            size,
            dims: geometry.dims,
            palette,
            blocks: BitStream::new(block_states, geometry.bits),
            volume: geometry.volume,
        })
    }

    /// The block state palette, in file order. Palette indices yielded by
    /// [`Region::iter_blocks`] refer to this slice.
    pub fn palette(&self) -> &[BlockState] {
        self.palette.states()
    }

    /// The absolute size per axis, with any negative direction folded away.
    pub fn dims(&self) -> BlockSize {
        self.dims
    }

    /// The number of voxels in this region, `dims().volume()` narrowed to
    /// `u64`.
    ///
    /// [`crate::Document::parse`] rejects regions whose volume does not fit
    /// into a `u64`, so this value is exact for any parsed region.
    pub fn volume(&self) -> u64 {
        self.volume
    }

    /// Bits used per block entry in the `BlockStates` bit stream: the smallest
    /// `n >= 2` such that `2^n >= palette().len()`.
    pub fn bits_per_entry(&self) -> u32 {
        self.palette.bits()
    }

    /// The palette index stored at `pos`, in region-local coordinates, or
    /// `None` when `pos` falls outside [`Region::dims`].
    ///
    /// The cost is one bit-stream decode, independent of the region's volume;
    /// the packed payload is never expanded.
    pub fn index_at(&self, pos: BlockPos) -> Option<u32> {
        let index = self.dims.index_of(pos)?;
        self.blocks.entry(index)
    }

    /// The block state at `pos`, in region-local coordinates, or `None` when
    /// `pos` is outside the region.
    ///
    /// A stored index that does not name a palette entry — possible only for
    /// a malformed file — also yields `None`, so this accessor is total.
    pub fn block_at(&self, pos: BlockPos) -> Option<&BlockState> {
        let index = self.index_at(pos)?;
        self.palette.states().get(index as usize)
    }

    /// Iterates over every voxel of the region in storage order, yielding
    /// palette indices.
    ///
    /// The voxel at local coordinates `(x, y, z)` is entry
    /// `i = (y * sz + z) * sx + x`; see [`BlockSize::index_of`]. The iterator
    /// yields exactly [`Region::volume`] items.
    ///
    /// # Malformed files
    ///
    /// Indices are not validated against the palette length: a hostile or
    /// corrupt file may yield indices `>= palette().len()`. Callers must
    /// bounds-check, e.g. with `palette().get(index as usize)`, or use
    /// [`Region::block_at`], which does it. This iterator never panics on such
    /// input.
    pub fn iter_blocks(&self) -> BlockIter<'_> {
        self.blocks.iter(self.volume)
    }
}

/// The facts both read modes derive before touching block data: the addressed
/// span, the bit width, and the voxel count.
///
/// Validation lives here so the random-access and streaming readers cannot
/// disagree about what a well-formed region is.
pub(crate) struct Geometry {
    pub(crate) dims: BlockSize,
    pub(crate) bits: u32,
    pub(crate) volume: u64,
}

/// Validates a region against its palette and packed payload, and derives the
/// facts every block read needs.
pub(crate) fn region_geometry(
    size: BlockPos,
    palette: &Palette,
    payload_bytes: u64,
    offset: usize,
) -> Result<Geometry> {
    let dims = BlockSize::of(size);
    let volume128 = dims.volume();
    let bits = palette.bits();

    if volume128 > 0 && palette.states().is_empty() {
        return Err(Error::EmptyPalette { offset });
    }
    if (1u64 << bits) < palette.states().len() as u64 {
        return Err(Error::PaletteTooLarge { offset });
    }
    // Checked before the u64 overflow check so that a hostile file with a
    // huge claimed Size is always reported as truncated block data.
    let required = (volume128 * u128::from(bits)).div_ceil(8);
    if u128::from(payload_bytes) < required {
        return Err(Error::TruncatedBlockStates {
            have: payload_bytes,
            need: required,
            offset,
        });
    }
    let volume = u64::try_from(volume128).map_err(|_| Error::VolumeOverflow { offset })?;
    Ok(Geometry { dims, bits, volume })
}

// Palette
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// The palette of one region, in file order, together with the bit width the
/// region's packed data uses.
///
/// The width is derived from the length and stored once, because every block
/// lookup needs it; deriving it per entry would repeat the same four
/// comparisons for the whole volume.
#[derive(Clone)]
pub(crate) struct Palette {
    states: Vec<BlockState>,
    bits: u32,
}

impl<S: Source> Decode<S> for Palette {
    const TAG: Tag = Tag::List;

    /// Reads a `BlockStatePalette`: a list of palette entry compounds.
    fn decode(r: &mut Reader<S>, _field: &'static str, enclosing: usize) -> Result<Self> {
        let elem = r.tag()?;
        r.expect("BlockStatePalette element", elem, Tag::Compound)?;
        let count = r.count()?;
        let entry_depth = r.nested(enclosing)?;
        // Cap the pre-reservation: a hostile count may be huge while the
        // payload is tiny; the payload bounds the real length.
        let mut states = Vec::with_capacity(count.min(256));
        for _ in 0..count {
            states.push(BlockState::decode(r, "palette entry", entry_depth)?);
        }
        Ok(Self::from_states(states))
    }
}

impl Palette {
    /// Wraps the states read from a file, deriving the region's bit width.
    fn from_states(states: Vec<BlockState>) -> Self {
        let bits = bits_for(states.len());
        Self { states, bits }
    }

    /// The states in file order; index 0 is the first entry.
    pub(crate) fn states(&self) -> &[BlockState] {
        &self.states
    }

    /// Bits per `BlockStates` entry.
    pub(crate) fn bits(&self) -> u32 {
        self.bits
    }
}

// Tests
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

#[cfg(test)]
mod tests {
    use super::{BlockPos, BlockSize};

    #[test]
    fn block_size_index_round_trips_through_positions() {
        let size = BlockSize::new(3, 2, 4);
        for (index, pos) in size.cartesian_positions().enumerate() {
            let index = index as u64;
            assert_eq!(size.index_of(pos), Some(index));
            assert_eq!(size.position_at(index), Some(pos));
        }
        assert_eq!(size.index_of(BlockPos::new(3, 0, 0)), None);
        assert_eq!(size.position_at(size.volume() as u64), None);
    }
}
