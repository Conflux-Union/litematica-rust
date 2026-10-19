//! One region of a schematic and the validation its block data needs.
//!
//! A [`Region`] is the shared model of parsed block data: the random-access
//! reader obtains one that borrows the decompressed file, and the streaming
//! reader obtains one that owns its payload. Both answer the same queries and
//! walk the same storage order; only the byte storage differs.

use std::fmt;

use crate::error::{Error, Result};
use crate::knowledge::mc::{BlockPos, BlockSize, BlockState};
use crate::knowledge::nbt::{FieldEntry, FieldOutcome, FieldTable, Reader};
use crate::knowledge::nbt::{SliceSource, Slot, Source, Tag};
use crate::model::bitstream::{BitStream, BlockIter};
use crate::model::palette::{Palette, PaletteId};

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

// Geometry
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// The facts every block read derives from a region's header: the addressed
/// span, the bit width, and the voxel count.
///
/// Validation lives here so the random-access and streaming readers cannot
/// disagree about what a well-formed region is.
pub(crate) struct Geometry {
    pub(crate) dims: BlockSize,
    pub(crate) bits: u32,
    pub(crate) volume: u64,
}

impl Geometry {
    /// Validates a region against its palette and packed payload.
    ///
    /// `payload_bytes` is the byte length of the `BlockStates` array;
    /// `offset` anchors the reported error.
    pub(crate) fn new(
        size: BlockPos,
        palette: &Palette,
        payload_bytes: u64,
        offset: usize,
    ) -> Result<Self> {
        let dims = BlockSize::of(size);
        let volume128 = dims.volume();
        let bits = palette.bits();

        if volume128 > 0 && palette.is_empty() {
            return Err(Error::EmptyPalette { offset });
        }
        if (1u64 << bits) < palette.len() as u64 {
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
        Ok(Self { dims, bits, volume })
    }
}

// Region
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// One region of a schematic.
///
/// The block data itself is never expanded: palette identifiers are decoded
/// from the packed `BlockStates` payload on demand, either one at a time in
/// storage order through [`Region::iter_blocks`] or at random coordinates
/// through [`Region::state_at`]. Both routes share one bit-stream decoder.
///
/// A region either borrows the file's payload (`Document`) or owns it (the
/// streaming reader). The variant is an implementation detail; every query
/// behaves identically.
#[derive(Clone)]
pub struct Region<'a> {
    /// Name of the region, unique within a schematic.
    pub name: String,
    /// Position of the region in schematic coordinates.
    pub position: BlockPos,
    /// Size of the region as declared in the file. Components may be negative,
    /// which means the region extends in the negative axis direction; the box
    /// actually addressed is [`Region::dims`].
    pub declared_size: BlockPos,
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
            .field("declared_size", &self.declared_size)
            .field("palette", &self.palette.states())
            .field("block_states_len", &self.blocks.byte_len())
            .finish()
    }
}

impl<'a> Region<'a> {
    /// Reads a region compound's fields, borrowing its `BlockStates` payload
    /// from the decompressed file. `name` is the compound's member name in
    /// `Regions`; `region_offset` anchors errors about missing fields.
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
        Self::from_borrowed(name, parts, block_states, states_offset)
    }

    /// Assembles a region whose payload is borrowed from the input buffer.
    pub(crate) fn from_borrowed(
        name: String,
        parts: RegionParts,
        block_states: &'a [u8],
        offset: usize,
    ) -> Result<Self> {
        Self::assemble(name, parts, BitStream::borrowed(block_states, 0), offset)
    }

    /// Assembles a region whose payload is owned by the region itself.
    pub(crate) fn from_owned(
        name: String,
        parts: RegionParts,
        block_states: Vec<u8>,
        offset: usize,
    ) -> Result<Self> {
        Self::assemble(name, parts, BitStream::owned(block_states, 0), offset)
    }

    /// Validates the payload against the header and fixes the bit width.
    fn assemble(
        name: String,
        parts: RegionParts,
        blocks: BitStream<'a>,
        offset: usize,
    ) -> Result<Self> {
        let RegionParts {
            position,
            size,
            palette,
        } = parts;
        let geometry = Geometry::new(size, &palette, blocks.byte_len() as u64, offset)?;
        Ok(Self {
            name,
            position,
            declared_size: size,
            dims: geometry.dims,
            palette,
            blocks: blocks.with_bits(geometry.bits),
            volume: geometry.volume,
        })
    }

    /// The block state palette, in file order. Palette identifiers yielded by
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
    /// [`crate::model::document::Document::parse`] rejects regions whose
    /// volume does not fit
    /// into a `u64`, so this value is exact for any parsed region.
    pub fn volume(&self) -> u64 {
        self.volume
    }

    /// Bits used per block entry in the `BlockStates` bit stream: the smallest
    /// `n >= 2` such that `2^n >= palette().len()`.
    pub fn bits_per_entry(&self) -> u32 {
        self.palette.bits()
    }

    /// The palette identifier stored at `pos`, in region-local coordinates, or
    /// `None` when `pos` falls outside [`Region::dims`].
    ///
    /// The cost is one bit-stream decode, independent of the region's volume;
    /// the packed payload is never expanded.
    pub fn palette_id_at(&self, pos: BlockPos) -> Option<PaletteId> {
        let index = self.dims.index_of(pos)?;
        self.blocks.entry(index)
    }

    /// The block state at `pos`, in region-local coordinates, or `None` when
    /// `pos` is outside the region.
    ///
    /// A stored identifier that does not name a palette entry — possible only
    /// for a malformed file — also yields `None`, so this accessor is total.
    pub fn state_at(&self, pos: BlockPos) -> Option<&BlockState> {
        let id = self.palette_id_at(pos)?;
        self.palette.state_at(id)
    }

    /// Iterates over every voxel of the region in storage order, yielding
    /// palette identifiers.
    ///
    /// The voxel at local coordinates `(x, y, z)` is entry
    /// `i = (y * sz + z) * sx + x`; see [`BlockSize::index_of`]. The iterator
    /// yields exactly [`Region::volume`] items.
    ///
    /// # Malformed files
    ///
    /// Identifiers are not validated against the palette length: a hostile or
    /// corrupt file may yield identifiers `>= palette().len()`. Callers must
    /// bounds-check, e.g. with `palette().get(id as usize)`, or use
    /// [`Region::state_at`], which does it. This iterator never panics on such
    /// input.
    pub fn iter_blocks(&self) -> BlockIter<'a> {
        self.blocks.clone().iter(self.volume)
    }

    /// Iterates over every voxel of the region in storage order, yielding
    /// resolved block states.
    ///
    /// Identifiers that do not name a palette entry — possible only for a
    /// malformed file — are skipped, so this iterator is total without
    /// requiring the caller to bounds-check. Use [`Region::iter_blocks`] when
    /// the raw identifiers themselves matter.
    pub fn iter_states(&self) -> impl Iterator<Item = &BlockState> + '_ {
        self.iter_blocks()
            .filter_map(|id| self.palette().get(id as usize))
    }
}
