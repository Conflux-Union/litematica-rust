//! Regions, block state palettes, and the bit-unpacking block iterator.

use std::fmt;

use crate::error::{Error, Result};
use crate::mc::{BlockPos, BlockSize};
use crate::nbt::{Decode, Reader, Tag};

// Block state
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// A single block state from a region palette.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockState {
    /// Namespaced block id, e.g. `minecraft:redstone_wire`.
    pub name: String,
    /// Block state properties as `(name, value)` pairs, exactly in the order
    /// they appear in the file.
    pub properties: Vec<(String, String)>,
}

impl<'a> Decode<'a> for BlockState {
    /// Reads one palette entry: a compound with a `Name` and an optional
    /// `Properties` compound.
    fn decode(r: &mut Reader<'a>, _field: &'static str, enclosing: usize) -> Result<Self> {
        let depth = r.nested(enclosing)?;
        let entry_offset = r.pos();
        let mut name: Option<String> = None;
        let mut properties: Vec<(String, String)> = Vec::new();
        loop {
            let tag = r.tag()?;
            if tag == Tag::End {
                break;
            }
            let field = r.string()?;
            match field.as_str() {
                "Name" => {
                    r.expect("palette entry Name", tag, Tag::String)?;
                    name = Some(r.string()?);
                }
                "Properties" => properties = Self::decode_properties(r, tag)?,
                _ => r.skip_value(tag, depth)?,
            }
        }
        let name = name.ok_or(Error::MissingField {
            field: "palette entry Name",
            offset: entry_offset,
        })?;
        Ok(Self { name, properties })
    }
}

impl BlockState {
    /// Reads a `Properties` compound: a map from property name to string value.
    fn decode_properties(r: &mut Reader<'_>, tag: Tag) -> Result<Vec<(String, String)>> {
        r.expect("palette entry Properties", tag, Tag::Compound)?;
        let mut properties = Vec::new();
        loop {
            let tag = r.tag()?;
            if tag == Tag::End {
                break;
            }
            let key = r.string()?;
            r.expect("palette property value", tag, Tag::String)?;
            properties.push((key, r.string()?));
        }
        Ok(properties)
    }
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
        r: &mut Reader<'a>,
        name: String,
        region_offset: usize,
        enclosing: usize,
    ) -> Result<Self> {
        let depth = r.nested(enclosing)?;
        let mut position: Option<BlockPos> = None;
        let mut size: Option<BlockPos> = None;
        let mut palette: Option<Palette> = None;
        let mut states: Option<&[u8]> = None;
        let mut states_offset = 0usize;

        loop {
            let tag = r.tag()?;
            if tag == Tag::End {
                break;
            }
            let field = r.string()?;
            match field.as_str() {
                "Position" => {
                    r.expect("Position", tag, Tag::Compound)?;
                    position = Some(BlockPos::decode(r, "Position", depth)?);
                }
                "Size" => {
                    r.expect("Size", tag, Tag::Compound)?;
                    size = Some(BlockPos::decode(r, "Size", depth)?);
                }
                "BlockStatePalette" => {
                    r.expect("BlockStatePalette", tag, Tag::List)?;
                    palette = Some(Palette::decode(r, "BlockStatePalette", depth)?);
                }
                "BlockStates" => {
                    r.expect("BlockStates", tag, Tag::LongArray)?;
                    states_offset = r.pos();
                    let count = r.count()?;
                    // Borrow the payload directly; never copy or expand it.
                    states = Some(r.take(count as u64 * 8)?);
                }
                _ => r.skip_value(tag, depth)?,
            }
        }

        let position = position.ok_or(Error::MissingField {
            field: "Position",
            offset: region_offset,
        })?;
        let size = size.ok_or(Error::MissingField {
            field: "Size",
            offset: region_offset,
        })?;
        let palette = palette.ok_or(Error::MissingField {
            field: "BlockStatePalette",
            offset: region_offset,
        })?;
        let states = states.ok_or(Error::MissingField {
            field: "BlockStates",
            offset: region_offset,
        })?;
        Self::from_parts(name, position, size, palette, states, states_offset)
    }

    /// Assembles a region from parsed pieces and validates the block data
    /// against the palette and volume. `offset` is the byte offset of the
    /// region's `BlockStates` payload and is used in error reports.
    fn from_parts(
        name: String,
        position: BlockPos,
        size: BlockPos,
        palette: Palette,
        block_states: &'a [u8],
        offset: usize,
    ) -> Result<Self> {
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
        if (block_states.len() as u128) < required {
            return Err(Error::TruncatedBlockStates {
                have: block_states.len() as u64,
                need: required,
                offset,
            });
        }
        let volume = u64::try_from(volume128).map_err(|_| Error::VolumeOverflow { offset })?;

        Ok(Self {
            name,
            position,
            size,
            dims,
            palette,
            blocks: BitStream::new(block_states, bits),
            volume,
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
        BlockIter::new(self.blocks, self.volume)
    }
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
struct Palette {
    states: Vec<BlockState>,
    bits: u32,
}

impl<'a> Decode<'a> for Palette {
    /// Reads a `BlockStatePalette`: a list of palette entry compounds.
    fn decode(r: &mut Reader<'a>, _field: &'static str, enclosing: usize) -> Result<Self> {
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
        let bits = Self::bits_for(states.len());
        Self { states, bits }
    }

    /// The litematica bit width for `len` palette entries: the smallest
    /// `n >= 2` with `2^n >= len`.
    fn bits_for(len: usize) -> u32 {
        (2..32).find(|n| (1u64 << n) >= len as u64).unwrap_or(32)
    }

    /// The states in file order; index 0 is the first entry.
    fn states(&self) -> &[BlockState] {
        &self.states
    }

    /// Bits per `BlockStates` entry.
    fn bits(&self) -> u32 {
        self.bits
    }
}

// Bit stream
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// The packed `BlockStates` payload viewed as fixed-width entries.
///
/// The payload is a run of big-endian longs, but entries are not aligned to
/// them: entry `i` starts at bit `i * bits` of one continuous stream, least
/// significant bit first, and may straddle two longs. [`BitCursor`] performs
/// that decode for both sequential and random access, so the layout of the
/// stream is stated once.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BitStream<'a> {
    longs: &'a [[u8; 8]],
    bits: u32,
    mask: u64,
}

impl<'a> BitStream<'a> {
    pub(crate) fn new(bytes: &'a [u8], bits: u32) -> Self {
        // A long array payload is a whole number of 8-byte entries, so the
        // remainder is empty for well-formed input and ignored otherwise.
        let (longs, _remainder) = bytes.as_chunks::<8>();
        Self {
            longs,
            bits,
            mask: (1u64 << bits) - 1,
        }
    }

    /// Entry `index`, or `None` when the payload does not reach it.
    fn entry(self, index: u64) -> Option<u32> {
        let bit = u128::from(index) * u128::from(self.bits);
        let long = usize::try_from(bit / 64).ok()?;
        let shift = (bit % 64) as u32;
        let first = u64::from_be_bytes(*self.longs.get(long)?);
        let mut cursor = BitCursor {
            stream: self,
            next_long: long + 1,
            cur: first >> shift,
            cur_bits: 64 - shift,
        };
        Some(cursor.next())
    }

    /// The `index`th long, zero-filled when the payload is shorter.
    fn long(self, index: usize) -> u64 {
        self.longs
            .get(index)
            .copied()
            .map(u64::from_be_bytes)
            .unwrap_or(0)
    }

    /// Number of payload bytes this stream windows, longs only.
    fn byte_len(self) -> usize {
        self.longs.len() * 8
    }
}

/// A forward cursor over a [`BitStream`], buffering one long at a time.
///
/// Sequential iteration is the cursor's natural use: each step consumes the
/// low `bits` of the buffer and refills only when the next entry crosses a
/// long boundary.
#[derive(Debug)]
struct BitCursor<'a> {
    stream: BitStream<'a>,
    next_long: usize,
    cur: u64,
    cur_bits: u32,
}

impl<'a> BitCursor<'a> {
    fn new(stream: BitStream<'a>) -> Self {
        Self {
            stream,
            next_long: 1,
            cur: stream.long(0),
            cur_bits: 64,
        }
    }

    #[inline]
    fn next(&mut self) -> u32 {
        if self.cur_bits >= self.stream.bits {
            // Fast path: enough buffered bits for this entry.
            let value = (self.cur & self.stream.mask) as u32;
            self.cur >>= self.stream.bits;
            self.cur_bits -= self.stream.bits;
            return value;
        }
        // Slow path: the entry straddles the long boundary, low bits from
        // the buffer, the rest from the next long.
        let low = self.cur_bits;
        let mut value = self.cur & self.stream.mask;
        self.cur = self.stream.long(self.next_long);
        self.next_long += 1;
        self.cur_bits = 64;
        value |= (self.cur & (self.stream.mask >> low)) << low;
        self.cur >>= self.stream.bits - low;
        self.cur_bits -= self.stream.bits - low;
        value as u32
    }
}

// Block iterator
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// An [`Iterator`] over the palette indices of a region, decoded on the fly
/// from the `BlockStates` bit stream.
///
/// The whole long array is treated as one continuous bit stream (entries may
/// cross long boundaries). Within a big-endian long, bit 0 is the least
/// significant bit; the first bit in the stream is the lowest bit of the
/// first entry. This matches the packing used by the Litematica mod and
/// rustmatica.
#[derive(Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct BlockIter<'a> {
    cursor: BitCursor<'a>,
    remaining: u64,
}

impl<'a> BlockIter<'a> {
    pub(crate) fn new(stream: BitStream<'a>, volume: u64) -> Self {
        Self {
            cursor: BitCursor::new(stream),
            remaining: volume,
        }
    }
}

impl Iterator for BlockIter<'_> {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<u32> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        Some(self.cursor.next())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.len();
        (n, Some(n))
    }
}

impl ExactSizeIterator for BlockIter<'_> {
    #[inline]
    fn len(&self) -> usize {
        usize::try_from(self.remaining).unwrap_or(usize::MAX)
    }
}

// Tests
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

#[cfg(test)]
mod tests {
    use super::{BitStream, BlockIter, BlockPos, BlockSize, Palette};

    #[test]
    fn bits_for_matches_litematica() {
        assert_eq!(Palette::bits_for(1), 2);
        assert_eq!(Palette::bits_for(2), 2);
        assert_eq!(Palette::bits_for(3), 2);
        assert_eq!(Palette::bits_for(4), 2);
        assert_eq!(Palette::bits_for(5), 3);
        assert_eq!(Palette::bits_for(8), 3);
        assert_eq!(Palette::bits_for(9), 4);
        assert_eq!(Palette::bits_for(16), 4);
        assert_eq!(Palette::bits_for(17), 5);
        assert_eq!(Palette::bits_for(64), 6);
        assert_eq!(Palette::bits_for(65), 7);
        assert_eq!(Palette::bits_for(100), 7);
    }

    #[test]
    fn iter_decodes_lsb_first_stream() {
        // 3 bits per entry, values 1, 2, 6, 5, 7 -> bits cross long boundary.
        let values = [1u32, 2, 6, 5, 7];
        let mut acc = 0u64;
        let mut filled = 0u32;
        for &v in &values {
            for k in 0..3 {
                acc |= u64::from((v >> k) & 1) << filled;
                filled += 1;
            }
        }
        let bytes = acc.to_be_bytes();
        let iter = BlockIter::new(BitStream::new(&bytes, 3), values.len() as u64);
        assert_eq!(iter.len(), values.len());
        let decoded: Vec<u32> = iter.collect();
        assert_eq!(decoded, values);
    }

    #[test]
    fn iter_entries_crossing_long_boundaries() {
        // 5 bits per entry, 20 entries -> 100 bits, crossing 64-bit borders.
        let values: Vec<u32> = (0..20).map(|i| (i * 7 + 3) % 32).collect();
        let mut longs = Vec::new();
        let mut acc = 0u64;
        let mut filled = 0u32;
        for &v in &values {
            for k in 0..5 {
                acc |= u64::from((v >> k) & 1) << filled;
                filled += 1;
                if filled == 64 {
                    longs.extend_from_slice(&acc.to_be_bytes());
                    acc = 0;
                    filled = 0;
                }
            }
        }
        if filled > 0 {
            longs.extend_from_slice(&acc.to_be_bytes());
        }
        let decoded: Vec<u32> =
            BlockIter::new(BitStream::new(&longs, 5), values.len() as u64).collect();
        assert_eq!(decoded, values);
    }

    #[test]
    fn random_entries_match_the_sequential_stream() {
        // 5 bits, entries straddling several long boundaries.
        let values: Vec<u32> = (0..20).map(|i| (i * 7 + 3) % 32).collect();
        let mut longs = Vec::new();
        let mut acc = 0u64;
        let mut filled = 0u32;
        for &v in &values {
            for k in 0..5 {
                acc |= u64::from((v >> k) & 1) << filled;
                filled += 1;
                if filled == 64 {
                    longs.extend_from_slice(&acc.to_be_bytes());
                    acc = 0;
                    filled = 0;
                }
            }
        }
        if filled > 0 {
            longs.extend_from_slice(&acc.to_be_bytes());
        }
        let stream = BitStream::new(&longs, 5);
        for (i, &want) in values.iter().enumerate() {
            assert_eq!(stream.entry(i as u64), Some(want), "entry {i}");
        }
        assert_eq!(stream.entry(100), None);
    }

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

    #[test]
    fn iter_zero_volume_is_empty() {
        let mut iter = BlockIter::new(BitStream::new(&[], 2), 0);
        assert_eq!(iter.len(), 0);
        assert!(iter.next().is_none());
    }
}
