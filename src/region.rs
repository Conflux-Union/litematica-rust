//! Regions, block state palettes, and the bit-unpacking block iterator.

use std::fmt;
use std::slice::ChunksExact;

use crate::error::{Error, Result};

/// A single block state from a region palette.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockState {
    /// Namespaced block id, e.g. `minecraft:redstone_wire`.
    pub name: String,
    /// Block state properties as `(name, value)` pairs, exactly in the order
    /// they appear in the file.
    pub properties: Vec<(String, String)>,
}

/// Computes the litematica bit width for a palette of `len` entries: the
/// smallest `n >= 2` with `2^n >= len`.
pub(crate) fn num_bits(len: usize) -> u32 {
    let mut n = 2u32;
    while n < 32 && (1u64 << n) < len as u64 {
        n += 1;
    }
    n
}

/// One region of a schematic.
///
/// The block data itself is *not* materialized: [`Region::iter_blocks`]
/// decodes palette indices on the fly directly from the borrowed
/// `BlockStates` payload of the input buffer.
#[derive(Clone)]
pub struct Region<'a> {
    /// Name of the region, unique within a schematic.
    pub name: String,
    /// Position of the region in schematic coordinates.
    pub position: [i32; 3],
    /// Size of the region as stored in the file. Components may be negative,
    /// which means the region extends in the negative axis direction.
    pub size: [i32; 3],
    palette: Vec<BlockState>,
    block_states: &'a [u8],
    volume: u64,
    bits: u32,
}

impl fmt::Debug for Region<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Region")
            .field("name", &self.name)
            .field("position", &self.position)
            .field("size", &self.size)
            .field("palette", &self.palette)
            .field("block_states_len", &self.block_states.len())
            .finish()
    }
}

impl<'a> Region<'a> {
    /// Assembles a region from parsed pieces and validates the block data
    /// against the palette and volume. `offset` is the byte offset of the
    /// region's `BlockStates` payload and is used in error reports.
    pub(crate) fn from_parts(
        name: String,
        position: [i32; 3],
        size: [i32; 3],
        palette: Vec<BlockState>,
        block_states: &'a [u8],
        offset: usize,
    ) -> Result<Self> {
        let dims = [
            size[0].unsigned_abs() as u128,
            size[1].unsigned_abs() as u128,
            size[2].unsigned_abs() as u128,
        ];
        let volume128 = dims[0] * dims[1] * dims[2];

        if volume128 > 0 && palette.is_empty() {
            return Err(Error::EmptyPalette { offset });
        }
        let bits = num_bits(palette.len());
        if (1u64 << bits) < palette.len() as u64 {
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
            palette,
            block_states,
            volume,
            bits,
        })
    }

    /// The block state palette, in file order. Palette indices yielded by
    /// [`Region::iter_blocks`] refer to this slice.
    pub fn palette(&self) -> &[BlockState] {
        &self.palette
    }

    /// The absolute size per axis: `[|size.x|, |size.y|, |size.z|]`.
    pub fn dims(&self) -> [u64; 3] {
        [
            self.size[0].unsigned_abs() as u64,
            self.size[1].unsigned_abs() as u64,
            self.size[2].unsigned_abs() as u64,
        ]
    }

    /// The number of voxels in this region, `dims()[0] * dims()[1] * dims()[2]`.
    ///
    /// [`crate::Document::parse`] rejects regions whose volume does not fit
    /// into a `u64`, so this value is exact for any parsed region.
    pub fn volume(&self) -> u64 {
        self.volume
    }

    /// Bits used per block entry in the `BlockStates` bit stream: the smallest
    /// `n >= 2` such that `2^n >= palette().len()`.
    pub fn bits_per_entry(&self) -> u32 {
        self.bits
    }

    /// Iterates over every voxel of the region in storage order, yielding
    /// palette indices.
    ///
    /// The voxel at local coordinates `(x, y, z)` is entry
    /// `i = y * (sx * sz) + z * sx + x` where `sx`/`sz` are the absolute sizes
    /// of the x/z axes; that is, `x` varies fastest and `y` slowest. The
    /// iterator yields exactly [`Region::volume`] items.
    ///
    /// # Malformed files
    ///
    /// Indices are not validated against the palette length: a hostile or
    /// corrupt file may yield indices `>= palette().len()`. Callers must
    /// bounds-check, e.g. with `palette().get(index as usize)`. This iterator
    /// never panics on such input.
    pub fn iter_blocks(&self) -> BlockIter<'_> {
        BlockIter::new(self.block_states, self.bits, self.volume)
    }
}

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
    longs: ChunksExact<'a, u8>,
    cur: u64,
    cur_bits: u32,
    mask: u64,
    bits: u32,
    remaining: u64,
}

impl<'a> BlockIter<'a> {
    pub(crate) fn new(bytes: &'a [u8], bits: u32, volume: u64) -> Self {
        Self {
            longs: bytes.chunks_exact(8),
            cur: 0,
            cur_bits: 0,
            mask: (1u64 << bits) - 1,
            bits,
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

        let bits = self.bits;
        let value = if self.cur_bits >= bits {
            // Fast path: enough buffered bits for this entry.
            let v = (self.cur & self.mask) as u32;
            self.cur >>= bits;
            self.cur_bits -= bits;
            v
        } else {
            // Slow path: refill from the next long; the entry straddles the
            // long boundary, low bits from the old long, high from the new.
            let low = self.cur_bits;
            let mut v = self.cur & self.mask;
            // After validation a refill is always available; map_or(0) keeps
            // this panic-free regardless.
            self.cur = self
                .longs
                .next()
                .map_or(0, |c| u64::from_be_bytes(c.try_into().unwrap_or_default()));
            self.cur_bits = 64;
            v |= (self.cur & (self.mask >> low)) << low;
            self.cur >>= bits - low;
            self.cur_bits -= bits - low;
            v as u32
        };
        Some(value)
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

#[cfg(test)]
mod tests {
    use super::{num_bits, BlockIter};

    #[test]
    fn num_bits_matches_litematica() {
        assert_eq!(num_bits(1), 2);
        assert_eq!(num_bits(2), 2);
        assert_eq!(num_bits(3), 2);
        assert_eq!(num_bits(4), 2);
        assert_eq!(num_bits(5), 3);
        assert_eq!(num_bits(8), 3);
        assert_eq!(num_bits(9), 4);
        assert_eq!(num_bits(16), 4);
        assert_eq!(num_bits(17), 5);
        assert_eq!(num_bits(64), 6);
        assert_eq!(num_bits(65), 7);
        assert_eq!(num_bits(100), 7);
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
        let iter = BlockIter::new(&bytes, 3, values.len() as u64);
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
        let decoded: Vec<u32> = BlockIter::new(&longs, 5, values.len() as u64).collect();
        assert_eq!(decoded, values);
    }

    #[test]
    fn iter_zero_volume_is_empty() {
        let mut iter = BlockIter::new(&[], 2, 0);
        assert_eq!(iter.len(), 0);
        assert!(iter.next().is_none());
    }
}
