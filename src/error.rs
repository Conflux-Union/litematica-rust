//! Error types for this crate.

use crate::knowledge::mc::{BlockPos, BlockState};
use std::fmt;
use std::io;

/// Convenient result alias used throughout this crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// All errors produced by this crate.
///
/// Every variant that describes a problem inside the decompressed NBT payload
/// carries the byte `offset` at which the problem was detected, so malformed
/// or hostile files can be diagnosed precisely.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Decompressing the gzip stream produced more than the configured number
    /// of bytes. Detection happens while decompressing, not after.
    #[error("decompressed output exceeds limit of {limit} bytes")]
    DecompressionLimitExceeded {
        /// The limit that was passed by the caller.
        limit: u64,
    },

    /// The gzip stream is corrupt or truncated. The underlying I/O error is
    /// preserved as the source.
    #[error("corrupt gzip stream: {source}")]
    CorruptGzip {
        /// The wrapped I/O error.
        #[from]
        source: io::Error,
    },

    /// The NBT payload ended before a value was complete.
    #[error("unexpected end of NBT payload at byte offset {offset}")]
    UnexpectedEof {
        /// Byte offset where more data was required.
        offset: usize,
    },

    /// A tag id outside the valid range 0..=12 was encountered.
    #[error("invalid NBT tag id {id} at byte offset {offset}")]
    InvalidTagId {
        /// The invalid tag id.
        id: u8,
        /// Byte offset of the tag id.
        offset: usize,
    },

    /// A signed i32 length or element count was negative.
    #[error("negative NBT length {length} at byte offset {offset}")]
    NegativeLength {
        /// The negative length that was read.
        length: i32,
        /// Byte offset of the length field.
        offset: usize,
    },

    /// NBT nesting exceeded the maximum depth of 64 levels.
    #[error("NBT nesting exceeds maximum depth of 64 at byte offset {offset}")]
    DepthLimitExceeded {
        /// Byte offset at which the nesting limit was hit.
        offset: usize,
    },

    /// A field required by the litematic schema is missing.
    #[error("missing required field `{field}` at byte offset {offset}")]
    MissingField {
        /// Name of the missing field.
        field: &'static str,
        /// Byte offset of the enclosing structure.
        offset: usize,
    },

    /// A field has an NBT type other than the one required by the schema.
    #[error("field `{field}` has unexpected NBT tag id {tag} at byte offset {offset}")]
    UnexpectedTag {
        /// Name of the offending field.
        field: &'static str,
        /// The tag id that was found.
        tag: u8,
        /// Byte offset of the field.
        offset: usize,
    },

    /// The `BlockStates` payload is shorter than required by the region volume
    /// and palette width.
    #[error(
        "BlockStates payload of {have} bytes is truncated, {need} bytes required \
         at byte offset {offset}"
    )]
    TruncatedBlockStates {
        /// Payload bytes actually present.
        have: u64,
        /// Payload bytes required by volume and bits per entry.
        need: u128,
        /// Byte offset of the payload start.
        offset: usize,
    },

    /// A region with non-zero volume has an empty block state palette, so no
    /// block index can ever be valid.
    #[error("region with non-zero volume has an empty palette at byte offset {offset}")]
    EmptyPalette {
        /// Byte offset of the region's `BlockStates` payload.
        offset: usize,
    },

    /// The region volume does not fit into a `u64`.
    #[error("region volume does not fit in u64 at byte offset {offset}")]
    VolumeOverflow {
        /// Byte offset of the region's `BlockStates` payload.
        offset: usize,
    },

    /// Serializing the schematic to the output stream failed. The underlying
    /// I/O error is preserved as the source.
    #[error("writing schematic output failed: {source}")]
    WriteIo {
        /// The wrapped I/O error.
        source: io::Error,
    },

    /// A `Layout::block_at` result is not covered by the layout's declared
    /// palette.
    #[error(
        "block state `{}` at {:?} is not present in the declared palette",
        .state.name,
        .pos
    )]
    BlockNotInPalette {
        /// The offending block state.
        state: BlockState,
        /// The position that produced it.
        pos: BlockPos,
    },

    /// A layout volume or a file quantity exceeds what a litematic file can
    /// store.
    #[error("layout {quantity} {value} exceeds the i32 limit of the litematic format")]
    DimensionOverflow {
        /// Which quantity overflowed.
        quantity: Quantity,
        /// The overflowing value.
        value: u128,
    },

    /// A string value exceeds the u16 byte-length limit of the NBT format.
    #[error("string of {len} bytes exceeds the NBT u16 length limit")]
    StringTooLong {
        /// The string length in bytes.
        len: usize,
    },

    /// The palette is so large that entries would need more than 32 bits,
    /// which cannot be represented by the `u32` indices this crate yields.
    #[error("block state palette requires more than 32 bits per entry at byte offset {offset}")]
    PaletteTooLarge {
        /// Byte offset of the palette list.
        offset: usize,
    },

    /// A region's packed payload is too long to address on this platform.
    #[error(
        "BlockStates payload of {longs} longs exceeds the platform address space \
         at byte offset {offset}"
    )]
    PayloadTooLarge {
        /// The declared number of longs.
        longs: usize,
        /// Byte offset of the payload length.
        offset: usize,
    },
}

/// A quantity that failed to fit the `i32` the litematic format stores.
///
/// Naming the quantity here keeps callers from matching on free-form strings
/// and gives every overflow report the same wording.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Quantity {
    /// One axis of a region size.
    SizeAxis,
    /// The number of palette entries.
    PaletteLength,
    /// The voxel count of a region.
    Volume,
    /// The number of longs in a packed `BlockStates` array.
    BlockStatesLength,
    /// The non-air block count written into the metadata.
    TotalBlocks,
}

impl fmt::Display for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::SizeAxis => "size axis",
            Self::PaletteLength => "BlockStatePalette length",
            Self::Volume => "volume",
            Self::BlockStatesLength => "BlockStates length",
            Self::TotalBlocks => "TotalBlocks",
        };
        f.write_str(name)
    }
}
