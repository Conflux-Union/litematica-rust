//! Streaming, low-memory parser for Minecraft `.litematic` schematic files.
//!
//! Tree-based parsers (such as rustmatica) materialize the full NBT structure
//! and then expand the packed block data, which peaks at ~15 GiB of RSS for a
//! large schematic (e.g. a 566×256×1423 bridge with 206 million voxels). This
//! crate takes a different route:
//!
//! - the gzip member is decompressed once into a bounded buffer
//!   ([`decompress_gzip_with_limit`]);
//! - the NBT payload is scanned by a hand-written forward-only cursor that
//!   never builds a tree ([`Document::parse`]);
//! - block state palettes are the only block data that is materialized;
//!   the packed `BlockStates` payload is borrowed from the input buffer and
//!   decoded on demand, either voxel by voxel by [`Region::iter_blocks`] or
//!   one entry at a time by [`Region::state_at`].
//!
//! Two read modes share that decoder:
//!
//! - [`Document`] decompresses the whole file once and gives random access to
//!   every region, including [`Region::state_at`] in O(1). Memory is the
//!   decompressed payload plus the palettes.
//! - [`StreamReader`] keeps memory bounded by the gzip buffer, the current
//!   palette, and one region's packed payload, independent of the schematic's
//!   volume. NBT field order does not matter: a region's payload is held only
//!   until its other fields have been read, then decoded voxel by voxel and
//!   released. `Metadata` may appear before or after `Regions`, so
//!   [`StreamReader::metadata`] returns `None` until it has been scanned.
//!
//! Unknown fields (`Entities`, `TileEntities`, pending ticks, preview image
//! data, ...) are skipped in O(length) without allocation, so parsing cost is
//! a single linear scan of the decompressed bytes plus lazy iteration.
//!
//! The crate also writes schematics with the same discipline: a
//! [`write::Layout`] answers point queries (`size`, `block_at`), and
//! [`write::write()`] serializes it
//! as a single-region file straight into any [`std::io::Write`] (file or
//! network stream), with memory bounded by the palette rather than the
//! volume — writing a 566x256x1423 layout (206 million voxels) peaks at a
//! few MiB of RSS.
//!
//! Every parse error reports the byte offset at which it was detected, and
//! hostile inputs (huge claimed sizes, negative lengths, over-deep nesting)
//! are rejected instead of being trusted.
//!
//! # Example
//!
//! ```no_run
//! use litematica::{decompress_gzip_with_limit, Document};
//!
//! # fn main() -> Result<(), litematica::Error> {
//! let raw = std::fs::read("bridge.litematic")?;
//! let nbt = decompress_gzip_with_limit(&raw, 512 * 1024 * 1024)?;
//! let doc = Document::parse(&nbt)?;
//!
//! println!("{} region(s), {} voxels total", doc.regions.len(), doc.total_volume());
//! for region in &doc.regions {
//!     for index in region.iter_blocks() {
//!         // Indices are not validated against the palette; bounds-check.
//!         if let Some(state) = region.palette().get(index as usize) {
//!             // inspect `state.name` and `state.properties`
//!         }
//!     }
//! }
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]
#![forbid(unsafe_code)]

mod bitstream;
mod document;
mod error;
mod nbt;
mod palette;
mod parse;
mod region;
mod schema;
mod stream;
use flate2::read::GzDecoder;
use std::io::Read;

pub mod mc;
pub mod write;
pub use bitstream::BlockIter;
pub use document::{Document, Metadata};
pub use error::{Error, Result};
pub use flate2::Compression;
pub use mc::{BlockPos, BlockSize, BlockState};
pub use palette::{Palette, PaletteId};
pub use region::Region;
pub use stream::StreamReader;

/// Decompresses a single gzip member with a hard cap on the output size.
///
/// The check happens while decompressing: as soon as the cumulative output
/// would exceed `limit` bytes, an [`Error::SizeLimitExceeded`] is returned and
/// no further data is inflated. This bounds the memory an untrusted file can
/// force the caller to allocate. A corrupt or truncated gzip stream surfaces
/// as [`Error::CorruptGzip`] wrapping the underlying I/O error.
///
/// Trailing bytes after the first gzip member are ignored.
pub fn decompress_gzip_with_limit(input: &[u8], limit: u64) -> Result<Vec<u8>> {
    // `Take` stops inflating as soon as one byte past the cap arrives, so the
    // bound is enforced during decompression rather than after it.
    let mut decoder = GzDecoder::new(input).take(limit.saturating_add(1));
    let mut out = Vec::new();
    decoder.read_to_end(&mut out)?;
    if out.len() as u64 > limit {
        return Err(Error::SizeLimitExceeded { limit });
    }
    Ok(out)
}
