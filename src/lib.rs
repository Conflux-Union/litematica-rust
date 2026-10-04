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
//!   decoded voxel by voxel by [`Region::iter_blocks`].
//!
//! Unknown fields (`Entities`, `TileEntities`, pending ticks, preview image
//! data, ...) are skipped in O(length) without allocation, so parsing cost is
//! a single linear scan of the decompressed bytes plus lazy iteration.
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

mod error;
mod nbt;
mod parse;
mod region;

pub use error::{Error, Result};
pub use region::{BlockIter, BlockState, Region};

use std::io::Read;

use flate2::read::GzDecoder;

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
    const CHUNK: usize = 64 * 1024;
    let mut out = Vec::new();
    let mut decoder = GzDecoder::new(input);
    let mut chunk = vec![0u8; CHUNK];
    loop {
        let n = decoder.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        if out.len() as u64 + n as u64 > limit {
            return Err(Error::SizeLimitExceeded { limit });
        }
        out.extend_from_slice(&chunk[..n]);
    }
    Ok(out)
}

/// Metadata of a litematic schematic, as declared by the file.
///
/// Optional fields are `None` when absent; string fields default to empty and
/// integer fields to `0`, so a stripped-down file still parses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Metadata {
    /// Schematic name (`Metadata.Name`).
    pub name: String,
    /// Free-form description (`Metadata.Description`).
    pub description: String,
    /// Author (`Metadata.Author`).
    pub author: String,
    /// Minecraft data version (`MinecraftDataVersion`), e.g. 4189 for 1.21.4.
    pub minecraft_data_version: i32,
    /// Litematica format version (`Version`), 6 for current files.
    pub version: i32,
    /// Optional litematica format sub-version (`SubVersion`).
    pub sub_version: Option<i32>,
    /// Region count claimed by the file (`Metadata.RegionCount`).
    pub region_count: Option<i32>,
    /// Total volume claimed by the file (`Metadata.TotalVolume`).
    pub total_volume: Option<i64>,
    /// Non-air block count claimed by the file (`Metadata.TotalBlocks`).
    pub total_blocks: Option<i64>,
    /// Creation time in milliseconds since the Unix epoch
    /// (`Metadata.TimeCreated`).
    pub time_created: i64,
    /// Last-modified time in milliseconds since the Unix epoch
    /// (`Metadata.TimeModified`).
    pub time_modified: i64,
    /// Enclosing size claimed by the file (`Metadata.EnclosingSize`).
    pub enclosing_size: Option<[i32; 3]>,
}

/// A parsed litematic document.
///
/// Regions borrow their packed block data from the buffer that
/// [`Document::parse`] was given; see the method documentation for details.
#[derive(Debug)]
pub struct Document<'a> {
    /// The schematic metadata.
    pub metadata: Metadata,
    /// The regions of the schematic, in file order.
    pub regions: Vec<Region<'a>>,
}

impl Document<'_> {
    /// Total voxel count across all regions (sum of [`Region::volume`]).
    ///
    /// Each region's volume is validated during parsing to fit a `u64`; the
    /// sum saturates rather than wrapping, which in practice can only happen
    /// for inputs far beyond any real file.
    pub fn total_volume(&self) -> u64 {
        self.regions
            .iter()
            .map(Region::volume)
            .fold(0u64, u64::saturating_add)
    }

    /// Size of the axis-aligned box enclosing all regions, per axis.
    ///
    /// This replicates the Litematica mod / rustmatica formula: for each
    /// region and axis, `mn = min(pos, pos + size + 1)` and
    /// `mx = max(pos, pos + size - 1)`; the document bounds start at the
    /// origin and fold every region's `mn`/`mx` in, and each axis reports
    /// `|mx - mn + 1|`.
    ///
    /// Returns `None` if the document has no regions.
    pub fn enclosing_size(&self) -> Option<[i32; 3]> {
        if self.regions.is_empty() {
            return None;
        }
        let mut lo = [0i64; 3];
        let mut hi = [0i64; 3];
        for region in &self.regions {
            for axis in 0..3 {
                let p = i64::from(region.position[axis]);
                let s = i64::from(region.size[axis]);
                lo[axis] = lo[axis].min(p.min(p + s + 1));
                hi[axis] = hi[axis].max(p.max(p + s - 1));
            }
        }
        let extent = |axis: usize| {
            (hi[axis] - lo[axis] + 1).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
        };
        Some([extent(0), extent(1), extent(2)])
    }
}
